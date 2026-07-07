"""Model definition for ERG-TBNet."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .config import ERGTBNetConfig
from .geometry import box_cxcywh, cxcywh_to_xyxy, spatial_constraint_vector, union_boxes


class MLP(nn.Module):
    """A compact feed-forward block with LayerNorm for stable relation learning."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ERGTBNet(nn.Module):
    """Equipment-Relation Guided Teaching Behavior Network.

    The model consumes relation candidates built from person, equipment, and
    workstation detections. Each candidate carries instance-level features and
    normalized boxes, which allows this implementation to remain detector-agnostic.
    """

    def __init__(self, config: ERGTBNetConfig) -> None:
        super().__init__()
        self.config = config
        d = config.feature_dim
        h = config.hidden_dim
        s = config.spatial_dim

        self.person_encoder = MLP(d, h, h, dropout=config.dropout)
        self.equipment_encoder = MLP(d, h, h, dropout=config.dropout)
        self.workstation_encoder = MLP(d, h, h, dropout=config.dropout)

        self.relation_unit = MLP(h * 3 + s, h, h, dropout=config.dropout)
        self.relation_prior_head = nn.Linear(h, 1)

        self.operation_mlp = MLP(h * 2 + s, h, h, dropout=config.dropout)
        self.operation_head = nn.Linear(h, 1)
        self.confusion_mlp = MLP(h * 2 + s, h, h, dropout=config.dropout)
        self.confusion_head = nn.Linear(h, 1)

        self.joint_feature = MLP(h * 2 + s, h, h, dropout=config.dropout)
        self.temporal_logits = nn.Parameter(torch.zeros(config.temporal_window))
        self.temporal_norm = nn.LayerNorm(h)
        self.class_head = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, config.behavior_classes))
        self.box_head = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, 4))

    @staticmethod
    def _flatten_time_relations(x: torch.Tensor) -> torch.Tensor:
        """Flatten batch, time, and relation dimensions before MLP encoding."""
        return x.reshape(-1, x.shape[-1])

    @staticmethod
    def _restore_time_relations(x: torch.Tensor, batch: int, time: int, relations: int) -> torch.Tensor:
        """Restore batch, time, and relation dimensions after MLP encoding."""
        return x.reshape(batch, time, relations, -1)

    def _encode_entities(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Encode person, equipment, and workstation features independently."""
        person_features = batch["person_features"]
        batch_size, time_steps, relations, _ = person_features.shape
        p = self._restore_time_relations(
            self.person_encoder(self._flatten_time_relations(batch["person_features"])),
            batch_size,
            time_steps,
            relations,
        )
        e = self._restore_time_relations(
            self.equipment_encoder(self._flatten_time_relations(batch["equipment_features"])),
            batch_size,
            time_steps,
            relations,
        )
        w = self._restore_time_relations(
            self.workstation_encoder(self._flatten_time_relations(batch["workstation_features"])),
            batch_size,
            time_steps,
            relations,
        )
        return {"person": p, "equipment": e, "workstation": w}

    def _temporal_aggregate(self, gated_features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Aggregate gated relation evidence within a short causal temporal window."""
        batch_size, time_steps, relations, hidden = gated_features.shape
        k = min(self.config.temporal_window, time_steps)
        weights = torch.softmax(self.temporal_logits[:k], dim=0)
        memory = gated_features.new_zeros(batch_size, time_steps, relations, hidden)
        weight_sum = gated_features.new_zeros(time_steps)
        for offset in range(k):
            if offset == 0:
                memory = memory + weights[offset] * gated_features
                weight_sum = weight_sum + weights[offset]
            else:
                memory[:, offset:] = memory[:, offset:] + weights[offset] * gated_features[:, :-offset]
                weight_sum[offset:] = weight_sum[offset:] + weights[offset]
        memory = memory / weight_sum.clamp(min=1.0e-6).view(1, time_steps, 1, 1)
        return self.temporal_norm(memory) * mask.unsqueeze(-1)

    def _predict_boxes(self, memory: torch.Tensor, person_boxes: torch.Tensor, equipment_boxes: torch.Tensor) -> torch.Tensor:
        """Predict behavior-event boxes as residual refinements over person-equipment unions."""
        base_boxes = union_boxes(person_boxes, equipment_boxes)
        base_cxcywh = box_cxcywh(base_boxes)
        residual = torch.tanh(self.box_head(memory))
        center = base_cxcywh[..., :2] + 0.12 * residual[..., :2]
        size = base_cxcywh[..., 2:] * torch.exp(0.20 * residual[..., 2:])
        refined = torch.cat([center, size], dim=-1)
        return cxcywh_to_xyxy(refined)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run a full ERG-TBNet forward pass."""
        mask = batch["mask"].float()
        person_boxes = batch["person_boxes"]
        equipment_boxes = batch["equipment_boxes"]
        workstation_boxes = batch["workstation_boxes"]
        batch_size, time_steps, relations = mask.shape

        encoded = self._encode_entities(batch)
        p_feat = encoded["person"]
        e_feat = encoded["equipment"]
        w_feat = encoded["workstation"]
        spatial = spatial_constraint_vector(person_boxes, equipment_boxes, workstation_boxes)

        relation_input = torch.cat([p_feat, e_feat, w_feat, spatial], dim=-1)
        relation_repr = self.relation_unit(relation_input)
        relation_prior = torch.sigmoid(self.relation_prior_head(relation_repr)).squeeze(-1) * mask

        operation_input = torch.cat([p_feat, e_feat, spatial], dim=-1)
        operation_hidden = self.operation_mlp(operation_input)
        operation_response = torch.sigmoid(self.operation_head(operation_hidden)).squeeze(-1) * mask

        confusion_input = torch.cat([e_feat, w_feat, spatial], dim=-1)
        confusion_hidden = self.confusion_mlp(confusion_input)
        confusion_response = torch.sigmoid(self.confusion_head(confusion_hidden)).squeeze(-1) * mask

        gate_logit = (
            self.config.alpha_operation * operation_response
            + self.config.beta_relation * relation_prior
            - self.config.gamma_confusion * confusion_response
            - self.config.gate_threshold
        ) * self.config.gate_temperature
        gate = torch.sigmoid(gate_logit) * mask

        joint_raw = self.joint_feature(operation_input)
        high_gate = (gate >= self.config.gate_threshold).float().unsqueeze(-1)
        joint_filtered = high_gate * joint_raw + (1.0 - high_gate) * joint_raw.detach()
        gated_features = gate.unsqueeze(-1) * joint_filtered
        temporal_memory = self._temporal_aggregate(gated_features, mask)
        temporal_memory = self.temporal_norm(temporal_memory + 0.35 * relation_repr) * mask.unsqueeze(-1)

        behavior_logits = self.class_head(temporal_memory)
        event_boxes = self._predict_boxes(temporal_memory, person_boxes, equipment_boxes)

        return {
            "relation_repr": relation_repr,
            "relation_prior": relation_prior,
            "operation_response": operation_response,
            "confusion_response": confusion_response,
            "gate": gate,
            "gated_features": gated_features,
            "temporal_memory": temporal_memory,
            "behavior_logits": behavior_logits,
            "event_boxes": event_boxes,
        }
