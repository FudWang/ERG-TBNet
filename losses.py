"""Losses and diagnostic metrics for ERG-TBNet."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from .config import ERGTBNetConfig
from .geometry import box_iou_aligned, generalized_iou_aligned


def masked_mean(values: torch.Tensor, mask: torch.Tensor, eps: float = 1.0e-7) -> torch.Tensor:
    """Compute a numerically stable masked mean."""
    return (values * mask).sum() / mask.sum().clamp(min=eps)


class ERGTBNetLoss(nn.Module):
    """Composite loss for detection, relation supervision, and temporal stability."""

    def __init__(self, config: ERGTBNetConfig) -> None:
        super().__init__()
        self.config = config

    def detection_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute classification, box regression, and generalized-IoU losses."""
        mask = batch["mask"].float()
        mask_bool = mask > 0.0
        labels = batch["behavior_labels"].long()
        logits = outputs["behavior_logits"]
        class_weights = torch.ones(self.config.behavior_classes, dtype=logits.dtype, device=logits.device)
        class_weights[0] = 0.65

        if mask_bool.any():
            cls_loss = F.cross_entropy(logits[mask_bool], labels[mask_bool], weight=class_weights)
        else:
            cls_loss = logits.sum() * 0.0

        positive_mask = (labels > 0) & mask_bool
        if positive_mask.any():
            pred_boxes = outputs["event_boxes"][positive_mask]
            target_boxes = batch["event_boxes"][positive_mask]
            reg_loss = F.smooth_l1_loss(pred_boxes, target_boxes, reduction="none").sum(dim=-1).mean()
            giou_loss = (1.0 - generalized_iou_aligned(pred_boxes, target_boxes)).mean()
        else:
            reg_loss = logits.sum() * 0.0
            giou_loss = logits.sum() * 0.0

        det_loss = cls_loss + self.config.bbox_loss_weight * reg_loss + self.config.iou_loss_weight * giou_loss
        return {
            "loss_detection": det_loss,
            "loss_cls": cls_loss,
            "loss_bbox": reg_loss,
            "loss_giou": giou_loss,
        }

    def relation_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute binary supervision for relation prior scores."""
        mask = batch["mask"].float()
        target = batch["relation_labels"].float()
        pred = outputs["relation_prior"].clamp(1.0e-6, 1.0 - 1.0e-6)
        bce = F.binary_cross_entropy(pred, target, reduction="none")
        weights = torch.where(target > 0.5, self.config.focal_positive_weight, 1.0).to(bce.dtype)
        return masked_mean(bce * weights, mask)

    def temporal_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Penalize prediction jitter for adjacent frames with stable relation labels."""
        stable = batch["stable_mask"].float()
        if stable.shape[1] < 2:
            return outputs["behavior_logits"].sum() * 0.0
        probs = torch.softmax(outputs["behavior_logits"], dim=-1)
        diff = (probs[:, 1:] - probs[:, :-1]).pow(2).sum(dim=-1)
        temporal_mask = stable[:, 1:]
        if temporal_mask.sum() <= 0:
            return outputs["behavior_logits"].sum() * 0.0
        return masked_mean(diff, temporal_mask)

    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute the total ERG-TBNet objective."""
        det = self.detection_loss(outputs, batch)
        rel = self.relation_loss(outputs, batch)
        temp = self.temporal_loss(outputs, batch)
        total = det["loss_detection"] + self.config.relation_loss_weight * rel + self.config.temporal_loss_weight * temp
        det.update(
            {
                "loss_relation": rel,
                "loss_temporal": temp,
                "loss_total": total,
            }
        )
        return det


@torch.no_grad()
def compute_metrics(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    config: ERGTBNetConfig,
) -> Dict[str, float]:
    """Compute relation-level diagnostic metrics for validation."""
    mask = batch["mask"] > 0.0
    labels = batch["behavior_labels"].long()
    target_positive = (labels > 0) & mask
    probs = torch.softmax(outputs["behavior_logits"], dim=-1)
    scores, pred_labels = probs.max(dim=-1)
    pred_positive = (pred_labels > 0) & mask & (scores >= config.score_threshold) & (outputs["gate"] >= config.score_threshold)
    class_correct = pred_positive & target_positive & (pred_labels == labels)
    iou = box_iou_aligned(outputs["event_boxes"], batch["event_boxes"])
    strict_correct = class_correct & (iou >= 0.5)

    tp = class_correct.sum().item()
    fp = (pred_positive & ~class_correct).sum().item()
    fn = (target_positive & ~class_correct).sum().item()
    strict_tp = strict_correct.sum().item()
    strict_fp = (pred_positive & ~strict_correct).sum().item()
    strict_fn = (target_positive & ~strict_correct).sum().item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
    strict_precision = strict_tp / max(strict_tp + strict_fp, 1)
    strict_recall = strict_tp / max(strict_tp + strict_fn, 1)
    strict_f1 = 2.0 * strict_precision * strict_recall / max(strict_precision + strict_recall, 1.0e-12)

    relation_pred = outputs["relation_prior"] >= config.score_threshold
    relation_target = batch["relation_labels"].bool() & mask
    relation_tp = (relation_pred & relation_target).sum().item()
    relation_fp = (relation_pred & ~relation_target & mask).sum().item()
    relation_fn = (~relation_pred & relation_target).sum().item()
    relation_precision = relation_tp / max(relation_tp + relation_fp, 1)
    relation_recall = relation_tp / max(relation_tp + relation_fn, 1)
    relation_f1 = 2.0 * relation_precision * relation_recall / max(relation_precision + relation_recall, 1.0e-12)

    target_iou = iou[target_positive].mean().item() if target_positive.any() else 0.0
    gate_mean = outputs["gate"][mask].mean().item() if mask.any() else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "event_f1_iou50": float(strict_f1),
        "relation_f1": float(relation_f1),
        "target_iou": float(target_iou),
        "gate_mean": float(gate_mean),
    }


def average_metric_dicts(metrics: Iterable[Mapping[str, float]]) -> Dict[str, float]:
    """Average a list of metric dictionaries."""
    metrics = list(metrics)
    if not metrics:
        return {}
    keys = metrics[0].keys()
    return {key: float(sum(item[key] for item in metrics) / len(metrics)) for key in keys}


def count_trainable_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
