"""Dataset and data-loading utilities for ERG-TBNet.

The implementation supports a JSONL relation-clip format and a deterministic
synthetic dataset used by the bundled cross-validation smoke test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .config import ERGTBNetConfig


BATCH_KEYS = (
    "person_features",
    "equipment_features",
    "workstation_features",
    "person_boxes",
    "equipment_boxes",
    "workstation_boxes",
    "relation_labels",
    "behavior_labels",
    "event_boxes",
    "mask",
    "stable_mask",
)


def _as_tensor(value: Any, dtype: torch.dtype) -> torch.Tensor:
    """Convert nested arrays or tensors to a tensor with the requested dtype."""
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype)
    return torch.tensor(value, dtype=dtype)


def _make_box(center: np.ndarray, size: np.ndarray) -> np.ndarray:
    """Create a valid normalized xyxy box from center and size arrays."""
    half = size * 0.5
    xy1 = np.clip(center - half, 0.0, 0.98)
    xy2 = np.clip(center + half, xy1 + 1.0e-3, 1.0)
    return np.concatenate([xy1, xy2], axis=-1).astype(np.float32)


def _union_box(box_a: np.ndarray, box_b: np.ndarray) -> np.ndarray:
    """Return the smallest xyxy box containing two numpy boxes."""
    return np.array(
        [
            min(box_a[0], box_b[0]),
            min(box_a[1], box_b[1]),
            max(box_a[2], box_b[2]),
            max(box_a[3], box_b[3]),
        ],
        dtype=np.float32,
    )


def _derive_stable_mask(behavior_labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Derive a temporal stability mask from adjacent positive labels."""
    stable = torch.zeros_like(mask, dtype=torch.float32)
    if behavior_labels.shape[0] > 1:
        same_label = behavior_labels[1:] == behavior_labels[:-1]
        positive = (behavior_labels[1:] > 0) & (behavior_labels[:-1] > 0)
        stable[1:] = (same_label & positive).float() * mask[1:] * mask[:-1]
    return stable


class TVETRelationDataset(Dataset):
    """Read relation clips from a JSONL file.

    Expected keys per line are listed in ``BATCH_KEYS``. Feature tensors should
    have shape ``[T, R, D]``, box tensors should have shape ``[T, R, 4]``, and
    label or mask tensors should have shape ``[T, R]``.
    """

    def __init__(self, jsonl_path: str | Path, feature_dim: Optional[int] = None) -> None:
        self.path = Path(jsonl_path)
        self.feature_dim = feature_dim
        self.records: List[Dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
        if not self.records:
            raise ValueError(f"No relation clips were found in {self.path}.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records[index]
        item: Dict[str, torch.Tensor] = {}
        for key in BATCH_KEYS:
            if key not in record:
                continue
            dtype = torch.long if key.endswith("labels") else torch.float32
            item[key] = _as_tensor(record[key], dtype=dtype)

        if "mask" not in item:
            item["mask"] = torch.ones_like(item["relation_labels"], dtype=torch.float32)
        if "stable_mask" not in item:
            item["stable_mask"] = _derive_stable_mask(item["behavior_labels"], item["mask"])
        if self.feature_dim is not None:
            got = item["person_features"].shape[-1]
            if got != self.feature_dim:
                raise ValueError(f"Feature dim mismatch: expected {self.feature_dim}, got {got}.")
        return item


class SyntheticTVETDataset(Dataset):
    """Generate deterministic relation clips for cross-validation smoke tests.

    The synthetic labels are intentionally learnable from both features and
    geometry. This allows the validation script to check that gradients, losses,
    temporal aggregation, and metrics are wired correctly without requiring the
    private TVET-IM or TVET-EE datasets.
    """

    def __init__(
        self,
        size: int = 72,
        clip_length: int = 5,
        max_relations: int = 8,
        config: Optional[ERGTBNetConfig] = None,
        seed: int = 7,
        positive_rate: float = 0.48,
    ) -> None:
        self.size = int(size)
        self.clip_length = int(clip_length)
        self.max_relations = int(max_relations)
        self.config = config or ERGTBNetConfig()
        self.seed = int(seed)
        self.positive_rate = float(positive_rate)
        proto_rng = np.random.default_rng(self.seed + 12345)
        self.prototypes = proto_rng.normal(0.0, 1.0, size=(self.config.behavior_classes, self.config.feature_dim)).astype(np.float32)
        self.prototypes[0] *= 0.35

    def __len__(self) -> int:
        return self.size

    def _generate_relation(self, rng: np.random.Generator, positive: bool) -> Dict[str, np.ndarray | int]:
        """Generate one relation trajectory across a short clip."""
        behavior = int(rng.integers(1, self.config.behavior_classes)) if positive else 0
        station_center = rng.uniform([0.25, 0.25], [0.75, 0.75])
        station_size = rng.uniform([0.25, 0.20], [0.42, 0.34])
        equipment_center = station_center + rng.normal(0.0, 0.08, size=2)
        equipment_center = np.clip(equipment_center, station_center - station_size * 0.35, station_center + station_size * 0.35)
        equipment_size = rng.uniform([0.07, 0.06], [0.16, 0.14])

        if positive:
            person_center = equipment_center + rng.normal(0.0, 0.045, size=2)
        else:
            if rng.random() < 0.55:
                person_center = equipment_center + rng.normal(0.16, 0.08, size=2)
            else:
                person_center = station_center + rng.normal(0.0, 0.18, size=2)
        person_center = np.clip(person_center, 0.05, 0.95)
        person_size = rng.uniform([0.08, 0.18], [0.14, 0.28])

        base_proto = self.prototypes[behavior]
        hard_negative = (not positive) and (rng.random() < 0.35)
        if hard_negative:
            confusing_behavior = int(rng.integers(1, self.config.behavior_classes))
            base_proto = self.prototypes[confusing_behavior] * 0.65

        return {
            "behavior": behavior,
            "person_center": person_center,
            "equipment_center": equipment_center,
            "station_center": station_center,
            "person_size": person_size,
            "equipment_size": equipment_size,
            "station_size": station_size,
            "base_proto": base_proto,
            "hard_negative": int(hard_negative),
        }

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.seed + index * 7919)
        t_len = self.clip_length
        rels = self.max_relations
        dim = self.config.feature_dim

        person_features = np.zeros((t_len, rels, dim), dtype=np.float32)
        equipment_features = np.zeros_like(person_features)
        workstation_features = np.zeros_like(person_features)
        person_boxes = np.zeros((t_len, rels, 4), dtype=np.float32)
        equipment_boxes = np.zeros_like(person_boxes)
        workstation_boxes = np.zeros_like(person_boxes)
        event_boxes = np.zeros_like(person_boxes)
        relation_labels = np.zeros((t_len, rels), dtype=np.int64)
        behavior_labels = np.zeros((t_len, rels), dtype=np.int64)
        mask = np.ones((t_len, rels), dtype=np.float32)

        for rel_idx in range(rels):
            active = rng.random() > 0.08
            mask[:, rel_idx] = 1.0 if active else 0.0
            positive = bool(active and rng.random() < self.positive_rate)
            rel = self._generate_relation(rng, positive=positive)

            for t_idx in range(t_len):
                jitter_scale = 0.012 if positive else 0.020
                p_center = rel["person_center"] + rng.normal(0.0, jitter_scale, size=2)
                e_center = rel["equipment_center"] + rng.normal(0.0, jitter_scale * 0.5, size=2)
                w_center = rel["station_center"] + rng.normal(0.0, jitter_scale * 0.25, size=2)
                p_box = _make_box(np.clip(p_center, 0.03, 0.97), rel["person_size"])
                e_box = _make_box(np.clip(e_center, 0.03, 0.97), rel["equipment_size"])
                w_box = _make_box(np.clip(w_center, 0.03, 0.97), rel["station_size"])
                person_boxes[t_idx, rel_idx] = p_box
                equipment_boxes[t_idx, rel_idx] = e_box
                workstation_boxes[t_idx, rel_idx] = w_box
                event_boxes[t_idx, rel_idx] = _union_box(p_box, e_box)

                visibility_drop = 0.55 if positive and rng.random() < 0.18 else 1.0
                proto = rel["base_proto"] * visibility_drop
                person_features[t_idx, rel_idx] = proto + rng.normal(0.0, 0.22, size=dim)
                equipment_features[t_idx, rel_idx] = proto + rng.normal(0.0, 0.20, size=dim)
                workstation_features[t_idx, rel_idx] = proto * 0.55 + rng.normal(0.0, 0.20, size=dim)

                relation_labels[t_idx, rel_idx] = int(positive)
                behavior_labels[t_idx, rel_idx] = int(rel["behavior"])

        relation_labels = relation_labels * mask.astype(np.int64)
        behavior_labels = behavior_labels * mask.astype(np.int64)
        stable_mask = _derive_stable_mask(torch.tensor(behavior_labels), torch.tensor(mask)).numpy().astype(np.float32)

        return {
            "person_features": torch.tensor(person_features, dtype=torch.float32),
            "equipment_features": torch.tensor(equipment_features, dtype=torch.float32),
            "workstation_features": torch.tensor(workstation_features, dtype=torch.float32),
            "person_boxes": torch.tensor(person_boxes, dtype=torch.float32),
            "equipment_boxes": torch.tensor(equipment_boxes, dtype=torch.float32),
            "workstation_boxes": torch.tensor(workstation_boxes, dtype=torch.float32),
            "relation_labels": torch.tensor(relation_labels, dtype=torch.long),
            "behavior_labels": torch.tensor(behavior_labels, dtype=torch.long),
            "event_boxes": torch.tensor(event_boxes, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
            "stable_mask": torch.tensor(stable_mask, dtype=torch.float32),
        }


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """Move every tensor in a mini-batch to the selected device."""
    return {key: value.to(device) for key, value in batch.items()}


def collate_relation_batch(samples: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack relation clips into a mini-batch."""
    return {key: torch.stack([sample[key] for sample in samples], dim=0) for key in BATCH_KEYS}


def make_kfold_loaders(
    dataset: Dataset,
    fold: int,
    folds: int,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation loaders for one deterministic fold."""
    indices = np.arange(len(dataset))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    split_indices = np.array_split(indices, folds)
    val_indices = split_indices[fold]
    train_indices = np.concatenate([part for idx, part in enumerate(split_indices) if idx != fold])
    train_loader = DataLoader(
        Subset(dataset, train_indices.tolist()),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_relation_batch,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_relation_batch,
    )
    return train_loader, val_loader
