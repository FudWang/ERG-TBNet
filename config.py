"""Configuration utilities for ERG-TBNet.

The defaults follow the method description whenever possible, while keeping the
synthetic validation entry point lightweight enough for CPU execution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch


@dataclass
class ERGTBNetConfig:
    """Hyper-parameters for Equipment-Relation Guided Teaching Behavior Network."""

    feature_dim: int = 32
    hidden_dim: int = 96
    behavior_classes: int = 6
    spatial_dim: int = 8
    relation_radius: float = 0.55
    gate_threshold: float = 0.55
    gate_temperature: float = 4.0
    alpha_operation: float = 1.0
    beta_relation: float = 1.0
    gamma_confusion: float = 1.0
    temporal_window: int = 5
    bbox_loss_weight: float = 5.0
    iou_loss_weight: float = 2.0
    relation_loss_weight: float = 0.8
    temporal_loss_weight: float = 0.4
    focal_positive_weight: float = 1.5
    learning_rate: float = 1.0e-4
    weight_decay: float = 5.0e-4
    batch_size: int = 8
    epochs: int = 120
    dropout: float = 0.0
    max_grad_norm: float = 2.0
    score_threshold: float = 0.35

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return asdict(self)

    def save_json(self, path: str) -> None:
        """Save the configuration as a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "ERGTBNetConfig":
        """Load the configuration from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(**payload)

    def clone_with(self, **kwargs: Any) -> "ERGTBNetConfig":
        """Create a shallow copy with selected fields overwritten."""
        data = self.to_dict()
        data.update(kwargs)
        return ERGTBNetConfig(**data)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def resolve_device(device: Optional[str] = None) -> torch.device:
    """Resolve the target torch device."""
    if device and device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
