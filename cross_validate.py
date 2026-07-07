"""Cross-validation entry point for ERG-TBNet.

This script runs a deterministic synthetic k-fold validation. It is designed to
verify code integrity, gradient flow, loss wiring, and metric computation before
replacing the synthetic dataset with real TVET relation clips.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter
from typing import Dict, List, Tuple

import torch
from torch import nn

from .config import ERGTBNetConfig, resolve_device, set_seed
from .data import SyntheticTVETDataset, make_kfold_loaders, move_batch_to_device
from .losses import ERGTBNetLoss, average_metric_dicts, compute_metrics, count_trainable_parameters
from .model import ERGTBNet


def train_one_epoch(
    model: ERGTBNet,
    criterion: ERGTBNetLoss,
    optimizer: torch.optim.Optimizer,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_grad_norm: float,
) -> Dict[str, float]:
    """Train the model for one epoch."""
    model.train()
    loss_logs: List[Dict[str, float]] = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        losses = criterion(outputs, batch)
        losses["loss_total"].backward()
        if max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        loss_logs.append({key: float(value.detach().cpu()) for key, value in losses.items()})
    return average_metric_dicts(loss_logs)


@torch.no_grad()
def evaluate(
    model: ERGTBNet,
    criterion: ERGTBNetLoss,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate the model on a validation loader."""
    model.eval()
    logs: List[Dict[str, float]] = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch)
        losses = criterion(outputs, batch)
        metrics = compute_metrics(outputs, batch, model.config)
        metrics.update({key: float(value.detach().cpu()) for key, value in losses.items()})
        logs.append(metrics)
    return average_metric_dicts(logs)


def run_fold(
    config: ERGTBNetConfig,
    dataset: SyntheticTVETDataset,
    fold: int,
    folds: int,
    args: argparse.Namespace,
    device: torch.device,
    cycle_seed: int,
) -> Dict[str, float]:
    """Run one fold of synthetic cross-validation."""
    set_seed(cycle_seed + fold)
    train_loader, val_loader = make_kfold_loaders(
        dataset=dataset,
        fold=fold,
        folds=folds,
        batch_size=config.batch_size,
        seed=cycle_seed,
        num_workers=args.num_workers,
    )
    model = ERGTBNet(config).to(device)
    criterion = ERGTBNetLoss(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    last_train: Dict[str, float] = {}
    for _ in range(args.epochs):
        last_train = train_one_epoch(model, criterion, optimizer, train_loader, device, config.max_grad_norm)
    val_metrics = evaluate(model, criterion, val_loader, device)
    val_metrics["train_loss_total"] = last_train.get("loss_total", 0.0)
    val_metrics["params_m"] = count_trainable_parameters(model) / 1.0e6
    return val_metrics


def summarize_folds(fold_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Summarize metrics with means and population standard deviations."""
    if not fold_metrics:
        return {}
    keys = fold_metrics[0].keys()
    summary: Dict[str, float] = {}
    for key in keys:
        values = [item[key] for item in fold_metrics]
        summary[f"{key}_mean"] = float(mean(values))
        summary[f"{key}_std"] = float(pstdev(values)) if len(values) > 1 else 0.0
    return summary


def build_cycle_config(base: ERGTBNetConfig, cycle_index: int) -> Tuple[str, ERGTBNetConfig, str]:
    """Return the configuration profile for one refinement cycle."""
    if cycle_index == 1:
        return (
            "cycle_1_method_faithful",
            base,
            "Initial method-faithful profile with relation unit, participation gate, and temporal aggregation.",
        )
    if cycle_index == 2:
        return (
            "cycle_2_relation_balanced",
            base.clone_with(relation_loss_weight=1.0, focal_positive_weight=1.8, score_threshold=0.10),
            "Refined profile emphasizing positive relation supervision after recall-oriented validation.",
        )
    return (
        "cycle_3_temporal_stabilized",
        base.clone_with(relation_loss_weight=1.0, temporal_loss_weight=0.55, focal_positive_weight=1.8, score_threshold=0.10, max_grad_norm=1.5),
        "Final profile with stronger temporal consistency and conservative gradient clipping.",
    )


def run_validation_cycles(args: argparse.Namespace) -> Dict[str, object]:
    """Run repeated cross-validation cycles and return a JSON-ready report."""
    device = resolve_device(args.device)
    base_config = ERGTBNetConfig(
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        behavior_classes=args.behavior_classes,
        temporal_window=args.temporal_window,
        gate_threshold=args.gate_threshold,
        relation_radius=args.relation_radius,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
    )
    dataset = SyntheticTVETDataset(
        size=args.synthetic_size,
        clip_length=args.clip_length,
        max_relations=args.max_relations,
        config=base_config,
        seed=args.seed,
    )

    cycles: List[Dict[str, object]] = []
    started = perf_counter()
    for cycle in range(1, args.cycles + 1):
        name, cycle_config, note = build_cycle_config(base_config, cycle)
        fold_metrics: List[Dict[str, float]] = []
        cycle_seed = args.seed + cycle * 1000
        for fold in range(args.folds):
            fold_metrics.append(run_fold(cycle_config, dataset, fold, args.folds, args, device, cycle_seed))
        cycles.append(
            {
                "name": name,
                "note": note,
                "config": cycle_config.to_dict(),
                "fold_metrics": fold_metrics,
                "summary": summarize_folds(fold_metrics),
            }
        )
    elapsed = perf_counter() - started
    return {
        "validation_type": "synthetic_kfold_smoke_validation",
        "device": str(device),
        "elapsed_seconds": round(elapsed, 3),
        "folds": args.folds,
        "epochs_per_fold": args.epochs,
        "cycles": cycles,
        "important_note": "Synthetic validation checks implementation correctness and training behavior; it is not a substitute for TVET-IM/TVET-EE benchmark reporting.",
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run ERG-TBNet synthetic cross-validation.")
    parser.add_argument("--synthetic-size", type=int, default=72)
    parser.add_argument("--clip-length", type=int, default=5)
    parser.add_argument("--max-relations", type=int, default=8)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--behavior-classes", type=int, default=6)
    parser.add_argument("--temporal-window", type=int, default=5)
    parser.add_argument("--relation-radius", type=float, default=0.55)
    parser.add_argument("--gate-threshold", type=float, default=0.55)
    parser.add_argument("--learning-rate", type=float, default=7.0e-4)
    parser.add_argument("--weight-decay", type=float, default=5.0e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--output", type=str, default="validation_report.json")
    return parser.parse_args()


def main() -> None:
    """Run validation and write a report."""
    args = parse_args()
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    report = run_validation_cycles(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    final_summary = report["cycles"][-1]["summary"]
    print(json.dumps({"final_cycle_summary": final_summary, "report": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
