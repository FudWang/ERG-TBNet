# ERG-TBNet 🧩⚙️

**Equipment-Relation Guided Teaching Behavior Network**  
*A detector-agnostic PyTorch reference implementation for relation-guided visual scene understanding in industry-education practical training environments.*

<p align="center">
  <b>Person</b> → <b>Equipment</b> → <b>Workstation</b> → <b>Participation Gate</b> → <b>Short-Term Temporal Evidence</b> → <b>Teaching Behavior Event</b>
</p>

---

## ✨ Highlights

ERG-TBNet converts independent detection outputs into structured teaching-behavior evidence. Instead of deciding whether a behavior occurs from isolated object boxes, the model explicitly builds **person-equipment-workstation relation units**, filters pseudo-interactions with an **equipment participation gate**, and stabilizes predictions with **short-term temporal consistency**.

The implementation is intentionally **detector-agnostic**: any detector or visual encoder can provide person, equipment, and workstation features/boxes, and ERG-TBNet then performs relation reasoning and behavior-event prediction.

---

## 🧠 Method Overview

### 1. Person-Equipment-Workstation Relation Unit

Each candidate triplet is represented by three entity embeddings and a spatial constraint vector:

- aligned person-equipment, person-workstation, and equipment-workstation overlaps;
- normalized center distances;
- relative person-equipment offsets.

The resulting relation representation estimates whether the triplet can become a valid teaching-behavior relation.

### 2. Equipment Participation Gate

The gate combines:

- local operation evidence between a person and equipment;
- the relation prior from the triplet unit;
- background confusion between equipment and workstation context.

Low-evidence relations are suppressed with stop-gradient filtering to reduce noise amplification from invalid proximity, idle equipment, or passing-by persons.

### 3. Short-Term Temporal Consistency

A lightweight causal temporal aggregation window accumulates gated relation evidence across adjacent frames. This reduces prediction jitter under short occlusion, weak operation cues, and brief behavior transitions without requiring long-sequence modeling.

---

## 📁 Repository Structure

```text
erg_tbnet_final/
├── erg_tbnet/
│   ├── __init__.py          # Public package API
│   ├── config.py            # Hyper-parameters and reproducibility helpers
│   ├── geometry.py          # Box geometry, IoU/GIoU, and spatial constraints
│   ├── data.py              # JSONL dataset reader and synthetic validation dataset
│   ├── model.py             # ERG-TBNet modules and forward pass
│   ├── losses.py            # Detection, relation, temporal losses and metrics
│   └── cross_validate.py    # Three-cycle synthetic k-fold validation script
├── README.md
├── requirements.txt
└── validation_report.json   # Generated validation log from the packaged run
```

The package contains **7 Python source files** plus this README and a minimal requirements file.

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python -m erg_tbnet.cross_validate \
  --synthetic-size 36 \
  --epochs 8 \
  --folds 3 \
  --cycles 3 \
  --hidden-dim 32 \
  --output validation_report.json
```

The bundled validation uses deterministic synthetic relation clips. It verifies that the implementation is executable, differentiable, and internally consistent. It is **not** a replacement for reporting TVET-IM or TVET-EE benchmark results.

---

## ✅ Packaged Validation Summary

A three-cycle cross-validation refinement was run before packaging. The final cycle used relation-positive balancing, stronger temporal stability, and conservative gradient clipping.

| Cycle | Main profile | F1 ↑ | Recall ↑ | Event F1@IoU50 ↑ | Target IoU ↑ |
|---|---:|---:|---:|---:|---:|
| 1 | Method-faithful baseline | 0.0926 | 0.0624 | 0.0926 | 0.9524 |
| 2 | Relation-balanced refinement | 0.7492 | 0.8502 | 0.7492 | 0.9406 |
| 3 | Temporal-stabilized final | **0.7632** | 0.8398 | **0.7632** | 0.9358 |

These numbers come from the included synthetic validation script and are meant as code-level diagnostics only. Real paper reproduction requires real detector outputs, TVET-style annotations, and the full training schedule.

---

## 🧾 JSONL Data Format

For real data, prepare one JSON object per clip. Each tensor-like field should be stored as nested lists.

```json
{
  "person_features": [[[...]]],
  "equipment_features": [[[...]]],
  "workstation_features": [[[...]]],
  "person_boxes": [[[x1, y1, x2, y2]]],
  "equipment_boxes": [[[x1, y1, x2, y2]]],
  "workstation_boxes": [[[x1, y1, x2, y2]]],
  "relation_labels": [[0, 1, 1]],
  "behavior_labels": [[0, 2, 4]],
  "event_boxes": [[[x1, y1, x2, y2]]],
  "mask": [[1, 1, 1]],
  "stable_mask": [[0, 1, 1]]
}
```

Expected shapes:

| Field group | Shape | Description |
|---|---:|---|
| `*_features` | `[T, R, D]` | Entity features extracted by a detector/backbone |
| `*_boxes` | `[T, R, 4]` | Normalized xyxy boxes |
| `relation_labels` | `[T, R]` | Binary validity label for each triplet |
| `behavior_labels` | `[T, R]` | `0` for background, `1..C-1` for behavior classes |
| `event_boxes` | `[T, R, 4]` | Behavior-event box targets |
| `mask` | `[T, R]` | Valid candidate mask |
| `stable_mask` | `[T, R]` | Adjacent-frame stability supervision mask |

---

## 🔬 Minimal Usage Example

```python
import torch
from erg_tbnet import ERGTBNet, ERGTBNetConfig, ERGTBNetLoss
from erg_tbnet.data import SyntheticTVETDataset, collate_relation_batch

cfg = ERGTBNetConfig(feature_dim=32, hidden_dim=96, behavior_classes=6)
dataset = SyntheticTVETDataset(size=8, config=cfg)
batch = collate_relation_batch([dataset[0], dataset[1]])

model = ERGTBNet(cfg)
criterion = ERGTBNetLoss(cfg)
outputs = model(batch)
losses = criterion(outputs, batch)
losses["loss_total"].backward()
```

---

## 🏛️ Academic Reproducibility Notes

- **Detector interface.** This implementation starts from relation candidates, entity features, and entity boxes. It can be coupled with YOLO, DETR, RT-DETR, D-FINE, or any detector that exports aligned features and normalized boxes.
- **Default method parameters.** The default relation radius, gate threshold, temporal window, and loss weights follow the method description: `rho=0.55`, `tau=0.55`, `K=5`, `lambda_box=5.0`, `lambda_iou=2.0`, `lambda_rel=0.8`, and `lambda_temp=0.4`.
- **Validation policy.** The packaged synthetic validation is a code-level sanity check. For a manuscript-grade experiment, use the original dataset split, five random seeds, strict event-box metrics, significance tests, and complexity reporting.
- **Extensibility.** The relation unit, gate, temporal aggregation, and loss components are modular and can be replaced independently for ablation studies.

---

## 📌 Citation Placeholder

```bibtex
@article{ergtbnet2026,
  title   = {Equipment-Relation Guided Visual Scene Understanding and Teaching Behavior Perception for Industry-Education Integrated Practical Training},
  author  = {Anonymous},
  journal = {Under Review},
  year    = {2026}
}
```

---

## ⚠️ Scope

This repository is a clean, research-oriented implementation template. It does not include private TVET-IM/TVET-EE images, detector checkpoints, or production deployment scripts.
