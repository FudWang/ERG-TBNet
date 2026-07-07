"""Vectorized geometry functions used by ERG-TBNet."""

from __future__ import annotations

import torch


def clamp_boxes(boxes: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Clamp normalized xyxy boxes to a valid image range."""
    boxes = boxes.clamp(0.0, 1.0)
    x1 = torch.minimum(boxes[..., 0], boxes[..., 2] - eps)
    y1 = torch.minimum(boxes[..., 1], boxes[..., 3] - eps)
    x2 = torch.maximum(boxes[..., 2], x1 + eps)
    y2 = torch.maximum(boxes[..., 3], y1 + eps)
    return torch.stack([x1, y1, x2, y2], dim=-1).clamp(0.0, 1.0)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """Compute aligned box areas for normalized xyxy boxes."""
    boxes = clamp_boxes(boxes)
    width = (boxes[..., 2] - boxes[..., 0]).clamp(min=0.0)
    height = (boxes[..., 3] - boxes[..., 1]).clamp(min=0.0)
    return width * height


def box_iou_aligned(boxes_a: torch.Tensor, boxes_b: torch.Tensor, eps: float = 1.0e-7) -> torch.Tensor:
    """Compute IoU between aligned boxes with the same leading shape."""
    boxes_a = clamp_boxes(boxes_a)
    boxes_b = clamp_boxes(boxes_b)
    inter_x1 = torch.maximum(boxes_a[..., 0], boxes_b[..., 0])
    inter_y1 = torch.maximum(boxes_a[..., 1], boxes_b[..., 1])
    inter_x2 = torch.minimum(boxes_a[..., 2], boxes_b[..., 2])
    inter_y2 = torch.minimum(boxes_a[..., 3], boxes_b[..., 3])
    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
    inter = inter_w * inter_h
    union = box_area(boxes_a) + box_area(boxes_b) - inter
    return inter / union.clamp(min=eps)


def generalized_iou_aligned(boxes_a: torch.Tensor, boxes_b: torch.Tensor, eps: float = 1.0e-7) -> torch.Tensor:
    """Compute generalized IoU between aligned normalized xyxy boxes."""
    boxes_a = clamp_boxes(boxes_a)
    boxes_b = clamp_boxes(boxes_b)
    iou = box_iou_aligned(boxes_a, boxes_b, eps=eps)
    enc_x1 = torch.minimum(boxes_a[..., 0], boxes_b[..., 0])
    enc_y1 = torch.minimum(boxes_a[..., 1], boxes_b[..., 1])
    enc_x2 = torch.maximum(boxes_a[..., 2], boxes_b[..., 2])
    enc_y2 = torch.maximum(boxes_a[..., 3], boxes_b[..., 3])
    enc_area = ((enc_x2 - enc_x1).clamp(min=0.0) * (enc_y2 - enc_y1).clamp(min=0.0)).clamp(min=eps)

    inter_x1 = torch.maximum(boxes_a[..., 0], boxes_b[..., 0])
    inter_y1 = torch.maximum(boxes_a[..., 1], boxes_b[..., 1])
    inter_x2 = torch.minimum(boxes_a[..., 2], boxes_b[..., 2])
    inter_y2 = torch.minimum(boxes_a[..., 3], boxes_b[..., 3])
    inter = (inter_x2 - inter_x1).clamp(min=0.0) * (inter_y2 - inter_y1).clamp(min=0.0)
    union = box_area(boxes_a) + box_area(boxes_b) - inter
    return iou - (enc_area - union) / enc_area


def box_centers(boxes: torch.Tensor) -> torch.Tensor:
    """Return normalized box centers."""
    boxes = clamp_boxes(boxes)
    return (boxes[..., :2] + boxes[..., 2:]) * 0.5


def box_cxcywh(boxes: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Convert normalized xyxy boxes to normalized cxcywh boxes."""
    boxes = clamp_boxes(boxes)
    centers = box_centers(boxes)
    sizes = (boxes[..., 2:] - boxes[..., :2]).clamp(min=eps)
    return torch.cat([centers, sizes], dim=-1)


def cxcywh_to_xyxy(cxcywh: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Convert normalized cxcywh boxes to normalized xyxy boxes."""
    centers = cxcywh[..., :2].clamp(0.0, 1.0)
    sizes = cxcywh[..., 2:].clamp(min=eps, max=1.0)
    half = sizes * 0.5
    boxes = torch.cat([centers - half, centers + half], dim=-1)
    return clamp_boxes(boxes, eps=eps)


def union_boxes(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Return the smallest aligned xyxy box containing both inputs."""
    boxes_a = clamp_boxes(boxes_a)
    boxes_b = clamp_boxes(boxes_b)
    return torch.stack(
        [
            torch.minimum(boxes_a[..., 0], boxes_b[..., 0]),
            torch.minimum(boxes_a[..., 1], boxes_b[..., 1]),
            torch.maximum(boxes_a[..., 2], boxes_b[..., 2]),
            torch.maximum(boxes_a[..., 3], boxes_b[..., 3]),
        ],
        dim=-1,
    )


def normalized_center_distance(boxes_a: torch.Tensor, boxes_b: torch.Tensor, eps: float = 1.0e-7) -> torch.Tensor:
    """Compute Euclidean distance between box centers normalized by image diagonal."""
    center_a = box_centers(boxes_a)
    center_b = box_centers(boxes_b)
    return torch.linalg.norm(center_a - center_b, dim=-1) / (2.0**0.5 + eps)


def spatial_constraint_vector(
    person_boxes: torch.Tensor,
    equipment_boxes: torch.Tensor,
    workstation_boxes: torch.Tensor,
) -> torch.Tensor:
    """Build the spatial constraint vector for person-equipment-workstation triplets.

    The vector contains three aligned overlaps, three normalized center distances,
    and two person-equipment offsets. It operationalizes the spatial affiliation,
    proximity, and relative-offset terms described by the relation unit.
    """
    pe_iou = box_iou_aligned(person_boxes, equipment_boxes)
    pw_iou = box_iou_aligned(person_boxes, workstation_boxes)
    ew_iou = box_iou_aligned(equipment_boxes, workstation_boxes)
    d_pe = normalized_center_distance(person_boxes, equipment_boxes)
    d_pw = normalized_center_distance(person_boxes, workstation_boxes)
    d_ew = normalized_center_distance(equipment_boxes, workstation_boxes)
    offset = box_centers(person_boxes) - box_centers(equipment_boxes)
    return torch.stack([pe_iou, pw_iou, ew_iou, d_pe, d_pw, d_ew, offset[..., 0], offset[..., 1]], dim=-1)
