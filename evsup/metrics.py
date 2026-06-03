from __future__ import annotations

import math

import numpy as np
import torch


def iou(pred_mask: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    intersection = (pred_mask & target_mask).sum().float()
    union = (pred_mask | target_mask).sum().float()
    if union == 0:
        return torch.tensor(float("nan"), device=pred_mask.device)
    return intersection / union


def mask_metrics(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    event_voxel: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float | list[float]]:
    pred_probs = torch.sigmoid(pred_logits)
    pred_dynamic = pred_probs > threshold
    target_dynamic = target > threshold
    pred_bg = ~pred_dynamic
    target_bg = ~target_dynamic

    iou_dynamic = iou(pred_dynamic, target_dynamic)
    iou_background = iou(pred_bg, target_bg)
    miou = torch.nanmean(torch.stack([iou_dynamic, iou_background]))

    if event_voxel.ndim == 4:
        event_mask = torch.any(event_voxel != 0, dim=1)
    else:
        event_mask = torch.any(event_voxel != 0, dim=0)
    piou = iou(pred_dynamic & event_mask, target_dynamic & event_mask)

    return {
        "IoU": [float(iou_dynamic.item()), float(iou_background.item())],
        "mIoU": float(miou.item()),
        "pIoU": float(piou.item()),
    }


def success_at(iou_value: float, threshold: float = 0.5) -> float:
    if math.isnan(iou_value):
        return float("nan")
    return float(iou_value >= threshold)


def nanmean_percent(values) -> float | list[float]:
    value = np.nanmean(np.asarray(values, dtype=float), axis=0) * 100.0
    return value.tolist() if hasattr(value, "tolist") else float(value)
