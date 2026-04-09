"""
Custom IoU Loss Function.
Inherits from nn.Module. No external IoU libraries used.
"""

import torch
import torch.nn as nn


class IoULoss(nn.Module):
    """
    Custom Intersection over Union (IoU) Loss.

    Inputs are expected in [x_center, y_center, width, height] format,
    normalized to [0, 1].

    IoU Loss = 1 - IoU
    (Minimizing this maximizes overlap between predicted and GT boxes.)

    Mathematical derivation:
    ------------------------
    Given pred/gt in [cx, cy, w, h] format:
      x1 = cx - w/2,  x2 = cx + w/2
      y1 = cy - h/2,  y2 = cy + h/2

    Intersection:
      inter_x1 = max(pred_x1, gt_x1)
      inter_y1 = max(pred_y1, gt_y1)
      inter_x2 = min(pred_x2, gt_x2)
      inter_y2 = min(pred_y2, gt_y2)
      inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    Union:
      union_area = pred_area + gt_area - inter_area

    IoU = inter_area / (union_area + eps)    [eps for numerical stability]
    Loss = 1 - IoU

    Gradient viability:
    - All operations are differentiable w.r.t. predicted coordinates
      (clamp/max produce subgradients at 0)
    - eps prevents division by zero
    """
    def __init__(self, eps: float = 1e-6, reduction: str = 'mean'):
        super(IoULoss, self).__init__()
        self.eps = eps
        assert reduction in ('mean', 'sum', 'none')
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred   : [B, 4] — predicted [cx, cy, w, h], values in [0, 1]
            target : [B, 4] — ground truth [cx, cy, w, h], values in [0, 1]
        Returns:
            loss   : scalar IoU loss
        """
        # ── Convert [cx, cy, w, h] → [x1, y1, x2, y2] ───────────────
        pred_x1 = pred[:, 0] - pred[:, 2] / 2
        pred_y1 = pred[:, 1] - pred[:, 3] / 2
        pred_x2 = pred[:, 0] + pred[:, 2] / 2
        pred_y2 = pred[:, 1] + pred[:, 3] / 2

        tgt_x1 = target[:, 0] - target[:, 2] / 2
        tgt_y1 = target[:, 1] - target[:, 3] / 2
        tgt_x2 = target[:, 0] + target[:, 2] / 2
        tgt_y2 = target[:, 1] + target[:, 3] / 2

        # ── Intersection ──────────────────────────────────────────────
        inter_x1 = torch.max(pred_x1, tgt_x1)
        inter_y1 = torch.max(pred_y1, tgt_y1)
        inter_x2 = torch.min(pred_x2, tgt_x2)
        inter_y2 = torch.min(pred_y2, tgt_y2)

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        # ── Union ─────────────────────────────────────────────────────
        pred_area = (pred_x2 - pred_x1).clamp(min=0) * (pred_y2 - pred_y1).clamp(min=0)
        tgt_area  = (tgt_x2 - tgt_x1).clamp(min=0) * (tgt_y2 - tgt_y1).clamp(min=0)
        union_area = pred_area + tgt_area - inter_area

        # ── IoU ───────────────────────────────────────────────────────
        iou = inter_area / (union_area + self.eps)
        loss = 1.0 - iou

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss   # [B]

    def compute_iou(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Returns raw IoU values (for metric logging, not for loss)."""
        with torch.no_grad():
            return 1.0 - self.forward(pred, target)
