import torch
import torch.nn as nn


class IoULoss(nn.Module):
    """
    Custom IoU Loss — inherits from nn.Module.
    No external IoU libraries used.

    Input format: [cx, cy, w, h] normalized to [0, 1]

    Math:
        Convert to [x1,y1,x2,y2] → compute intersection → compute union
        IoU = intersection / (union + eps)
        Loss = 1 - IoU

    Gradient viability:
        All ops (max, min, clamp) have subgradients → backprop works fine.
        eps prevents division by zero.
    """
    def __init__(self, eps: float = 1e-6, reduction: str = 'mean'):
        super(IoULoss, self).__init__()
        self.eps = eps
        assert reduction in ('mean', 'sum', 'none')
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred   : [B, 4]  predicted  [cx, cy, w, h]
            target : [B, 4]  ground truth [cx, cy, w, h]
        Returns:
            loss   : scalar
        """
        # Convert [cx, cy, w, h] → [x1, y1, x2, y2]
        pred_x1 = pred[:, 0] - pred[:, 2] / 2
        pred_y1 = pred[:, 1] - pred[:, 3] / 2
        pred_x2 = pred[:, 0] + pred[:, 2] / 2
        pred_y2 = pred[:, 1] + pred[:, 3] / 2

        tgt_x1 = target[:, 0] - target[:, 2] / 2
        tgt_y1 = target[:, 1] - target[:, 3] / 2
        tgt_x2 = target[:, 0] + target[:, 2] / 2
        tgt_y2 = target[:, 1] + target[:, 3] / 2

        # Intersection
        inter_x1 = torch.max(pred_x1, tgt_x1)
        inter_y1 = torch.max(pred_y1, tgt_y1)
        inter_x2 = torch.min(pred_x2, tgt_x2)
        inter_y2 = torch.min(pred_y2, tgt_y2)

        inter_w    = (inter_x2 - inter_x1).clamp(min=0)
        inter_h    = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        # Union
        pred_area = (pred_x2 - pred_x1).clamp(min=0) * \
                    (pred_y2 - pred_y1).clamp(min=0)
        tgt_area  = (tgt_x2  - tgt_x1).clamp(min=0)  * \
                    (tgt_y2  - tgt_y1).clamp(min=0)
        union_area = pred_area + tgt_area - inter_area

        # IoU and Loss
        iou  = inter_area / (union_area + self.eps)
        loss = 1.0 - iou

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss   # [B]

    def compute_iou(self, pred: torch.Tensor,
                    target: torch.Tensor) -> torch.Tensor:
        """Returns raw IoU values (for logging, not training)."""
        with torch.no_grad():
            return 1.0 - self.forward(pred, target)
