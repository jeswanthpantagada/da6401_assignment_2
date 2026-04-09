"""
Unified Multi-Task Learning Pipeline.
Single forward pass → Classification + BBox Regression + Segmentation.
"""

import torch
import torch.nn as nn
from models.vgg11 import VGG11, CustomDropout


def double_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
    )


class MultiTaskVGG(nn.Module):
    """
    Unified Multi-Task Model with single shared VGG11 backbone.

    Architecture:
    ─────────────────────────────────────────────────────────────
    Input Image [B, 3, H, W]
         │
         ▼
    Shared Encoder (VGG11 blocks 1-5)
    [saves skip connections: s1, s2, s3, s4, s5]
         │
         ├──► Task A: Classification Head
         │         GlobalAvgPool → FC → 37-class logits
         │
         ├──► Task B: Localization Head
         │         GlobalAvgPool → FC → 4 bbox coords (Sigmoid)
         │
         └──► Task C: Segmentation Decoder (U-Net)
                   Bottleneck → Transposed Conv + skip concat → [B, 3, H, W]

    Multi-task loss:
        L_total = λ1 * L_cls + λ2 * L_bbox + λ3 * L_seg
    """

    def __init__(self, num_classes: int = 37, seg_classes: int = 3,
                 dropout_p: float = 0.5):
        super(MultiTaskVGG, self).__init__()

        # ── Shared Encoder (VGG11 backbone, without FC) ───────────────
        self.enc1_conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2_conv = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.enc3_conv = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        self.enc4_conv = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.pool4 = nn.MaxPool2d(2, 2)

        self.enc5_conv = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.pool5 = nn.MaxPool2d(2, 2)

        self.global_pool = nn.AdaptiveAvgPool2d((7, 7))

        # ── Task A: Classification Head ───────────────────────────────
        self.cls_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(4096, num_classes)   # raw logits
        )

        # ── Task B: Localization Head ─────────────────────────────────
        self.bbox_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 4),
            nn.Sigmoid()   # normalize to [0,1]
        )

        # ── Task C: Segmentation Decoder (U-Net) ─────────────────────
        self.bottleneck = double_conv(512, 1024)

        self.up5  = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec5 = double_conv(512 + 512, 512)

        self.up4  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = double_conv(256 + 512, 256)

        self.up3  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = double_conv(128 + 256, 128)

        self.up2  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = double_conv(64 + 128, 64)

        self.up1  = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = double_conv(32 + 64, 64)

        self.seg_out = nn.Conv2d(64, seg_classes, kernel_size=1)

        self._initialize_weights()

    def forward(self, x: torch.Tensor):
        """
        Single forward pass → all three task outputs.

        Args:
            x: [B, 3, H, W]

        Returns:
            cls_logits : [B, 37]         — breed classification
            bbox_pred  : [B, 4]          — bounding box [cx, cy, w, h]
            seg_logits : [B, 3, H, W]    — segmentation mask logits
        """
        # ── Shared Encoder ────────────────────────────────────────────
        s1 = self.enc1_conv(x);   e1 = self.pool1(s1)
        s2 = self.enc2_conv(e1);  e2 = self.pool2(s2)
        s3 = self.enc3_conv(e2);  e3 = self.pool3(s3)
        s4 = self.enc4_conv(e3);  e4 = self.pool4(s4)
        s5 = self.enc5_conv(e4);  e5 = self.pool5(s5)

        # ── Global feature for classification & localization ──────────
        feat = self.global_pool(e5)           # [B, 512, 7, 7]
        feat_flat = torch.flatten(feat, 1)    # [B, 25088]

        # ── Task A: Classification ────────────────────────────────────
        cls_logits = self.cls_head(feat_flat)

        # ── Task B: Localization ──────────────────────────────────────
        bbox_pred = self.bbox_head(feat_flat)

        # ── Task C: Segmentation ──────────────────────────────────────
        d = self.bottleneck(e5)

        d = self.up5(d);  d = torch.cat([d, s5], dim=1); d = self.dec5(d)
        d = self.up4(d);  d = torch.cat([d, s4], dim=1); d = self.dec4(d)
        d = self.up3(d);  d = torch.cat([d, s3], dim=1); d = self.dec3(d)
        d = self.up2(d);  d = torch.cat([d, s2], dim=1); d = self.dec2(d)
        d = self.up1(d);  d = torch.cat([d, s1], dim=1); d = self.dec1(d)

        seg_logits = self.seg_out(d)

        return cls_logits, bbox_pred, seg_logits

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.constant_(m.bias, 0)

    def load_from_vgg11(self, vgg: VGG11):
        """Load pretrained VGG11 weights into shared encoder."""
        def copy_weights(src, dst):
            src_mods = [m for m in src.modules() if isinstance(m, (nn.Conv2d, nn.BatchNorm2d))]
            dst_mods = [m for m in dst.modules() if isinstance(m, (nn.Conv2d, nn.BatchNorm2d))]
            for s, d in zip(src_mods, dst_mods):
                d.weight.data.copy_(s.weight.data)
                if s.bias is not None and d.bias is not None:
                    d.bias.data.copy_(s.bias.data)

        copy_weights(vgg.block1, self.enc1_conv)
        copy_weights(vgg.block2, self.enc2_conv)
        copy_weights(vgg.block3, self.enc3_conv)
        copy_weights(vgg.block4, self.enc4_conv)
        copy_weights(vgg.block5, self.enc5_conv)
        print("[MultiTaskVGG] Loaded VGG11 pretrained weights.")


# ── Multi-Task Loss ────────────────────────────────────────────────────────
class MultiTaskLoss(nn.Module):
    """
    Weighted sum of all three task losses.
    L = λ1 * L_cls + λ2 * L_bbox + λ3 * L_seg
    """
    def __init__(self, lambda_cls=1.0, lambda_bbox=1.0, lambda_seg=1.0,
                 num_seg_classes=3):
        super().__init__()
        from losses import IoULoss
        from models.segmentation import SegmentationLoss

        self.cls_loss  = nn.CrossEntropyLoss()
        self.bbox_loss = IoULoss()
        self.seg_loss  = SegmentationLoss(num_seg_classes)

        self.lam_cls  = lambda_cls
        self.lam_bbox = lambda_bbox
        self.lam_seg  = lambda_seg

    def forward(self, cls_pred, cls_gt, bbox_pred, bbox_gt, seg_pred, seg_gt):
        l_cls  = self.cls_loss(cls_pred, cls_gt)
        l_bbox = self.bbox_loss(bbox_pred, bbox_gt)
        l_seg  = self.seg_loss(seg_pred, seg_gt)

        total = (self.lam_cls  * l_cls +
                 self.lam_bbox * l_bbox +
                 self.lam_seg  * l_seg)

        return total, {'cls': l_cls.item(),
                       'bbox': l_bbox.item(),
                       'seg': l_seg.item()}
