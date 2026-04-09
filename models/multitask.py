import torch
import torch.nn as nn
from .vgg11          import VGG11Encoder
from .layers         import CustomDropout
from .segmentation   import _double_conv, SegmentationLoss
from losses.iou_loss import IoULoss


class MultiTaskVGG(nn.Module):
    """
    Unified Multi-Task model — single forward pass yields all 3 outputs.

    Shared backbone: VGG11Encoder
    Task A: Classification  → 37-class logits
    Task B: Localization    → 4 bbox coords [cx,cy,w,h]
    Task C: Segmentation    → [B, 3, H, W] pixel-wise logits
    """
    def __init__(self, num_classes: int = 37,
                 seg_classes: int = 3, dropout_p: float = 0.5):
        super(MultiTaskVGG, self).__init__()

        # Shared encoder
        self.encoder = VGG11Encoder(in_channels=3)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # Task A: Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(4096, num_classes),
        )

        # Task B: Localization head
        self.bbox_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
            nn.Sigmoid(),
        )

        # Task C: Segmentation decoder (U-Net)
        self.bottleneck = _double_conv(512, 1024)

        self.up5  = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec5 = _double_conv(512 + 512, 512)
        self.up4  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = _double_conv(256 + 512, 256)
        self.up3  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = _double_conv(128 + 256, 128)
        self.up2  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = _double_conv(64 + 128, 64)
        self.up1  = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = _double_conv(32 + 64, 64)
        self.seg_out = nn.Conv2d(64, seg_classes, kernel_size=1)

        self._initialize_weights()

    def forward(self, x: torch.Tensor):
        """
        Single forward pass → all three task outputs.

        Args:
            x: [B, 3, H, W]
        Returns:
            cls_logits : [B, 37]
            bbox_pred  : [B, 4]
            seg_logits : [B, 3, H, W]
        """
        # Shared encoder with skip connections
        e5, skips = self.encoder(x, return_features=True)
        s1, s2, s3, s4, s5 = (skips['s1'], skips['s2'],
                               skips['s3'], skips['s4'], skips['s5'])

        # Global pooled features for cls + bbox
        feat      = self.avgpool(e5)         # [B, 512, 7, 7]
        feat_flat = torch.flatten(feat, 1)   # [B, 25088]

        # Task A
        cls_logits = self.cls_head(feat_flat)

        # Task B
        bbox_pred = self.bbox_head(feat_flat)

        # Task C — U-Net decoder
        d = self.bottleneck(e5)
        d = self.up5(d); d = torch.cat([d, s5], dim=1); d = self.dec5(d)
        d = self.up4(d); d = torch.cat([d, s4], dim=1); d = self.dec4(d)
        d = self.up3(d); d = torch.cat([d, s3], dim=1); d = self.dec3(d)
        d = self.up2(d); d = torch.cat([d, s2], dim=1); d = self.dec2(d)
        d = self.up1(d); d = torch.cat([d, s1], dim=1); d = self.dec1(d)
        seg_logits = self.seg_out(d)

        return cls_logits, bbox_pred, seg_logits

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


class MultiTaskLoss(nn.Module):
    """Weighted sum: L = λ1*L_cls + λ2*L_bbox + λ3*L_seg"""
    def __init__(self, lambda_cls=1.0, lambda_bbox=1.0, lambda_seg=1.0):
        super(MultiTaskLoss, self).__init__()
        self.cls_loss  = nn.CrossEntropyLoss()
        self.bbox_loss = IoULoss()
        self.seg_loss  = SegmentationLoss(num_classes=3)
        self.lam_cls   = lambda_cls
        self.lam_bbox  = lambda_bbox
        self.lam_seg   = lambda_seg

    def forward(self, cls_pred, cls_gt, bbox_pred, bbox_gt,
                seg_pred, seg_gt):
        l_cls  = self.cls_loss(cls_pred, cls_gt)
        l_bbox = self.bbox_loss(bbox_pred, bbox_gt)
        l_seg  = self.seg_loss(seg_pred, seg_gt)
        total  = (self.lam_cls  * l_cls +
                  self.lam_bbox * l_bbox +
                  self.lam_seg  * l_seg)
        return total, {
            'cls' : l_cls.item(),
            'bbox': l_bbox.item(),
            'seg' : l_seg.item(),
        }
