import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder


def _double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNetVGG11(nn.Module):
    """
    U-Net style segmentation with VGG11 encoder (contracting path).

    Upsampling: strictly ConvTranspose2d (learnable). No bilinear allowed.
    Skip connections: concatenation of encoder + decoder feature maps.

    Loss: CrossEntropy + Dice (combined)
        - CE: stable gradients, per-pixel classification
        - Dice: directly optimizes DSC metric, handles class imbalance
        - Combined outperforms either alone empirically

    Output: [B, num_classes, H, W] — 3 classes (foreground, background, boundary)
    """
    def __init__(self, encoder: VGG11Encoder, num_classes: int = 3):
        super(UNetVGG11, self).__init__()
        self.encoder  = encoder
        self.avgpool  = nn.AdaptiveAvgPool2d((7, 7))

        # Bottleneck
        self.bottleneck = _double_conv(512, 1024)

        # Decoder — TransposedConv + skip concat + double_conv
        self.up5  = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec5 = _double_conv(512 + 512, 512)

        self.up4  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = _double_conv(256 + 512, 256)

        self.up3  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = _double_conv(128 + 256, 128)

        self.up2  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = _double_conv(64 + 128, 64)

        self.up1  = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = _double_conv(32 + 64, 64)

        # 1×1 output conv
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            logits: [B, num_classes, H, W]
        """
        # Encoder — get skip connections
        e5, skips = self.encoder(x, return_features=True)
        s1, s2, s3, s4, s5 = (skips['s1'], skips['s2'],
                               skips['s3'], skips['s4'], skips['s5'])

        # Bottleneck
        d = self.bottleneck(e5)

        # Decoder with skip connections
        d = self.up5(d);  d = torch.cat([d, s5], dim=1); d = self.dec5(d)
        d = self.up4(d);  d = torch.cat([d, s4], dim=1); d = self.dec4(d)
        d = self.up3(d);  d = torch.cat([d, s3], dim=1); d = self.dec3(d)
        d = self.up2(d);  d = torch.cat([d, s2], dim=1); d = self.dec2(d)
        d = self.up1(d);  d = torch.cat([d, s1], dim=1); d = self.dec1(d)

        return self.out_conv(d)   # [B, num_classes, H, W]


class DiceLoss(nn.Module):
    """Soft Dice Loss — differentiable, handles class imbalance."""
    def __init__(self, num_classes: int = 3, eps: float = 1e-6):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)   # [B, C, H, W]
        B, C  = probs.shape[:2]

        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(1, targets.unsqueeze(1), 1)

        dice_loss = 0.0
        for c in range(C):
            p = probs[:, c].reshape(B, -1)
            g = one_hot[:, c].reshape(B, -1)
            inter = (p * g).sum(dim=1)
            dice  = (2 * inter + self.eps) / \
                    (p.sum(dim=1) + g.sum(dim=1) + self.eps)
            dice_loss += (1 - dice).mean()

        return dice_loss / C


class SegmentationLoss(nn.Module):
    """Combined CrossEntropy + Dice Loss."""
    def __init__(self, num_classes: int = 3, alpha: float = 0.5):
        super(SegmentationLoss, self).__init__()
        self.ce   = nn.CrossEntropyLoss()
        self.dice = DiceLoss(num_classes)
        self.alpha = alpha

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        return (self.alpha * self.ce(logits, targets) +
                (1 - self.alpha) * self.dice(logits, targets))
