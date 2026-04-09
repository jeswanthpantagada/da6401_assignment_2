"""
U-Net Style Semantic Segmentation using VGG11 as encoder.
Uses Transposed Convolutions (NO bilinear interpolation) for upsampling.
Uses skip connections (feature fusion via concatenation).
"""

import torch
import torch.nn as nn
from models.vgg11 import VGG11


def double_conv(in_ch, out_ch):
    """Two consecutive Conv-BN-ReLU blocks."""
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
    U-Net with VGG11 Encoder.

    Architecture:
    ─────────────
    Encoder (VGG11 blocks with MaxPool):
      Input      → Block1 → 64  ch, H/2
      Block1 out → Block2 → 128 ch, H/4
      Block2 out → Block3 → 256 ch, H/8
      Block3 out → Block4 → 512 ch, H/16
      Block4 out → Block5 → 512 ch, H/32

    Bottleneck:
      Block5 out → double_conv → 1024 ch

    Decoder (TransposedConv + skip concat):
      Upsample → concat(skip) → double_conv  [mirrors encoder]
      512 up + 512 skip → 512
      512 up + 256 skip → 256
      256 up + 128 skip → 128
      128 up + 64  skip → 64

    Output Head:
      1×1 Conv → num_classes (3 for trimap: foreground, background, boundary)

    Loss Function Justification:
    ----------------------------
    We use a COMBINED loss: CrossEntropyLoss + Dice Loss.

    - CrossEntropyLoss: Standard pixel-wise multi-class classification.
      Works well when class distribution is roughly balanced.
    - Dice Loss: Directly optimizes the Dice Similarity Coefficient (DSC),
      which is the evaluation metric. DSC is invariant to class imbalance,
      penalizing false negatives and false positives equally — critical
      for the pet trimap where background pixels may dominate.
    - Combined (CE + Dice): Provides stable gradients from CE early in training
      while Dice Loss fine-tunes the overlap quality. Empirically shown to
      outperform either loss alone on segmentation benchmarks.

    Upsampling:
    -----------
    Strictly uses ConvTranspose2d (learnable). No bilinear interpolation.
    ConvTranspose2d learns optimal upsampling kernels, allowing the model
    to reconstruct spatial information that's task-specific.
    """

    def __init__(self, pretrained_vgg: VGG11, num_classes: int = 3):
        super(UNetVGG11, self).__init__()

        # ── Encoder (from VGG11) ──────────────────────────────────────
        # We extract each block BEFORE its MaxPool for skip connections
        vgg = pretrained_vgg

        # VGG11 blocks (each includes Conv+BN+ReLU + MaxPool at the end)
        # We need the pre-pool feature maps for skip connections.
        # Re-define encoder to separate conv and pool stages:

        # Block 1: conv only (no pool)
        self.enc1_conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        # Block 2
        self.enc2_conv = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        # Block 3 (2 convs)
        self.enc3_conv = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        # Block 4 (2 convs)
        self.enc4_conv = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.pool4 = nn.MaxPool2d(2, 2)

        # Block 5 (2 convs)
        self.enc5_conv = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.pool5 = nn.MaxPool2d(2, 2)

        # Load pretrained weights from VGG11 into encoder
        self._load_pretrained_encoder(pretrained_vgg)

        # ── Bottleneck ────────────────────────────────────────────────
        self.bottleneck = double_conv(512, 1024)

        # ── Decoder (Transposed Conv + skip concat) ───────────────────
        # Each decoder stage: upsample → concat skip → double_conv

        # Stage 5: 1024 → 512, then concat 512 (enc5) → 1024 → 512
        self.up5   = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec5  = double_conv(512 + 512, 512)

        # Stage 4: 512 → 256, then concat 512 (enc4) → 768 → 256
        self.up4   = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4  = double_conv(256 + 512, 256)

        # Stage 3: 256 → 128, then concat 256 (enc3) → 384 → 256
        self.up3   = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3  = double_conv(128 + 256, 128)

        # Stage 2: 128 → 64, then concat 128 (enc2) → 192 → 128
        self.up2   = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2  = double_conv(64 + 128, 64)

        # Stage 1: 64 → 32, then concat 64 (enc1) → 96 → 64
        self.up1   = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1  = double_conv(32 + 64, 64)

        # ── Output Head ───────────────────────────────────────────────
        # 1×1 conv: channel → num_classes (3: foreground, background, boundary)
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, 3, H, W]  (H=W=224 recommended)
        Returns:
            logits: [B, num_classes, H, W]
        """
        # ── Encoder (save skip connections) ──────────────────────────
        s1 = self.enc1_conv(x)          # [B, 64, H, W]
        x  = self.pool1(s1)             # [B, 64, H/2, W/2]

        s2 = self.enc2_conv(x)          # [B, 128, H/2, W/2]
        x  = self.pool2(s2)

        s3 = self.enc3_conv(x)          # [B, 256, H/4, W/4]
        x  = self.pool3(s3)

        s4 = self.enc4_conv(x)          # [B, 512, H/8, W/8]
        x  = self.pool4(s4)

        s5 = self.enc5_conv(x)          # [B, 512, H/16, W/16]
        x  = self.pool5(s5)             # [B, 512, H/32, W/32]

        # ── Bottleneck ────────────────────────────────────────────────
        x = self.bottleneck(x)          # [B, 1024, H/32, W/32]

        # ── Decoder ───────────────────────────────────────────────────
        x = self.up5(x)                         # [B, 512, H/16, W/16]
        x = torch.cat([x, s5], dim=1)           # [B, 1024, H/16, W/16]
        x = self.dec5(x)                        # [B, 512, H/16, W/16]

        x = self.up4(x)                         # [B, 256, H/8, W/8]
        x = torch.cat([x, s4], dim=1)           # [B, 768, H/8, W/8]
        x = self.dec4(x)                        # [B, 256, H/8, W/8]

        x = self.up3(x)                         # [B, 128, H/4, W/4]
        x = torch.cat([x, s3], dim=1)           # [B, 384, H/4, W/4]
        x = self.dec3(x)                        # [B, 128, H/4, W/4]

        x = self.up2(x)                         # [B, 64, H/2, W/2]
        x = torch.cat([x, s2], dim=1)           # [B, 192, H/2, W/2]
        x = self.dec2(x)                        # [B, 64, H/2, W/2]

        x = self.up1(x)                         # [B, 32, H, W]
        x = torch.cat([x, s1], dim=1)           # [B, 96, H, W]
        x = self.dec1(x)                        # [B, 64, H, W]

        return self.out_conv(x)                 # [B, num_classes, H, W]

    def _load_pretrained_encoder(self, vgg: VGG11):
        """
        Copy weights from trained VGG11 blocks into UNet encoder.
        """
        def copy_conv_weights(src_seq, dst_seq):
            src_mods = [m for m in src_seq.modules()
                        if isinstance(m, (nn.Conv2d, nn.BatchNorm2d))]
            dst_mods = [m for m in dst_seq.modules()
                        if isinstance(m, (nn.Conv2d, nn.BatchNorm2d))]
            for s, d in zip(src_mods, dst_mods):
                d.weight.data.copy_(s.weight.data)
                if s.bias is not None and d.bias is not None:
                    d.bias.data.copy_(s.bias.data)

        copy_conv_weights(vgg.block1, self.enc1_conv)
        copy_conv_weights(vgg.block2, self.enc2_conv)
        copy_conv_weights(vgg.block3, self.enc3_conv)
        copy_conv_weights(vgg.block4, self.enc4_conv)
        copy_conv_weights(vgg.block5, self.enc5_conv)
        print("[UNetVGG11] Loaded pretrained VGG11 weights into encoder.")


# ── Combined Loss: CrossEntropy + Dice ───────────────────────────────────────
class DiceLoss(nn.Module):
    """
    Soft Dice Loss for multi-class segmentation.

    Dice = 2 * |P ∩ G| / (|P| + |G|)
    Loss = 1 - Dice

    Uses softmax probabilities, not hard predictions → differentiable.
    """
    def __init__(self, num_classes: int = 3, eps: float = 1e-6):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : [B, C, H, W] — raw network output
            targets: [B, H, W]    — integer class labels
        """
        probs = torch.softmax(logits, dim=1)   # [B, C, H, W]
        B, C, H, W = probs.shape

        # One-hot encode targets: [B, C, H, W]
        targets_one_hot = torch.zeros_like(probs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)

        dice_loss = 0.0
        for c in range(C):
            p = probs[:, c].reshape(B, -1)         # [B, H*W]
            g = targets_one_hot[:, c].reshape(B, -1)
            intersection = (p * g).sum(dim=1)
            dice = (2 * intersection + self.eps) / (p.sum(dim=1) + g.sum(dim=1) + self.eps)
            dice_loss += (1 - dice).mean()

        return dice_loss / C


class SegmentationLoss(nn.Module):
    """Combined CrossEntropy + Dice Loss for segmentation."""
    def __init__(self, num_classes=3, alpha=0.5):
        super().__init__()
        self.ce   = nn.CrossEntropyLoss()
        self.dice = DiceLoss(num_classes)
        self.alpha = alpha  # weight for CE; (1-alpha) for Dice

    def forward(self, logits, targets):
        return self.alpha * self.ce(logits, targets) + (1 - self.alpha) * self.dice(logits, targets)
