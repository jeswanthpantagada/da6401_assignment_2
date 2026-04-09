"""
Encoder-Decoder for Bounding Box Regression.
Uses VGG11 backbone as encoder, attaches a regression head.
"""

import torch
import torch.nn as nn
from models.vgg11 import VGG11


class LocalizationModel(nn.Module):
    """
    Object Localization via Bounding Box Regression.

    Backbone Strategy:
    ------------------
    We use PARTIAL FINE-TUNING:
    - Freeze block1, block2, block3 (low-level: edges, textures, colors)
    - Unfreeze block4, block5 (high-level: object parts, semantic features)

    Justification:
    - Early layers of VGG learn generic features (Gabor-like filters, color blobs)
      that transfer well across all visual tasks — freezing them prevents overfitting
      on the small pet dataset and significantly speeds up training.
    - Later layers encode task-specific, high-level representations. For localization,
      the network must learn *where* an object is (spatial relationship), which differs
      from classification. Fine-tuning later blocks lets the backbone adapt its
      representations to be more spatially aware.
    - This is a middle ground: better performance than a fully frozen encoder,
      lower risk of catastrophic forgetting vs. full fine-tuning.

    Regression Head:
    ----------------
    Outputs [x_center, y_center, width, height] normalized to [0, 1].
    Sigmoid activation ensures outputs are bounded in [0, 1] (normalized coords).
    """
    def __init__(self, pretrained_vgg: VGG11, freeze_early: bool = True):
        super(LocalizationModel, self).__init__()

        # ── Encoder: VGG11 convolutional backbone ────────────────────
        self.block1 = pretrained_vgg.block1
        self.block2 = pretrained_vgg.block2
        self.block3 = pretrained_vgg.block3
        self.block4 = pretrained_vgg.block4
        self.block5 = pretrained_vgg.block5
        self.avgpool = pretrained_vgg.avgpool

        # ── Freeze early blocks if requested ─────────────────────────
        if freeze_early:
            for block in [self.block1, self.block2, self.block3]:
                for param in block.parameters():
                    param.requires_grad = False
            print("[LocalizationModel] Froze blocks 1-3. Fine-tuning blocks 4-5.")
        else:
            print("[LocalizationModel] Full fine-tuning of all backbone layers.")

        # ── Regression Decoder / Head ─────────────────────────────────
        # Input: 512 * 7 * 7 = 25088 features
        self.regression_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            nn.Linear(256, 4),   # [x_center, y_center, width, height]
            nn.Sigmoid()         # Normalize to [0, 1] — matches normalized bbox coords
        )

        # Initialize only the regression head
        self._init_regression_head()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor [B, 3, H, W]
        Returns:
            bbox: [B, 4] — [x_center, y_center, width, height], values in [0,1]
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        bbox = self.regression_head(x)
        return bbox

    def _init_regression_head(self):
        for m in self.regression_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
