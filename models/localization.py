import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder


class LocalizationModel(nn.Module):
    """
    Bounding Box Regression using VGG11 encoder.

    Backbone strategy: PARTIAL FINE-TUNING
    - Freeze blocks 1-3 (low-level: edges, textures — universal features)
    - Unfreeze blocks 4-5 (high-level: object parts — need spatial adaptation)

    Justification:
        Early layers detect generic primitives that transfer across all
        visual tasks. Freezing them prevents overfitting on the small pet
        dataset. Later layers encode task-specific semantics; for localization
        (where IS the object?) they need to adapt beyond ImageNet features.

    Output: [cx, cy, w, h] normalized to [0,1] via Sigmoid activation.
    """
    def __init__(self, encoder: VGG11Encoder, freeze_early: bool = True):
        super(LocalizationModel, self).__init__()

        self.encoder = encoder
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        if freeze_early:
            for block in [self.encoder.block1,
                          self.encoder.block2,
                          self.encoder.block3]:
                for param in block.parameters():
                    param.requires_grad = False
            print('[LocalizationModel] Froze encoder blocks 1-3.')

        # Regression head: 25088 → 4
        self.regression_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 4),
            nn.Sigmoid(),   # output in [0,1] for normalized coords
        )

        self._init_head()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            bbox: [B, 4]  [cx, cy, w, h] in [0,1]
        """
        feat = self.encoder(x)              # [B, 512, H/32, W/32]
        feat = self.avgpool(feat)           # [B, 512, 7, 7]
        feat = torch.flatten(feat, 1)       # [B, 25088]
        return self.regression_head(feat)   # [B, 4]

    def _init_head(self):
        for m in self.regression_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
