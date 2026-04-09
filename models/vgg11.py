"""
VGG11 Implementation from Scratch with:
- Custom Dropout (inherits from nn.Module)
- Batch Normalization
- Justification for design choices
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# CUSTOM DROPOUT LAYER
# Must NOT use nn.Dropout or F.dropout

class CustomDropout(nn.Module):
    """
    Custom Dropout implementation using inverted dropout scaling.

    During training:
      - A binary mask is sampled from Bernoulli(1 - p)
      - The mask is applied element-wise to the input
      - Output is scaled by 1/(1-p) to preserve expected activation magnitude
        (this is "inverted dropout" — ensures no scaling needed at test time)

    During evaluation (self.training = False):
      - Input is passed through unchanged (deterministic behavior)

    Args:
        p (float): Probability of an element being zeroed. Default: 0.5
    """
    def __init__(self, p: float = 0.5):
        super(CustomDropout, self).__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.p > 0.0:
            # Step 1: Sample binary mask ~ Bernoulli(1 - p)
            # Each element kept with probability (1 - p)
            keep_prob = 1.0 - self.p
            # torch.bernoulli samples 1 with probability = value in input tensor
            mask = torch.bernoulli(torch.full(x.shape, keep_prob, device=x.device, dtype=x.dtype))
            # Step 2: Apply mask & scale (inverted dropout)
            x = x * mask / keep_prob
        return x

    def extra_repr(self) -> str:
        return f'p={self.p}'


# VGG11 ARCHITECTURE FROM SCRATCH
# Standard VGG11 config: [64, 'M', 128, 'M', 256, 256, 'M',
#                          512, 512, 'M', 512, 512, 'M']

class VGG11(nn.Module):
    """
    VGG11 Architecture with BatchNorm and Custom Dropout.

    Design Choices & Justification:
    1. BatchNorm2d placement: After every Conv2d, before ReLU.
       - Theoretical: BN normalizes pre-activations, reducing internal covariate
         shift. Placing it before ReLU ensures the normalization acts on
         unbounded pre-activations, which is more stable than post-ReLU.
       - Empirical: This placement allows higher learning rates (faster convergence)
         and reduces sensitivity to weight initialization.

    2. Custom Dropout placement: Only in the classifier (FC layers), NOT in conv layers.
       - Theoretical: Convolutional layers share weights spatially; applying dropout
         to feature maps causes nearby pixels to compensate, reducing its effectiveness.
         Spatial Dropout (per-channel) would be needed for conv layers, but standard
         dropout is most effective and well-studied in FC layers.
       - Empirical: Adding dropout after the first two FC layers (with p=0.5) is the
         standard VGG practice, validated across many benchmarks.

    3. Dropout p=0.5 in classifier:
       - Balances regularization strength vs. capacity. At p=0.5, exactly half
         the neurons are dropped, maximally diversifying the ensemble of sub-networks.
    """
    def __init__(self, num_classes: int = 37, dropout_p: float = 0.5):
        super(VGG11, self).__init__()
        self.num_classes = num_classes

        # Convolutional Feature Extractor
        # Block 1: 3 → 64
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)   # 224→112
        )

        # Block 2: 64 → 128
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)   # 112→56
        )

        # Block 3: 128 → 256, 256 → 256
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)   # 56→28
        )

        # Block 4: 256 → 512, 512 → 512
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)   # 28→14
        )

        # Block 5: 512 → 512, 512 → 512
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)   # 14→7
        )

        # Adaptive pooling → always 7×7 regardless of input size 
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        #  Classifier (FC layers)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),          # Custom dropout after FC1

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),          # Custom dropout after FC2

            nn.Linear(4096, num_classes)         # No activation: raw logits
        )

        # Weight Initialization 
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolutional backbone
        x = self.block1(x)   # [B, 64, 112, 112]
        x = self.block2(x)   # [B, 128, 56, 56]
        x = self.block3(x)   # [B, 256, 28, 28]
        x = self.block4(x)   # [B, 512, 14, 14]
        x = self.block5(x)   # [B, 512, 7, 7]
        x = self.avgpool(x)  # [B, 512, 7, 7]

        # Flatten for FC
        x = torch.flatten(x, 1)  # [B, 25088]

        # Classification head
        x = self.classifier(x)   # [B, 37]
        return x

    def get_backbone(self):
        """Returns only the convolutional feature extractor blocks."""
        return nn.Sequential(
            self.block1, self.block2, self.block3,
            self.block4, self.block5
        )

    def _initialize_weights(self):
        """He/Kaiming init for conv; Xavier for linear."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
