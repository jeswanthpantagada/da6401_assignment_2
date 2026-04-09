import torch
import torch.nn as nn
from .layers import CustomDropout


class VGG11Classifier(nn.Module):
    """
    Full VGG11 Classification model (encoder + classifier head).

    Design Choices:
    ---------------
    BatchNorm2d: placed after every Conv2d, before ReLU.
        - Reduces internal covariate shift
        - Allows higher learning rates
        - Placed before ReLU so normalization acts on unbounded values

    CustomDropout: placed only in FC layers (p=0.5).
        - Not in conv layers: spatial weight sharing makes standard
          dropout ineffective there
        - In FC layers: prevents co-adaptation, maximally diversifies
          the implicit ensemble of sub-networks
    """
    def __init__(self, num_classes: int = 37, dropout_p: float = 0.5):
        super(VGG11Classifier, self).__init__()
        self.num_classes = num_classes

        # Block 1: 3→64, pool→112×112
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Block 2: 64→128, pool→56×56
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Block 3: 128→256→256, pool→28×28
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Block 4: 256→512→512, pool→14×14
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Block 5: 512→512→512, pool→7×7
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # Classifier head with CustomDropout
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, num_classes),
        )

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)    # [B, 64,  112, 112]
        x = self.block2(x)    # [B, 128,  56,  56]
        x = self.block3(x)    # [B, 256,  28,  28]
        x = self.block4(x)    # [B, 512,  14,  14]
        x = self.block5(x)    # [B, 512,   7,   7]
        x = self.avgpool(x)   # [B, 512,   7,   7]
        x = torch.flatten(x, 1)         # [B, 25088]
        x = self.classifier(x)          # [B, 37]
        return x

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
