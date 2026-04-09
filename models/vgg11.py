import torch
import torch.nn as nn
from typing import Dict, Tuple, Union


class VGG11Encoder(nn.Module):
    """
    VGG11-style encoder with optional intermediate feature returns.
    Used as the shared backbone for all tasks.

    When return_features=True, returns skip connection feature maps
    needed by the U-Net decoder (segmentation task).
    """
    def __init__(self, in_channels: int = 3):
        super(VGG11Encoder, self).__init__()

        # Block 1: 3→64
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: 64→128
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3: 128→256→256
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4: 256→512→512
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 5: 512→512→512
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self._initialize_weights()

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass.

        Args:
            x: input image tensor [B, 3, H, W].
            return_features: if True, also return skip maps for U-Net decoder.

        Returns:
            If return_features=False: output tensor [B, 512, H/32, W/32]
            If return_features=True : (output, dict of skip tensors)
        """
        s1 = self.block1(x);    e1 = self.pool1(s1)   # 64,  H/2
        s2 = self.block2(e1);   e2 = self.pool2(s2)   # 128, H/4
        s3 = self.block3(e2);   e3 = self.pool3(s3)   # 256, H/8
        s4 = self.block4(e3);   e4 = self.pool4(s4)   # 512, H/16
        s5 = self.block5(e4);   e5 = self.pool5(s5)   # 512, H/32

        if return_features:
            skips = {'s1': s1, 's2': s2, 's3': s3, 's4': s4, 's5': s5}
            return e5, skips
        return e5

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
