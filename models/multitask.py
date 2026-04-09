"""Unified multi-task model."""

import os

import torch
import torch.nn as nn

from .classification import VGG11Classifier
from .localization import LocalizationModel
from .segmentation import UNetVGG11
from .vgg11 import VGG11Encoder


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        classifier_path: str = "classifier.pth",
        localizer_path: str = "localizer.pth",
        unet_path: str = "unet.pth",
        num_classes: int = None,
    ):
        """
        Initialize the shared backbone/heads using these trained weights.
        Args:
            num_breeds: Number of output classes for classification head.
            seg_classes: Number of output classes for segmentation head.
            in_channels: Number of input channels.
            classifier_path: Path to trained classifier weights.
            localizer_path: Path to trained localizer weights.
            unet_path: Path to trained unet weights.
        """
        super(MultiTaskPerceptionModel, self).__init__()

        if num_classes is not None:
            num_breeds = num_classes

        os.makedirs("checkpoints", exist_ok=True)
        classifier_path = os.path.join("checkpoints", os.path.basename(classifier_path))
        localizer_path = os.path.join("checkpoints", os.path.basename(localizer_path))
        unet_path = os.path.join("checkpoints", os.path.basename(unet_path))

        import gdown
        gdown.download(id="1JgctJgD9EP8PgL8--0kGrKhfRCWdhRlA", output=classifier_path, quiet=False)
        gdown.download(id="1M7Lp9zrOneDXCcxlB3-8JNJwJl7zKhCM", output=localizer_path, quiet=False)
        gdown.download(id="13pAD3ziJMXZAzwWxFSPjQlA6VvvE2-6D", output=unet_path, quiet=False)

        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        classifier_model = VGG11Classifier(num_classes=num_breeds)
        localization_model = LocalizationModel(
            VGG11Encoder(in_channels=in_channels), freeze_early=False
        )
        segmentation_model = UNetVGG11(
            VGG11Encoder(in_channels=in_channels), num_classes=seg_classes
        )

        self.classifier_head = classifier_model.classifier
        self.regression_head = localization_model.regression_head
        self.bottleneck = segmentation_model.bottleneck
        self.up5 = segmentation_model.up5
        self.dec5 = segmentation_model.dec5
        self.up4 = segmentation_model.up4
        self.dec4 = segmentation_model.dec4
        self.up3 = segmentation_model.up3
        self.dec3 = segmentation_model.dec3
        self.up2 = segmentation_model.up2
        self.dec2 = segmentation_model.dec2
        self.up1 = segmentation_model.up1
        self.dec1 = segmentation_model.dec1
        self.out_conv = segmentation_model.out_conv

        self._load_pretrained_weights(
            classifier_path, localizer_path, unet_path
        )

    def _load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        cleaned_state_dict = {}
        for key, value in checkpoint.items():
            if key.startswith("module."):
                key = key[len("module."):]
            cleaned_state_dict[key] = value

        return cleaned_state_dict

    def _load_module_weights(
        self,
        module: nn.Module,
        state_dict,
        prefix: str = "",
    ):
        module_state = module.state_dict()
        filtered_state_dict = {}

        for key, value in state_dict.items():
            if prefix:
                if not key.startswith(prefix):
                    continue
                key = key[len(prefix):]

            if key in module_state and module_state[key].shape == value.shape:
                filtered_state_dict[key] = value

        if filtered_state_dict:
            module.load_state_dict(filtered_state_dict, strict=False)

    def _load_pretrained_weights(
        self,
        classifier_path: str,
        localizer_path: str,
        unet_path: str,
    ):
        classifier_state = self._load_checkpoint(classifier_path)
        localizer_state = self._load_checkpoint(localizer_path)
        unet_state = self._load_checkpoint(unet_path)

        self._load_module_weights(self.encoder, classifier_state)
        self._load_module_weights(self.classifier_head, classifier_state, "classifier.")
        self._load_module_weights(self.regression_head, localizer_state, "regression_head.")
        self._load_module_weights(self.bottleneck, unet_state, "bottleneck.")
        self._load_module_weights(self.up5, unet_state, "up5.")
        self._load_module_weights(self.dec5, unet_state, "dec5.")
        self._load_module_weights(self.up4, unet_state, "up4.")
        self._load_module_weights(self.dec4, unet_state, "dec4.")
        self._load_module_weights(self.up3, unet_state, "up3.")
        self._load_module_weights(self.dec3, unet_state, "dec3.")
        self._load_module_weights(self.up2, unet_state, "up2.")
        self._load_module_weights(self.dec2, unet_state, "dec2.")
        self._load_module_weights(self.up1, unet_state, "up1.")
        self._load_module_weights(self.dec1, unet_state, "dec1.")
        self._load_module_weights(self.out_conv, unet_state, "out_conv.")

    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor.
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor
        """
        encoded, skips = self.encoder(x, return_features=True)

        pooled = self.avgpool(encoded)
        flattened = torch.flatten(pooled, 1)

        classification = self.classifier_head(flattened)
        localization = self.regression_head(flattened)

        segmentation = self.bottleneck(encoded)
        segmentation = self.up5(segmentation)
        segmentation = torch.cat([segmentation, skips["s5"]], dim=1)
        segmentation = self.dec5(segmentation)
        segmentation = self.up4(segmentation)
        segmentation = torch.cat([segmentation, skips["s4"]], dim=1)
        segmentation = self.dec4(segmentation)
        segmentation = self.up3(segmentation)
        segmentation = torch.cat([segmentation, skips["s3"]], dim=1)
        segmentation = self.dec3(segmentation)
        segmentation = self.up2(segmentation)
        segmentation = torch.cat([segmentation, skips["s2"]], dim=1)
        segmentation = self.dec2(segmentation)
        segmentation = self.up1(segmentation)
        segmentation = torch.cat([segmentation, skips["s1"]], dim=1)
        segmentation = self.dec1(segmentation)
        segmentation = self.out_conv(segmentation)

        return {
            "classification": classification,
            "localization": localization,
            "segmentation": segmentation,
        }


MultiTaskVGG = MultiTaskPerceptionModel
