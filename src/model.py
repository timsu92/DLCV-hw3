from __future__ import annotations
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet101_Weights


def _enable_checkpointing(layer: nn.Sequential) -> None:
    """Wrap each block in a ResNet layer with gradient checkpointing."""
    for block in layer:
        orig = block.forward
        block.forward = lambda x, _f=orig: checkpoint(_f, x, use_reentrant=False)


def build_model(
    num_classes: int = 5,
    max_size: int = 2000,
    grad_checkpoint: bool = False,
) -> MaskRCNN:
    """Build ResNet101-FPN Mask R-CNN.

    num_classes: 4 cell types + 1 background = 5.
    max_size: maximum image side length after resizing (default 2000 for training).
    grad_checkpoint: enable gradient checkpointing on ResNet layer2/3/4 to
        reduce activation memory ~30-40% at the cost of one extra forward pass.
    """
    backbone = resnet_fpn_backbone(
        backbone_name="resnet101",
        weights=ResNet101_Weights.IMAGENET1K_V2,
        trainable_layers=5,  # fine-tune entire backbone for domain adaptation
    )

    if grad_checkpoint:
        for layer_name in ("layer2", "layer3", "layer4"):
            _enable_checkpointing(getattr(backbone.body, layer_name))

    anchor_generator = AnchorGenerator(
        sizes=((8, 16), (32, 64), (64, 128), (128, 256), (256, 512)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )

    model = MaskRCNN(
        backbone,
        num_classes=num_classes,
        min_size=(640, 704, 768, 832, 896, 1024),
        max_size=max_size,
        rpn_anchor_generator=anchor_generator,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )
    return model
