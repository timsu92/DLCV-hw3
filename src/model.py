from __future__ import annotations
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet101_Weights


def build_model(num_classes: int = 5, max_size: int = 2000) -> MaskRCNN:
    """Build ResNet101-FPN Mask R-CNN.

    num_classes: 4 cell types + 1 background = 5.
    Anchor sizes are extended with small anchors (8px, 16px) on high-resolution
    FPN levels to detect tiny cells. 5 tuples match 5 FPN feature maps (P2–P6).
    max_size: maximum image side length after resizing (default 2000 for training).
    """
    backbone = resnet_fpn_backbone(
        backbone_name="resnet101",
        weights=ResNet101_Weights.IMAGENET1K_V2,
        trainable_layers=5,  # fine-tune entire backbone for domain adaptation
    )

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
