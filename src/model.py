from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from torchvision.models import ResNet101_Weights
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        r = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(r, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        b, c = x.shape[:2]
        avg = self.fc(self.avg_pool(x).view(b, c))
        mx = self.fc(self.max_pool(x).view(b, c))
        return x * self.sigmoid(avg + mx).view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.spatial(self.channel(x))


class CBAMBackboneWrapper(nn.Module):
    """Wraps BackboneWithFPN to apply CBAM between body and FPN."""

    def __init__(self, backbone, cbam_keys: dict[str, int]):
        super().__init__()
        self.wrapped = backbone
        self.cbams = nn.ModuleDict({k: CBAM(c) for k, c in cbam_keys.items()})
        self.out_channels = backbone.out_channels

    def forward(self, x: Tensor):
        features = self.wrapped.body(x)
        for key, mod in self.cbams.items():
            if key in features:
                features[key] = mod(features[key])
        return self.wrapped.fpn(features)


def _enable_checkpointing(layer: nn.Sequential) -> None:
    """Wrap each block in a ResNet layer with gradient checkpointing."""
    for block in layer:
        orig = block.forward
        block.forward = lambda x, _f=orig: checkpoint(_f, x, use_reentrant=False)


def build_model(
    num_classes: int = 5,
    min_size: tuple[int, ...] = (640, 768, 896, 1024),
    max_size: int = 1024,
    grad_checkpoint: bool = False,
    use_cbam: bool = False,
) -> MaskRCNN:
    """Build ResNet101-FPN Mask R-CNN.

    num_classes: 4 cell types + 1 background = 5.
    min_size: shorter-side targets for multi-scale training (multiples of 32 align
        cleanly with FPN strides 4/8/16/32/64). Eval uses min_size[-1]=1024.
    max_size: maximum image side length after resizing.
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

    if use_cbam:
        backbone = CBAMBackboneWrapper(backbone, {"2": 1024, "3": 2048})

    # Anchor sizes shifted down by one octave compared to torchvision default to
    # add size=4 coverage at P2 — ~5% of instances have sqrt(area) < 16 at model
    # input. Trade-off: anchor-256/512 dropped (0.22% of instances > sqrt(256)
    # rely on box regression from the anchor-128 match instead). Keeps 6 anchors
    # per FPN level uniformly so RPNHead's single num_anchors works.
    anchor_generator = AnchorGenerator(
        sizes=((4, 8), (16, 32), (32, 64), (64, 128), (128, 256)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )

    model = MaskRCNN(
        backbone,
        num_classes=num_classes,
        min_size=min_size,
        max_size=max_size,
        rpn_anchor_generator=anchor_generator,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        # Dataset has up to 772 instances per image; raise the three-stage funnel
        # (pre-NMS → post-NMS → final detections) to avoid capping recall.
        rpn_pre_nms_top_n_test=2600,
        rpn_post_nms_top_n_test=1500,
        box_detections_per_img=1000,
    )
    return model
