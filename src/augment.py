import torch
from torchvision.transforms import v2

_PRE_RESIZE = 1024
_MAX_SIZE = 1025  # torchvision v2 requires max_size > size strictly


def get_train_transform():
    """Train pipeline:
    - Resize first to bound CPU memory of the per-instance mask stack.
    - Photometric distort runs on uint8 (faster + numerically cleaner than on float).
    - RandomIoUCrop introduces real (not synthetic) aspect-ratio variety:
        * sampler_options[0]=0.0 keeps "no crop" as a valid choice (~14% of samples).
        * aspect 0.2-5.0 covers 99.5% of test image aspect ratios.
        * scale 0.5-1.0 prevents cells (P10 sqrt-area 19px at 1024 input) from
          shrinking past P2 anchor coverage when followed by model multi-scale resize.
    - SanitizeBoundingBoxes removes degenerate fragments after the crop clips
      partial cells. Default min_size=1 keeps all real cells (smallest is 5x5).
    - ToDtype last so prior ops run on uint8.
    """
    return v2.Compose([
        v2.Resize(_PRE_RESIZE, max_size=_MAX_SIZE, antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomPhotometricDistort(p=1.0),
        v2.RandomIoUCrop(
            min_scale=0.5,
            max_scale=1.0,
            min_aspect_ratio=0.2,
            max_aspect_ratio=5.0,
            sampler_options=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
            trials=40,
        ),
        v2.SanitizeBoundingBoxes(),
        v2.ToDtype(torch.float32, scale=True),
    ])


def get_val_transform():
    """Validation: pre-resize + dtype conversion only (no augmentation).

    Resize(1024, max_size=1025) effectively caps both dims near 1024 - matches
    the model's eval min_size[-1]=1024, max_size=1024 (model max_size has no
    `> size` constraint, so model uses 1024 directly while transform uses 1025).
    The model's GeneralizedRCNNTransform may do a tiny additional rescale for
    aspect-ratio > 1 images. After inference, evaluate() scales predicted masks
    back to the true original size before RLE-encoding so COCOeval IoU is
    computed correctly.
    """
    return v2.Compose([
        v2.Resize(_PRE_RESIZE, max_size=_MAX_SIZE, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
    ])
