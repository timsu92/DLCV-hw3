import torch
from torchvision.transforms import v2

_PRE_RESIZE = (
    640  # pre-resize longer side to cap GPU mask memory in _resize_image_and_masks
)


def get_train_transform():
    """Augmentation for training: pre-resize → spatial flips → colour distortion → float.

    The CPU-side Resize(640) caps image/mask dimensions before GPU transfer so that
    torchvision's internal _resize_image_and_masks never creates a massive float32
    mask tensor from the original high-res image. Flips and photometric distortion
    run after resize for efficiency.
    """
    return v2.Compose(
        [
            v2.Resize(_PRE_RESIZE, antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomPhotometricDistort(p=1.0),
            v2.ToDtype(torch.float32, scale=True),  # uint8 → float [0, 1]
        ]
    )


def get_val_transform():
    """Validation: pre-resize + dtype conversion only (no augmentation).

    Resize(640) is required here to prevent paste_masks_in_image OOM: during
    eval the model pastes predicted masks onto the "original" input canvas. At
    full resolution (~1772×1731) with 500-1000 predictions this requires
    500×1772×1731×4 B ≈ 6+ GiB. At 640 px it's ~1 GiB.

    After inference, evaluate() scales predicted masks back to the true original
    size before RLE-encoding so COCOeval IoU is computed correctly.
    """
    return v2.Compose(
        [
            v2.Resize(_PRE_RESIZE, antialias=True),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
