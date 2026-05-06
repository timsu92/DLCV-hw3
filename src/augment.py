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
    """Validation: dtype conversion only (no augmentation, no resize).

    No pre-resize here: during inference target=None so _resize_image_and_masks
    skips the mask float32 cast entirely. Keeping original resolution ensures
    MaskRCNN postprocess rescales predictions back to the GT annotation size.
    """
    return v2.Compose(
        [
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
