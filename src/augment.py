import torch
from torchvision.transforms import v2


def get_train_transform():
    """Augmentation for training: spatial flips + colour distortion + dtype conversion.

    MaskRCNN's internal GeneralizedRCNNTransform handles normalisation and
    multi-scale resizing, so we only apply spatial and photometric augmentation here.
    """
    return v2.Compose(
        [
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomPhotometricDistort(p=1.0),
            v2.ToDtype(torch.float32, scale=True),  # uint8 → float [0, 1]
        ]
    )


def get_val_transform():
    """Validation: dtype conversion only (no augmentation)."""
    return v2.Compose(
        [
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
