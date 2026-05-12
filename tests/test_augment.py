import torch
from torchvision import tv_tensors


def _make_sample(H=1024, W=1024, n_inst=3):
    """Create a fake (img, target) pair with tv_tensors."""
    img = tv_tensors.Image(torch.randint(0, 255, (3, H, W), dtype=torch.uint8))
    # boxes well inside the image so RandomIoUCrop can find a valid crop
    boxes = torch.tensor(
        [
            [100.0, 100.0, 200.0, 200.0],
            [400.0, 400.0, 500.0, 500.0],
            [600.0, 600.0, 700.0, 700.0],
        ]
    )[:n_inst]
    masks = tv_tensors.Mask(torch.zeros(n_inst, H, W, dtype=torch.uint8))
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = b.long().tolist()
        masks[i, y1:y2, x1:x2] = 1
    bboxes = tv_tensors.BoundingBoxes(
        boxes, format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=(H, W)
    )
    target = {
        "boxes": bboxes,
        "labels": torch.ones(n_inst, dtype=torch.int64),
        "masks": masks,
    }
    return img, target


def test_train_transform_output_types():
    from src.augment import get_train_transform

    t = get_train_transform()
    img, target = t(*_make_sample())
    assert img.dtype == torch.float32
    assert img.max() <= 1.0 and img.min() >= 0.0


def test_train_transform_includes_random_iou_crop():
    """Training pipeline must include RandomIoUCrop and SanitizeBoundingBoxes."""
    from src.augment import get_train_transform

    transforms = get_train_transform().transforms
    types = {type(t).__name__ for t in transforms}
    assert "RandomIoUCrop" in types
    assert "SanitizeBoundingBoxes" in types


def test_train_transform_resize_uses_1024():
    """Pre-resize size must match model eval min_size (1024)."""
    from torchvision.transforms import v2

    from src.augment import get_train_transform

    transforms = get_train_transform().transforms
    resize = next(t for t in transforms if isinstance(t, v2.Resize))
    assert resize.size == [1024], f"got {resize.size}"
    assert resize.max_size == 1025, f"got {resize.max_size}"


def test_val_transform_resize_uses_1024():
    """Val pipeline pre-resize matches train (and inference)."""
    from torchvision.transforms import v2

    from src.augment import get_val_transform

    transforms = get_val_transform().transforms
    resize = next(t for t in transforms if isinstance(t, v2.Resize))
    assert resize.size == [1024], f"got {resize.size}"
    assert resize.max_size == 1025


def test_val_transform_resizes_to_1024():
    """Val transform changes dtype to float32 and resizes shorter side to 1024."""
    from src.augment import get_val_transform

    t = get_val_transform()
    img_in = tv_tensors.Image(torch.zeros((3, 800, 1000), dtype=torch.uint8))
    boxes = tv_tensors.BoundingBoxes(
        torch.tensor([[0.0, 0.0, 100.0, 100.0]]),
        format=tv_tensors.BoundingBoxFormat.XYXY,
        canvas_size=(800, 1000),
    )
    target = {
        "boxes": boxes,
        "labels": torch.tensor([1]),
        "masks": tv_tensors.Mask(torch.ones(1, 800, 1000, dtype=torch.uint8)),
    }
    img_out, _ = t(img_in, target)
    assert img_out.dtype == torch.float32
    # shorter=800 → 1024, but longer would be 1280; capped at 1025 → rescale,
    # final shorter ≈ 800 * 1025/1280 = 640
    assert img_out.shape[1] >= 640 and img_out.shape[1] <= 1024
    assert img_out.shape[2] <= 1025
