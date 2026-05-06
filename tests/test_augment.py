import torch
import numpy as np
from torchvision import tv_tensors


def _make_sample(H=64, W=64, n_inst=3):
    """Create a fake (img, target) pair with tv_tensors."""
    img = tv_tensors.Image(torch.randint(0, 255, (3, H, W), dtype=torch.uint8))
    boxes = torch.tensor([[5., 5., 20., 20.], [30., 30., 50., 50.], [10., 40., 40., 60.]])[:n_inst]
    masks = tv_tensors.Mask(torch.randint(0, 2, (n_inst, H, W), dtype=torch.uint8))
    bboxes = tv_tensors.BoundingBoxes(boxes, format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=(H, W))
    target = {"boxes": bboxes, "labels": torch.ones(n_inst, dtype=torch.int64), "masks": masks}
    return img, target


def test_train_transform_output_types():
    from src.augment import get_train_transform
    t = get_train_transform()
    img, target = t(*_make_sample())
    assert img.dtype == torch.float32
    assert img.max() <= 1.0 and img.min() >= 0.0


def test_train_transform_preserves_instance_count():
    from src.augment import get_train_transform
    t = get_train_transform()
    _, original_target = _make_sample(n_inst=3)
    _, transformed_target = t(*_make_sample(n_inst=3))
    assert len(transformed_target["boxes"]) == len(original_target["boxes"])
    assert len(transformed_target["masks"]) == len(original_target["masks"])


def test_val_transform_no_spatial_change():
    """Val transform only changes dtype, not spatial content."""
    from src.augment import get_val_transform
    t = get_val_transform()
    img_in = tv_tensors.Image(torch.arange(0, 3*4*4).reshape(3, 4, 4).to(torch.uint8))
    boxes = tv_tensors.BoundingBoxes(torch.tensor([[0., 0., 2., 2.]]),
                                     format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=(4, 4))
    target = {"boxes": boxes, "labels": torch.tensor([1]), "masks": tv_tensors.Mask(torch.ones(1, 4, 4, dtype=torch.uint8))}
    img_out, _ = t(img_in, target)
    assert img_out.dtype == torch.float32
    assert img_out.shape == img_in.shape
