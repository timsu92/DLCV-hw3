import numpy as np
import pytest
from pycocotools import mask as mask_utils


def test_mask_to_instances_counts():
    """Mask with values [0,1,2,3] → 3 binary masks."""
    from src.utils import mask_to_instances
    mask = np.array([[0, 1, 2], [3, 0, 1], [2, 3, 0]], dtype=np.float64)
    instances = mask_to_instances(mask)
    assert len(instances) == 3


def test_mask_to_instances_binary():
    """Each returned mask is binary and covers exactly the right pixels."""
    from src.utils import mask_to_instances
    mask = np.array([[0, 1, 1], [2, 0, 1]], dtype=np.float64)
    instances = mask_to_instances(mask)
    # instance for value 1: pixels (0,1),(0,2),(1,2)
    combined = sum(m.astype(int) for m in instances)
    assert combined.max() == 1  # no pixel belongs to two instances
    assert combined.sum() == (mask > 0).sum()


def test_binary_mask_to_bbox():
    """Known mask → known [x, y, w, h] bbox."""
    from src.utils import binary_mask_to_bbox
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 3:7] = True  # rows 2-4, cols 3-6
    x, y, w, h = binary_mask_to_bbox(mask)
    assert x == 3.0
    assert y == 2.0
    assert w == 4.0  # cols 3,4,5,6 → width 4
    assert h == 3.0  # rows 2,3,4 → height 3


def test_encode_mask_roundtrip():
    """encode_mask → pycocotools decode → original mask."""
    from src.utils import encode_mask
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:10, 5:10] = True
    rle = encode_mask(mask)
    assert isinstance(rle["counts"], str)
    assert rle["size"] == [20, 20]
    decoded = mask_utils.decode({"size": rle["size"], "counts": rle["counts"].encode("utf-8")})
    np.testing.assert_array_equal(decoded, mask.astype(np.uint8))


def test_load_rgb_drops_alpha(tmp_path):
    """load_rgb returns (H, W, 3) uint8, dropping the 4th channel."""
    import tifffile
    from src.utils import load_rgb
    img_4ch = np.random.randint(0, 255, (8, 8, 4), dtype=np.uint8)
    img_path = tmp_path / "test.tif"
    tifffile.imwrite(str(img_path), img_4ch)
    rgb = load_rgb(img_path)
    assert rgb.shape == (8, 8, 3)
    assert rgb.dtype == np.uint8
    np.testing.assert_array_equal(rgb, img_4ch[:, :, :3])


def test_binary_mask_to_bbox_empty_raises():
    """Empty mask raises ValueError."""
    from src.utils import binary_mask_to_bbox
    mask = np.zeros((10, 10), dtype=bool)
    with pytest.raises(ValueError, match="empty mask"):
        binary_mask_to_bbox(mask)
