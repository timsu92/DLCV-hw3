from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from torchvision.ops import nms as _box_nms
from torchvision.transforms import v2


def load_rgb(path: str | Path) -> np.ndarray:
    """Load TIFF image. Returns (H, W, 3) uint8. Converts grayscale or drops alpha."""
    img = tifffile.imread(str(path))
    if img.ndim == 2:
        # Grayscale → repeat across 3 channels
        img = np.stack([img, img, img], axis=-1)
    return img[:, :, :3]


def load_mask(path: str | Path) -> np.ndarray:
    """Load instance mask TIFF. Returns (H, W) float64 where 0 = background."""
    return tifffile.imread(str(path))


def mask_to_instances(mask: np.ndarray) -> list[np.ndarray]:
    """Return list of boolean binary masks, one per unique non-zero value."""
    ids = np.unique(mask)
    ids = ids[ids > 0]
    return [(mask == v) for v in ids]


def binary_mask_to_bbox(binary_mask: np.ndarray) -> list[float]:
    """Return [x, y, w, h] bounding box (COCO format) from boolean mask."""
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    if not rows.any() or not cols.any():
        raise ValueError("binary_mask_to_bbox called on empty mask (all zeros)")
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [float(cmin), float(rmin), float(cmax - cmin + 1), float(rmax - rmin + 1)]


def encode_mask(binary_mask: np.ndarray) -> dict:
    """Encode boolean mask to COCO RLE dict with counts as UTF-8 string."""
    arr = np.asfortranarray(binary_mask).astype(np.uint8)
    rle = mask_utils.encode(arr)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def rle_to_bytes(rle: dict) -> dict:
    """Convert JSON-serialised RLE (counts as str) back to pycocotools format (counts as bytes)."""
    return {"size": rle["size"], "counts": rle["counts"].encode("utf-8")}


def resize_binary_mask(binary: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Nearest-neighbour resize of a boolean mask to (target_h, target_w).

    Used to scale model output masks (at inference resolution) back to the
    original image resolution required by COCOeval.
    """
    if binary.shape == (target_h, target_w):
        return binary
    pil = Image.fromarray(binary.astype(np.uint8))
    return np.array(pil.resize((target_w, target_h), Image.NEAREST), dtype=bool)


def pre_resize_image(
    img: np.ndarray, size: int = 1024, max_size: int = 1025
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Resize image so shorter side == `size`, then cap longer side to `max_size`.

    Mirrors `v2.Resize(size, max_size, antialias=True)` + `v2.ToDtype(float32, scale=True)`
    from `get_val_transform` (so val/inference share identical preprocessing).
    Both default to 1024 to match the model's eval `min_size[-1]=1024, max_size=1024`,
    so no further resize happens inside `GeneralizedRCNNTransform`. The cap also
    bounds `paste_masks_in_image` memory on extreme-aspect-ratio test images.

    Returns a (3, H, W) float32 tensor in [0, 1] and the original (h, w) tuple.
    """
    orig_h, orig_w = img.shape[:2]
    img_t = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W) uint8
    img_t = v2.functional.resize(img_t, size=[size], max_size=max_size, antialias=True)
    return img_t.to(torch.float32) / 255.0, (orig_h, orig_w)


def cross_class_nms(pred: dict, iou_threshold: float = 0.5) -> dict:
    """Class-agnostic NMS to suppress cross-class duplicate detections.

    Mask R-CNN's built-in NMS is per-class, so the same cell can appear as two
    different categories. This applies a second class-agnostic pass: if two boxes
    overlap by > iou_threshold, only the higher-score prediction is kept.
    """
    boxes = pred["boxes"]
    if len(boxes) == 0:
        return pred
    keep = _box_nms(boxes, pred["scores"], iou_threshold)
    return {k: v[keep] for k, v in pred.items()}
