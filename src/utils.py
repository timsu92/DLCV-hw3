from __future__ import annotations
from pathlib import Path
import numpy as np
import tifffile
from pycocotools import mask as mask_utils


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
