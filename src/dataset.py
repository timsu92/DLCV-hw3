from __future__ import annotations
import json
import random
from pathlib import Path
import numpy as np
import tifffile
from src.utils import load_mask, mask_to_instances, binary_mask_to_bbox, encode_mask

CATEGORIES = [
    {"id": 1, "name": "class1", "supercategory": "cell"},
    {"id": 2, "name": "class2", "supercategory": "cell"},
    {"id": 3, "name": "class3", "supercategory": "cell"},
    {"id": 4, "name": "class4", "supercategory": "cell"},
]


def split_images(
    train_dir: Path,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Return (train_folders, val_folders) split. Shuffled with fixed seed."""
    all_folders = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    rng = random.Random(seed)
    shuffled = all_folders.copy()
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_fraction))
    return shuffled[n_val:], shuffled[:n_val]


def build_coco_annotations(train_dir: Path, folders: list[str]) -> dict:
    """Build COCO-format annotations dict for the given folder list.

    Instance IDs are globally unique across class files for a given image
    (class1.tif owns IDs 1..N, class2.tif owns N+1..M).
    category_id is determined by which class file the instance appears in.
    """
    images = []
    annotations = []
    ann_id = 1

    for img_idx, folder in enumerate(folders, start=1):
        img_path = train_dir / folder / "image.tif"
        img_arr = tifffile.imread(str(img_path))
        H, W = img_arr.shape[:2]
        images.append({"id": img_idx, "file_name": folder, "height": H, "width": W})

        for cat_id in range(1, 5):
            mask_path = train_dir / folder / f"class{cat_id}.tif"
            if not mask_path.exists():
                continue
            mask = load_mask(mask_path)
            for binary in mask_to_instances(mask):
                bbox = binary_mask_to_bbox(binary)
                rle = encode_mask(binary)
                annotations.append({
                    "id": ann_id,
                    "image_id": img_idx,
                    "category_id": cat_id,
                    "segmentation": rle,
                    "bbox": bbox,
                    "area": float(binary.sum()),
                    "iscrowd": 0,
                })
                ann_id += 1

    return {"images": images, "annotations": annotations, "categories": CATEGORIES}


def load_or_build_annotations(
    train_dir: Path,
    cache_train: Path,
    cache_val: Path,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[dict, dict]:
    """Return (train_coco, val_coco), building and caching if needed."""
    if cache_train.exists() and cache_val.exists():
        with open(cache_train) as f:
            train_coco = json.load(f)
        with open(cache_val) as f:
            val_coco = json.load(f)
        return train_coco, val_coco

    train_folders, val_folders = split_images(train_dir, val_fraction, seed)
    print(f"Building annotations: {len(train_folders)} train, {len(val_folders)} val...")
    train_coco = build_coco_annotations(train_dir, train_folders)
    val_coco = build_coco_annotations(train_dir, val_folders)

    cache_train.parent.mkdir(parents=True, exist_ok=True)
    cache_train.write_text(json.dumps(train_coco))
    cache_val.write_text(json.dumps(val_coco))
    print("Annotations cached.")
    return train_coco, val_coco
