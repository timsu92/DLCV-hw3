from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import tifffile
import torch
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset
from torchvision import tv_tensors

from src.utils import (
    binary_mask_to_bbox,
    encode_mask,
    load_mask,
    load_rgb,
    mask_to_instances,
    rle_to_bytes,
)

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
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": img_idx,
                        "category_id": cat_id,
                        "segmentation": rle,
                        "bbox": bbox,
                        "area": float(binary.sum()),
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

    return {"images": images, "annotations": annotations, "categories": CATEGORIES}


class CellDataset(Dataset):
    """Torchvision-compatible detection dataset wrapping COCO-format annotations."""

    def __init__(
        self,
        train_dir: Path,
        coco_data: dict,
        transforms=None,
    ):
        self.train_dir = train_dir
        self.transforms = transforms
        self.images = coco_data["images"]
        self._ann_by_image: dict[int, list[dict]] = {}
        for ann in coco_data["annotations"]:
            self._ann_by_image.setdefault(ann["image_id"], []).append(ann)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        info = self.images[idx]
        img_arr = load_rgb(
            self.train_dir / info["file_name"] / "image.tif"
        )  # (H, W, 3) uint8
        H, W = img_arr.shape[:2]

        anns = self._ann_by_image.get(info["id"], [])
        boxes, labels, masks = [], [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])  # XYXY
            labels.append(ann["category_id"])
            decoded = mask_utils.decode(
                rle_to_bytes(ann["segmentation"])
            )  # (H, W) uint8
            masks.append(decoded)

        img_t = tv_tensors.Image(
            torch.from_numpy(img_arr).permute(2, 0, 1)  # (3, H, W) uint8
        )

        if boxes:
            boxes_t = tv_tensors.BoundingBoxes(
                torch.tensor(boxes, dtype=torch.float32),
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(H, W),
            )
            masks_t = tv_tensors.Mask(
                torch.from_numpy(np.stack(masks, axis=0).astype(np.uint8))
            )
            labels_t = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_t = tv_tensors.BoundingBoxes(
                torch.zeros((0, 4), dtype=torch.float32),
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(H, W),
            )
            masks_t = tv_tensors.Mask(torch.zeros((0, H, W), dtype=torch.uint8))
            labels_t = torch.zeros(0, dtype=torch.int64)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "masks": masks_t,
            "image_id": torch.tensor([info["id"]]),
        }

        if self.transforms is not None:
            img_t, target = self.transforms(img_t, target)

        return img_t, target


def oversample_rare_classes(
    train_dir: Path,
    folders: list[str],
    rare_classes: tuple[int, ...] = (3, 4),
    factor: int = 3,
) -> list[str]:
    """Return folder list with rare-class images repeated `factor` times total."""
    result = list(folders)
    for folder in folders:
        has_rare = any(
            (train_dir / folder / f"class{c}.tif").exists() for c in rare_classes
        )
        if has_rare:
            result.extend([folder] * (factor - 1))
    return result


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
    print(
        f"Building annotations: {len(train_folders)} train, {len(val_folders)} val..."
    )
    train_coco = build_coco_annotations(train_dir, train_folders)
    val_coco = build_coco_annotations(train_dir, val_folders)

    cache_train.parent.mkdir(parents=True, exist_ok=True)
    cache_train.write_text(json.dumps(train_coco))
    cache_val.write_text(json.dumps(val_coco))
    print("Annotations cached.")
    return train_coco, val_coco
