"""Interactive ground-truth visualizer.

Controls:
  →  /  d  : next image
  ←  /  a  : previous image
  0-9      : build jump number
  Enter    : jump to typed number (1-based)
  Backspace: delete last digit of jump number
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from pycocotools import mask as mask_utils

from src.dataset import load_or_build_annotations
from src.utils import load_rgb, rle_to_bytes

TRAIN_DIR = Path("data/train")
CACHE_TRAIN = Path("data/train_annotations.json")
CACHE_VAL = Path("data/val_annotations.json")


class GTViewer:
    def __init__(self, coco_data: dict, train_dir: Path) -> None:
        # Sort images alphabetically by folder name for consistent ordering
        self.images = sorted(coco_data["images"], key=lambda x: x["file_name"])
        self.ann_by_image: dict[int, list[dict]] = {}
        for ann in coco_data["annotations"]:
            self.ann_by_image.setdefault(ann["image_id"], []).append(ann)

        self.train_dir = train_dir
        self.n = len(self.images)
        self.idx = 0
        self.num_buf = ""

        self.fig, self.ax = plt.subplots(figsize=(12, 9))
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._render()
        plt.tight_layout()
        plt.show()

    def _on_key(self, event) -> None:
        key = event.key
        if key in ("right", "d"):
            self.idx = (self.idx + 1) % self.n
            self.num_buf = ""
        elif key in ("left", "a"):
            self.idx = (self.idx - 1) % self.n
            self.num_buf = ""
        elif key in "0123456789":
            self.num_buf += key
        elif key == "enter" and self.num_buf:
            target = int(self.num_buf) - 1  # 1-based → 0-based
            self.idx = max(0, min(target, self.n - 1))
            self.num_buf = ""
        elif key == "backspace":
            self.num_buf = self.num_buf[:-1]
        else:
            return
        self._render()

    def _render(self) -> None:
        self.ax.clear()
        info = self.images[self.idx]
        img = load_rgb(self.train_dir / info["file_name"] / "image.tif")
        self.ax.imshow(img)

        anns = self.ann_by_image.get(info["id"], [])
        rng = np.random.default_rng(seed=0)  # fixed seed → same colour per image across renders

        overlay = np.zeros((*img.shape[:2], 4), dtype=np.float32)
        for ann in anns:
            colour = rng.random(3)
            binary = mask_utils.decode(rle_to_bytes(ann["segmentation"])).astype(bool)
            overlay[binary, :3] = colour
            overlay[binary, 3] = 0.45

            x, y, w, h = ann["bbox"]
            self.ax.add_patch(
                mpatches.Rectangle((x, y), w, h, linewidth=1,
                                   edgecolor=colour, facecolor="none")
            )
            self.ax.text(x, max(0, y - 3), f"class{ann['category_id']}",
                         color=colour, fontsize=7, clip_on=True)

        self.ax.imshow(overlay, interpolation="nearest")

        jump_hint = f"  (jump: {self.num_buf}▌)" if self.num_buf else ""
        self.ax.set_title(
            f"[{self.idx + 1}/{self.n}]  {info['file_name']}  —  {len(anns)} instances{jump_hint}",
            fontsize=10,
        )
        self.ax.axis("off")
        self.fig.canvas.draw_idle()


def main() -> None:
    train_coco, val_coco = load_or_build_annotations(
        TRAIN_DIR, CACHE_TRAIN, CACHE_VAL
    )
    # Remap train and val IDs separately before merging — both splits use IDs
    # starting from 1, so a single shared id_map would have key collisions.
    n_train = len(train_coco["images"])
    train_remap = {img["id"]: i + 1 for i, img in enumerate(train_coco["images"])}
    val_remap   = {img["id"]: n_train + i + 1 for i, img in enumerate(val_coco["images"])}

    for img in train_coco["images"]:
        img["id"] = train_remap[img["id"]]
    for img in val_coco["images"]:
        img["id"] = val_remap[img["id"]]
    for ann in train_coco["annotations"]:
        ann["image_id"] = train_remap[ann["image_id"]]
    for ann in val_coco["annotations"]:
        ann["image_id"] = val_remap[ann["image_id"]]

    all_images = train_coco["images"] + val_coco["images"]
    all_anns   = train_coco["annotations"] + val_coco["annotations"]
    merged = {"images": all_images, "annotations": all_anns, "categories": train_coco["categories"]}
    GTViewer(merged, TRAIN_DIR)


if __name__ == "__main__":
    main()
