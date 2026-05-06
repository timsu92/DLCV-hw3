"""Interactive submission visualizer.

Usage:
    uv run python analysis/visualize_pred.py
    uv run python analysis/visualize_pred.py --results path/to/test-results.json

Controls:
  →  /  d  : next image
  ←  /  a  : previous image
  0-9      : build jump number
  Enter    : jump to that image (1-based index in sorted test filenames)
  Backspace: delete last digit
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from pycocotools import mask as mask_utils

from src.utils import load_rgb, rle_to_bytes

TEST_DIR = Path("data/test_release")
DEFAULT_RESULTS = Path("test-results.json")
ID_MAP_PATH = Path("data/test_image_name_to_ids.json")

CATEGORY_NAMES = {1: "class1", 2: "class2", 3: "class3", 4: "class4"}


class PredViewer:
    def __init__(self, test_dir: Path, results_path: Path, id_map_path: Path) -> None:
        with open(id_map_path) as f:
            id_list = json.load(f)
        # filename → image_id (1-based)
        self.name_to_id: dict[str, int] = {e["file_name"]: e["id"] for e in id_list}
        # Sort test images alphabetically — same convention as GT viewer
        self.filenames = sorted(self.name_to_id.keys())
        self.n = len(self.filenames)

        with open(results_path) as f:
            all_preds = json.load(f)
        # Group predictions by image_id
        self.preds_by_id: dict[int, list[dict]] = {}
        for p in all_preds:
            self.preds_by_id.setdefault(p["image_id"], []).append(p)

        self.test_dir = test_dir
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
            target = int(self.num_buf) - 1
            self.idx = max(0, min(target, self.n - 1))
            self.num_buf = ""
        elif key == "backspace":
            self.num_buf = self.num_buf[:-1]
        else:
            return
        self._render()

    def _render(self) -> None:
        self.ax.clear()
        filename = self.filenames[self.idx]
        image_id = self.name_to_id[filename]
        img = load_rgb(self.test_dir / filename)
        self.ax.imshow(img)

        preds = self.preds_by_id.get(image_id, [])
        # Sort by score ascending so higher-confidence masks draw on top
        preds = sorted(preds, key=lambda p: p["score"])

        rng = np.random.default_rng(seed=0)
        overlay = np.zeros((*img.shape[:2], 4), dtype=np.float32)

        for pred in preds:
            colour = rng.random(3)
            binary = mask_utils.decode(rle_to_bytes(pred["segmentation"])).astype(bool)
            overlay[binary, :3] = colour
            overlay[binary, 3] = 0.45

            x, y, w, h = pred["bbox"]
            cat_name = CATEGORY_NAMES.get(
                pred["category_id"], f"cat{pred['category_id']}"
            )
            self.ax.add_patch(
                mpatches.Rectangle(
                    (x, y), w, h, linewidth=1, edgecolor=colour, facecolor="none"
                )
            )
            self.ax.text(
                x,
                max(0, y - 3),
                f"{cat_name} {pred['score']:.2f}",
                color=colour,
                fontsize=7,
                clip_on=True,
            )

        self.ax.imshow(overlay, interpolation="nearest")

        jump_hint = f"  (jump: {self.num_buf}▌)" if self.num_buf else ""
        self.ax.set_title(
            f"[{self.idx + 1}/{self.n}]  {filename}  —  {len(preds)} predictions{jump_hint}",
            fontsize=10,
        )
        self.ax.axis("off")
        self.fig.canvas.draw_idle()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--test-dir", type=Path, default=TEST_DIR)
    parser.add_argument("--id-map", type=Path, default=ID_MAP_PATH)
    args = parser.parse_args()
    PredViewer(args.test_dir, args.results, args.id_map)


if __name__ == "__main__":
    main()
