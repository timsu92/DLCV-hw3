"""Dataset statistics — kept in repo for report reference."""

from __future__ import annotations
import json
import os
import sys
from pathlib import Path
import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_mask, mask_to_instances

TRAIN_DIR = Path("data/train")


def main() -> None:
    folders = sorted(os.listdir(TRAIN_DIR))

    class_instance_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    images_with_class = {1: 0, 2: 0, 3: 0, 4: 0}
    image_sizes: list[tuple[int, int]] = []
    instances_per_image: list[int] = []

    for folder in folders:
        img = tifffile.imread(str(TRAIN_DIR / folder / "image.tif"))
        H, W = img.shape[:2]
        image_sizes.append((H, W))

        total = 0
        for c in range(1, 5):
            mask_path = TRAIN_DIR / folder / f"class{c}.tif"
            if not mask_path.exists():
                continue
            mask = load_mask(mask_path)
            n = len(mask_to_instances(mask))
            class_instance_counts[c] += n
            images_with_class[c] += 1
            total += n
        instances_per_image.append(total)

    sizes = np.array(image_sizes)
    print(f"Total images: {len(folders)}")
    print(
        f"Image H range: {sizes[:, 0].min()}–{sizes[:, 0].max()}, mean {sizes[:, 0].mean():.1f}"
    )
    print(
        f"Image W range: {sizes[:, 1].min()}–{sizes[:, 1].max()}, mean {sizes[:, 1].mean():.1f}"
    )
    print(f"Images containing each class: {images_with_class}")
    print(f"Total instances per class:    {class_instance_counts}")
    print(
        f"Instances per image — min: {min(instances_per_image)}, "
        f"max: {max(instances_per_image)}, mean: {np.mean(instances_per_image):.1f}"
    )

    stats = {
        "n_images": len(folders),
        "h_min": int(sizes[:, 0].min()),
        "h_max": int(sizes[:, 0].max()),
        "w_min": int(sizes[:, 1].min()),
        "w_max": int(sizes[:, 1].max()),
        "images_with_class": images_with_class,
        "total_instances_per_class": class_instance_counts,
        "instances_per_image_min": min(instances_per_image),
        "instances_per_image_max": max(instances_per_image),
        "instances_per_image_mean": float(np.mean(instances_per_image)),
    }
    out = Path("analysis/output/dataset_stats.json")
    out.write_text(json.dumps(stats, indent=2))
    print(f"\nSaved stats to {out}")


if __name__ == "__main__":
    main()
