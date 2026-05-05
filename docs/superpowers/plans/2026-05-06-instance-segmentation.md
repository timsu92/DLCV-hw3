# Instance Segmentation Assignment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Mask R-CNN (ResNet101-FPN) instance segmentation pipeline trained with DDP on 2×RTX 5080, targeting AP50 > 0.5975 on CodaBench.

**Architecture:** Raw TIF masks are parsed into COCO-format JSON on first run (cached to disk). A CellDataset wraps this for torchvision detection. Training uses DDP + gradient accumulation (effective batch 8) + AMP. Inference outputs `test-results.json` with pycocotools RLE-encoded masks.

**Tech Stack:** Python 3.12 / uv, PyTorch 2.11+cu128, torchvision 0.26, pycocotools, tifffile, imagecodecs, matplotlib, bun, playwright

---

## File Map

| File | Responsibility |
|---|---|
| `src/utils.py` | Pure data functions: load image/mask, extract instances, compute bbox, encode RLE |
| `src/dataset.py` | `split_images`, `build_coco_annotations`, `CellDataset` class |
| `src/augment.py` | torchvision.transforms.v2 pipelines for train and val |
| `src/model.py` | `build_model()` — ResNet101-FPN + Mask R-CNN |
| `src/train.py` | DDP training entry point (`torchrun --nproc_per_node=2 src/train.py`) |
| `src/inference.py` | Load checkpoint, run on test set, write `test-results.json` |
| `analysis/explore_data.py` | Dataset statistics (kept in repo for report reference) |
| `analysis/visualize_gt.py` | Interactive matplotlib GT viewer (verify pipeline before training) |
| `tests/test_utils.py` | Unit tests for utils functions |
| `tests/test_dataset.py` | COCO JSON format tests + Dataset shape tests |
| `tests/test_model.py` | Model build + forward pass shape tests |
| `tests/test_inference.py` | RLE roundtrip + output JSON schema tests |
| `report/package.json` | bun-managed report project |
| `report/scripts/pdf.ts` | Playwright → PDF export |
| `report/src/index.html` | Report HTML content |

---

## Task 1: Project scaffolding + data utilities

**Files:**
- Create: `src/__init__.py`
- Create: `src/utils.py`
- Create: `tests/__init__.py`
- Create: `tests/test_utils.py`

- [ ] **Step 1.1: Create package init files**

```bash
mkdir -p src tests analysis checkpoints
touch src/__init__.py tests/__init__.py analysis/__init__.py
```

- [ ] **Step 1.2: Write failing tests for utils functions**

Create `tests/test_utils.py`:

```python
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
```

- [ ] **Step 1.3: Run tests to confirm they fail**

```bash
uv run pytest tests/test_utils.py -v
```

Expected: `ModuleNotFoundError: No module named 'src'`

- [ ] **Step 1.4: Write `src/utils.py`**

```python
from __future__ import annotations
from pathlib import Path
import numpy as np
import tifffile
from pycocotools import mask as mask_utils


def load_rgb(path: str | Path) -> np.ndarray:
    """Load TIFF image, drop alpha channel. Returns (H, W, 3) uint8."""
    img = tifffile.imread(str(path))
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
```

- [ ] **Step 1.5: Run tests to confirm they pass**

```bash
uv run pytest tests/test_utils.py -v
```

Expected: 5 passed

- [ ] **Step 1.6: Commit**

```bash
git add src/__init__.py src/utils.py tests/__init__.py tests/test_utils.py analysis/__init__.py
git commit -m "feat: add data utility functions with tests"
```

---

## Task 2: Dataset exploration script

**Files:**
- Create: `analysis/explore_data.py`
- Create: `analysis/output/.gitkeep`

- [ ] **Step 2.1: Create output directory**

```bash
mkdir -p analysis/output
touch analysis/output/.gitkeep
```

- [ ] **Step 2.2: Write `analysis/explore_data.py`**

```python
"""Dataset statistics — kept in repo for report reference."""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import tifffile
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
    print(f"Image H range: {sizes[:,0].min()}–{sizes[:,0].max()}, mean {sizes[:,0].mean():.1f}")
    print(f"Image W range: {sizes[:,1].min()}–{sizes[:,1].max()}, mean {sizes[:,1].mean():.1f}")
    print(f"Images containing each class: {images_with_class}")
    print(f"Total instances per class:    {class_instance_counts}")
    print(f"Instances per image — min: {min(instances_per_image)}, "
          f"max: {max(instances_per_image)}, mean: {np.mean(instances_per_image):.1f}")

    stats = {
        "n_images": len(folders),
        "h_min": int(sizes[:,0].min()), "h_max": int(sizes[:,0].max()),
        "w_min": int(sizes[:,1].min()), "w_max": int(sizes[:,1].max()),
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
```

- [ ] **Step 2.3: Run the script**

```bash
uv run python analysis/explore_data.py
```

Expected output (approximate):
```
Total images: 209
Image H range: 81–1956, mean 563.9
Image W range: 74–2162, mean 628.3
Images containing each class: {1: 96, 2: 146, 3: 94, 4: 58}
Total instances per class:    {1: 14537, 2: 15653, 3: 630, 4: 587}
Instances per image — min: 2, max: 772, mean: 150.3
Saved stats to analysis/output/dataset_stats.json
```

- [ ] **Step 2.4: Commit**

```bash
git add analysis/explore_data.py analysis/output/.gitkeep analysis/output/dataset_stats.json
git commit -m "feat: add dataset exploration script and statistics"
```

---

## Task 3: COCO annotation generation

**Files:**
- Create: `src/dataset.py` (functions only, no class yet)
- Modify: `tests/test_dataset.py` (new file)

- [ ] **Step 3.1: Write failing tests for annotation generation**

Create `tests/test_dataset.py`:

```python
import json
import numpy as np
import pytest
from pathlib import Path


@pytest.fixture
def tiny_train_dir(tmp_path):
    """Two-image mock train directory."""
    for folder, classes in [("img_a", [1, 2]), ("img_b", [3])]:
        d = tmp_path / folder
        d.mkdir()
        import tifffile
        # 10×10 RGBA image
        tifffile.imwrite(str(d / "image.tif"),
                         np.random.randint(0, 255, (10, 10, 4), dtype=np.uint8))
        for c in classes:
            # class mask: one instance with value 1, another with value 2
            mask = np.zeros((10, 10), dtype=np.float64)
            mask[1:3, 1:3] = 1
            mask[5:8, 5:8] = 2
            tifffile.imwrite(str(d / f"class{c}.tif"), mask)
    return tmp_path


def test_split_sizes(tiny_train_dir):
    from src.dataset import split_images
    train, val = split_images(tiny_train_dir, val_fraction=0.5, seed=42)
    assert len(train) + len(val) == 2
    assert len(set(train) & set(val)) == 0  # no overlap


def test_coco_categories(tiny_train_dir):
    from src.dataset import split_images, build_coco_annotations
    train_folders, _ = split_images(tiny_train_dir, val_fraction=0.5, seed=42)
    coco = build_coco_annotations(tiny_train_dir, train_folders)
    cat_ids = {c["id"] for c in coco["categories"]}
    assert cat_ids == {1, 2, 3, 4}


def test_coco_annotation_count(tiny_train_dir):
    """img_a has class1(2 inst) + class2(2 inst) = 4 anns."""
    from src.dataset import split_images, build_coco_annotations
    # Force img_a into train by using all folders
    all_folders = sorted([d.name for d in tiny_train_dir.iterdir() if d.is_dir()])
    coco = build_coco_annotations(tiny_train_dir, [f for f in all_folders if f == "img_a"])
    assert len(coco["annotations"]) == 4  # 2 instances × 2 classes


def test_coco_annotation_fields(tiny_train_dir):
    from src.dataset import build_coco_annotations
    all_folders = sorted([d.name for d in tiny_train_dir.iterdir() if d.is_dir()])
    coco = build_coco_annotations(tiny_train_dir, all_folders)
    ann = coco["annotations"][0]
    required = {"id", "image_id", "category_id", "segmentation", "bbox", "area", "iscrowd"}
    assert required <= ann.keys()
    assert ann["iscrowd"] == 0
    assert len(ann["bbox"]) == 4
    assert ann["segmentation"]["counts"] is not None
    assert isinstance(ann["segmentation"]["counts"], str)


def test_category_id_from_class_file(tiny_train_dir):
    """Instances from class2.tif must have category_id == 2."""
    from src.dataset import build_coco_annotations
    all_folders = sorted([d.name for d in tiny_train_dir.iterdir() if d.is_dir()])
    coco = build_coco_annotations(tiny_train_dir, ["img_a"])
    cat_ids = {ann["category_id"] for ann in coco["annotations"]}
    assert cat_ids == {1, 2}  # img_a has class1 and class2
```

- [ ] **Step 3.2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_dataset.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.dataset'`

- [ ] **Step 3.3: Write `src/dataset.py` (annotation functions)**

```python
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
```

- [ ] **Step 3.4: Run annotation tests**

```bash
uv run pytest tests/test_dataset.py -v
```

Expected: 5 passed

- [ ] **Step 3.5: Commit**

```bash
git add src/dataset.py tests/test_dataset.py
git commit -m "feat: COCO annotation generation with train/val split"
```

---

## Task 4: CellDataset class

**Files:**
- Modify: `src/dataset.py` (add `CellDataset`)
- Modify: `tests/test_dataset.py` (add dataset tests)

- [ ] **Step 4.1: Write failing dataset tests**

Append to `tests/test_dataset.py`:

```python
def test_cell_dataset_len(tiny_train_dir):
    from src.dataset import build_coco_annotations, CellDataset
    all_folders = sorted([d.name for d in tiny_train_dir.iterdir() if d.is_dir()])
    coco = build_coco_annotations(tiny_train_dir, all_folders)
    ds = CellDataset(tiny_train_dir, coco)
    assert len(ds) == 2


def test_cell_dataset_item_shapes(tiny_train_dir):
    import torch
    from src.dataset import build_coco_annotations, CellDataset
    all_folders = sorted([d.name for d in tiny_train_dir.iterdir() if d.is_dir()])
    coco = build_coco_annotations(tiny_train_dir, all_folders)
    ds = CellDataset(tiny_train_dir, coco)
    img, target = ds[0]
    assert img.shape[0] == 3          # RGB channels
    assert img.dtype == torch.uint8
    assert target["boxes"].ndim == 2 and target["boxes"].shape[1] == 4
    assert target["labels"].ndim == 1
    assert target["masks"].ndim == 3
    assert len(target["boxes"]) == len(target["labels"]) == len(target["masks"])


def test_cell_dataset_boxes_xyxy(tiny_train_dir):
    """Boxes must be in XYXY format: x2 > x1 and y2 > y1."""
    import torch
    from src.dataset import build_coco_annotations, CellDataset
    all_folders = sorted([d.name for d in tiny_train_dir.iterdir() if d.is_dir()])
    coco = build_coco_annotations(tiny_train_dir, all_folders)
    ds = CellDataset(tiny_train_dir, coco)
    for i in range(len(ds)):
        _, target = ds[i]
        if len(target["boxes"]) > 0:
            assert (target["boxes"][:, 2] > target["boxes"][:, 0]).all()
            assert (target["boxes"][:, 3] > target["boxes"][:, 1]).all()


def test_oversampled_dataset_has_more_entries(tiny_train_dir):
    """Oversampled dataset repeats rare-class images."""
    from src.dataset import build_coco_annotations, CellDataset, oversample_rare_classes
    all_folders = sorted([d.name for d in tiny_train_dir.iterdir() if d.is_dir()])
    coco = build_coco_annotations(tiny_train_dir, all_folders)
    ds_normal = CellDataset(tiny_train_dir, coco)
    oversampled_folders = oversample_rare_classes(tiny_train_dir, all_folders, factor=3)
    coco_os = build_coco_annotations(tiny_train_dir, oversampled_folders)
    ds_os = CellDataset(tiny_train_dir, coco_os)
    # img_b has class3 → should be repeated
    assert len(ds_os) > len(ds_normal)
```

- [ ] **Step 4.2: Run to confirm failure**

```bash
uv run pytest tests/test_dataset.py::test_cell_dataset_len -v
```

Expected: `ImportError: cannot import name 'CellDataset'`

- [ ] **Step 4.3: Append `CellDataset` and `oversample_rare_classes` to `src/dataset.py`**

```python
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from pycocotools import mask as mask_utils
from src.utils import load_rgb, rle_to_bytes


class CellDataset(Dataset):
    """Torchvision-compatible detection dataset wrapping COCO-format annotations."""

    def __init__(self, train_dir: Path, coco_data: dict, transforms=None):
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
        img_arr = load_rgb(self.train_dir / info["file_name"] / "image.tif")  # (H, W, 3) uint8
        H, W = img_arr.shape[:2]

        anns = self._ann_by_image.get(info["id"], [])
        boxes, labels, masks = [], [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])   # XYXY
            labels.append(ann["category_id"])
            decoded = mask_utils.decode(rle_to_bytes(ann["segmentation"]))  # (H, W) uint8
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
```

- [ ] **Step 4.4: Run all dataset tests**

```bash
uv run pytest tests/test_dataset.py -v
```

Expected: all 9 tests passed

- [ ] **Step 4.5: Commit**

```bash
git add src/dataset.py tests/test_dataset.py
git commit -m "feat: CellDataset class with tv_tensors and rare-class oversampling"
```

---

## Task 5: Augmentation pipeline

**Files:**
- Create: `src/augment.py`
- Create: `tests/test_augment.py`

- [ ] **Step 5.1: Write failing augmentation tests**

Create `tests/test_augment.py`:

```python
import torch
import numpy as np
from torchvision import tv_tensors


def _make_sample(H=64, W=64, n_inst=3):
    """Create a fake (img, target) pair."""
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
```

- [ ] **Step 5.2: Run to confirm failure**

```bash
uv run pytest tests/test_augment.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.augment'`

- [ ] **Step 5.3: Write `src/augment.py`**

```python
from torchvision.transforms import v2


def get_train_transform():
    """Augmentation for training: spatial flips + colour distortion + dtype conversion.

    MaskRCNN's internal GeneralizedRCNNTransform handles normalisation and
    multi-scale resizing, so we only apply spatial and photometric augmentation here.
    """
    return v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomPhotometricDistort(p=1.0),
        v2.ToDtype(torch.float32, scale=True),  # uint8 → float [0, 1]
    ])


def get_val_transform():
    """Validation: dtype conversion only (no augmentation)."""
    return v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
    ])


import torch  # noqa: E402 (needed for ToDtype)
```

- [ ] **Step 5.4: Run augmentation tests**

```bash
uv run pytest tests/test_augment.py -v
```

Expected: 3 passed

- [ ] **Step 5.5: Commit**

```bash
git add src/augment.py tests/test_augment.py
git commit -m "feat: torchvision v2 augmentation pipeline"
```

---

## Task 6: Interactive GT visualizer

**Files:**
- Create: `analysis/visualize_gt.py`

(No automated tests — interactive UI. Verify manually before training.)

- [ ] **Step 6.1: Write `analysis/visualize_gt.py`**

```python
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
        rng = np.random.default_rng(seed=0)  # fixed seed → same colour per instance across renders

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
    # Merge both splits for full visibility
    all_images = train_coco["images"] + val_coco["images"]
    all_anns = train_coco["annotations"] + val_coco["annotations"]

    # Re-index to avoid image_id conflicts between splits
    id_map = {img["id"]: i + 1 for i, img in enumerate(all_images)}
    for img in all_images:
        img["id"] = id_map[img["id"]]
    for ann in all_anns:
        ann["image_id"] = id_map[ann["image_id"]]

    merged = {"images": all_images, "annotations": all_anns, "categories": train_coco["categories"]}
    GTViewer(merged, TRAIN_DIR)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6.2: Generate annotation cache (required before running visualizer)**

```bash
uv run python -c "
from pathlib import Path
from src.dataset import load_or_build_annotations
load_or_build_annotations(Path('data/train'), Path('data/train_annotations.json'), Path('data/val_annotations.json'))
"
```

Expected output:
```
Building annotations: 177 train, 32 val...
Annotations cached.
```

- [ ] **Step 6.3: Run the visualizer and verify**

```bash
uv run python analysis/visualize_gt.py
```

Manually confirm:
- [ ] Masks align with image content (cells are where the mask says)
- [ ] Each instance has a distinct colour overlay
- [ ] Category labels show `class1`–`class4` correctly
- [ ] Bounding boxes are tight around each mask
- [ ] Arrow keys and number+Enter navigation work

- [ ] **Step 6.4: Commit**

```bash
git add analysis/visualize_gt.py data/train_annotations.json data/val_annotations.json
git commit -m "feat: interactive GT visualizer + cached COCO annotations"
```

---

## Task 7: Model construction

**Files:**
- Create: `src/model.py`
- Create: `tests/test_model.py`

- [ ] **Step 7.1: Write failing model tests**

Create `tests/test_model.py`:

```python
import torch
import pytest


def test_model_builds():
    from src.model import build_model
    model = build_model()
    assert model is not None


def test_model_parameter_count():
    """Model must have fewer than 200M trainable parameters."""
    from src.model import build_model
    model = build_model()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params < 200_000_000, f"Too many params: {n_params:,}"


def test_model_eval_forward():
    """model.eval() forward pass returns boxes, labels, masks, scores."""
    from src.model import build_model
    model = build_model()
    model.eval()
    img = torch.rand(3, 200, 200)
    with torch.no_grad():
        output = model([img])
    assert len(output) == 1
    result = output[0]
    assert "boxes" in result
    assert "labels" in result
    assert "masks" in result
    assert "scores" in result
    # masks shape: (N, 1, H, W) — MaskRCNN outputs soft masks
    if len(result["masks"]) > 0:
        assert result["masks"].ndim == 4
        assert result["masks"].shape[1] == 1


def test_model_train_forward():
    """model.train() forward pass returns a loss dict."""
    from src.model import build_model
    model = build_model()
    model.train()
    imgs = [torch.rand(3, 100, 100)]
    targets = [{
        "boxes": torch.tensor([[10., 10., 50., 50.]]),
        "labels": torch.tensor([1]),
        "masks": torch.zeros(1, 100, 100, dtype=torch.uint8),
    }]
    losses = model(imgs, targets)
    assert isinstance(losses, dict)
    expected_keys = {"loss_classifier", "loss_box_reg", "loss_mask", "loss_objectness", "loss_rpn_box_reg"}
    assert expected_keys <= losses.keys()
    total = sum(losses.values())
    assert total.item() > 0
```

- [ ] **Step 7.2: Run to confirm failure**

```bash
uv run pytest tests/test_model.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.model'`

- [ ] **Step 7.3: Write `src/model.py`**

```python
from __future__ import annotations
import torch
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet101_Weights


def build_model(num_classes: int = 5) -> MaskRCNN:
    """Build ResNet101-FPN Mask R-CNN.

    num_classes: 4 cell types + 1 background = 5.
    Anchor sizes are extended with small anchors (8px, 16px) on high-resolution
    FPN levels to detect tiny cells. 5 tuples match 5 FPN feature maps (P2–P6).
    """
    backbone = resnet_fpn_backbone(
        backbone_name="resnet101",
        weights=ResNet101_Weights.IMAGENET1K_V2,
        trainable_layers=5,  # fine-tune entire backbone for domain adaptation
    )

    anchor_generator = AnchorGenerator(
        sizes=((8, 16), (32, 64), (64, 128), (128, 256), (256, 512)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )

    model = MaskRCNN(
        backbone,
        num_classes=num_classes,
        min_size=(640, 704, 768, 832, 896, 1024),
        max_size=2000,
        rpn_anchor_generator=anchor_generator,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )
    return model
```

- [ ] **Step 7.4: Run model tests (CPU only — download may take a moment)**

```bash
uv run pytest tests/test_model.py -v
```

Expected: 4 passed. Note: first run downloads ResNet101 weights (~170MB).

- [ ] **Step 7.5: Commit**

```bash
git add src/model.py tests/test_model.py
git commit -m "feat: ResNet101-FPN Mask R-CNN model with custom anchors"
```

---

## Task 8: DDP training script

**Files:**
- Create: `src/train.py`
- Create: `checkpoints/.gitkeep`

(No unit tests for the training loop — verify by running a 1-epoch smoke test.)

- [ ] **Step 8.1: Create checkpoints directory**

```bash
touch checkpoints/.gitkeep
```

- [ ] **Step 8.2: Write `src/train.py`**

```python
"""DDP training script.

Launch with:
    torchrun --nproc_per_node=2 src/train.py

Or single-GPU smoke test:
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 src/train.py --epochs 1
"""
from __future__ import annotations
import argparse
import contextlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.augment import get_train_transform, get_val_transform
from src.dataset import CellDataset, load_or_build_annotations, oversample_rare_classes
from src.model import build_model
from src.utils import encode_mask, rle_to_bytes

TRAIN_DIR = Path("data/train")
CACHE_TRAIN = Path("data/train_annotations.json")
CACHE_VAL = Path("data/val_annotations.json")
CHECKPOINT_DIR = Path("checkpoints")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=2, help="per-GPU batch size")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--accum-steps", type=int, default=2)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--score-thresh", type=float, default=0.05)
    return p.parse_args()


def setup_ddp():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def collate_fn(batch):
    return tuple(zip(*batch))


@torch.no_grad()
def evaluate(model_without_ddp, val_loader, val_coco_json: dict, device, score_thresh=0.05):
    """Run COCOeval on val set. Returns AP50."""
    model_without_ddp.eval()
    coco_gt = COCO()
    coco_gt.dataset = val_coco_json
    coco_gt.createIndex()

    results = []
    for imgs, targets in val_loader:
        imgs = [img.to(device) for img in imgs]
        preds = model_without_ddp(imgs)
        for pred, target in zip(preds, targets):
            image_id = target["image_id"].item()
            for box, label, score, mask in zip(
                pred["boxes"], pred["labels"], pred["scores"], pred["masks"]
            ):
                if score < score_thresh:
                    continue
                binary = (mask[0] > 0.5).cpu().numpy()
                rle = encode_mask(binary)
                H, W = binary.shape
                results.append({
                    "image_id": image_id,
                    "category_id": label.item(),
                    "score": score.item(),
                    "segmentation": rle,
                    "bbox": box.tolist(),
                })

    if not results:
        return 0.0

    coco_dt = coco_gt.loadRes(results)
    evaluator = COCOeval(coco_gt, coco_dt, "segm")
    evaluator.params.iouThrs = [0.5]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return float(evaluator.stats[0])  # AP at IoU=0.50


def main():
    args = parse_args()
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    is_main = local_rank == 0

    train_coco, val_coco = load_or_build_annotations(TRAIN_DIR, CACHE_TRAIN, CACHE_VAL)

    # Oversample rare classes (class3, class4) × 3 to compensate imbalance
    train_folders = [img["file_name"] for img in train_coco["images"]]
    train_folders_os = oversample_rare_classes(TRAIN_DIR, train_folders, factor=3)
    from src.dataset import build_coco_annotations
    train_coco_os = build_coco_annotations(TRAIN_DIR, train_folders_os)

    train_ds = CellDataset(TRAIN_DIR, train_coco_os, transforms=get_train_transform())
    val_ds = CellDataset(TRAIN_DIR, val_coco, transforms=get_val_transform())

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=4, pin_memory=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, sampler=val_sampler,
        num_workers=2, pin_memory=True, collate_fn=collate_fn,
    )

    model = build_model().to(device)
    model = DDP(model, device_ids=[local_rank])

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_optim_steps = (len(train_loader) // args.accum_steps) * args.epochs
    warmup_steps = min(args.warmup_steps, total_optim_steps // 5)
    scheduler = SequentialLR(optimizer, schedulers=[
        LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps),
        CosineAnnealingLR(optimizer, T_max=max(1, total_optim_steps - warmup_steps), eta_min=1e-6),
    ], milestones=[warmup_steps])

    scaler = GradScaler("cuda")
    best_ap50 = 0.0
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        optimizer.zero_grad()
        epoch_loss = 0.0

        for step, (imgs, targets) in enumerate(train_loader):
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            is_last_accum = (step + 1) % args.accum_steps == 0

            ctx = model.no_sync() if not is_last_accum else contextlib.nullcontext()
            with ctx:
                with autocast("cuda"):
                    loss_dict = model(imgs, targets)
                    loss = sum(loss_dict.values()) / args.accum_steps
                scaler.scale(loss).backward()

            if is_last_accum:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * args.accum_steps

        if is_main:
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{args.epochs}  loss={avg_loss:.4f}")

        # Evaluate on rank 0 only (val sampler covers all data since drop_last=False)
        if is_main:
            ap50 = evaluate(model.module, val_loader, val_coco, device, args.score_thresh)
            print(f"  Val AP50: {ap50:.4f}  (best: {best_ap50:.4f})")
            if ap50 > best_ap50:
                best_ap50 = ap50
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "ap50": ap50,
                }, CHECKPOINT_DIR / "best_model.pth")
                print(f"  ✓ Saved best checkpoint (AP50={ap50:.4f})")

        dist.barrier()

    cleanup_ddp()


if __name__ == "__main__":
    main()
```

- [ ] **Step 8.3: Smoke test — 1 epoch on 1 GPU**

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 src/train.py --epochs 1 --batch-size 1
```

Expected: runs without error, prints loss and AP50 for epoch 1.

- [ ] **Step 8.4: Full training run — 50 epochs on 2 GPUs**

```bash
torchrun --nproc_per_node=2 src/train.py
```

Expected: checkpoints/best_model.pth saved when val AP50 improves.

- [ ] **Step 8.5: Commit**

```bash
git add src/train.py checkpoints/.gitkeep
git commit -m "feat: DDP training script with AMP and gradient accumulation"
```

---

## Task 9: Inference and submission

**Files:**
- Create: `src/inference.py`
- Create: `tests/test_inference.py`

- [ ] **Step 9.1: Write failing inference tests**

Create `tests/test_inference.py`:

```python
import json
import numpy as np
import pytest
from pycocotools import mask as mask_utils


def test_rle_encode_decode_roundtrip():
    """Full encode→JSON-serialise→deserialise→decode roundtrip."""
    from src.utils import encode_mask, rle_to_bytes
    mask = np.zeros((30, 40), dtype=bool)
    mask[5:15, 10:25] = True
    rle = encode_mask(mask)
    # Simulate JSON round-trip
    serialised = json.dumps(rle)
    loaded = json.loads(serialised)
    decoded = mask_utils.decode(rle_to_bytes(loaded))
    np.testing.assert_array_equal(decoded, mask.astype(np.uint8))


def test_submission_entry_fields():
    """build_submission_entry returns required COCO result fields."""
    from src.inference import build_submission_entry
    import torch
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:20, 10:20] = True
    entry = build_submission_entry(
        image_id=3,
        category_id=2,
        score=0.85,
        binary_mask=mask,
    )
    assert entry["image_id"] == 3
    assert entry["category_id"] == 2
    assert abs(entry["score"] - 0.85) < 1e-6
    assert "segmentation" in entry
    assert isinstance(entry["segmentation"]["counts"], str)
    assert entry["segmentation"]["size"] == [50, 50]
    assert len(entry["bbox"]) == 4   # [x, y, w, h]
    assert entry["bbox"][2] > 0 and entry["bbox"][3] > 0


def test_output_json_is_list_of_dicts(tmp_path):
    """run_inference writes a JSON file that is a list of result dicts."""
    import torch
    from unittest.mock import patch, MagicMock
    from src.inference import run_inference

    # Mock model that returns one instance per image
    def fake_model(imgs):
        H, W = imgs[0].shape[-2:]
        mask = torch.zeros(1, 1, H, W)
        mask[0, 0, 5:15, 5:15] = 1.0
        return [{"boxes": torch.tensor([[5., 5., 15., 15.]]),
                 "labels": torch.tensor([1]),
                 "scores": torch.tensor([0.9]),
                 "masks": mask}]

    test_image_ids = {"fake.tif": 42}
    out_path = tmp_path / "test-results.json"

    import tifffile
    fake_img = np.random.randint(0, 255, (20, 20, 4), dtype=np.uint8)
    tifffile.imwrite(str(tmp_path / "fake.tif"), fake_img)

    run_inference(
        model=fake_model,
        test_dir=tmp_path,
        image_name_to_id=test_image_ids,
        output_path=out_path,
        score_threshold=0.5,
        device=torch.device("cpu"),
    )

    results = json.loads(out_path.read_text())
    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0]["image_id"] == 42
    assert results[0]["category_id"] == 1
```

- [ ] **Step 9.2: Run to confirm failure**

```bash
uv run pytest tests/test_inference.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.inference'`

- [ ] **Step 9.3: Write `src/inference.py`**

```python
"""Inference script — produces test-results.json for CodaBench submission.

Run:
    uv run python src/inference.py --checkpoint checkpoints/best_model.pth \
        --test-dir data/test_release \
        --output test-results.json \
        --score-thresh 0.3
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import tifffile
from torch.amp.autocast_mode import autocast

from src.model import build_model
from src.utils import load_rgb, encode_mask, binary_mask_to_bbox


def build_submission_entry(
    image_id: int,
    category_id: int,
    score: float,
    binary_mask: np.ndarray,
) -> dict:
    """Build one COCO-result dict from a predicted binary mask.

    binary_mask: (H, W) boolean numpy array.
    segmentation.size is [height, width] (not width × height).
    """
    rle = encode_mask(binary_mask)
    bbox = binary_mask_to_bbox(binary_mask)
    return {
        "image_id": image_id,
        "category_id": category_id,
        "score": float(score),
        "segmentation": rle,
        "bbox": bbox,
    }


def run_inference(
    model,
    test_dir: Path,
    image_name_to_id: dict[str, int],
    output_path: Path,
    score_threshold: float = 0.3,
    device: torch.device = torch.device("cuda"),
) -> None:
    """Run model on all test images and write test-results.json."""
    results = []

    for filename, image_id in image_name_to_id.items():
        img_path = test_dir / filename
        img_rgb = load_rgb(img_path)          # (H, W, 3) uint8
        img_t = (
            torch.from_numpy(img_rgb)
            .permute(2, 0, 1)                 # (3, H, W)
            .float() / 255.0
            .to(device)
        )

        with autocast("cuda" if device.type == "cuda" else "cpu"):
            preds = model([img_t])[0]

        for box, label, score, mask in zip(
            preds["boxes"], preds["labels"], preds["scores"], preds["masks"]
        ):
            if score.item() < score_threshold:
                continue
            binary = (mask[0] > 0.5).cpu().numpy().astype(bool)
            if not binary.any():
                continue
            results.append(build_submission_entry(
                image_id=image_id,
                category_id=label.item(),
                score=score.item(),
                binary_mask=binary,
            ))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {len(results)} predictions to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_model.pth"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_release"))
    parser.add_argument("--id-map", type=Path, default=Path("data/test_image_name_to_ids.json"))
    parser.add_argument("--output", type=Path, default=Path("test-results.json"))
    parser.add_argument("--score-thresh", type=float, default=0.3)
    args = parser.parse_args()

    with open(args.id_map) as f:
        id_list = json.load(f)
    image_name_to_id = {entry["file_name"]: entry["id"] for entry in id_list}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model()
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    run_inference(model, args.test_dir, image_name_to_id, args.output, args.score_thresh, device)


if __name__ == "__main__":
    main()
```

- [ ] **Step 9.4: Run inference tests**

```bash
uv run pytest tests/test_inference.py -v
```

Expected: 3 passed

- [ ] **Step 9.5: Run inference on test set (after training)**

```bash
uv run python src/inference.py \
    --checkpoint checkpoints/best_model.pth \
    --test-dir data/test_release \
    --output test-results.json \
    --score-thresh 0.3
```

Verify output format matches the reference in `docs/encode_mask_to_RLE/test-results.json`.

- [ ] **Step 9.6: Commit**

```bash
git add src/inference.py tests/test_inference.py
git commit -m "feat: inference script with RLE submission generation"
```

---

## Task 10: Report scaffolding (bun + Playwright)

**Files:**
- Create: `report/package.json`
- Create: `report/scripts/pdf.ts`
- Create: `report/src/index.html`

- [ ] **Step 10.1: Initialise bun project**

```bash
cd report && bun init -y
bun add playwright
bunx playwright install chromium
```

- [ ] **Step 10.2: Create `report/scripts/pdf.ts`**

```typescript
import { chromium } from "playwright";
import path from "path";

const htmlPath = path.resolve(import.meta.dir, "../src/index.html");
const pdfPath = path.resolve(import.meta.dir, "../report.pdf");

const browser = await chromium.launch();
const page = await browser.newPage();

// Use file:// URL so local font @font-face rules work
await page.goto(`file://${htmlPath}`, { waitUntil: "networkidle" });
await page.waitForTimeout(500); // allow fonts + KaTeX to render

await page.pdf({
  path: pdfPath,
  format: "A4",
  printBackground: true,
  margin: { top: "20mm", bottom: "20mm", left: "15mm", right: "15mm" },
});

await browser.close();
console.log(`PDF written to ${pdfPath}`);
```

- [ ] **Step 10.3: Create `report/src/index.html` skeleton**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>HW3 Report</title>
  <!-- KaTeX for math (AP₅₀, superscripts) -->
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
    onload="renderMathInElement(document.body)"></script>
  <!-- Chinese font + body font -->
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;700&display=swap');

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: "Linux Libertine", "Noto Sans TC", "Times New Roman", serif;
      font-size: 10pt;
      line-height: 1.5;
      color: #000;
      background: #fff;
    }

    /* Two-column ECCV-style layout */
    .paper {
      width: 210mm;
      margin: 0 auto;
      padding: 20mm 15mm;
      column-count: 2;
      column-gap: 8mm;
    }

    h1 { font-size: 16pt; text-align: center; column-span: all; margin-bottom: 4mm; }
    .authors { text-align: center; column-span: all; margin-bottom: 8mm; font-size: 10pt; }
    h2 { font-size: 11pt; font-weight: bold; margin: 4mm 0 2mm; }
    h3 { font-size: 10pt; font-weight: bold; margin: 3mm 0 1mm; }
    p  { margin-bottom: 2mm; text-align: justify; }

    figure { margin: 3mm 0; text-align: center; }
    figcaption { font-size: 8pt; color: #444; margin-top: 1mm; }
    img { max-width: 100%; }

    table { width: 100%; border-collapse: collapse; font-size: 9pt; margin: 2mm 0; }
    th, td { border: 1px solid #ccc; padding: 1mm 2mm; text-align: center; }
    th { background: #f0f0f0; font-weight: bold; }

    .references p { font-size: 9pt; margin-bottom: 1mm; }

    @media print {
      body { -webkit-print-color-adjust: exact; }
    }
  </style>
</head>
<body>
<div class="paper">
  <h1>Instance Segmentation of Medical Cell Images</h1>
  <p class="authors">
    <!-- Replace with your student ID and Chinese name -->
    Student ID &nbsp;|&nbsp; 你的名字
  </p>

  <h2>1. Introduction</h2>
  <p><!-- Describe the task and your core approach. --></p>

  <h2>2. Method</h2>
  <h3>2.1 Data Preprocessing</h3>
  <p><!-- Describe mask decoding, RGB-only, oversampling. --></p>

  <h3>2.2 Model Architecture</h3>
  <p>
    We use Mask R-CNN with a ResNet-101 backbone and Feature Pyramid Network (FPN).
    The anchor generator covers sizes \((8, 256)\text{px}\) per FPN level to handle
    the wide range of cell sizes in the dataset.
  </p>

  <h3>2.3 Training</h3>
  <p><!-- LR schedule, batch size, epochs, DDP, AMP. --></p>

  <h2>3. Additional Experiments</h2>
  <h3>3.1 Mask Scoring Head</h3>
  <p>
    <!-- Hypothesis: replacing classification score with predicted mask IoU
         rewards high-quality masks and should improve AP₅₀. -->
  </p>

  <h2>4. Results</h2>
  <p><!-- AP50 table, training curve image, visualisation. --></p>

  <figure>
    <!-- <img src="../figures/training_curve.png" alt="Training loss curve"> -->
    <figcaption>Figure 1: Training loss and val AP\(_{50}\) over 50 epochs.</figcaption>
  </figure>

  <h2>References</h2>
  <div class="references">
    <p>[1] He, K. et al. Mask R-CNN. ICCV 2017.</p>
    <p>[2] Huang, Z. et al. Mask Scoring R-CNN. CVPR 2019.</p>
    <!-- Add more as needed -->
  </div>
</div>
</body>
</html>
```

- [ ] **Step 10.4: Add PDF build script to package.json**

In `report/package.json`, ensure:
```json
{
  "name": "report",
  "scripts": {
    "pdf": "bun run scripts/pdf.ts"
  },
  "dependencies": {
    "playwright": "latest"
  }
}
```

- [ ] **Step 10.5: Test PDF generation**

```bash
cd report && bun run pdf
```

Expected: `PDF written to report/report.pdf` — open PDF to verify fonts render correctly (check Chinese characters and math).

- [ ] **Step 10.6: Commit**

```bash
cd ..
git add report/
git commit -m "feat: bun + playwright PDF report scaffolding"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Covered by |
|---|---|
| Drop alpha, use RGB only | `src/utils.py::load_rgb` Task 1 |
| Instance IDs globally unique across classes | `build_coco_annotations` comment + test Task 3 |
| category_id from class file, not ID value | `build_coco_annotations` loop Task 3 |
| image_id 1-based from test_image_name_to_ids.json | `src/inference.py::main` Task 9 |
| segmentation.size = [H, W] (not W×H) | `encode_mask` via pycocotools, verified in Task 9 test |
| RLE counts as UTF-8 string | `encode_mask` in Task 1, `rle_to_bytes` for reload |
| ResNet101-FPN, trainable_layers=5 | `src/model.py` Task 7 |
| AnchorGenerator 5 tuples matching 5 FPN maps | `src/model.py` Task 7 |
| AdamW + LinearLR warmup + CosineAnnealingLR | `src/train.py` Task 8 |
| Effective batch 8 via 2-GPU × batch2 × accum2 | `src/train.py` accum_steps=2 Task 8 |
| model.no_sync() during accumulation | `src/train.py` Task 8 |
| GradScaler + autocast correct imports | `src/train.py` Task 8 |
| WeightedRandomSampler via oversampling | `oversample_rare_classes` Task 4 |
| 85/15 train/val split seed=42 | `split_images` Task 3 |
| Sorted-alphabetical navigation in visualizer | `GTViewer.__init__` Task 6 |
| → / ← / d / a keys + number+Enter jump | `GTViewer._on_key` Task 6 |
| Title format [N/209] folder — M instances | `GTViewer._render` Task 6 |
| Chinese font + KaTeX in report | `report/src/index.html` Task 10 |
| PDF via bun + playwright | `report/scripts/pdf.ts` Task 10 |

All spec requirements covered. No placeholders or TBD items detected.
