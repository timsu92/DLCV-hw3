# Augmentation + Resolution Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the four changes in `docs/superpowers/specs/2026-05-10-augmentation-and-resolution-design.md` to push leaderboard AP50 above 0.5975 (current 0.4958, val 0.6997 sanity-checked).

**Architecture:** Five files modified in dependency order: `utils.py` (pre-resize default) → `augment.py` (new train pipeline) → `model.py` (anchor + scale) → `dataset.py` (skip dense images) → `train.py` (wire defaults). Each task is TDD where unit tests fit, smoke-test where they don't (full-model forward passes are too expensive for unit tests).

**Tech Stack:** Python 3.12, PyTorch + torchvision v2 transforms, pycocotools, AdamW + AMP. Uses `uv` for dependency management. Single-GPU dev/test (`CUDA_VISIBLE_DEVICES=0`); production run on 2 GPUs.

---

## Files Modified

| File | Responsibility | Section in spec |
|---|---|---|
| `src/utils.py` | `pre_resize_image` defaults `size=1024, max_size=1024` (function body unchanged) | §4.2 |
| `src/augment.py` | `_PRE_RESIZE=1024`, `_MAX_SIZE=1024`, new train pipeline with `RandomIoUCrop` + `SanitizeBoundingBoxes` | §4.1 |
| `src/model.py` | `build_model` defaults: `min_size=(640,768,896,1024), max_size=1024`; anchor sizes shifted to add size=4 at P2 | §4.3 |
| `src/dataset.py` | `CellDataset.__init__` accepts `skip_above_instances: int \| None = None`; filters `self.images` at init | §4.4 |
| `src/train.py` | Update default `--min-size`, `--max-size`, `--epochs`; pass `skip_above_instances=400` to train dataset only | §4.5 |

`src/inference.py` is **not modified** — `pre_resize_image` is called with no args, so the default-arg update propagates automatically (§4.6).

---

## Task 1: Update `pre_resize_image` defaults (`src/utils.py`)

**Files:**
- Modify: `src/utils.py:69-86`
- Test: `tests/test_utils.py`

- [ ] **Step 1.1: Write failing test for new defaults**

Append to `tests/test_utils.py`:
```python
def test_pre_resize_image_default_size_is_1024():
    """New default: shorter side resized to 1024 (was 640)."""
    import numpy as np
    from src.utils import pre_resize_image

    img = np.zeros((1771, 1760, 3), dtype=np.uint8)
    out, (orig_h, orig_w) = pre_resize_image(img)
    # shorter side 1760 → 1024; aspect preserved
    assert out.shape == (3, 1031, 1024), f"got {out.shape}"
    assert (orig_h, orig_w) == (1771, 1760)


def test_pre_resize_image_default_max_size_is_1024():
    """Extreme aspect (1:9.4) capped to longer side 1024."""
    import numpy as np
    from src.utils import pre_resize_image

    img = np.zeros((160, 1500, 3), dtype=np.uint8)
    out, (orig_h, orig_w) = pre_resize_image(img)
    # shorter→1024 would give 9600 long side; capped at max_size=1024
    # scale = 1024/1500 = 0.683, h = 160*0.683 = 109
    assert out.shape[2] == 1024, f"long side not capped: {out.shape}"
    assert out.shape[1] == 109, f"unexpected short side: {out.shape}"
```

- [ ] **Step 1.2: Run test to confirm it fails**

Run: `uv run pytest tests/test_utils.py::test_pre_resize_image_default_size_is_1024 tests/test_utils.py::test_pre_resize_image_default_max_size_is_1024 -v`

Expected: FAIL — current defaults are 640/1333, output shape will be `(3, 644, 640)` and `(3, 142, 1333)`.

- [ ] **Step 1.3: Update defaults in `src/utils.py`**

In `src/utils.py`, change the `pre_resize_image` signature only. The body stays the same:
```python
def pre_resize_image(
    img: np.ndarray, size: int = 1024, max_size: int = 1024
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Resize image so shorter side == `size`, then cap longer side to `max_size`.

    Mirrors `v2.Resize(size, max_size, antialias=True)` + `v2.ToDtype(float32, scale=True)`
    from `get_val_transform` (so val/inference share identical preprocessing),
    plus a `max_size` cap on the longer side that train/val never triggered
    because images are near-square — but extreme-aspect-ratio test images would
    expand to e.g. 1024×9600 and make `paste_masks_in_image` allocate 15+ GB.

    Returns a (3, H, W) float32 tensor in [0, 1] and the original (h, w) tuple.
    """
    orig_h, orig_w = img.shape[:2]
    img_t = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W) uint8
    img_t = v2.functional.resize(
        img_t, size=[size], max_size=max_size, antialias=True
    )
    return img_t.to(torch.float32) / 255.0, (orig_h, orig_w)
```

- [ ] **Step 1.4: Run tests to confirm they pass**

Run: `uv run pytest tests/test_utils.py -v`

Expected: PASS (all utils tests).

- [ ] **Step 1.5: Commit**

```bash
git add src/utils.py tests/test_utils.py
git commit -m "feat(utils): bump pre_resize_image defaults to 1024 to match new model eval"
```

---

## Task 2: Augmentation pipeline overhaul (`src/augment.py`)

**Files:**
- Modify: `src/augment.py` (entire file)
- Test: `tests/test_augment.py`

- [ ] **Step 2.1: Write failing test for new pipeline structure**

Append to `tests/test_augment.py`:
```python
def test_train_transform_includes_random_iou_crop():
    """Training pipeline must include RandomIoUCrop and SanitizeBoundingBoxes."""
    from torchvision.transforms import v2

    from src.augment import get_train_transform

    transforms = get_train_transform().transforms
    types = {type(t).__name__ for t in transforms}
    assert "RandomIoUCrop" in types
    assert "SanitizeBoundingBoxes" in types


def test_train_transform_resize_uses_1024():
    """Pre-resize size must match model eval min_size (1024)."""
    from torchvision.transforms import v2

    from src.augment import get_train_transform

    transforms = get_train_transform().transforms
    resize = next(t for t in transforms if isinstance(t, v2.Resize))
    assert resize.size == [1024], f"got {resize.size}"
    assert resize.max_size == 1024, f"got {resize.max_size}"


def test_val_transform_resize_uses_1024():
    """Val pipeline pre-resize matches train (and inference)."""
    from torchvision.transforms import v2

    from src.augment import get_val_transform

    transforms = get_val_transform().transforms
    resize = next(t for t in transforms if isinstance(t, v2.Resize))
    assert resize.size == [1024], f"got {resize.size}"
    assert resize.max_size == 1024
```

- [ ] **Step 2.2: Run tests to confirm they fail**

Run: `uv run pytest tests/test_augment.py -v`

Expected: FAIL — current `_PRE_RESIZE=640`, no `RandomIoUCrop`.

- [ ] **Step 2.3: Rewrite `src/augment.py`**

Replace the entire file content with:
```python
import torch
from torchvision.transforms import v2

_PRE_RESIZE = 1024
_MAX_SIZE = 1024


def get_train_transform():
    """Train pipeline:
    - Resize first to bound CPU memory of the per-instance mask stack.
    - Photometric distort runs on uint8 (faster + numerically cleaner than on float).
    - RandomIoUCrop introduces real (not synthetic) aspect-ratio variety:
        * sampler_options[0]=0.0 keeps "no crop" as a valid choice (~14% of samples).
        * aspect 0.2-5.0 covers 99.5% of test image aspect ratios.
        * scale 0.5-1.0 prevents cells (P10≈19px at 1024 input) from shrinking past
          P2 anchor coverage when followed by model multi-scale resize.
    - SanitizeBoundingBoxes removes degenerate fragments after the crop clips
      partial cells. Default min_size=1 keeps all real cells (smallest is 5×5).
    - ToDtype last so prior ops run on uint8.
    """
    return v2.Compose([
        v2.Resize(_PRE_RESIZE, max_size=_MAX_SIZE, antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomPhotometricDistort(p=1.0),
        v2.RandomIoUCrop(
            min_scale=0.5,
            max_scale=1.0,
            min_aspect_ratio=0.2,
            max_aspect_ratio=5.0,
            sampler_options=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
            trials=40,
        ),
        v2.SanitizeBoundingBoxes(),
        v2.ToDtype(torch.float32, scale=True),
    ])


def get_val_transform():
    """Validation: pre-resize + dtype conversion only (no augmentation).

    `Resize(1024, max_size=1024)` matches the model's eval `min_size[-1]=1024,
    max_size=1024` exactly — so the model's internal GeneralizedRCNNTransform
    does not need to resize again. After inference, `evaluate()` scales
    predicted masks back to the true original size before RLE-encoding so
    COCOeval IoU is computed correctly.
    """
    return v2.Compose([
        v2.Resize(_PRE_RESIZE, max_size=_MAX_SIZE, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
    ])
```

- [ ] **Step 2.4: Run tests to confirm they pass**

Run: `uv run pytest tests/test_augment.py -v`

Expected: PASS (3 new tests + any pre-existing).

- [ ] **Step 2.5: Functional smoke test — apply transform to a real sample**

Run this one-shot in shell to verify the pipeline runs end-to-end on a real training sample without errors:
```bash
uv run python << 'EOF'
from pathlib import Path
from src.dataset import CellDataset, load_or_build_annotations
from src.augment import get_train_transform

train_coco, _ = load_or_build_annotations(
    Path("data/train"),
    Path("data/train_annotations.json"),
    Path("data/val_annotations.json"),
)
ds = CellDataset(Path("data/train"), train_coco, transforms=get_train_transform())
img, target = ds[0]
print(f"img: {img.shape} {img.dtype}, range [{img.min():.2f}, {img.max():.2f}]")
print(f"boxes: {target['boxes'].shape}, masks: {target['masks'].shape}, labels: {target['labels'].shape}")
assert img.dtype.is_floating_point
assert img.shape[0] == 3
assert target["boxes"].shape[0] == target["masks"].shape[0] == target["labels"].shape[0]
print("OK")
EOF
```

Expected output ends with `OK`, image float32, shape `(3, ~1024, ~1024)` (varies by crop), counts match.

- [ ] **Step 2.6: Commit**

```bash
git add src/augment.py tests/test_augment.py
git commit -m "feat(augment): add RandomIoUCrop + SanitizeBoundingBoxes; align pre-resize to 1024"
```

---

## Task 3: Model anchor + scale config (`src/model.py`)

**Files:**
- Modify: `src/model.py:18-62`
- Test: `tests/test_model.py`

- [ ] **Step 3.1: Write failing test for new anchor sizes and defaults**

Append to `tests/test_model.py`:
```python
def test_build_model_default_min_size_and_max_size():
    """Defaults shifted to (640, 768, 896, 1024) and max=1024."""
    from src.model import build_model

    model = build_model(grad_checkpoint=False)
    # GeneralizedRCNNTransform stores min_size/max_size on the transform
    transform = model.transform
    assert transform.min_size == (640, 768, 896, 1024), f"got {transform.min_size}"
    assert transform.max_size == 1024, f"got {transform.max_size}"


def test_build_model_anchor_sizes_shifted():
    """Anchor sizes start at 4 (was 8); 6 anchors per location uniformly."""
    from src.model import build_model

    model = build_model(grad_checkpoint=False)
    sizes = model.rpn.anchor_generator.sizes
    expected = ((4, 8), (16, 32), (32, 64), (64, 128), (128, 256))
    assert sizes == expected, f"got {sizes}"
    counts = model.rpn.anchor_generator.num_anchors_per_location()
    assert counts == [6, 6, 6, 6, 6], f"non-uniform anchor count breaks RPNHead: {counts}"


def test_build_model_trainable_params_under_200m():
    """Assignment hard limit: < 200M trainable parameters."""
    from src.model import build_model

    model = build_model()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable < 200_000_000, f"{trainable/1e6:.2f}M exceeds 200M cap"
```

- [ ] **Step 3.2: Run tests to confirm they fail**

Run: `uv run pytest tests/test_model.py -v`

Expected: FAIL — current defaults are `(480, 512, 544)` / `640` and anchors start at 8.

- [ ] **Step 3.3: Update `src/model.py`**

Replace the `build_model` signature defaults and the `anchor_generator` block. The full updated `build_model`:
```python
def build_model(
    num_classes: int = 5,
    min_size: tuple[int, ...] = (640, 768, 896, 1024),
    max_size: int = 1024,
    grad_checkpoint: bool = False,
) -> MaskRCNN:
    """Build ResNet101-FPN Mask R-CNN.

    num_classes: 4 cell types + 1 background = 5.
    min_size: shorter-side targets for multi-scale training (multiples of 32 align
        cleanly with FPN strides 4/8/16/32/64). Eval uses min_size[-1]=1024.
    max_size: maximum image side length after resizing.
    grad_checkpoint: enable gradient checkpointing on ResNet layer2/3/4 to
        reduce activation memory ~30-40% at the cost of one extra forward pass.
    """
    backbone = resnet_fpn_backbone(
        backbone_name="resnet101",
        weights=ResNet101_Weights.IMAGENET1K_V2,
        trainable_layers=5,  # fine-tune entire backbone for domain adaptation
    )

    if grad_checkpoint:
        for layer_name in ("layer2", "layer3", "layer4"):
            _enable_checkpointing(getattr(backbone.body, layer_name))

    # Anchor sizes shifted down by one octave compared to torchvision default to
    # add size=4 coverage at P2 — ~5% of instances have sqrt(area) < 16 at model
    # input. Trade-off: anchor-256/512 dropped (0.22% of instances > sqrt(256)
    # rely on box regression from the anchor-128 match instead). Keeps 6 anchors
    # per FPN level uniformly so RPNHead's single num_anchors works.
    anchor_generator = AnchorGenerator(
        sizes=((4, 8), (16, 32), (32, 64), (64, 128), (128, 256)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )

    model = MaskRCNN(
        backbone,
        num_classes=num_classes,
        min_size=min_size,
        max_size=max_size,
        rpn_anchor_generator=anchor_generator,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        # Dataset has up to 772 instances per image; raise the three-stage funnel
        # (pre-NMS → post-NMS → final detections) to avoid capping recall.
        rpn_pre_nms_top_n_test=2600,
        rpn_post_nms_top_n_test=1500,
        box_detections_per_img=1000,
    )
    return model
```

- [ ] **Step 3.4: Run tests to confirm they pass**

Run: `uv run pytest tests/test_model.py -v`

Expected: PASS.

- [ ] **Step 3.5: Commit**

```bash
git add src/model.py tests/test_model.py
git commit -m "feat(model): shift anchors to add size=4 at P2; bump min_size/max_size to 1024"
```

---

## Task 4: Skip dense training images (`src/dataset.py`)

**Files:**
- Modify: `src/dataset.py:88-101` (CellDataset.__init__)
- Test: `tests/test_dataset.py`

- [ ] **Step 4.1: Write failing test for skip behaviour**

Append to `tests/test_dataset.py`:
```python
def test_cell_dataset_skip_above_instances_drops_dense_images(tmp_path):
    """Images with > skip_above_instances annotations are excluded at init."""
    import numpy as np
    import tifffile

    from src.dataset import CellDataset, build_coco_annotations

    # Build two images: one with 1 instance ("img_a"), one with 3 ("img_b")
    for folder, n_instances in [("img_a", 1), ("img_b", 3)]:
        d = tmp_path / folder
        d.mkdir()
        tifffile.imwrite(
            str(d / "image.tif"),
            np.random.randint(0, 255, (10, 10, 4), dtype=np.uint8),
        )
        # one class file with `n_instances` distinct mask ids
        mask = np.zeros((10, 10), dtype=np.float64)
        for i in range(n_instances):
            mask[i, i] = i + 1
        tifffile.imwrite(str(d / "class1.tif"), mask)

    coco = build_coco_annotations(tmp_path, ["img_a", "img_b"])

    # Skip threshold = 2 → drops img_b (which has 3 instances), keeps img_a (1)
    ds = CellDataset(tmp_path, coco, transforms=None, skip_above_instances=2)
    assert len(ds) == 1
    assert ds.images[0]["file_name"] == "img_a"


def test_cell_dataset_skip_above_instances_none_keeps_all(tmp_path):
    """skip_above_instances=None keeps every image (default behaviour)."""
    import numpy as np
    import tifffile

    from src.dataset import CellDataset, build_coco_annotations

    for folder, n in [("img_a", 1), ("img_b", 3)]:
        d = tmp_path / folder
        d.mkdir()
        tifffile.imwrite(
            str(d / "image.tif"),
            np.random.randint(0, 255, (10, 10, 4), dtype=np.uint8),
        )
        mask = np.zeros((10, 10), dtype=np.float64)
        for i in range(n):
            mask[i, i] = i + 1
        tifffile.imwrite(str(d / "class1.tif"), mask)

    coco = build_coco_annotations(tmp_path, ["img_a", "img_b"])
    ds = CellDataset(tmp_path, coco, transforms=None)  # default None
    assert len(ds) == 2
```

- [ ] **Step 4.2: Run tests to confirm they fail**

Run: `uv run pytest tests/test_dataset.py::test_cell_dataset_skip_above_instances_drops_dense_images tests/test_dataset.py::test_cell_dataset_skip_above_instances_none_keeps_all -v`

Expected: FAIL — `skip_above_instances` not in `CellDataset.__init__` signature.

- [ ] **Step 4.3: Add `skip_above_instances` to `CellDataset.__init__`**

In `src/dataset.py`, update the `CellDataset.__init__` to:
```python
    def __init__(
        self,
        train_dir: Path,
        coco_data: dict,
        transforms=None,
        skip_above_instances: int | None = None,
    ):
        self.train_dir = train_dir
        self.transforms = transforms
        self._ann_by_image: dict[int, list[dict]] = {}
        for ann in coco_data["annotations"]:
            self._ann_by_image.setdefault(ann["image_id"], []).append(ann)

        # Filter out dense images that would OOM during RPN target assignment.
        # Skipping (rather than subsampling per-epoch) avoids creating false-
        # negative anchor supervision on dropped GT.
        if skip_above_instances is not None:
            self.images = [
                img for img in coco_data["images"]
                if len(self._ann_by_image.get(img["id"], [])) <= skip_above_instances
            ]
        else:
            self.images = coco_data["images"]
```

- [ ] **Step 4.4: Run tests to confirm they pass**

Run: `uv run pytest tests/test_dataset.py -v`

Expected: PASS — both new tests + all existing dataset tests.

- [ ] **Step 4.5: Commit**

```bash
git add src/dataset.py tests/test_dataset.py
git commit -m "feat(dataset): add skip_above_instances param to drop dense training images"
```

---

## Task 5: Train script — wire up new defaults (`src/train.py`)

**Files:**
- Modify: `src/train.py:46-73, 169-174`

- [ ] **Step 5.1: Update default arg values**

In `src/train.py`'s `parse_args()`, change three defaults:
- `--epochs`: `50` → `37`
- `--min-size`: `[480, 512, 544]` → `[640, 768, 896, 1024]`
- `--max-size`: `640` → `1024`

The full updated lines:
```python
    p.add_argument("--epochs", type=int, default=37)
    ...
    p.add_argument(
        "--min-size",
        type=int,
        nargs="+",
        default=[640, 768, 896, 1024],
        help="shorter-side targets for multi-scale training (multiples of 32)",
    )
    p.add_argument(
        "--max-size",
        type=int,
        default=1024,
        help="max image side after resizing",
    )
```

- [ ] **Step 5.2: Pass `skip_above_instances=400` to train dataset only**

In `src/train.py`, find the train_ds construction (around line 169) and update:
```python
    train_ds = CellDataset(
        TRAIN_DIR,
        train_coco_os,
        transforms=get_train_transform(),
        skip_above_instances=400,
    )
    val_ds = CellDataset(TRAIN_DIR, val_coco, transforms=get_val_transform())
```

`val_ds` deliberately omits `skip_above_instances` — val uses model.eval(), no RPN target assignment, so dense images don't OOM and we want them in the AP50 calculation.

- [ ] **Step 5.3: Smoke-test arg parsing**

Run:
```bash
uv run python -c "
import sys; sys.argv = ['x']
from src.train import parse_args
a = parse_args()
print(f'epochs={a.epochs}, min_size={a.min_size}, max_size={a.max_size}')
assert a.epochs == 37
assert a.min_size == [640, 768, 896, 1024]
assert a.max_size == 1024
print('OK')
"
```

Expected output ends with `OK`.

- [ ] **Step 5.4: Commit**

```bash
git add src/train.py
git commit -m "feat(train): wire new defaults (37 epochs, multi-scale 640-1024, skip dense)"
```

---

## Task 6: Single-GPU smoke test (1 epoch end-to-end)

**Files:** None modified; this is verification only.

- [ ] **Step 6.1: Verify trainable param count + VRAM with realistic load**

Run this end-to-end smoke test on a single GPU. It runs 5 training iterations on real data and prints peak VRAM:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
    timeout 600 uv run torchrun --nproc_per_node=1 -m src.train \
    --epochs 1 --grad-checkpoint 2>&1 | tail -50
```

Expected:
- No OOM during forward/backward
- Loss prints (e.g., `Epoch 1/1  loss=...`)
- Val AP50 prints at end of epoch (likely low, ~0.05-0.20 after 1 epoch — that's fine, this is a pipeline check)
- "Peak GPU Memory Statistics" block at end shows GPU 0 at < 14 GiB

If OOM appears: either `expandable_segments` env var was missed, or memory is genuinely tight. Check `nvidia-smi` shows the second GPU is free of large processes; try `--batch-size 1 --accum-steps 4` as a fallback.

- [ ] **Step 6.2: Run val-check on the 1-epoch checkpoint**

After step 6.1 finishes, find the new checkpoint dir (`ls -t checkpoints | head -1`) and confirm inference pipeline still produces non-zero AP50:

```bash
CKPT_DIR=$(ls -t checkpoints | head -1)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
    uv run python -m src.inference \
    --checkpoint "checkpoints/$CKPT_DIR/best_model.pth" \
    --output "checkpoints/$CKPT_DIR/val-check-results.json" \
    --score-thresh 0 \
    --val-check 2>&1 | tail -10
```

Expected: prints `Val AP50 (sanity check): X.XXXX` matching the AP50 logged during the 1-epoch training (within 0.001).

- [ ] **Step 6.3: Commit smoke-test artifacts (optional)**

If you want to keep a reference smoke-test result:
```bash
git add checkpoints/$(ls -t checkpoints | head -1)/val-check-results.json
git commit -m "test: smoke-test results for new aug+resolution pipeline (1 epoch)"
```

Skip this step if no commit needed.

---

## Task 7: Full 37-epoch training run (2 GPUs)

**Files:** None modified.

- [ ] **Step 7.1: Confirm both GPUs available**

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
```

Expected: both GPU 0 and GPU 1 show < 1 GiB used. If GPU 1 is still in use, fall back to single-GPU training (`torchrun --nproc_per_node=1 ...`) and accept the 2× wall-clock cost.

- [ ] **Step 7.2: Launch DDP training**

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup uv run torchrun --nproc_per_node=2 -m src.train \
    --epochs 37 --grad-checkpoint \
    > checkpoints/train.log 2>&1 &
```

Note `nohup` + `&` so the run survives terminal disconnects. The `train.log` lives at the top level temporarily; the actual checkpoint dir is `checkpoints/<TIMESTAMP>/` and gets its own `train.log` written to by `print(..., flush=True)` lines? Currently the script prints to stdout — review `src/train.py` to confirm log location, and if it's just stdout, the redirected `train.log` is the source of truth.

Single-GPU fallback if needed:
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
    nohup uv run torchrun --nproc_per_node=1 -m src.train \
    --epochs 37 --grad-checkpoint \
    > checkpoints/train.log 2>&1 &
```

- [ ] **Step 7.3: Monitor training**

```bash
tail -f checkpoints/train.log
```

Look for:
- Each epoch ends with `Epoch X/37  loss=...` and `Val AP50: ...`
- No OOM messages
- "Saved best checkpoint" lines as AP50 improves
- Total runtime expected ~5 hr on 2 GPUs (~10 hr on 1 GPU)

If val AP50 has not exceeded 0.5 by epoch 10, something is wrong — check loss is decreasing, augmentation is working (visualize a sample with `analysis/visualize_gt.py`), and the new anchor isn't producing all-background predictions.

- [ ] **Step 7.4: Identify final checkpoint**

After training completes:
```bash
CKPT_DIR=$(ls -t checkpoints | head -1)
echo "Final checkpoint dir: checkpoints/$CKPT_DIR"
ls "checkpoints/$CKPT_DIR/best_model.pth"
grep "Val AP50" "checkpoints/$CKPT_DIR/train.log" | tail -5
```

Note the best Val AP50 — this is the number to confirm during val-check.

---

## Task 8: Inference + leaderboard upload

**Files:** None modified.

- [ ] **Step 8.1: Val-check on the new best_model.pth**

```bash
CKPT_DIR=$(ls -t checkpoints | head -1)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
    uv run python -m src.inference \
    --checkpoint "checkpoints/$CKPT_DIR/best_model.pth" \
    --output "checkpoints/$CKPT_DIR/val-check-results.json" \
    --score-thresh 0 \
    --val-check 2>&1 | tail -15
```

Expected: `Val AP50 (sanity check): X.XXXX` matches the best-checkpoint AP50 from `train.log` within 0.001. This proves the new model + new pre-resize + FP32 inference still reproduce training measurements.

If the AP50 mismatches by > 0.005, **stop** — the inference pipeline isn't aligned with training. Likely culprit is `pre_resize_image` defaults vs `_PRE_RESIZE` in `augment.py` getting out of sync.

- [ ] **Step 8.2: Generate test-set submission**

```bash
CKPT_DIR=$(ls -t checkpoints | head -1)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
    uv run python -m src.inference \
    --checkpoint "checkpoints/$CKPT_DIR/best_model.pth" \
    --output "checkpoints/$CKPT_DIR/test-results.json" \
    --score-thresh 0.05 2>&1 | tail -5
```

Expected: `Wrote N predictions to ...test-results.json` with N in the range 15k–25k (similar order of magnitude to the 19,801 from the previous run).

- [ ] **Step 8.3: Upload to CodaBench**

The previous run zipped the JSON; check the existing pattern:
```bash
ls checkpoints/*/com-*.zip 2>/dev/null
```

If a `.zip` was used previously, mirror that:
```bash
CKPT_DIR=$(ls -t checkpoints | head -1)
SHA=$(git rev-parse --short HEAD)
zip -j "checkpoints/$CKPT_DIR/com-$SHA-thr0.05.zip" "checkpoints/$CKPT_DIR/test-results.json"
```

Upload the resulting `.zip` (or the raw `.json` if CodaBench accepts that) and record the leaderboard score.

- [ ] **Step 8.4: Decide next step based on leaderboard**

- **Score > 0.5975** → goal hit. Move to Task 10 (report) from the original plan.
- **Score 0.50–0.5975** → improvement but short of target. Try multi-scale TTA (run inference at 768/1024/1280 and average) before training again.
- **Score ≤ 0.50** → marginal change, the augmentation didn't help enough. Investigate: visualize predictions on a few test images via `analysis/visualize_pred.py` and compare against val visuals to identify what's failing.

Record the result and any next-step decision in a follow-up commit message or notes.

---

## Self-Review Notes

Spec coverage check:
- §4.1 (augment) → Task 2 ✓
- §4.2 (utils pre_resize_image) → Task 1 ✓
- §4.3 (model) → Task 3 ✓
- §4.4 (dataset skip) → Task 4 ✓
- §4.5 (train) → Task 5 ✓
- §4.6 (inference unchanged) → Documented in Task 1 (default arg propagation) and verified in Tasks 6.2 and 8.1 ✓
- §5 (memory budget) → Task 6 verifies under 1-epoch real load ✓
- §6 (training strategy) → Task 7 ✓
- §7 (acceptance criteria) → Tasks 6.2, 8.1, 8.2, 8.4 cover all four criteria ✓
