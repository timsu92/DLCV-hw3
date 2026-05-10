# Augmentation + Resolution Upgrade — Design Spec

**Date**: 2026-05-10
**Deadline**: 2026-05-12 23:59
**Goal**: Beat AP50 = 0.5975 on CodaBench leaderboard (current: 0.4958)

---

## 1. Motivation

The previous training run reached **Val AP50 = 0.6997** (best) but only **0.4958** on the test leaderboard — a 0.20 gap. After verifying the inference pipeline reproduces val AP50 to within 0.0001, the gap is attributable to **train→test generalization**, not to a pipeline bug.

Key observations from analysing the current dataset stats:

- 209 training images is small; the model overfits to in-distribution samples.
- Train/val image aspect ratio max is 8.07 (P50=1.22), test images go up to 9.4 — so extreme aspect ratios are not unique to test, but the long tail is sparse in train.
- Image short-side range is 74–1731 px, long-side 98–2162 px — wide variation that benefits from explicit multi-scale augmentation.
- Cell `sqrt(area)` distribution at model input: P10=18.5, P50=32.6, P90=61.9, max=970. **5.43% of instances have sqrt area < 16 px** at model input — these are below the smallest current anchor (size 8 with aspect 1).

Hypothesis: tighter alignment between augmentation, model resolution, and inference resolution — combined with explicit aspect-ratio-aware cropping and a small-anchor adjustment — closes part of the 0.20 gap. The training script otherwise stays unchanged.

---

## 2. Scope

This spec covers four orthogonal changes that ship as one training run:

1. **Augmentation pipeline**: add `v2.RandomIoUCrop` (aspect 0.2–5.0) and `v2.SanitizeBoundingBoxes`; remove `v2.Resize(640)` from the train transform composition.
2. **Pre-resize alignment**: train/val/inference pre-resize all switch to `size=1024, max_size=1024` to match the model's eval `min_size[-1]=1024, max_size=1024`.
3. **Model anchor + scale config**: shift anchors to introduce a `size=4` at P2 (`((4, 8), (16, 32), (32, 64), (64, 128), (128, 256))`) and bump `min_size` to `(640, 768, 896, 1024)`, `max_size=1024`.
4. **Skip dense training images**: exclude images with > 400 GT instances from the train split (not val) at `CellDataset.__init__` time. Dropping samples avoids creating false-negative anchor supervision; subsampling at `__getitem__` does not.

Out of scope: Mosaic (rejected — shrinks 20×20 cells past detection threshold when combined with multi-scale resize), ScaleJitter on top of model multi-scale (rejected — combined range pushes cells too small), random GT subsampling (rejected — creates false-negative anchor supervision on dropped cells), MS R-CNN head (deferred; could be a follow-up if there is time).

---

## 3. Current State

### Files involved

| File | Current responsibility |
|---|---|
| `src/augment.py` | `_PRE_RESIZE=640` constant; `get_train_transform` (Resize+flips+photometric+ToDtype); `get_val_transform` (Resize+ToDtype). |
| `src/utils.py` | `pre_resize_image(size=640, max_size=1333)` — torchvision v2 functional resize, returns float32 (3,H,W) tensor + original (h,w). Used by `src/inference.py`. |
| `src/model.py` | `build_model(min_size=(480,512,544), max_size=640)` — ResNet101-FPN Mask R-CNN; anchor sizes `((8,16),(32,64),(64,128),(128,256),(256,512))`. |
| `src/dataset.py` | `CellDataset` returns `(img_tensor, target_dict)` where `target_dict["masks"]` is `(N, H, W) uint8`, `target_dict["boxes"]` is `(N, 4) float`. No GT cap. |
| `src/train.py` | Default `--min-size 480 512 544 --max-size 640`; uses `get_train_transform()`/`get_val_transform()`. |
| `src/inference.py` | Calls `pre_resize_image(img_rgb)` with default args. |

### Verified constraints (single GPU, RTX 5080 16 GiB)

| Config | Peak VRAM | Notes |
|---|---|---|
| Old (`min=480..544, max=640`) | 9.21 GiB | Production training, leaderboard 0.4958 |
| New, no GT cap, GT=400 in dense img | 12.56 GiB | OK with `expandable_segments:True` |
| New, no GT cap, GT=600 | OOM | RPN box-IoU matrix dominates |
| New, no GT cap, GT=772 (dataset max) | OOM | Even at batch=1 |

13 of 178 training+val images (7.3%) have > 500 instances; 19 (10.7%) have > 400; 4 (2.2%) have > 700. Sub-sampling GT to 400 affects only ~11% of images.

### Trainable parameter count

`build_model` with new anchor config: **62.88M trainable parameters** (well under the 200M assignment limit).

---

## 4. Design

### 4.1 Augmentation pipeline (`src/augment.py`)

Replace `_PRE_RESIZE = 640` with `_PRE_RESIZE = 1024` and `_MAX_SIZE = 1024`. Rewrite the transforms:

```python
import torch
from torchvision.transforms import v2

_PRE_RESIZE = 1024
_MAX_SIZE = 1024


def get_train_transform():
    """RandomIoUCrop introduces aspect-ratio variety on real cells (no synthesis).

    Order matters:
    - Resize first to bound CPU mask memory (mask tensors scale with H*W*N).
    - Photometric distort uses uint8 input (faster + more accurate than on float).
    - RandomIoUCrop with sampler_options=[0.0, ...] keeps "no crop" as a valid choice
      so ~14% of samples stay full-image — preserves the original distribution.
    - SanitizeBoundingBoxes removes degenerate post-crop fragments (default
      min_size=1 keeps all real cells, even 5×5).
    - ToDtype last for numerical correctness of subsequent ops.
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
    return v2.Compose([
        v2.Resize(_PRE_RESIZE, max_size=_MAX_SIZE, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
    ])
```

`RandomIoUCrop` parameters explained:
- `min_scale=0.5, max_scale=1.0`: crop area is between 50% and 100% of input area. Below 0.5 cells become too small at typical 0.36× model resize.
- `min_aspect_ratio=0.2, max_aspect_ratio=5.0`: crop aspect ratio range 1:5 to 5:1, covers 99.5% of test image aspect ratios (max 9.4 still partially handled by IoUCrop's ratio sampling).
- `sampler_options`: list of minimum-IoU thresholds for the crop. `0.0` means "any crop OK" and effectively pass-through (no crop). The other values force the crop to keep at least N% IoU with at least one GT box.
- `trials=40`: per call, RandomIoUCrop tries up to 40 candidate crops to satisfy IoU. Default 40 is sufficient.

### 4.2 Pre-resize alignment (`src/utils.py`)

Change `pre_resize_image` defaults so train/val/inference all hit the same model-input size:

```python
def pre_resize_image(
    img: np.ndarray, size: int = 1024, max_size: int = 1024
) -> tuple[torch.Tensor, tuple[int, int]]:
    # body unchanged — only defaults updated
```

Same `v2.functional.resize` call, just `size=1024, max_size=1024`. `src/inference.py` calls this with no kwargs, so updating the default is sufficient.

### 4.3 Model anchor + scale (`src/model.py`)

```python
def build_model(
    num_classes: int = 5,
    min_size: tuple[int, ...] = (640, 768, 896, 1024),
    max_size: int = 1024,
    grad_checkpoint: bool = False,
) -> MaskRCNN:
    # ... backbone unchanged ...
    anchor_generator = AnchorGenerator(
        sizes=((4, 8), (16, 32), (32, 64), (64, 128), (128, 256)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )
    # ... rest unchanged (rpn_pre_nms_top_n_test=2600, etc.) ...
```

Why shift anchors instead of adding a 3rd size per level: RPN head requires uniform `num_anchors_per_location` across all FPN levels. Adding 3 sizes per level inflates the anchor count by 50%, which OOMs at training resolution. Shifting all levels down by one octave keeps 6 anchors per location (same compute) and adds size-4 coverage at P2. Cost: lose anchor-256 explicit coverage (instances with `sqrt(area) > 256` — 0.22% of dataset). These are still detected via box regression from the anchor-128 match.

### 4.4 Skip dense training images (`src/dataset.py`)

Add an optional `skip_above_instances: int | None = None` parameter to `CellDataset.__init__` (or to `build_coco_annotations` / `oversample_rare_classes`, whichever is the cleanest filter point). At init time — **before** the `__getitem__` path — drop folders whose annotated instance count exceeds the threshold:

```python
# In CellDataset.__init__ after building self.image_ids / self.coco
if skip_above_instances is not None:
    keep_ids = [
        img_id for img_id in self.image_ids
        if len(self.coco.getAnnIds(imgIds=img_id)) <= skip_above_instances
    ]
    self.image_ids = keep_ids
```

In `train.py`, pass `skip_above_instances=400` only to the train dataset; pass `None` for val.

**Why skip rather than subsample**: torchvision's RPN computes loss against every anchor. Anchors at locations of dropped GT are matched as background (no GT to assign), so the model learns "no cell here" at locations where there *is* a cell. Skipping the whole image removes that signal entirely, cleanly. This is also why RandomIoUCrop is safe: it removes both pixels and GT, so cropped-out cells contribute neither pixels nor anchors.

**Impact** (verified):
- 15 / 178 train images (8.4%) have > 400 instances and are dropped. Training set: 163 images.
- Val keeps all val images — eval is forward-only and does not allocate the RPN IoU matrix.
- Per-class GT loss in train split:
  - class1 (common): 11574 → 8448 (-27.0%)
  - class2 (common): 14637 → 9305 (-36.4%)
  - class3 (rare): 538 → 522 (-3.0%)
  - class4 (rare): 550 → 549 (-0.2%)

The rare classes (class3, class4) — which already get oversampled ×3 — are essentially unaffected. Class1/class2 still have 8k–9k training samples each, more than sufficient.

### 4.5 Training script (`src/train.py`)

Update default args:
- `--min-size 640 768 896 1024`
- `--max-size 1024`
- `--epochs 37`
- `--grad-checkpoint` (set to True by default? Or keep as flag — keep as flag for explicitness, document it must be set)

No structural changes to the loop. Continue from-scratch (no `--resume`).

### 4.6 Inference (`src/inference.py`)

No code changes. The default-arg update to `pre_resize_image` automatically aligns inference. The `--val-check` mode still works for sanity-checking the new model against val AP50 before submitting.

---

## 5. Memory Budget

Verified peak VRAM @ batch=2, worst-case square 1024×1024 input, 400 GT, AMP, grad_checkpoint=True, on RTX 5080 16 GiB:

| Configuration | Peak VRAM |
|---|---|
| New config, GT=400, batch=2, single-GPU | **12.56 GiB** |
| Headroom under 15 GiB cap | 2.9 GiB |

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is **required** — without it, fragmentation can push intermittent allocations over 13.5 GiB.

For the production 2-GPU run, per-GPU memory is unchanged (DDP replicates the model and shards data; each GPU sees its own batch). Same 2.9 GiB headroom per GPU.

---

## 6. Training Strategy

| Setting | Value |
|---|---|
| Mode | From-scratch |
| Epochs | 37 |
| Optimizer | AdamW, lr=1e-4, weight_decay=1e-4 (unchanged) |
| LR schedule | Linear warmup 100 steps + Cosine to 1e-6 (unchanged, recomputed for 37 epochs) |
| Effective batch | 8 (2 GPU × batch 2 × accum 2) |
| Mixed precision | `autocast("cuda")` + `GradScaler("cuda")` (unchanged) |
| Gradient clipping | max_norm=1.0 (unchanged) |
| `--grad-checkpoint` | **Required** |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` |
| Validation | Every epoch, rank 0 only (unchanged) |
| Best checkpoint | `best_model.pth` saved when val AP50 improves |
| Periodic checkpoints | every 3 epochs (unchanged) |

Expected training time: ~5 hr on 2× RTX 5080 (~25% slower than baseline due to grad_checkpoint).

Deadline buffer: training start at T-48 hr → finish T-43 hr → leaves ~40 hr for inference, debugging, multi-scale TTA fallback, and report write-up.

---

## 7. Acceptance Criteria

This spec is implemented correctly when:

1. **Sanity check via `--val-check`**: after training completes, running `uv run python -m src.inference --checkpoint <new_best> --val-check --score-thresh 0` produces a Val AP50 within 0.001 of the value logged during training (proves inference pipeline parity).
2. **Trainable params remain < 200M**: `build_model()` reports 62.88M (unchanged by anchor shift).
3. **No OOM during training**: with `expandable_segments:True`, no batch raises OOM across 37 epochs (subject to GT cap honoured by `CellDataset`).
4. **Test leaderboard improves**: leaderboard AP50 strictly greater than 0.4958 (current baseline). Goal is > 0.5975 (passing threshold).

---

## 8. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Anchor-256 loss hurts large-cell recall | Low — only 0.22% of instances exceed sqrt-area 256 | Box regression from anchor-128 still produces large-cell predictions; effect bounded |
| Skipping 15 dense images costs rare-class supervision | **Verified low**: class3 -3.0%, class4 -0.2%, class1 -27%, class2 -36%. Rare classes essentially unaffected; common classes retain 8k–9k samples each | None needed; threshold=400 is the verified safe operating point |
| `RandomIoUCrop` produces empty samples | Mitigated by `sampler_options=[0.0, ...]` — falls back to no-crop | `SanitizeBoundingBoxes` removes degenerate post-crop boxes; `box_batch_size_per_image=512` provides padding |
| New aug pipeline introduces bugs (mask alignment, etc.) | Medium | Run 1 full epoch with `--epochs 1` first; check `--val-check` AP50 is non-zero; visualize a batch via `analysis/visualize_gt.py` after applying transforms |
| Training underconverges in 37 epochs | Low — old run plateaued at epoch 23 | If at epoch 30 val AP50 still climbing, extend to 50; if plateaued at 25, accept |
| Memory headroom too tight (2.9 GiB) at runtime | Medium — empirical buffer | If OOM appears, drop GT cap to 300 (peak 10.13 GiB observed) |

---

## 9. Implementation Files to Modify

| File | Change |
|---|---|
| `src/augment.py` | Update `_PRE_RESIZE=1024`, add `_MAX_SIZE=1024`; rewrite `get_train_transform` with `RandomIoUCrop` + `SanitizeBoundingBoxes`; update `get_val_transform` with `max_size`. |
| `src/utils.py` | Change `pre_resize_image` default `size=1024, max_size=1024`. |
| `src/model.py` | Update `build_model` defaults: `min_size=(640,768,896,1024), max_size=1024`; anchor sizes shifted. |
| `src/dataset.py` | Add `skip_above_instances: int \| None = None` to `CellDataset.__init__`; filter `image_ids` at init. |
| `src/train.py` | Update default `--min-size`, `--max-size`, `--epochs`; pass `skip_above_instances=400` to train dataset (val gets `None`). |
| `src/inference.py` | No changes (pre-resize default propagates). |

No files need to be created. No tests break (tests in `tests/` use small synthetic shapes that are independent of these defaults).

---

## 10. References

- [1] He, K. et al. Mask R-CNN. ICCV 2017.
- [2] torchvision v2 transforms — `RandomIoUCrop`, `SanitizeBoundingBoxes`. https://pytorch.org/vision/main/transforms.html
- [3] Singh, B. & Davis, L. S. Large Scale Jitter (rejected here). CVPR 2018.
