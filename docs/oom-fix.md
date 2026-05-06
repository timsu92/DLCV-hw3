# Training OOM: Root Cause and Final Fix

## Root Cause

`torchvision`'s `GeneralizedRCNNTransform._resize_image_and_masks` (in `transform.py:79`) resizes GT masks by doing:

```python
mask = torch.nn.functional.interpolate(
    mask[:, None].float(), ...  # uint8 → float32: 4× memory, at ORIGINAL resolution
)
```

The float32 cast happens at the **original image resolution** (1772×1731), before downsampling.
For a 809-instance image this is `809 × 1772 × 1731 × 4 ≈ 9.9 GB` — impossible on a 16 GB GPU.

Reducing the model's `min_size`/`max_size` does not help because the bottleneck is
the cast at original resolution, not the output size.

### Why `--max-anns` was rejected

Random sub-sampling of GT instances creates false negatives: the model is penalised
for predicting instances that were silently dropped from the targets, corrupting training.

### Why 2-GPU DDP didn't help

GPU 1 had ~7 GB occupied by unrelated display processes. Each DDP rank needs a full
model copy, so GPU 1 was always too full.

---

## Final Fix: CPU-side pre-resize in the augmentation pipeline

Add `v2.Resize(640)` as the **first step in the training transform** (`src/augment.py`).
This resizes image + masks to ≤640 px **on CPU before GPU transfer**, so the float32 cast
in `_resize_image_and_masks` operates on a 640 px image:

```
772 instances × 640 × 655 × float32 = 1.3 GB   ← safe on 16 GB GPU
```

### Why the val transform must NOT include Resize

During inference, `target=None` is passed to the model so `_resize_image_and_masks`
skips the mask cast entirely — no OOM risk. More importantly, MaskRCNN's `postprocess`
rescales predictions back to the "original" input resolution. If val images are pre-resized
to 640 px, predictions land at 640 px, but COCO GT annotations are at 1772×1731 px →
IoU = 0 → AP50 = 0.0000. The val transform must keep original resolution.

### Additional stability fix

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` prevents CUDA memory fragmentation
(observed as "4 GiB reserved but unallocated" in earlier attempts).

---

## Working Launch Command

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 -m src.train \
  --batch-size 2 \
  --accum-steps 4
```

## AP50 Progression (final working config)

| Epoch | Loss   | Val AP50 |
|-------|--------|----------|
| 1     | 3.0156 | 0.1748   |
| 2     | 1.6355 | 0.2991   |
| 3     | 1.4350 | 0.4241   |
| 4     | 1.3143 | 0.4225   |

Target: AP50 > 0.5975 (CodaBench leaderboard threshold).
VRAM stable at ~10.5 GB (GPU 0, RTX 5080 16 GB).
