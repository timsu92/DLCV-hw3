# Threshold=500 Experiment — Results & Analysis

**Date**: 2026-05-11  
**Run**: 37 epochs, single GPU (RTX 5080, CUDA_VISIBLE_DEVICES=0), `--grad-checkpoint`  
**Change from previous run**: `--skip-above-instances 500` (was 400)  
**Val AP50**: 0.7695 (epoch 22) — up from 0.7599  
**Leaderboard AP50**: **0.5348** — down from 0.5675

---

## 1. What Changed

Only one hyperparameter differed from the threshold=400 run:

| Setting | threshold=400 run | threshold=500 run |
|---|---|---|
| `--skip-above-instances` | 400 | **500** |
| Training images | 163 / 178 | **168 / 178** (+5) |
| class2 annotations | 8,534 / 14,637 (58.3%) | **10,242 / 14,637 (70.0%)** |
| Peak VRAM | 11.94 GiB | **13.94 GiB** |
| Best val AP50 | 0.7599 (epoch 26) | **0.7695 (epoch 22)** |
| Leaderboard AP50 | **0.5675** | **0.5348** |

The 5 additional images (those with 400–499 GT instances) added 1,708 class2 annotations to training. Peak VRAM rose from 11.94 → 13.94 GiB (still within the 15.46 GiB GPU limit).

Note: threshold=600 was also attempted but crashed with OOM at epoch 1. The mask head's memory cost for >500 GT instances was much higher than the IoU-matrix-only estimate.

---

## 2. Result: Val Improved, Leaderboard Dropped

| Metric | threshold=400 | threshold=500 | Δ |
|---|---|---|---|
| Best val AP50 | 0.7599 | **0.7695** | +0.0096 |
| Leaderboard AP50 | **0.5675** | 0.5348 | **−0.0327** |
| Val–test gap | 0.192 | **0.235** | +0.043 |

The val–test gap widened from 0.192 to 0.235 — a clear sign that this run overfit more to the training/val distribution despite (or because of) the additional training data.

---

## 3. Hypothesis: Why Adding Data Hurt

### 3.1 The 5 added images are high-density images

The 5 images newly included (GT count 400–499) are among the densest in the training split. Their instance patterns (tightly packed cells at high density) may not be representative of the test set distribution.

### 3.2 Val improvement may be partially spurious

The val set includes dense images from its own split. After training with more dense images, the model has seen more variety of high-density scenes — which directly overlaps with what the val evaluator measures, but may not help with test-set images that have different density or scale characteristics.

### 3.3 The val–test gap widened, not narrowed

The core problem diagnosed in the original run was distribution shift between train/val (both from the same ~1760×1760 px nearly-square images) and the test set (wider aspect ratios, different scale distribution). Adding more high-density images from the training distribution makes the model better at that distribution — but the test set's difficulty comes from geometric variety (aspect ratio, scale), not from density.

### 3.4 Possible confirmation

If this hypothesis is correct, training with `RandomIoUCrop` already provides the aspect-ratio variety that matters for the test set. The 5 additional dense images add density supervision but not aspect-ratio variety, and the noise they introduce in the model's convergence path slightly hurts.

---

## 4. Decision: Revert to threshold=400

The threshold=400 checkpoint (`checkpoints/20260510T135904Z/best_model.pth`) produced:
- Val AP50 = 0.7599
- Leaderboard AP50 = 0.5675

This is the current best leaderboard result. Future experiments should use threshold=400 as the baseline.

---

## 5. Lessons Learned

| Lesson | Detail |
|---|---|
| Val AP50 ≠ leaderboard AP50 | The val set is drawn from the same distribution as training data; improvements on val do not reliably predict leaderboard improvement when the test distribution differs. |
| Dense images may not help | High-density training images (400–500 GT) add data but also potentially shift the model toward density-specific features that are not general. |
| VRAM estimate was wrong for threshold=600 | The simple IoU-matrix model under-estimated mask-head memory. Actual OOM for GT=600 images happened at epoch 1; safe upper bound is threshold~500 with grad_checkpoint. |
| The val–test gap is the right metric to watch | A widening gap (0.192 → 0.235) signals overfitting to the train/val distribution even when val AP50 improved. |
