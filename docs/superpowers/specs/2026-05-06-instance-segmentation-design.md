# Instance Segmentation Assignment — Design Spec

**Date**: 2026-05-06  
**Deadline**: 2026-05-12 23:59  
**Goal**: Beat AP50 = 0.5975 on CodaBench leaderboard

---

## 1. Dataset

| Property | Value |
|---|---|
| Train images | 209 (85% train / 15% val, seed=42) |
| Test images | 101 |
| Image format | TIFF, shape (H, W, 4) uint8 — **drop alpha (always 255), use RGB only** |
| Mask format | TIFF, float64; each non-zero pixel value = one instance |
| Classes | 4 cell types (class1–class4) |
| Image size range | H: 81–1956px, W: 74–2162px |
| Class imbalance | class1/2: ~15K instances each; class3/4: ~600 each |

**Mask decoding rule**: Instance IDs are globally unique across all class files for a given image (class1.tif owns IDs 1..N, class2.tif owns N+1..M, etc.). Category is determined by *which file* the instance appears in, not the ID value. Background = 0.

**ID mapping**: `data/test_image_name_to_ids.json` maps test filenames to **1-based** `image_id`.

---

## 2. Project Structure

```
project/
├── src/
│   ├── dataset.py      # Dataset class + COCO JSON generation + train/val split
│   ├── model.py        # ResNet101-FPN + Mask R-CNN construction
│   ├── augment.py      # torchvision.transforms.v2 augmentation pipeline
│   ├── train.py        # DDP training script (2-GPU)
│   └── inference.py    # Inference → RLE → test-results.json
├── analysis/
│   ├── explore_data.py # Dataset statistics (keep in repo for report reference)
│   ├── visualize_gt.py # Verify dataset pipeline BEFORE training (see § 2.5)
│   └── visualize_pred.py # Visualize model predictions after training
├── report/
│   ├── package.json    # bun-managed
│   ├── src/index.html  # Report (HTML + CSS, ECCV 2026 two-column style)
│   └── scripts/pdf.ts  # Playwright → PDF export
└── pyproject.toml      # uv-managed; imagecodecs and python-pptx already added
```

---

## 3. Data Preprocessing

### Mask → COCO annotations

For each training image:
- For each `class{N}.tif` (N = 1–4):
  - For each unique non-zero value `v` in the mask:
    - Extract binary mask: `mask == v`
    - `category_id = N` (1-based)
    - Compute bbox from binary mask
    - Encode to RLE via `pycocotools.mask.encode()`

Generate two COCO-format JSON files at startup:
- `data/train_annotations.json`
- `data/val_annotations.json`

### Training augmentation pipeline

```python
torchvision.transforms.v2.Compose([
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),        # medical images have no orientation bias
    RandomPhotometricDistort(),        # brightness/contrast/saturation/hue jitter
    ScaleJitter(target_size=(1024, 1024), scale_range=(0.5, 2.0)),
    RandomShortestSize(min_size=(640, 704, 768, 832, 896, 1024), max_size=2000),
])
```

Validation: resize only (no augmentation), `min_size=1024, max_size=2000`.

### Class imbalance

`WeightedRandomSampler`: images containing class3 or class4 get weight × 3, others weight = 1.

### 2.5 Dataset Verification (run before training)

`analysis/visualize_gt.py` is an **interactive matplotlib viewer**. Images are sorted alphabetically by folder name; index = position in that sorted order (0-based display as 1-based to the user).

**Navigation**:
- `→` / `←` (or `d` / `a`): next / previous image
- Type a number then `Enter`: jump to that image index (1-based)

**Display**: for each image, draw per-instance coloured mask overlays + bounding boxes + category label text (`class1`–`class4`) + instance count in title bar: `[42/209] <folder-name> — 37 instances`.

Must confirm before proceeding to training:
- [ ] Masks align with image content
- [ ] Each instance has a distinct colour overlay
- [ ] Category labels are correct (class1–class4)
- [ ] Bounding boxes are tight around each mask

---

## 4. Model Architecture

### Main model (Approach B)

```python
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet101_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator

backbone = resnet_fpn_backbone(
    backbone_name='resnet101',
    weights=ResNet101_Weights.IMAGENET1K_V2,
    trainable_layers=5,  # fine-tune entire backbone
)

model = MaskRCNN(
    backbone,
    num_classes=5,  # 4 cell types + background (class 0)
    min_size=(640, 704, 768, 832, 896, 1024),
    max_size=2000,
    rpn_anchor_generator=AnchorGenerator(
        # 5 tuples = 5 FPN feature maps (P2–P6); smaller anchors on high-res levels
        sizes=((8, 16), (32, 64), (64, 128), (128, 256), (256, 512)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    ),
)
```

**Parameter count**: ~64M (well within 200M limit).

### Additional experiment (Approach C)

Add a Mask Scoring Head (MS R-CNN, CVPR 2019): a small conv+FC branch that predicts the IoU between each predicted mask and its GT mask. At inference, replace the classification score with this IoU score for ranking — improves AP by rewarding high-quality masks.

---

## 5. Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Epochs | 50 |
| Effective batch size | 8 (2 GPU × batch 2 × accum steps 2) |
| Gradient accumulation | 2 steps; use `model.no_sync()` during intermediate steps |
| Gradient clipping | max norm = 1.0 |
| Mixed precision | `from torch.amp.autocast_mode import autocast` + `from torch.amp.grad_scaler import GradScaler` |
| LR warmup | Linear over 500 steps (start_factor=0.1) |
| LR decay | CosineAnnealingLR → eta_min=1e-6 |
| Train/val split | 85% / 15%, random_seed=42 |

**Checkpoint**: save best `val_ap50` checkpoint as `checkpoints/best_model.pth`.

**DDP launch**: `torchrun --nproc_per_node=2 src/train.py`

---

## 6. Inference & Submission

**Pipeline**:
1. Load test image → drop alpha → RGB tensor
2. `model.eval()` + `autocast('cuda')`
3. Filter predictions by score threshold (start at 0.3, tune via COCOeval)
4. For each predicted instance: encode binary mask → RLE via `pycocotools`
5. Dump to `test-results.json`

**Output format** (per entry):
```json
{
  "image_id": 1,
  "category_id": 2,
  "score": 0.714,
  "bbox": [x, y, w, h],
  "segmentation": { "size": [H, W], "counts": "<utf-8 rle string>" }
}
```

**Key correctness checks**:
- `image_id` is **1-based** (from `test_image_name_to_ids.json`)
- `category_id` is **1-based** (1–4, matching class1–class4)
- `segmentation.size` is `[height, width]` (not width × height)
- RLE counts: `rle["counts"].decode("utf-8")` after `pycocotools.mask.encode()`
- Masks are Fortran-contiguous (`np.asfortranarray`) before encoding

---

## 7. PDF Report

**Stack**: bun + Playwright (chromium)

```typescript
// report/scripts/pdf.ts
import { chromium } from "playwright";
const browser = await chromium.launch();
const page = await browser.newPage();
await page.goto(`file://${process.cwd()}/src/index.html`);
await page.pdf({ path: "report.pdf", format: "A4", printBackground: true });
await browser.close();
```

**Font requirements**:
- Body: Linux Libertine or similar (ECCV style)
- Chinese characters (author name): `Noto Sans TC` via Google Fonts or local `@font-face`
- Math (AP₅₀, superscripts): KaTeX (installed via bun) — do not rely on `<sup>`/`<sub>` alone for math

**Report sections** (per slide 7 grading):
1. Introduction & task overview
2. Method: preprocessing, model architecture, hyperparameters
3. Additional experiments: Mask Scoring Head hypothesis, results, implications
4. Results: val AP50 curve, visualization, comparison table
5. References (ECCV 2026 citation format)
