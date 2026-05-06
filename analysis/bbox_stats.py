"""Bounding box size analysis — helps validate/tune anchor generator sizes.

Usage:
    uv run python -m analysis.bbox_stats

Outputs:
- Per-class and overall bbox size distributions (original image scale)
- Estimated bbox sizes after MaskRCNN's internal resize
- Anchor coverage summary vs. actual bbox sizes
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

CACHE_TRAIN = Path("data/train_annotations.json")
CACHE_VAL   = Path("data/val_annotations.json")

# MaskRCNN training resize params (from src/model.py)
MIN_SIZES = (640, 704, 768, 832, 896, 1024)
MAX_SIZE  = 2000

# Anchor sizes from src/model.py (one value per FPN level, at scaled image resolution)
ANCHOR_SIZES = [8, 16, 32, 64, 64, 128, 128, 256, 256, 512]  # flattened per-level pairs
ANCHOR_MIN = 8
ANCHOR_MAX = 512

CATEGORY_NAMES = {1: "class1", 2: "class2", 3: "class3", 4: "class4"}


def maskrcnn_scale(H: int, W: int, min_size: int = 832, max_size: int = MAX_SIZE) -> float:
    """Estimate the resize scale MaskRCNN applies to an (H, W) image."""
    scale = min_size / min(H, W)
    if max(H, W) * scale > max_size:
        scale = max_size / max(H, W)
    return scale


def percentile_table(values: np.ndarray, label: str) -> None:
    p = np.percentile(values, [0, 10, 25, 50, 75, 90, 95, 99, 100])
    print(f"\n  {label}  (n={len(values):,})")
    print(f"    min={p[0]:.1f}  p10={p[1]:.1f}  p25={p[2]:.1f}  median={p[3]:.1f}"
          f"  p75={p[4]:.1f}  p90={p[5]:.1f}  p95={p[6]:.1f}  p99={p[7]:.1f}  max={p[8]:.1f}")


def main() -> None:
    # Load both splits
    with open(CACHE_TRAIN) as f:
        train = json.load(f)
    with open(CACHE_VAL) as f:
        val = json.load(f)

    # Build image_id → (H, W) lookup
    img_hw: dict[int, tuple[int, int]] = {}
    for coco in (train, val):
        for img in coco["images"]:
            img_hw[img["id"]] = (img["height"], img["width"])

    all_anns = train["annotations"] + val["annotations"]
    print(f"Total annotations: {len(all_anns):,}")

    # Collect per-category bbox data
    orig_wh:   dict[int, list] = {c: [] for c in range(1, 5)}
    scaled_wh: dict[int, list] = {c: [] for c in range(1, 5)}

    for ann in all_anns:
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            continue
        cat = ann["category_id"]
        img_id = ann["image_id"]
        H, W = img_hw[img_id]

        orig_wh[cat].append((w, h))

        # Use median min_size for a representative estimate
        scale = maskrcnn_scale(H, W, min_size=832)
        scaled_wh[cat].append((w * scale, h * scale))

    # ── Original scale ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BBOX SIZES — ORIGINAL IMAGE SCALE (pixels)")
    print("=" * 70)

    all_orig = []
    for cat in range(1, 5):
        arr = np.array(orig_wh[cat])          # (N, 2): [w, h]
        if len(arr) == 0:
            continue
        all_orig.extend(arr.tolist())
        sides = np.sqrt(arr[:, 0] * arr[:, 1])   # geometric mean side (proxy for object size)
        percentile_table(sides, f"{CATEGORY_NAMES[cat]} — √(w×h)")

    all_orig_arr = np.array(all_orig)
    sides_all = np.sqrt(all_orig_arr[:, 0] * all_orig_arr[:, 1])
    percentile_table(sides_all, "ALL CLASSES — √(w×h)")

    # ── Scaled (after MaskRCNN resize) ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("BBOX SIZES — AFTER MASKRCNN RESIZE (min_size=832, max_size=2000)")
    print("=" * 70)

    all_scaled = []
    for cat in range(1, 5):
        arr = np.array(scaled_wh[cat])
        if len(arr) == 0:
            continue
        all_scaled.extend(arr.tolist())
        sides = np.sqrt(arr[:, 0] * arr[:, 1])
        percentile_table(sides, f"{CATEGORY_NAMES[cat]} — √(w×h) scaled")

    all_scaled_arr = np.array(all_scaled)
    sides_scaled = np.sqrt(all_scaled_arr[:, 0] * all_scaled_arr[:, 1])
    percentile_table(sides_scaled, "ALL CLASSES — √(w×h) scaled")

    # ── Anchor coverage ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ANCHOR COVERAGE vs. SCALED BBOX SIZES")
    print("=" * 70)
    print(f"\n  Current anchor range: {ANCHOR_MIN}px – {ANCHOR_MAX}px")
    print(f"  (aspect ratios 0.5, 1.0, 2.0 → effective range ×{1/math.sqrt(2):.2f}–×{math.sqrt(2):.2f})")

    below = (sides_scaled < ANCHOR_MIN).mean() * 100
    above = (sides_scaled > ANCHOR_MAX).mean() * 100
    within = 100 - below - above
    print(f"\n  Scaled bboxes within anchor range [{ANCHOR_MIN}, {ANCHOR_MAX}]: {within:.1f}%")
    print(f"  Below anchor min (<{ANCHOR_MIN}px): {below:.1f}%")
    print(f"  Above anchor max (>{ANCHOR_MAX}px): {above:.1f}%")

    # Suggest if adjustment needed
    p1_scaled  = float(np.percentile(sides_scaled, 1))
    p99_scaled = float(np.percentile(sides_scaled, 99))
    print(f"\n  p1={p1_scaled:.1f}px  p99={p99_scaled:.1f}px")
    if p1_scaled < ANCHOR_MIN * 0.7:
        print(f"  ⚠  p1 is well below anchor min — consider adding smaller anchors "
              f"(e.g. sizes starting at {int(p1_scaled * 0.8)}px)")
    if p99_scaled > ANCHOR_MAX * 1.3:
        print(f"  ⚠  p99 exceeds anchor max — consider adding larger anchors "
              f"(e.g. sizes up to {int(p99_scaled * 1.2)}px)")
    if p1_scaled >= ANCHOR_MIN * 0.7 and p99_scaled <= ANCHOR_MAX * 1.3:
        print("  ✓  Current anchor range looks appropriate for this dataset.")

    # Save summary JSON
    out = Path("analysis/output/bbox_stats.json")
    out.parent.mkdir(exist_ok=True)
    summary = {
        "total_annotations": len(all_anns),
        "original_scale": {
            CATEGORY_NAMES[c]: {
                "n": len(orig_wh[c]),
                "sqrt_wh_percentiles": {
                    str(p): round(float(np.percentile(np.sqrt(np.array(orig_wh[c])[:, 0] * np.array(orig_wh[c])[:, 1]), p)), 2)
                    for p in [0, 10, 25, 50, 75, 90, 95, 99, 100]
                }
            } for c in range(1, 5) if orig_wh[c]
        },
        "scaled_min832": {
            "anchor_range": [ANCHOR_MIN, ANCHOR_MAX],
            "pct_within": round(within, 2),
            "pct_below_min": round(below, 2),
            "pct_above_max": round(above, 2),
            "p1": round(p1_scaled, 2),
            "p99": round(p99_scaled, 2),
        },
    }
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary to {out}")


if __name__ == "__main__":
    main()
