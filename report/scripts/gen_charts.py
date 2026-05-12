"""Generate visualization charts and prediction overlays for the report.

Outputs 5 PNGs into /project/report/src/images/:
  - training_curves.png
  - class_distribution.png
  - instance_sizes.png
  - pred_viz_1.png
  - pred_viz_2.png

Run from the project root:
  cd /project && uv run python report/scripts/gen_charts.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from pycocotools import mask as mask_util

PROJECT_ROOT = Path("/project")
CHECKPOINTS = PROJECT_ROOT / "checkpoints"
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
OUT_DIR = PROJECT_ROOT / "report" / "src" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Chart 1: training curves
# ---------------------------------------------------------------------------

# Hardcoded final model: 20260511T071433Z, commit 81be0cf, "CBAM (final)"
CBAM_VAL_AP50 = [
    0.3419, 0.4650, 0.3843, 0.5360, 0.6040, 0.6552, 0.6113, 0.6303, 0.6705, 0.6797,
    0.5898, 0.6795, 0.7243, 0.7453, 0.7051, 0.7183, 0.7680, 0.7513, 0.7495, 0.7737,
    0.7500, 0.7330, 0.7776, 0.7601, 0.7716, 0.7738, 0.7805, 0.7745, 0.7738, 0.7692,
    0.7699, 0.7713, 0.7797, 0.7720, 0.7733, 0.7730, 0.7732,
]
CBAM_TRAIN_LOSS = [
    1.7936, 1.3327, 1.2430, 1.1574, 1.0784, 1.0301, 1.0029, 0.9667, 0.9269, 0.8985,
    0.8799, 0.8849, 0.8380, 0.8247, 0.8170, 0.7822, 0.7625, 0.7516, 0.7428, 0.7233,
    0.7051, 0.6958, 0.6845, 0.6691, 0.6594, 0.6419, 0.6248, 0.6172, 0.6178, 0.6121,
    0.6040, 0.5926, 0.5890, 0.5915, 0.5855, 0.5867, 0.5793,
]

# Map each "other" checkpoint dir to a human-readable run label.
# The zip filename in each checkpoint dir encodes the commit: com-<sha>-ep<N>-thr<X>.zip
OTHER_RUNS = [
    # 20260510T135904Z, commit 1dbd1a2 -> Aug + Resolution upgrade (skip_above=400)
    ("20260510T135904Z", "Aug + Resolution"),
    # 20260511T083204Z, commit 17a849c -> CBAM applied to mask head too
    ("20260511T083204Z", "CBAM (mask head)"),
    # 20260509T133755Z, commit 8e6a22a -> earlier baseline (pre aug/resolution)
    ("20260509T133755Z", "Baseline"),
]

EPOCH_LOSS_RE = re.compile(r"Epoch (\d+)/\d+\s+loss=([0-9.]+)")
VAL_AP_RE = re.compile(r"Val AP50:\s*([0-9.]+)")


def parse_train_log(log_path: Path) -> tuple[list[int], list[float], list[float]]:
    """Parse train.log and return (epochs, train_loss, val_ap50).

    Assumes "Val AP50:" lines appear after the corresponding "Epoch X/Y loss=..." line.
    Returns lists of equal length covering the epochs for which both metrics exist.
    """
    if not log_path.exists():
        return [], [], []
    text = log_path.read_text(errors="replace")
    epochs, losses, ap50s = [], [], []
    # Walk through the log linearly: when we see Epoch X loss=..., remember it; the next
    # Val AP50: line is its validation result.
    cur_epoch: int | None = None
    cur_loss: float | None = None
    for line in text.splitlines():
        m_epoch = EPOCH_LOSS_RE.search(line)
        if m_epoch:
            cur_epoch = int(m_epoch.group(1))
            cur_loss = float(m_epoch.group(2))
            continue
        m_ap = VAL_AP_RE.search(line)
        if m_ap and cur_epoch is not None and cur_loss is not None:
            epochs.append(cur_epoch)
            losses.append(cur_loss)
            ap50s.append(float(m_ap.group(1)))
            cur_epoch = None
            cur_loss = None
    return epochs, losses, ap50s


def make_training_curves():
    fig, ax_loss = plt.subplots(figsize=(8, 4), dpi=150)
    ax_ap = ax_loss.twinx()

    # Color palette: CBAM is brightest, others muted.
    other_colors = ["#d97706", "#0891b2", "#6b7280"]  # amber, cyan, grey
    cbam_loss_color = "#1d4ed8"   # bright blue
    cbam_ap_color = "#dc2626"     # bright red

    # Plot "other" runs first so CBAM draws on top.
    for (ckpt_dir, label), color in zip(OTHER_RUNS, other_colors):
        log_path = CHECKPOINTS / ckpt_dir / "train.log"
        epochs, losses, ap50s = parse_train_log(log_path)
        if not epochs:
            print(f"  [skip] {ckpt_dir}: no parseable data")
            continue
        ax_loss.plot(epochs, losses, color=color, alpha=0.4,
                     linewidth=1.5, linestyle="-",
                     label=f"{label} loss")
        ax_ap.plot(epochs, ap50s, color=color, alpha=0.4,
                   linewidth=1.5, linestyle="--",
                   label=f"{label} AP50")
        print(f"  [ok]   {ckpt_dir} -> {label}: {len(epochs)} epochs")

    # CBAM (final) on top, full alpha.
    cbam_epochs = list(range(1, len(CBAM_TRAIN_LOSS) + 1))
    ax_loss.plot(cbam_epochs, CBAM_TRAIN_LOSS, color=cbam_loss_color, alpha=1.0,
                 linewidth=2.0, label="CBAM (final) loss")
    ax_ap.plot(cbam_epochs, CBAM_VAL_AP50, color=cbam_ap_color, alpha=1.0,
               linewidth=2.0, linestyle="--", label="CBAM (final) AP50")

    # Mark best epoch (ep27, 0.7805).
    best_epoch = 27
    ax_ap.axvline(best_epoch, color="black", linestyle=":", linewidth=1.0, alpha=0.7)
    ax_ap.annotate(
        f"best ep{best_epoch} (0.7805)",
        xy=(best_epoch, 0.7805),
        xytext=(best_epoch + 1, 0.55),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
    )

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Train loss")
    ax_ap.set_ylabel("Val AP50 (segm)")
    ax_loss.set_xlim(left=0)
    ax_ap.set_ylim(0.0, 1.0)
    ax_loss.grid(True, alpha=0.3)

    # Merge legends from both axes.
    h1, l1 = ax_loss.get_legend_handles_labels()
    h2, l2 = ax_ap.get_legend_handles_labels()
    ax_loss.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower right", ncol=2)

    fig.tight_layout()
    out = OUT_DIR / "training_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# Chart 2: class distribution
# ---------------------------------------------------------------------------

def make_class_distribution():
    classes = ["Class 1", "Class 2", "Class 3", "Class 4"]
    instances = [14537, 15653, 630, 587]
    images = [96, 146, 94, 58]

    fig, ax1 = plt.subplots(figsize=(7, 4), dpi=150)
    x = np.arange(len(classes))
    width = 0.4

    bars_inst = ax1.bar(x - width / 2, instances, width,
                        color="#2563eb", label="Instances",
                        edgecolor="white", linewidth=0.5)
    ax1.set_yscale("log")
    ax1.set_ylabel("Number of instances (log)", color="#2563eb")
    ax1.tick_params(axis="y", labelcolor="#2563eb")
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes)
    # Headroom so top labels fit
    ax1.set_ylim(top=max(instances) * 3)

    for b, v in zip(bars_inst, instances):
        ax1.text(b.get_x() + b.get_width() / 2, v * 1.1, f"{v:,}",
                 ha="center", va="bottom", fontsize=8, color="#1e3a8a")

    ax2 = ax1.twinx()
    bars_img = ax2.bar(x + width / 2, images, width,
                       color="#f59e0b", label="Images",
                       edgecolor="white", linewidth=0.5)
    ax2.set_ylabel("Number of images", color="#b45309")
    ax2.tick_params(axis="y", labelcolor="#b45309")
    ax2.set_ylim(top=max(images) * 1.25)

    for b, v in zip(bars_img, images):
        ax2.text(b.get_x() + b.get_width() / 2, v + max(images) * 0.02, f"{v}",
                 ha="center", va="bottom", fontsize=8, color="#92400e")

    ax1.set_title("Class distribution (instances vs images)")

    handles = [bars_inst, bars_img]
    labels = ["Instances (log)", "Images"]
    ax1.legend(handles, labels, fontsize=8, loc="upper right")

    fig.tight_layout()
    out = OUT_DIR / "class_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# Chart 3: instance sizes box plot
# ---------------------------------------------------------------------------

def make_instance_sizes():
    class_stats = [
        {"label": "Class 1", "med": 30.6, "q1": 26.8, "q3": 35.4,
         "whislo": 23.4, "whishi": 40.4, "fliers": []},
        {"label": "Class 2", "med": 19.2, "q1": 17.3, "q3": 21.0,
         "whislo": 15.4, "whishi": 23.4, "fliers": []},
        {"label": "Class 3", "med": 27.0, "q1": 24.5, "q3": 29.5,
         "whislo": 22.5, "whishi": 32.4, "fliers": []},
        {"label": "Class 4", "med": 53.4, "q1": 42.9, "q3": 68.9,
         "whislo": 36.9, "whishi": 92.8, "fliers": []},
    ]

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    colors = ["#3b82f6", "#10b981", "#ef4444", "#f59e0b"]

    bp = ax.bxp(class_stats, patch_artist=True, showfliers=False, widths=0.55)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
        patch.set_edgecolor("#1f2937")
    for median in bp["medians"]:
        median.set_color("#1f2937")
        median.set_linewidth(1.5)
    for whisker in bp["whiskers"]:
        whisker.set_color("#1f2937")
    for cap in bp["caps"]:
        cap.set_color("#1f2937")

    ax.axhline(16, color="red", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.text(0.55, 16.6, "Stride-16 limit (layer3)",
            color="red", fontsize=8, va="bottom")

    ax.set_ylabel(r"Instance size $\sqrt{\mathrm{area}}$ [px]")
    ax.set_title("Per-class instance size distribution")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    out = OUT_DIR / "instance_sizes.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# Charts 4 & 5: prediction overlays
# ---------------------------------------------------------------------------

CLASS_COLORS = {
    1: (0, 100, 255),    # blue
    2: (0, 200, 0),      # green
    3: (255, 50, 50),    # red
    4: (255, 165, 0),    # orange
}
CLASS_NAMES = {1: "class1", 2: "class2", 3: "class3", 4: "class4"}


def decode_rle(seg) -> np.ndarray:
    """Decode a COCO segmentation (RLE dict or compressed string) to a HxW uint8 mask."""
    if isinstance(seg, dict):
        rle = seg
        if isinstance(rle.get("counts"), str):
            rle = {"size": rle["size"], "counts": rle["counts"].encode("ascii")}
        return mask_util.decode(rle)
    # str fallback (uncommon)
    raise ValueError(f"Unsupported segmentation type: {type(seg)}")


def render_overlay(img_rgb: np.ndarray, preds: list[dict], score_thr: float = 0.5,
                   alpha: float = 0.5) -> np.ndarray:
    out = img_rgb.astype(np.float32).copy()
    for p in preds:
        if p["score"] < score_thr:
            continue
        cat = p["category_id"]
        color = np.array(CLASS_COLORS.get(cat, (255, 255, 255)), dtype=np.float32)
        try:
            m = decode_rle(p["segmentation"])
        except Exception as e:
            print(f"  warn: failed to decode mask: {e}")
            continue
        sel = m > 0
        if sel.shape != out.shape[:2]:
            continue
        out[sel] = (1 - alpha) * out[sel] + alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)


def save_pred_viz(img_id: int, val_imgs: dict, preds_by_img: dict,
                  out_path: Path, title: str):
    info = val_imgs[img_id]
    folder = info["file_name"]
    img_path = TRAIN_DIR / folder / "image.tif"
    img = tifffile.imread(str(img_path))
    if img.ndim == 3 and img.shape[2] >= 3:
        img_rgb = img[:, :, :3]
    else:
        img_rgb = np.stack([img] * 3, axis=-1)

    preds = preds_by_img.get(img_id, [])
    overlay = render_overlay(img_rgb, preds, score_thr=0.5, alpha=0.5)

    H, W = overlay.shape[:2]
    # figsize ~ image dims scaled, dpi=100
    fig_w = max(4.0, W / 100)
    fig_h = max(3.0, H / 100) + 0.6  # extra space for title/legend
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
    ax.imshow(overlay)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)

    # Legend swatches
    patches = [
        mpatches.Patch(color=np.array(CLASS_COLORS[c]) / 255.0, label=CLASS_NAMES[c])
        for c in (1, 2, 3, 4)
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=7,
              framealpha=0.85, ncol=4, handlelength=1.2, handletextpad=0.4,
              columnspacing=0.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def make_pred_visualizations():
    val_ann_path = DATA_DIR / "val_annotations.json"
    results_path = CHECKPOINTS / "20260511T071433Z" / "val-results.json"

    with open(val_ann_path) as f:
        val_data = json.load(f)
    val_imgs = {img["id"]: img for img in val_data["images"]}

    with open(results_path) as f:
        all_results = json.load(f)
    preds_by_img: dict[int, list[dict]] = {}
    for r in all_results:
        preds_by_img.setdefault(r["image_id"], []).append(r)

    # Image 22: 841x1177, dense (~272 preds @ 0.5), mostly class1/2 -> "many instances".
    # Image 25: 348x271, mixed with class3/4 -> "fewer larger cells / rare classes".
    pick1, pick2 = 22, 25
    print(f"  Using val image_id={pick1} ({val_imgs[pick1]['file_name']}) "
          f"and image_id={pick2} ({val_imgs[pick2]['file_name']})")

    save_pred_viz(pick1, val_imgs, preds_by_img,
                  OUT_DIR / "pred_viz_1.png",
                  "Prediction overlay (dense class1/2 sample)")
    save_pred_viz(pick2, val_imgs, preds_by_img,
                  OUT_DIR / "pred_viz_2.png",
                  "Prediction overlay (rare class3/4 sample)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Chart 1: training_curves.png")
    make_training_curves()
    print("=== Chart 2: class_distribution.png")
    make_class_distribution()
    print("=== Chart 3: instance_sizes.png")
    make_instance_sizes()
    print("=== Charts 4 & 5: pred_viz_*.png")
    make_pred_visualizations()
    print("\nAll charts written to", OUT_DIR)


if __name__ == "__main__":
    main()
