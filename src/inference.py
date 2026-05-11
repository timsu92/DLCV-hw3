"""Inference script — produces test-results.json for CodaBench submission.

Run:
    uv run python -m src.inference --checkpoint checkpoints/best_model.pth \
        --output test-results.json \
        --score-thresh 0.3

Sanity-check the inference pipeline against training Val AP50:
    uv run python -m src.inference --checkpoint checkpoints/best_model.pth \
        --output val-results.json --score-thresh 0 --val-check

The `--val-check` flag runs inference on the held-out val split (same images
that train.py's evaluate() saw) and prints AP50. If this number matches the
AP50 logged during training, the inference pipeline reproduces the training
measurement faithfully.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.dataset import load_or_build_annotations
from src.model import build_model
from src.utils import (
    binary_mask_to_bbox,
    cross_class_nms,
    encode_mask,
    load_rgb,
    pre_resize_image,
    resize_binary_mask,
)


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
    score_threshold: float = 0.3,
    device: torch.device = torch.device("cuda"),
) -> list[dict]:
    """Run model on all images and return COCO-result dicts.

    Runs in FP32 (no autocast) to match train.py's evaluate(), so this
    pipeline reproduces the validation measurement on the test set too.
    """
    results = []

    with torch.no_grad():
        for filename, image_id in image_name_to_id.items():
            torch.cuda.empty_cache()
            img_path = test_dir / filename
            img_rgb = load_rgb(img_path)  # (H, W, 3) uint8
            # pre_resize_image returns a (3, H, W) float32 tensor mirroring
            # get_val_transform (v2.Resize antialiased + ToDtype scale=True),
            # plus a longer-side cap to bound paste_masks_in_image memory.
            img_t, (orig_h, orig_w) = pre_resize_image(img_rgb)
            img_t = img_t.to(device)

            preds = model([img_t])[0]
            preds = cross_class_nms(preds)

            for box, label, score, mask in zip(
                preds["boxes"], preds["labels"], preds["scores"], preds["masks"]
            ):
                if score.item() < score_threshold:
                    continue
                binary = (mask[0] > 0.5).cpu().numpy().astype(bool)
                binary = resize_binary_mask(binary, orig_h, orig_w)
                if not binary.any():
                    continue
                results.append(
                    build_submission_entry(
                        image_id=image_id,
                        category_id=label.item(),
                        score=score.item(),
                        binary_mask=binary,
                    )
                )

    return results


def compute_segm_ap50(results: list[dict], val_coco: dict) -> float:
    """Run COCOeval (segm) on val results — mirrors train.py's evaluate()."""
    if not results:
        return 0.0
    coco_gt = COCO()
    coco_gt.dataset = val_coco
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(results)
    evaluator = COCOeval(coco_gt, coco_dt, "segm")
    evaluator.params.maxDets = [1, 10, 1500]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return float(evaluator.stats[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/best_model.pth")
    )
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_release"))
    parser.add_argument(
        "--id-map", type=Path, default=Path("data/test_image_name_to_ids.json")
    )
    parser.add_argument("--output", type=Path, default=Path("test-results.json"))
    parser.add_argument("--score-thresh", type=float, default=0.05)
    parser.add_argument(
        "--val-check",
        action="store_true",
        help=(
            "Run on val split (data/train/) and print AP50 to compare against "
            "the AP50 logged during training. Use --score-thresh 0 for an "
            "apples-to-apples match (train.py's evaluate() does not filter)."
        ),
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    use_cbam = ckpt.get("use_cbam", False)
    model = build_model(use_cbam=use_cbam)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    val_coco: dict | None = None
    if args.val_check:
        _, val_coco = load_or_build_annotations(
            Path("data/train"),
            Path("data/train_annotations.json"),
            Path("data/val_annotations.json"),
        )
        test_dir = Path("data/train")
        image_name_to_id = {
            f"{img['file_name']}/image.tif": img["id"] for img in val_coco["images"]
        }
    else:
        with open(args.id_map) as f:
            id_list = json.load(f)
        image_name_to_id = {entry["file_name"]: entry["id"] for entry in id_list}
        test_dir = args.test_dir

    results = run_inference(
        model, test_dir, image_name_to_id, args.score_thresh, device
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"Wrote {len(results)} predictions to {args.output}")

    if args.val_check:
        assert val_coco is not None
        ap50 = compute_segm_ap50(results, val_coco)
        print("\n" + "=" * 60)
        print(f"Val AP50 (sanity check): {ap50:.4f}")
        print("Compare this against the best Val AP50 logged during training.")
        print("=" * 60)


if __name__ == "__main__":
    main()
