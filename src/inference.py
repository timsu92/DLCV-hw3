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
            .float()
            .div(255.0)
            .to(device)
        )

        device_type = "cuda" if device.type == "cuda" else "cpu"
        with autocast(device_type):
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
