import json
import numpy as np
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
    assert len(entry["bbox"]) == 4  # [x, y, w, h]
    assert entry["bbox"][2] > 0 and entry["bbox"][3] > 0


def test_run_inference_returns_list_of_dicts(tmp_path):
    """run_inference returns a list of COCO-result dicts (no longer writes JSON)."""
    import torch
    import tifffile
    from src.inference import run_inference

    # Mock model that returns one instance covering most of the image
    # (so it survives the resize_binary_mask back to the original 200x200 size).
    def fake_model(imgs):
        H, W = imgs[0].shape[-2:]
        mask = torch.zeros(1, 1, H, W)
        mask[0, 0, H // 4 : 3 * H // 4, W // 4 : 3 * W // 4] = 1.0
        return [
            {
                "boxes": torch.tensor([[float(W // 4), float(H // 4), float(3 * W // 4), float(3 * H // 4)]]),
                "labels": torch.tensor([1]),
                "scores": torch.tensor([0.9]),
                "masks": mask,
            }
        ]

    test_image_ids = {"fake.tif": 42}

    fake_img = np.random.randint(0, 255, (200, 200, 4), dtype=np.uint8)
    tifffile.imwrite(str(tmp_path / "fake.tif"), fake_img)

    results = run_inference(
        model=fake_model,
        test_dir=tmp_path,
        image_name_to_id=test_image_ids,
        score_threshold=0.5,
        device=torch.device("cpu"),
    )

    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0]["image_id"] == 42
    assert results[0]["category_id"] == 1
