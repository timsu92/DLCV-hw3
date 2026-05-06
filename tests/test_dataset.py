import numpy as np
import pytest


@pytest.fixture
def tiny_train_dir(tmp_path):
    """Two-image mock train directory."""
    for folder, classes in [("img_a", [1, 2]), ("img_b", [3])]:
        d = tmp_path / folder
        d.mkdir()
        import tifffile

        # 10×10 RGBA image
        tifffile.imwrite(
            str(d / "image.tif"), np.random.randint(0, 255, (10, 10, 4), dtype=np.uint8)
        )
        for c in classes:
            # class mask: one instance with value 1, another with value 2
            mask = np.zeros((10, 10), dtype=np.float64)
            mask[1:3, 1:3] = 1
            mask[5:8, 5:8] = 2
            tifffile.imwrite(str(d / f"class{c}.tif"), mask)
    return tmp_path


def test_split_sizes(tiny_train_dir):
    from src.dataset import split_images

    train, val = split_images(tiny_train_dir, val_fraction=0.5, seed=42)
    assert len(train) + len(val) == 2
    assert len(set(train) & set(val)) == 0  # no overlap


def test_coco_categories(tiny_train_dir):
    from src.dataset import build_coco_annotations, split_images

    train_folders, _ = split_images(tiny_train_dir, val_fraction=0.5, seed=42)
    coco = build_coco_annotations(tiny_train_dir, train_folders)
    cat_ids = {c["id"] for c in coco["categories"]}
    assert cat_ids == {1, 2, 3, 4}


def test_coco_annotation_count(tiny_train_dir):
    """img_a has class1(2 inst) + class2(2 inst) = 4 anns."""
    from src.dataset import build_coco_annotations

    all_folders = sorted([d.name for d in tiny_train_dir.iterdir() if d.is_dir()])
    coco = build_coco_annotations(
        tiny_train_dir, [f for f in all_folders if f == "img_a"]
    )
    assert len(coco["annotations"]) == 4  # 2 instances × 2 classes


def test_coco_annotation_fields(tiny_train_dir):
    from src.dataset import build_coco_annotations

    all_folders = sorted([d.name for d in tiny_train_dir.iterdir() if d.is_dir()])
    coco = build_coco_annotations(tiny_train_dir, all_folders)
    ann = coco["annotations"][0]
    required = {
        "id",
        "image_id",
        "category_id",
        "segmentation",
        "bbox",
        "area",
        "iscrowd",
    }
    assert required <= ann.keys()
    assert ann["iscrowd"] == 0
    assert len(ann["bbox"]) == 4
    assert ann["segmentation"]["counts"] is not None
    assert isinstance(ann["segmentation"]["counts"], str)


def test_category_id_from_class_file(tiny_train_dir):
    """Instances from class2.tif must have category_id == 2."""
    from src.dataset import build_coco_annotations

    coco = build_coco_annotations(tiny_train_dir, ["img_a"])
    cat_ids = {ann["category_id"] for ann in coco["annotations"]}
    assert cat_ids == {1, 2}  # img_a has class1 and class2


def test_cell_dataset_len(tiny_train_dir):
    from src.dataset import CellDataset, build_coco_annotations

    all_folders = sorted([d.name for d in tiny_train_dir.iterdir() if d.is_dir()])
    coco = build_coco_annotations(tiny_train_dir, all_folders)
    ds = CellDataset(tiny_train_dir, coco)
    assert len(ds) == 2


def test_cell_dataset_item_shapes(tiny_train_dir):
    import torch

    from src.dataset import CellDataset, build_coco_annotations

    all_folders = sorted([d.name for d in tiny_train_dir.iterdir() if d.is_dir()])
    coco = build_coco_annotations(tiny_train_dir, all_folders)
    ds = CellDataset(tiny_train_dir, coco)
    img, target = ds[0]
    assert img.shape[0] == 3  # RGB channels
    assert img.dtype == torch.uint8
    assert target["boxes"].ndim == 2 and target["boxes"].shape[1] == 4
    assert target["labels"].ndim == 1
    assert target["masks"].ndim == 3
    assert len(target["boxes"]) == len(target["labels"]) == len(target["masks"])


def test_cell_dataset_boxes_xyxy(tiny_train_dir):
    """Boxes must be in XYXY format: x2 > x1 and y2 > y1."""
    from src.dataset import CellDataset, build_coco_annotations

    all_folders = sorted([d.name for d in tiny_train_dir.iterdir() if d.is_dir()])
    coco = build_coco_annotations(tiny_train_dir, all_folders)
    ds = CellDataset(tiny_train_dir, coco)
    for i in range(len(ds)):
        _, target = ds[i]
        if len(target["boxes"]) > 0:
            assert (target["boxes"][:, 2] > target["boxes"][:, 0]).all()
            assert (target["boxes"][:, 3] > target["boxes"][:, 1]).all()


def test_oversampled_dataset_has_more_entries(tiny_train_dir):
    """Oversampled dataset repeats rare-class images."""
    from src.dataset import CellDataset, build_coco_annotations, oversample_rare_classes

    all_folders = sorted([d.name for d in tiny_train_dir.iterdir() if d.is_dir()])
    coco = build_coco_annotations(tiny_train_dir, all_folders)
    ds_normal = CellDataset(tiny_train_dir, coco)
    oversampled_folders = oversample_rare_classes(tiny_train_dir, all_folders, factor=3)
    coco_os = build_coco_annotations(tiny_train_dir, oversampled_folders)
    ds_os = CellDataset(tiny_train_dir, coco_os)
    # img_b has class3 → should be repeated
    assert len(ds_os) > len(ds_normal)
