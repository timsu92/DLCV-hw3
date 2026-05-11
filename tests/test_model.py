import torch


def test_model_builds():
    from src.model import build_model

    model = build_model()
    assert model is not None


def test_model_parameter_count():
    """Model must have fewer than 200M trainable parameters."""
    from src.model import build_model

    model = build_model()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params < 200_000_000, f"Too many params: {n_params:,}"


def test_model_eval_forward():
    """model.eval() forward pass returns boxes, labels, masks, scores."""
    from src.model import build_model

    model = build_model()
    model.eval()
    img = torch.rand(3, 200, 200)
    with torch.no_grad():
        output = model([img])
    assert len(output) == 1
    result = output[0]
    assert "boxes" in result
    assert "labels" in result
    assert "masks" in result
    assert "scores" in result
    # masks shape: (N, 1, H, W) — MaskRCNN outputs soft masks
    if len(result["masks"]) > 0:
        assert result["masks"].ndim == 4
        assert result["masks"].shape[1] == 1


def test_model_train_forward():
    """model.train() forward pass returns a loss dict."""
    from src.model import build_model

    model = build_model()
    model.train()
    imgs = [torch.rand(3, 100, 100)]
    targets = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "labels": torch.tensor([1]),
            "masks": torch.zeros(1, 100, 100, dtype=torch.uint8),
        }
    ]
    losses = model(imgs, targets)
    assert isinstance(losses, dict)
    expected_keys = {
        "loss_classifier",
        "loss_box_reg",
        "loss_mask",
        "loss_objectness",
        "loss_rpn_box_reg",
    }
    assert expected_keys <= losses.keys()
    total = sum(losses.values())
    assert total.item() > 0


def test_build_model_default_min_size_and_max_size():
    """Defaults shifted to (640, 768, 896, 1024) and max=1024."""
    from src.model import build_model

    model = build_model(grad_checkpoint=False)
    transform = model.transform
    assert transform.min_size == (640, 768, 896, 1024), f"got {transform.min_size}"
    assert transform.max_size == 1024, f"got {transform.max_size}"


def test_build_model_anchor_sizes_shifted():
    """Anchor sizes start at 4 (was 8); 6 anchors per location uniformly."""
    from src.model import build_model

    model = build_model(grad_checkpoint=False)
    sizes = model.rpn.anchor_generator.sizes
    expected = ((4, 8), (16, 32), (32, 64), (64, 128), (128, 256))
    assert sizes == expected, f"got {sizes}"
    counts = model.rpn.anchor_generator.num_anchors_per_location()
    assert counts == [6, 6, 6, 6, 6], f"non-uniform anchor count breaks RPNHead: {counts}"


def test_build_model_with_cbam_builds():
    from src.model import build_model

    model = build_model(use_cbam=True)
    assert model is not None


def test_cbam_modules_present():
    from src.model import build_model, CBAM, CBAMBackboneWrapper, MaskHeadWithCBAM

    model = build_model(use_cbam=True)
    assert isinstance(model.backbone, CBAMBackboneWrapper)
    assert isinstance(model.backbone.cbams["2"], CBAM)
    assert isinstance(model.backbone.cbams["3"], CBAM)
    assert model.backbone.cbams["2"].channel.fc[0].in_features == 1024
    assert model.backbone.cbams["3"].channel.fc[0].in_features == 2048
    assert model.backbone.out_channels == 256
    assert isinstance(model.roi_heads.mask_head, MaskHeadWithCBAM)
    assert isinstance(model.roi_heads.mask_head.cbam, CBAM)
    assert model.roi_heads.mask_head.cbam.channel.fc[0].in_features == 256


def test_cbam_output_shapes_unchanged():
    from src.model import build_model

    model = build_model(use_cbam=True)
    model.eval()
    with torch.no_grad():
        out = model([torch.rand(3, 200, 200)])[0]
    assert set(out.keys()) >= {"boxes", "labels", "masks", "scores"}
    if len(out["masks"]) > 0:
        assert out["masks"].ndim == 4
        assert out["masks"].shape[1] == 1
