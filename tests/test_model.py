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
