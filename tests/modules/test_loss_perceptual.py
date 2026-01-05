import torch

from src.modules.loss.perceptual import (
    LPIPSLossConfig,
    DISTSConfig,
    PerceptualLoss,
)


def test_perceptual_loss_lpips():
    loss_type = "lpips"

    pred = torch.rand(2, 3, 64, 64) * 2 - 1
    target = torch.rand(2, 3, 64, 64) * 2 - 1

    loss_fn = PerceptualLoss(
        loss_configs=[
            LPIPSLossConfig(loss_type="lpips", model_type="alex"),
        ],
        convert_zero_to_one=True,
    )

    loss = loss_fn(pred, target)[loss_type]

    assert loss.shape == ()
    assert loss.item() >= 0.0


def test_perceptual_loss_lpips_cuda():
    if not torch.cuda.is_available():
        return

    loss_type = "lpips"

    pred = torch.rand(2, 3, 64, 64).cuda() * 2 - 1
    target = torch.rand(2, 3, 64, 64).cuda() * 2 - 1

    loss_fn = PerceptualLoss(
        loss_configs=[
            LPIPSLossConfig(model="alex"),
        ],
        convert_zero_to_one=True,
    ).cuda()

    loss = loss_fn(pred, target)[loss_type]

    assert loss.shape == ()
    assert loss.item() >= 0.0


def test_perceptual_loss_dists():
    loss_type = "dists"

    pred = torch.rand(2, 3, 64, 64) * 2 - 1
    target = torch.rand(2, 3, 64, 64) * 2 - 1

    loss_fn = PerceptualLoss(
        loss_configs=[
            DISTSConfig(),
        ],
        convert_zero_to_one=True,
    )

    loss = loss_fn(pred, target)[loss_type]

    assert loss.shape == ()
    assert loss.item() >= 0.0


def test_perceptual_loss_dists_cuda():
    if not torch.cuda.is_available():
        return

    loss_type = "dists"

    pred = torch.rand(2, 3, 64, 64).cuda() * 2 - 1
    target = torch.rand(2, 3, 64, 64).cuda() * 2 - 1

    loss_fn = PerceptualLoss(
        loss_configs=[
            DISTSConfig(),
        ],
        convert_zero_to_one=True,
    ).cuda()

    loss = loss_fn(pred, target)[loss_type]

    assert loss.shape == ()
    assert loss.item() >= 0.0
