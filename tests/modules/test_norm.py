import torch

from src.modules.norm import (
    NormType,
    get_norm_layer,
    FP32LayerNorm,
    FP32RMSNorm,
    DyTNrom,
    DerfNorm,
)


@torch.no_grad()
def test_fp32_layer_norm():
    norm = FP32LayerNorm(
        normalized_shape=128,
        elementwise_affine=True,
        eps=1e-6,
    )
    assert norm is not None

    input = torch.randn(4, 128)
    output = norm(input)
    assert output.shape == input.shape


@torch.no_grad()
def test_fp32_rms_norm():
    norm = FP32RMSNorm(
        normalized_shape=128,
        elementwise_affine=True,
        eps=1e-6,
    )
    assert norm is not None

    input = torch.randn(4, 128)
    output = norm(input)
    assert output.shape == input.shape


@torch.no_grad()
def test_dyt_norm():
    norm = DyTNrom(
        normalized_shape=128,
        elementwise_affine=True,
        alpha_init_value=0.5,
    )
    assert norm is not None

    input = torch.randn(4, 128)
    output = norm(input)
    assert output.shape == input.shape


@torch.no_grad()
def test_derf_norm():
    norm = DerfNorm(
        normalized_shape=128,
        elementwise_affine=True,
        alpha_init_value=0.5,
        shift_init_value=0.0,
    )
    assert norm is not None

    input = torch.randn(4, 128)
    output = norm(input)
    assert output.shape == input.shape


@torch.no_grad()
def test_get_norm_layer():
    norm_types: list[NormType] = ["layer", "rms", "dyt", "derf"]

    for norm_type in norm_types:
        norm = get_norm_layer(
            norm_type=norm_type,
            normalized_shape=128,
            elementwise_affine=True,
            eps=1e-6,
            alpha_init_value=0.5,
            shift_init_value=0.0,
        )
        assert norm is not None

        input = torch.randn(4, 128)
        output = norm(input)
        assert output.shape == input.shape

    for norm_type in norm_types:
        norm = get_norm_layer(
            norm_type=norm_type,
            normalized_shape=128,
            elementwise_affine=False,
            eps=1e-6,
            alpha_init_value=0.5,
            shift_init_value=0.0,
        )
        assert norm is not None

        input = torch.randn(4, 128)
        output = norm(input)
        assert output.shape == input.shape
