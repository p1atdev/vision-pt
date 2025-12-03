import torch

from typing import Literal
from pydantic import BaseModel

from ...utils.dtype import str_to_dtype
# from ...modules.attention import AttentionImplementation


DenoiserType = Literal["class2image", "text2image"]


class BaseDenoiserConfig(BaseModel):
    type: DenoiserType

    patch_size: int = 16
    in_channels: int = 3
    out_channels: int = 3
    hidden_size: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0

    bottleneck_dim: int = 128

    rope_theta: float = 256.0
    rope_axes_dims: list[int] = [16, 24, 24]
    rope_axes_lens: list[int] = [256, 128, 128]
    rope_zero_centered: list[bool] = [False, True, True]


# class to image
class C2IDenoiserConfig(BaseDenoiserConfig):
    type: Literal["class2image"] = "class2image"

    num_classes: int = 1000
    in_context_len: int = 32


class T2IDenoiserConfig(BaseDenoiserConfig):
    type: Literal["text2image"] = "text2image"

    text_embed_dim: int
    text_context_len: int = 64


DenoiserConfig = C2IDenoiserConfig | T2IDenoiserConfig


class C2I_B_16_Config(C2IDenoiserConfig):
    patch_size: int = 16

    depth: int = 12
    hidden_size: int = 768
    num_heads: int = 12
    bottleneck_dim: int = 128

    in_context_len: int = 32


class T2I_B_16_Config(T2IDenoiserConfig):
    patch_size: int = 16

    depth: int = 12
    hidden_size: int = 768
    num_heads: int = 12
    bottleneck_dim: int = 128

    text_embed_dim: int
    text_context_len: int = 128

    rope_axes_dims: list[int] = [16, 24, 24]  # sum = 64 = 768 / 12
    rope_axes_lens: list[int] = [
        256,  # max 256 token text
        128,  # 2048x2048 image size
        128,
    ]
