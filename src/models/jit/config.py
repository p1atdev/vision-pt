import torch
import json

from typing import Literal
from pydantic import BaseModel

from ...utils.dtype import str_to_dtype

# from ...modules.attention import AttentionImplementation
from ...modules.loss.flow_match import ModelPredictionType

PositionalEncoding = Literal["rope", "pope"]


class DenoiserConfig(BaseModel):
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
    use_output_bottleneck: bool = False
    use_pixel_shuffle: bool = False

    num_time_tokens: int = 4
    timestep_scale: float = 1.0  # or 1000.0 like diffusion

    positional_encoding: PositionalEncoding = "rope"
    rope_theta: float = 256.0
    rope_axes_dims: list[int] = [16, 24, 24]
    rope_axes_lens: list[int] = [256, 128, 128]
    rope_zero_centered: list[bool] = [False, True, True]

    context_dim: int = 768
    context_start_block: int = 0
    do_context_fuse: bool = False


class JiT_B_16_Config(DenoiserConfig):
    patch_size: int = 16

    depth: int = 12
    hidden_size: int = 768
    num_heads: int = 12
    bottleneck_dim: int = 128

    context_dim: int = 768
    context_start_block: int = 4  # 0, 1, 2, 3: no context, 4+: with context

    rope_axes_dims: list[int] = [16, 24, 24]  # sum = 64 = 768 / 12
    rope_axes_lens: list[int] = [
        256,  # max 256 token text
        128,  # 2048x2048 image size
        128,
    ]


ContextType = Literal["class", "text"]


class ClassContextConfig(BaseModel):
    type: Literal["class"] = "class"
    label2id_map_path: str

    splitter: str = " "  # ","

    do_mask_padding: bool = True

    @property
    def label2id(self) -> dict[str, int]:
        with open(self.label2id_map_path, "r") as f:
            label2id = json.load(f)

        return label2id


class TextContextConfig(BaseModel):
    type: Literal["text"] = "text"
    pretrained_model: str = "p1atdev/Qwen3-VL-2B-Instruct-Text-Only"


ContextConfig = ClassContextConfig | TextContextConfig


class JiTConfig(BaseModel):
    dtype: str = "float32"

    context_encoder: ContextConfig
    denoiser: DenoiserConfig = JiT_B_16_Config()

    # default JiT is x-pred (image prediction)
    model_pred: ModelPredictionType = "image"  # "image" | "velocity" | "noise"

    @property
    def torch_dtype(self) -> torch.dtype:
        return str_to_dtype(self.dtype)
