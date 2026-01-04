from typing import NamedTuple, Literal


import torch
import torch.nn as nn
import torch.nn.functional as F


class FP32LayerNorm(nn.LayerNorm):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            hidden_states.to(torch.float32),
            self.normalized_shape,
            self.weight.to(torch.float32) if self.weight is not None else None,
            self.bias.to(torch.float32) if self.bias is not None else None,
            self.eps,
        ).to(hidden_states.dtype)


class FP32RMSNorm(nn.RMSNorm):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(
            hidden_states.to(torch.float32),
            self.normalized_shape,
            weight=self.weight,
            eps=self.eps,
        ).to(hidden_states.dtype)


class SingleAdaLayerNormZeroOutput(NamedTuple):
    hidden_states: torch.Tensor
    scale: torch.Tensor
    shift: torch.Tensor
    gate: torch.Tensor


class SingleAdaLayerNormZero(nn.Module):
    def __init__(
        self,
        hidden_dim: int,  # which will be normalized
        gate_dim: int,  # after attention dim
        embedding_dim: int,
    ) -> None:
        super().__init__()

        self.act = nn.SiLU()
        self.norm = FP32LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.scale_shift = nn.Linear(  # time -> scale, shift
            embedding_dim,
            2 * hidden_dim,  # 2 for scale, shift
            bias=True,
        )
        self.gate = nn.Linear(  # time -> gate
            embedding_dim,
            gate_dim,  # gate
            bias=True,
        )

    def init_weights(self) -> None:
        self.scale_shift.to_empty(device=torch.device("cpu"))
        self.gate.to_empty(device=torch.device("cpu"))

        # init with zeros!
        nn.init.zeros_(self.scale_shift.weight)
        nn.init.zeros_(self.scale_shift.bias)

        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        time_embed: torch.Tensor,
    ) -> SingleAdaLayerNormZeroOutput:
        norm_hidden_states = self.norm(hidden_states)

        time_embed = self.act(time_embed)
        scale, shift = self.scale_shift(time_embed).chunk(2, dim=1)
        gate = self.gate(time_embed)

        hidden_states = norm_hidden_states * (1 + scale.unsqueeze(1)) + shift.unsqueeze(
            1
        )

        return SingleAdaLayerNormZeroOutput(
            hidden_states=hidden_states,
            scale=scale,
            shift=shift,
            gate=gate,  # will be used later
        )


# ref: https://github.com/jiachenzhu/DyT/blob/main/other_tasks/DiT/dynamic_tanh.py
class DyTNrom(nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        elementwise_affine: bool = True,
        alpha_init_value: float = 0.5,
    ):
        super().__init__()

        self.normalized_shape = normalized_shape
        self.elementwise_affine = elementwise_affine
        self.alpha_init_value = alpha_init_value

        self.alpha = nn.Parameter(torch.ones(1) * self.alpha_init_value)

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.weight = None
            self.bias = None

    def init_weights(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)  # type: ignore
            nn.init.zeros_(self.bias)  # type: ignore

        nn.init.constant_(self.alpha, self.alpha_init_value)

    @torch.compile
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.elementwise_affine:
            # weight and bias are not none here
            return torch.tanh(self.alpha * input) * self.weight + self.bias  # type: ignore
        else:
            return torch.tanh(self.alpha * input)


# https://github.com/zlab-princeton/Derf/blob/main/DiT/dynamic_erf.py
class DerfNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        elementwise_affine: bool = True,
        alpha_init_value: float = 0.5,
        shift_init_value: float = 0.0,
    ):
        super().__init__()

        self.normalized_shape = normalized_shape
        self.elementwise_affine = elementwise_affine
        self.alpha_init_value = alpha_init_value
        self.shift_init_value = shift_init_value

        self.alpha = nn.Parameter(torch.ones(1) * self.alpha_init_value)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.weight = None
            self.bias = None
        self.shift = nn.Parameter(torch.ones(1) * self.shift_init_value)

    def init_weights(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)  # type: ignore
            nn.init.zeros_(self.bias)  # type: ignore

        nn.init.constant_(self.alpha, self.alpha_init_value)
        nn.init.constant_(self.shift, self.shift_init_value)

    @torch.compile
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = self.alpha * input + self.shift

        if self.elementwise_affine:
            # weight and bias are not none here
            return torch.erf(input) * self.weight + self.bias  # type: ignore
        else:
            return torch.erf(input)


NormType = Literal["layer", "rms", "dyt", "derf"]


def get_norm_layer(
    norm_type: NormType,
    normalized_shape: int,
    elementwise_affine: bool = True,
    eps: float = 1e-6,
    alpha_init_value: float = 0.5,
    shift_init_value: float = 0.0,
) -> nn.Module:
    if norm_type == "layer":
        norm_layer = FP32LayerNorm(
            normalized_shape,
            elementwise_affine=elementwise_affine,
            eps=eps,
        )
    elif norm_type == "rms":
        norm_layer = FP32RMSNorm(
            normalized_shape,
            elementwise_affine=elementwise_affine,
            eps=eps,
        )
    elif norm_type == "dyt":
        norm_layer = DyTNrom(
            normalized_shape,
            elementwise_affine=elementwise_affine,
            alpha_init_value=alpha_init_value,
        )
    elif norm_type == "derf":
        norm_layer = DerfNorm(
            normalized_shape,
            elementwise_affine=elementwise_affine,
            alpha_init_value=alpha_init_value,
            shift_init_value=shift_init_value,
        )
    else:
        raise ValueError(f"Unsupported norm type: {norm_type}")

    return norm_layer
