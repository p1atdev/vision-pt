# Reference: https://github.com/LTH14/JiT/blob/main/model_jit.py


import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

import torch.nn.functional as F

from .config import DenoiserConfig
from ...modules.norm import FP32RMSNorm
from ...modules.attention import scaled_dot_product_attention
from ...modules.timestep.embedding import get_timestep_embedding


class BottleneckPatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        patch_size: int = 16,
        in_channels: int = 3,
        bottleneck_dim: int = 128,
        hidden_dim: int = 768,
        bias: bool = True,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.bottleneck_dim = bottleneck_dim
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.proj_1 = nn.Conv2d(
            in_channels,
            bottleneck_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.proj_2 = nn.Conv2d(
            bottleneck_dim,
            hidden_dim,
            kernel_size=1,
            stride=1,
            bias=bias,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # B, C, H, W = image.shape

        # [B, C, H, W]
        # -> [B, bottleneck_dim, H/patch_size, W/patch_size] (proj_1)
        # -> [B, hidden_dim, H/patch_size, W/patch_size] (proj_2)
        # -> [B, hidden_dim, num_patches] (flatten)
        # -> [B, num_patches, hidden_dim] (transpose)
        patches = (
            self.proj_2(
                self.proj_1(image),
            )
            .flatten(2)
            .transpose(1, 2)
        )

        return patches


class TimestepEmbedder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        freq_embedding_size: int = 256,
    ):
        super().__init__()

        self.freq_embedding_size = freq_embedding_size

        self.mlp = nn.Sequential(
            nn.Linear(freq_embedding_size, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        freq_emb = get_timestep_embedding(
            timestep,
            embedding_dim=self.freq_embedding_size,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        time_embed = self.mlp(freq_emb.to(dtype=self.mlp[0].weight.dtype))

        return time_embed


def apply_rope(
    inputs: torch.Tensor,  # (batch_size, num_heads, seq_len, dim)
    freqs_cis: torch.Tensor,  # (batch_size, seq_len, dim//2) complex64
) -> torch.Tensor:
    batch_size, num_heads, seq_len, dim = inputs.shape

    with torch.autocast(device_type="cuda", enabled=False):
        inputs_cis = torch.view_as_complex(
            inputs.float().view(batch_size, num_heads, seq_len, dim // 2, 2)
        )
        freqs_cis = freqs_cis.unsqueeze(1)  # (batch_size, 1, seq_len, dim//2)
        output = torch.view_as_real(inputs_cis * freqs_cis).flatten(3)

        return output.type_as(inputs)


class RopeEmbedder:
    def __init__(
        self,
        rope_theta: float = 256.0,  # ref: Z-Image
        axes_dims: list[int] = [32, 64, 64],  # text, height, width
        axes_lens: list[int] = [256, 128, 128],  # text, height, width
        zero_centered: list[bool] = [False, True, True],  # text, height, width
    ):
        self.rope_theta = rope_theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        self.zero_centered = zero_centered

        # text starts with 0, image axes are zero-centered

        self.freqs_cis = self.precompute_freqs_cis(
            theta=self.rope_theta,
            dims=self.axes_dims,
            lens=self.axes_lens,
        )

    def get_offset(self, axis: int) -> int:
        """Get position offset for given axis."""
        return self.axes_lens[axis] // 2 if self.zero_centered[axis] else 0

    @staticmethod
    def get_rope_freqs(
        dim: int,
        min_position: int = 0,
        max_position: int = 128,
        theta: float = 10000.0,
    ) -> torch.Tensor:
        freqs = 1.0 / (
            theta
            ** (
                torch.arange(0, dim, 2, dtype=torch.float64, device=torch.device("cpu"))
                / dim
            )
        )
        positions = torch.arange(
            start=min_position,
            end=max_position,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )

        freqs = torch.outer(positions, freqs).float()  # (max_position, dim//2)
        # ↓pos, → dim//2
        # [ min_position * [1/θ^(0/dim), 1/θ^(2/dim), 1/θ^(4/dim), ..., 1/θ^((dim-2)/dim)]
        #   ...
        #   0 * [1/θ^(0/dim), 1/θ^(2/dim), 1/θ^(4/dim), ..., 1/θ^((dim-2)/dim)]
        #   1 * [1/θ^(0/dim), 1/θ^(2/dim), 1/θ^(4/dim), ..., 1/θ^((dim-2)/dim)]
        #   2 * [1/θ^(0/dim), 1/θ^(2/dim), 1/θ^(4/dim), ..., 1/θ^((dim-2)/dim)]
        #   ...
        #   max_position * [1/θ^(0/dim), 1/θ^(2/dim), 1/θ^(4/dim), ..., 1/θ^((dim-2)/dim)] ]

        freqs_cis = torch.polar(
            abs=torch.ones_like(freqs),
            angle=freqs,
        ).to(torch.complex64)  # (min_position~max_position, dim//2) complex64

        # 大きさは変えずに回転を表す複素数
        return freqs_cis

    @staticmethod
    def precompute_freqs_cis(
        theta: float,
        dims: list[int],
        lens: list[int],
        zero_centered: list[bool] = [False, True, True],
    ):
        freqs_cis = []

        for i, (dim, len_) in enumerate(zip(dims, lens)):
            freq_cis = RopeEmbedder.get_rope_freqs(
                dim=dim,
                # len_: 128 -> -64 to 63
                min_position=(len_ // 2) - len_ if zero_centered[i] else 0,
                max_position=(len_ // 2) if zero_centered[i] else len_,
                theta=theta,
            )  # (len_, dim//2) complex64

            freqs_cis.append(freq_cis)

        return freqs_cis

    # get frequencies for given position ids
    def __call__(self, position_ids: torch.Tensor):
        # move to device
        freqs_cis = [fc.to(position_ids.device) for fc in self.freqs_cis]

        result = []
        for i in range(len(self.axes_dims)):
            index = (
                position_ids[..., i : i + 1]
                .repeat(
                    # match dimensions for each axis
                    1,  # batch size?
                    1,  # sequence length?
                    freqs_cis[i].shape[-1],
                )
                .to(torch.int64)
            ) + self.get_offset(axis=i)  # adjust for zero-centered axes
            result.append(
                torch.gather(
                    freqs_cis[i].unsqueeze(0).repeat(index.shape[0], 1, 1),
                    dim=1,
                    index=index,
                )
            )

        return torch.cat(result, dim=-1)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_norm = FP32RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = FP32RMSNorm(self.head_dim) if qk_norm else nn.Identity()

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(attn_dropout)

        self.to_o = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def _pre_attn_reshape(self, x: torch.Tensor):
        batch_size, seq_len, dim = x.shape

        # [B, N, D] -> [B, N, num_heads, D/num_heads] -> [B, num_heads, N, D/num_heads]
        x = x.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)  # [B, num_heads, N, head_dim]

        return x

    def _post_attn_reshape(self, x: torch.Tensor):
        batch_size, num_heads, seq_len, head_dim = x.shape

        # [B, num_heads, N, head_dim] -> [B, N, num_heads, head_dim] -> [B, N, D]
        x = (
            x.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, seq_len, num_heads * head_dim)
        )

        return x

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_freqs: torch.Tensor,
        mask: torch.Tensor | None = None,  # 1: attend, 0: ignore
    ) -> torch.Tensor:
        batch_size, seq_len, _dim = hidden_states.shape

        # QKV
        q = self.to_q(hidden_states)
        k = self.to_k(hidden_states)
        v = self.to_v(hidden_states)

        q = self._pre_attn_reshape(q)  # [B, num_heads, N, head_dim]
        k = self._pre_attn_reshape(k)
        v = self._pre_attn_reshape(v)

        # QKNorm
        q = self.q_norm(q)
        k = self.k_norm(k)

        q = apply_rope(q, rope_freqs)
        k = apply_rope(k, rope_freqs)

        if mask is not None:
            # mask: (batch_size, seq_len) -> (batch_size, num_heads, seq_len, seq_len)
            mask = (
                mask.bool()
                .view(batch_size, 1, 1, seq_len)
                .expand(-1, self.num_heads, seq_len, -1)
            )

        attn = scaled_dot_product_attention(
            q,
            k,
            v,
            dropout=self.attn_dropout.p if self.training else 0.0,
            mask=mask,
            is_causal=False,
        ).to(hidden_states.dtype)
        attn = self._post_attn_reshape(attn)

        # output
        out = self.to_o(attn)
        out = self.proj_dropout(out)

        return out


class SwiGLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        hidden_dim = int(hidden_dim * 2 / 3)

        self.w_1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_2 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_3 = nn.Linear(hidden_dim, dim, bias=bias)

        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x_1 = self.w_1(hidden_states)
        x_2 = self.w_2(hidden_states)

        x = F.silu(x_1) * x_2

        x = self.w_3(self.ffn_dropout(x))

        return x


class FinalLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        mlp_ratio: float,
        patch_size: int,
        out_channels: int,
    ):
        super().__init__()

        self.norm_final = FP32RMSNorm(hidden_dim)

        self.mlp = SwiGLU(
            dim=hidden_dim,
            hidden_dim=int(hidden_dim * mlp_ratio),
            dropout=0.0,
            bias=True,
        )

        self.linear = nn.Linear(
            hidden_dim,
            patch_size * patch_size * out_channels,
            bias=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        x = self.norm_final(hidden_states)
        x = self.mlp(x)
        x = self.linear(x)

        return x


class JiTBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        bias: bool = True,
    ):
        super().__init__()

        self.norm1 = FP32RMSNorm(hidden_dim, eps=1e-6)
        self.attn = Attention(
            dim=hidden_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
        )

        self.norm2 = FP32RMSNorm(hidden_dim)
        self.mlp = SwiGLU(
            dim=hidden_dim,
            hidden_dim=int(hidden_dim * mlp_ratio),
            dropout=ffn_dropout,
            bias=bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_freqs: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        # attn
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            rope_freqs,
            mask=mask,
        )

        # mlp
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))

        return hidden_states


class JiT(nn.Module):
    def __init__(self, config: DenoiserConfig):
        super().__init__()

        self.config = config

        assert (config.hidden_size // config.num_heads) == sum(config.rope_axes_dims), (
            "The sum of rope_axes_dims must equal to hidden_size / num_heads = head_dim."
        )

        self.num_axes = len(
            config.rope_axes_dims
        )  # 0: image_index, 1: height, 2: width

        # image patch embedder
        self.patch_embedder = BottleneckPatchEmbed(
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            bottleneck_dim=config.bottleneck_dim,
            hidden_dim=config.hidden_size,
            bias=True,
        )

        # timestep embedder
        self.time_embedder = TimestepEmbedder(
            hidden_dim=config.hidden_size,
            freq_embedding_size=256,
        )
        self.time_position_embeds = nn.Parameter(
            torch.randn(
                config.num_time_tokens,
                config.hidden_size,
            ),
            requires_grad=True,
        )

        # RoPE embedder
        self.rope_embedder = RopeEmbedder(
            rope_theta=config.rope_theta,
            axes_dims=config.rope_axes_dims,
            axes_lens=config.rope_axes_lens,
        )

        # class condition or text embedding
        self.context_embedder = nn.Linear(
            config.context_dim,
            config.hidden_size,
            bias=True,
        )

        self.blocks = nn.ModuleList(
            [
                JiTBlock(
                    hidden_dim=config.hidden_size,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    attn_dropout=config.attn_dropout,
                    proj_dropout=config.proj_dropout,
                    ffn_dropout=0.0,
                    qkv_bias=True,
                    qk_norm=True,
                    bias=True,
                )
                for _ in range(config.depth)
            ]
        )

        self.final_layer = FinalLayer(
            hidden_dim=config.hidden_size,
            mlp_ratio=config.mlp_ratio,
            patch_size=config.patch_size,
            out_channels=config.in_channels,
        )

        self.gradient_checkpointing = False

    def initialize_weights(self):
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.RMSNorm):
                nn.init.ones_(m.weight)

        # patch embed
        w_1 = self.patch_embedder.proj_1.weight
        nn.init.xavier_uniform_(w_1.view([w_1.shape[0], -1]))
        w_2 = self.patch_embedder.proj_2.weight
        nn.init.xavier_uniform_(w_2.view([w_2.shape[0], -1]))
        if self.patch_embedder.proj_2.bias is not None:
            nn.init.zeros_(self.patch_embedder.proj_2.bias)

        # time position embeds
        nn.init.normal_(
            self.time_position_embeds,
            std=0.02,
        )

        # time embedder
        nn.init.normal_(
            self.time_embedder.mlp[0].weight,  # type: ignore
            std=0.02,
        )
        nn.init.normal_(
            self.time_embedder.mlp[2].weight,  # type: ignore
            std=0.02,
        )

    def set_gradient_checkpointing(self, enable: bool = True):
        self.gradient_checkpointing = enable

    def prepare_image_position_ids(
        self,
        height: int,
        width: int,
        global_index: int,
    ) -> torch.Tensor:
        # [H/patch_size, W/patch_size]

        patch_size = self.config.patch_size
        h_patches = height // patch_size
        w_patches = width // patch_size

        position_ids = torch.zeros(
            h_patches,
            w_patches,
            self.num_axes,
        )

        # image_index
        position_ids[:, :, 0] = global_index  # image

        # height (y-index)
        position_ids[:, :, 1] = (
            torch.arange(
                start=h_patches // 2 - h_patches,
                end=h_patches // 2,
            )
            .unsqueeze(1)
            .repeat(1, w_patches)
        )
        # width (x-index)
        position_ids[:, :, 2] = (
            torch.arange(
                start=w_patches // 2 - w_patches,
                end=w_patches // 2,
            )
            .unsqueeze(0)
            .repeat(h_patches, 1)
        )

        return position_ids.view(-1, self.num_axes)  # (num_patches, n_axes)

    def prepare_context_position_ids(
        self,
        seq_len: int,
        global_index: int = 0,
    ) -> torch.Tensor:
        position_ids = torch.zeros(
            seq_len,
            self.num_axes,
        )

        # context_index (i, ..., i)
        position_ids[:, 0] = global_index  # text

        # token indices are (0, 0)...(seq_len-1, seq_len-1)
        position_ids[:, 1] = torch.arange(seq_len)
        position_ids[:, 2] = torch.arange(seq_len)

        return position_ids

    def prepare_time_position_ids(
        self,
        seq_len: int,
        global_index: int = 1,
    ) -> torch.Tensor:
        position_ids = torch.zeros(
            seq_len,
            self.num_axes,
        )

        # time_index (i, ..., i)
        position_ids[:, 0] = global_index  # time

        # token indices are (0, 0)...(seq_len-1, seq_len-1)
        position_ids[:, 1] = torch.arange(seq_len)
        position_ids[:, 2] = torch.arange(seq_len)

        return position_ids

    def unpatchify(
        self,
        patches: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        batch_size, num_patches, _patch_dim = patches.shape

        patch_size = self.config.patch_size
        out_channels = self.config.out_channels

        h_patches = height // patch_size
        w_patches = width // patch_size

        assert num_patches == h_patches * w_patches, "Mismatch in number of patches"

        # [B, N, patch_size*patch_size*C] -> [B, H_patch, W_patch, patch_size, patch_size, C]
        patches = patches.view(
            batch_size,
            h_patches,
            w_patches,
            patch_size,
            patch_size,
            out_channels,
        )

        # [B, H_patch, W_patch, patch_size, patch_size, C]
        # -> [B, C, H_patch, patch_size, W_patch, patch_size]
        patches = patches.permute(0, 5, 1, 3, 2, 4)
        # -> [B, C, H_img, W_img]
        images = patches.reshape(batch_size, out_channels, height, width)

        return images

    def forward(
        self,
        image: torch.Tensor,  # [B, C, H, W]
        timestep: torch.Tensor,  # [B]
        context: torch.Tensor,  # [B, context_len, context_dim]
        context_mask: torch.Tensor | None = None,  # [B, context_len]
    ):
        batch_size, _in_channels, height, width = image.shape

        time_embed: torch.Tensor = self.time_embedder(timestep)  # [B, hidden_dim]
        time_tokens = time_embed.unsqueeze(1).repeat(  # add seq_len dim
            1,
            self.time_position_embeds.shape[0],  # num_time_tokens
            1,
        ) + self.time_position_embeds.unsqueeze(0).repeat(  # add batch dim
            batch_size,
            1,
            1,
        )  # [B, num_time_tokens, hidden_dim]
        num_time_tokens = time_tokens.shape[1]

        context_embed = self.context_embedder(context)
        context_len = context_embed.shape[1]

        patches = self.patch_embedder(image)  # [B, N, hidden_dim]
        patches_len = patches.shape[1]

        # context -> time -> patches
        context_position_ids = self.prepare_context_position_ids(
            seq_len=context_len,
            global_index=0,
        )
        time_position_ids = self.prepare_time_position_ids(
            seq_len=num_time_tokens,
            global_index=1,
        )
        patches_position_ids = self.prepare_image_position_ids(
            height=height,
            width=width,
            global_index=2,  # after context and time tokens
        )

        # actually: patches -> time -> context
        position_ids = torch.cat(
            [
                patches_position_ids,
                time_position_ids,
                context_position_ids,
            ],
            dim=0,
        ).view(1, -1, self.num_axes)  # (1, total_seq_len, n_axes)

        # prepare RoPE
        freqs_cis = (
            self.rope_embedder(position_ids=position_ids)
            .repeat(
                batch_size,
                1,
                1,
            )
            .to(device=image.device)
        )

        # attention mask
        if context_mask is not None:
            patches_mask = torch.ones(batch_size, patches_len, device=image.device)
            time_mask = torch.ones(batch_size, num_time_tokens, device=image.device)
            mask = torch.cat(
                [
                    patches_mask,
                    time_mask,
                    context_mask.to(image.device),
                ],
                dim=1,
            )
        else:
            # attend all
            mask = torch.ones(
                batch_size,
                patches_len + num_time_tokens + context_len,
                device=image.device,
            )

        for _i, block in enumerate(self.blocks):
            tokens = torch.cat(
                [
                    patches,  # 16x16
                    time_tokens,  # 4
                    context_embed,  # 64
                ],
                dim=1,  # cat in seq_len dimension
            )

            if self.gradient_checkpointing and self.training:
                patches = checkpoint.checkpoint(  # type: ignore
                    block,
                    tokens,
                    freqs_cis,
                    mask,
                )[:, :patches_len, :]
            else:
                patches = block(
                    tokens,
                    rope_freqs=freqs_cis,
                    mask=mask,
                )[:, :patches_len, :]  # only keep patch tokens

        patches = self.final_layer(patches)

        pred_image = self.unpatchify(
            patches,
            height=height,
            width=width,
        )

        return pred_image
