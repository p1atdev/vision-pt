# ref: https://arxiv.org/pdf/2209.12152
# https://github.com/baofff/U-ViT

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from typing import Literal

from ....modules.norm import FP32RMSNorm
from ....modules.attention import scaled_dot_product_attention
from ..denoiser import (
    Attention,
    SwiGLU,
    BottleneckPatchEmbed,
    TimestepEmbedder,
    RopeEmbedder,
    FinalLayer,
    BottleneckFinalLayer,
    JiT,
)
from ..pipeline import JiTModel
from ..config import DenoiserConfig, JiTConfig
from .pope import PopeEmbedder, apply_pope

PositionalEncoding = Literal["rope", "pope"]


class PopeAttention(Attention):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
        )

        self.pope_bias = nn.Parameter(
            torch.zeros((num_heads, self.head_dim))
        )  # (num_heads, head_dim)

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

        q = apply_pope(q, rope_freqs)
        k = apply_pope(
            k,
            rope_freqs,
            learned_bias=self.pope_bias.clamp(-torch.pi, torch.pi),
        )  # apply learned bias to K only

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


class UJiTBlock(nn.Module):
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
        has_skip_connection: bool = False,
        positional_encoding: PositionalEncoding = "rope",
    ):
        super().__init__()

        # skip connection of U-ViT
        self.skip_linear = (
            nn.Linear(
                hidden_dim * 2,
                hidden_dim,
                bias=True,
            )
            if has_skip_connection
            else nn.Identity()
        )

        self.norm1 = FP32RMSNorm(hidden_dim, eps=1e-6)
        self.attn = (
            PopeAttention(
                dim=hidden_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                attn_dropout=attn_dropout,
                proj_dropout=proj_dropout,
            )
            if positional_encoding == "pope"
            else Attention(
                dim=hidden_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                attn_dropout=attn_dropout,
                proj_dropout=proj_dropout,
            )
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
        skip_hidden_states: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ):
        # skip connection
        if skip_hidden_states is not None:
            hidden_states = self.skip_linear(
                torch.cat([hidden_states, skip_hidden_states], dim=-1)
            )

        # attn
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            rope_freqs,
            mask=mask,
        )

        # mlp
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))

        return hidden_states


class UJiTDenoiserConfig(DenoiserConfig):
    positional_encoding: PositionalEncoding = "rope"


class UJiT(JiT):
    def __init__(self, config: UJiTDenoiserConfig):
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

        # image size embedder
        self.image_size_embedder = TimestepEmbedder(
            hidden_dim=config.hidden_size,
            freq_embedding_size=256,
        )

        # RoPE embedder
        if config.positional_encoding == "rope":
            self.rope_embedder = RopeEmbedder(
                rope_theta=config.rope_theta,
                axes_dims=config.rope_axes_dims,
                axes_lens=config.rope_axes_lens,
                zero_centered=config.rope_zero_centered,
            )
        elif config.positional_encoding == "pope":
            self.rope_embedder = PopeEmbedder(
                pope_theta=config.rope_theta,
                axes_dims=config.rope_axes_dims,
                axes_lens=config.rope_axes_lens,
                zero_centered=config.rope_zero_centered,
            )
        else:
            raise ValueError(
                f"Unknown positional_encoding: {config.positional_encoding}"
            )

        # class condition or text embedding
        self.context_embedder = nn.Linear(
            config.context_dim,
            config.hidden_size,
            bias=True,
        )

        # U-JiT blocks
        self.in_blocks = nn.ModuleList(
            [
                UJiTBlock(
                    hidden_dim=config.hidden_size,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    attn_dropout=config.attn_dropout,
                    proj_dropout=config.proj_dropout,
                    ffn_dropout=0.0,
                    qkv_bias=True,
                    qk_norm=True,
                    bias=True,
                    has_skip_connection=False,
                    positional_encoding=config.positional_encoding,
                )
                for _ in range(config.depth // 2)
            ]
        )
        self.mid_block = UJiTBlock(
            hidden_dim=config.hidden_size,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            attn_dropout=config.attn_dropout,
            proj_dropout=config.proj_dropout,
            ffn_dropout=0.0,
            qkv_bias=True,
            qk_norm=True,
            bias=True,
            has_skip_connection=False,
            positional_encoding=config.positional_encoding,
        )
        self.out_blocks = nn.ModuleList(
            [
                UJiTBlock(
                    hidden_dim=config.hidden_size,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    attn_dropout=config.attn_dropout,
                    proj_dropout=config.proj_dropout,
                    ffn_dropout=0.0,
                    qkv_bias=True,
                    qk_norm=True,
                    bias=True,
                    has_skip_connection=True,  # U-JiT skip connection
                    positional_encoding=config.positional_encoding,
                )
                for _ in range(config.depth // 2)
            ]
        )

        self.final_layer = (
            BottleneckFinalLayer(
                hidden_dim=config.hidden_size,
                bottleneck_dim=config.bottleneck_dim,
                patch_size=config.patch_size,
                out_channels=config.in_channels,
            )
            if self.config.use_output_bottleneck
            else FinalLayer(
                hidden_dim=config.hidden_size,
                mlp_ratio=config.mlp_ratio,
                patch_size=config.patch_size,
                out_channels=config.in_channels,
            )
        )

        self.gradient_checkpointing = False

    def initialize_weights(self):
        super().initialize_weights()

        # Initialize weights
        for m in self.modules():
            # PoPE bias
            if isinstance(m, PopeAttention):
                nn.init.zeros_(m.pope_bias)

        # U-JiT specific initialization
        for block in self.out_blocks:
            if hasattr(block, "skip_linear") and isinstance(
                block.skip_linear, nn.Linear
            ):
                # Init skip_linear as identity for main path, zero for skip path
                # weight: [hidden_dim, 2 * hidden_dim]
                # bias: [hidden_dim]
                nn.init.zeros_(block.skip_linear.bias)
                nn.init.zeros_(block.skip_linear.weight)

                hidden_dim = block.skip_linear.weight.shape[0]
                # The first hidden_dim columns correspond to the main path (hidden_states)
                # The second hidden_dim columns correspond to the skip path (skip_hidden_states)

                # Set main path to identity
                block.skip_linear.weight.data[:, :hidden_dim] = torch.eye(hidden_dim)

        # Zero-init output projections for better gradient flow at init
        # This makes each block an identity function initially
        for block in [*self.in_blocks, self.mid_block, *self.out_blocks]:
            nn.init.zeros_(block.attn.to_o.weight)
            nn.init.zeros_(block.attn.to_o.bias)
            nn.init.zeros_(block.mlp.w_3.weight)
            nn.init.zeros_(block.mlp.w_3.bias)

    def forward_block(
        self,
        block: UJiTBlock,
        tokens: torch.Tensor,
        freqs_cis: torch.Tensor,
        context_embed: torch.Tensor,
        mask: torch.Tensor | None = None,
        skip_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.config.do_context_fuse:
            # add context tokens from this block
            tokens = torch.cat(
                [
                    tokens,
                    context_embed,  # 32 | 64
                ],
                dim=1,  # cat in seq_len dimension
            )

        if self.gradient_checkpointing and self.training:
            tokens = checkpoint.checkpoint(  # type: ignore
                block,
                tokens,
                freqs_cis,
                skip_tokens,
                mask,
            )
        else:
            tokens = block(
                tokens,
                rope_freqs=freqs_cis,
                skip_hidden_states=skip_tokens,
                mask=mask,
            )

        skip = tokens

        if not self.config.do_context_fuse:
            # remove context tokens after each block
            tokens = tokens[:, : -context_embed.shape[1], :]

        return tokens, skip

    def forward(
        self,
        image: torch.Tensor,  # [B, C, H, W]
        timestep: torch.Tensor,  # [B]
        context: torch.Tensor,  # [B, context_len, context_dim]
        original_size: torch.Tensor,  # [B, 2] (H, W)
        target_size: torch.Tensor,  # [B, 2] (H, W)
        crop_coords: torch.Tensor,  # [B, 2] (top, left)
        context_mask: torch.Tensor | None = None,  # [B, context_len]
    ):
        batch_size, _in_channels, height, width = image.shape

        # time embed + time position embed
        time_embed: torch.Tensor = self.time_embedder(
            timestep * self.config.timestep_scale  # 0~1 -> 0~1000 if needed
        )  # [B, hidden_dim]
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

        # text / class context embed
        context_embed = self.context_embedder(context)
        context_len = context_embed.shape[1]

        # image size embed
        imagesize_embed = self.get_imagesize_embed(
            original_size=original_size,
            target_size=target_size,
            crop_coords=crop_coords,
        )  # [B, 6, hidden_dim]
        num_imagesize_tokens = imagesize_embed.shape[1]

        # image patch embed
        patches = self.patch_embedder(image)  # [B, N, hidden_dim]
        patches_len = patches.shape[1]

        # context -> time -> imagesize -> patches
        context_position_ids = self.prepare_context_position_ids(
            seq_len=context_len,
            global_index=0,
        )
        time_position_ids = self.prepare_context_position_ids(
            seq_len=num_time_tokens,
            global_index=1,
        )
        imagesize_position_ids = self.prepare_context_position_ids(
            seq_len=num_imagesize_tokens,
            global_index=2,
        )
        patches_position_ids = self.prepare_image_position_ids(
            height=height,
            width=width,
            global_index=3,  # after context and time tokens
        )

        # actually: patches -> imagesize -> time -> context
        position_ids = torch.cat(
            [
                patches_position_ids,
                imagesize_position_ids,
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
            imagesize_mask = torch.ones(
                batch_size, num_imagesize_tokens, device=image.device
            )
            time_mask = torch.ones(batch_size, num_time_tokens, device=image.device)
            mask = torch.cat(
                [
                    patches_mask,
                    imagesize_mask,
                    time_mask,
                    context_mask.to(image.device),
                ],
                dim=1,
            )
        else:
            # attend all
            mask = torch.ones(
                batch_size,
                patches_len + num_imagesize_tokens + num_time_tokens + context_len,
                device=image.device,
            )

        # no context at initial blocks
        tokens = torch.cat(
            [
                patches,  # 16x16
                imagesize_embed,  # 6
                time_tokens,  # 4 | 8
            ],
            dim=1,  # cat in seq_len dimension
        )

        # if context fuse at initial blocks, add context now
        if self.config.do_context_fuse:
            tokens = torch.cat(
                [
                    tokens,
                    context_embed,
                ],
                dim=1,
            )

        # transformer blocks

        # in blocks
        skip_connections: list[torch.Tensor] = []
        for i, block in enumerate(self.in_blocks):  # type: ignore
            block: UJiTBlock

            tokens, full_tokens = self.forward_block(
                block,
                tokens,
                context_embed=context_embed,
                freqs_cis=freqs_cis,
                mask=mask,
            )
            skip_connections.append(full_tokens)

        # mid block
        tokens, _ = self.forward_block(
            self.mid_block,
            tokens,
            context_embed=context_embed,
            freqs_cis=freqs_cis,
            mask=mask,
        )

        # out blocks
        for i, block in enumerate(self.out_blocks):  # type: ignore
            block: UJiTBlock

            # get skip connection
            skip_tokens = skip_connections.pop()

            tokens, _ = self.forward_block(
                block,
                tokens,
                context_embed=context_embed,
                freqs_cis=freqs_cis,
                mask=mask,
                skip_tokens=skip_tokens,
            )

        patches = tokens[:, :patches_len, :]  # only keep patch tokens
        patches = self.final_layer(patches)

        pred_image = (
            self.pixel_shuffle(
                patches,
                height=height,
                width=width,
            )
            if self.config.use_pixel_shuffle
            else self.unpatchify(
                patches,
                height=height,
                width=width,
            )
        )

        return pred_image


class Denoiser(UJiT):
    def __init__(self, config: UJiTDenoiserConfig):
        nn.Module.__init__(self)
        UJiT.__init__(self, config)


class UJiTConfig(JiTConfig):
    denoiser: UJiTDenoiserConfig = UJiTDenoiserConfig()


class UJiTModel(JiTModel):
    denoiser: UJiT
    denoiser_class: type[JiT] = Denoiser

    @classmethod
    def new_with_config(
        cls,
        config: UJiTConfig,
    ) -> "UJiTModel":
        return super().new_with_config(config)  # type: ignore

    @classmethod
    def from_pretrained(
        cls,
        config: JiTConfig,
        checkpoint_path: str,
    ) -> "UJiTModel":
        return super().from_pretrained(  # type: ignore
            config,
            checkpoint_path,
        )
