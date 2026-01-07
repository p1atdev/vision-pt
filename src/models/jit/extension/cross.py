# ref: https://arxiv.org/pdf/2209.12152
# https://github.com/baofff/U-ViT

from typing import Literal

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint


from ....modules.norm import NormType, get_norm_layer
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
    apply_rope,
    PopeAttention,
)
from ..pipeline import JiTModel
from ..config import DenoiserConfig, JiTConfig, PositionalEncoding
from .pope import PopeEmbedder, apply_pope

NormPosition = Literal["pre", "post", "sandwich"]


class CrossAttention(Attention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor,
        query_rope_freqs: torch.Tensor,
        key_rope_freqs: torch.Tensor,
        query_mask: torch.Tensor | None = None,  # 1: attend, 0: ignore
        key_mask: torch.Tensor | None = None,  # 1: attend, 0: ignore
    ) -> torch.Tensor:
        batch_size, query_len, _dim = hidden_states.shape
        _kv_batch, kv_len, _kv_dim = key_value_states.shape

        # QKV
        q = self.to_q(hidden_states)
        k = self.to_k(key_value_states)
        v = self.to_v(key_value_states)

        q = self._pre_attn_reshape(q)  # [B, num_heads, N, head_dim]
        k = self._pre_attn_reshape(k)
        v = self._pre_attn_reshape(v)

        # QKNorm
        q = self.q_norm(q)
        k = self.k_norm(k)

        q = apply_rope(q, query_rope_freqs)
        k = apply_rope(k, key_rope_freqs)

        mask = None
        if query_mask is not None and key_mask is not None:
            # mask: (batch_size, query_len) x (batch_size, key_len) -> (batch_size, num_heads, query_len, key_len)
            mask = (
                query_mask.bool()
                .view(batch_size, 1, query_len, 1)
                .expand(-1, self.num_heads, -1, kv_len)
            ) & (
                key_mask.bool()
                .view(batch_size, 1, 1, kv_len)
                .expand(-1, self.num_heads, query_len, -1)
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


class PopeCrossAttention(PopeAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor,
        query_rope_freqs: torch.Tensor,
        key_rope_freqs: torch.Tensor,
        query_mask: torch.Tensor | None = None,  # 1: attend, 0: ignore
        key_mask: torch.Tensor | None = None,  # 1: attend, 0: ignore
    ) -> torch.Tensor:
        batch_size, query_len, _dim = hidden_states.shape
        _kv_batch, kv_len, _kv_dim = key_value_states.shape

        # QKV
        q = self.to_q(hidden_states)
        k = self.to_k(key_value_states)
        v = self.to_v(key_value_states)

        q = self._pre_attn_reshape(q)  # [B, num_heads, N, head_dim]
        k = self._pre_attn_reshape(k)
        v = self._pre_attn_reshape(v)

        # QKNorm
        q = self.q_norm(q)
        k = self.k_norm(k)

        q = apply_pope(q, query_rope_freqs)
        k = apply_pope(
            k,
            key_rope_freqs,
            learned_bias=self.pope_bias.clamp(-torch.pi, torch.pi),
        )  # apply learned bias to K only

        mask = None
        if query_mask is not None and key_mask is not None:
            # mask: (batch_size, query_len) x (batch_size, key_len) -> (batch_size, num_heads, query_len, key_len)
            mask = (
                query_mask.bool()
                .view(batch_size, 1, query_len, 1)
                .expand(-1, self.num_heads, -1, kv_len)
            ) & (
                key_mask.bool()
                .view(batch_size, 1, 1, kv_len)
                .expand(-1, self.num_heads, query_len, -1)
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
        eps: float = 1e-6,
        positional_encoding: PositionalEncoding = "rope",
        norm_type: NormType = "rms",
        norm_position: NormPosition = "sandwich",
    ):
        super().__init__()

        self.has_pre_norm = norm_position == "pre" or norm_position == "sandwich"
        self.has_post_norm = norm_position == "post" or norm_position == "sandwich"

        self.norm_attn_pre = (
            get_norm_layer(norm_type, hidden_dim, eps=eps)
            if self.has_pre_norm
            else nn.Identity()
        )
        self.norm_attn_post = (
            get_norm_layer(norm_type, hidden_dim, eps=eps)
            if self.has_post_norm
            else nn.Identity()
        )
        self.attn = (
            PopeAttention(
                dim=hidden_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                attn_dropout=attn_dropout,
                proj_dropout=proj_dropout,
                eps=eps,
                norm_type=norm_type,
            )
            if positional_encoding == "pope"
            else Attention(
                dim=hidden_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                attn_dropout=attn_dropout,
                proj_dropout=proj_dropout,
                eps=eps,
                norm_type=norm_type,
            )
        )

        self.norm_mlp_pre = (
            get_norm_layer(norm_type, hidden_dim, eps=eps)
            if self.has_pre_norm
            else nn.Identity()
        )
        self.norm_mlp_post = (
            get_norm_layer(norm_type, hidden_dim, eps=eps)
            if self.has_post_norm
            else nn.Identity()
        )
        self.mlp = SwiGLU(
            dim=hidden_dim,
            hidden_dim=int(hidden_dim * mlp_ratio),
            dropout=ffn_dropout,
            bias=bias,
        )

    def forward(
        self,
        image_hidden_states: torch.Tensor,
        context_hidden_states: torch.Tensor,
        image_rope_freqs: torch.Tensor,
        context_rope_freqs: torch.Tensor,
        image_mask: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ):
        image_len = image_hidden_states.shape[1]
        hidden_states = torch.cat(
            [
                image_hidden_states,
                context_hidden_states,
            ],
            dim=1,
        )
        rope_freqs = torch.cat(
            [
                image_rope_freqs,
                context_rope_freqs,
            ],
            dim=1,
        )
        if image_mask is not None and context_mask is not None:
            mask = torch.cat(
                [
                    image_mask,
                    context_mask,
                ],
                dim=1,
            )
        else:
            mask = None

        # attn
        hidden_states = hidden_states + self.norm_attn_post(
            self.attn(
                self.norm_attn_pre(hidden_states),
                rope_freqs,
                mask=mask,
            )
        )
        # mlp
        hidden_states = hidden_states + self.norm_mlp_post(
            self.mlp(self.norm_mlp_pre(hidden_states))
        )

        image_states = hidden_states[:, :image_len, :]
        context_states = hidden_states[:, image_len:, :]

        return image_states, context_states


class CrossJiTBlock(nn.Module):
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
        eps: float = 1e-6,
        positional_encoding: PositionalEncoding = "rope",
        norm_type: NormType = "rms",
        norm_position: NormPosition = "sandwich",
    ):
        super().__init__()

        self.has_pre_norm = norm_position == "pre" or norm_position == "sandwich"
        self.has_post_norm = norm_position == "post" or norm_position == "sandwich"

        self.norm_attn_image_pre = (
            get_norm_layer(norm_type, hidden_dim, eps=eps)
            if self.has_pre_norm
            else nn.Identity()
        )
        self.norm_attn_post = (
            get_norm_layer(norm_type, hidden_dim, eps=eps)
            if self.has_post_norm
            else nn.Identity()
        )
        self.norm_attn_context_pre = (
            get_norm_layer(norm_type, hidden_dim, eps=eps)
            if self.has_pre_norm
            else nn.Identity()
        )

        self.attn = (
            PopeCrossAttention(
                dim=hidden_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                attn_dropout=attn_dropout,
                proj_dropout=proj_dropout,
                norm_type="rms",
            )
            if positional_encoding == "pope"
            else CrossAttention(
                dim=hidden_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                attn_dropout=attn_dropout,
                proj_dropout=proj_dropout,
                norm_type="rms",
            )
        )

        self.norm_mlp_pre = (
            get_norm_layer(norm_type, hidden_dim, eps=eps)
            if self.has_pre_norm
            else nn.Identity()
        )
        self.norm_mlp_post = (
            get_norm_layer(norm_type, hidden_dim, eps=eps)
            if self.has_post_norm
            else nn.Identity()
        )

        self.mlp = SwiGLU(
            dim=hidden_dim,
            hidden_dim=int(hidden_dim * mlp_ratio),
            dropout=ffn_dropout,
            bias=bias,
        )

    def forward(
        self,
        image_hidden_states: torch.Tensor,
        context_hidden_states: torch.Tensor,
        image_rope_freqs: torch.Tensor,
        context_rope_freqs: torch.Tensor,
        image_mask: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ):
        # attn
        hidden_states = image_hidden_states + self.norm_attn_post(
            self.attn(
                self.norm_attn_image_pre(image_hidden_states),
                self.norm_attn_context_pre(context_hidden_states),
                image_rope_freqs,
                context_rope_freqs,
                query_mask=image_mask,
                key_mask=context_mask,
            )
        )

        # mlp
        hidden_states = hidden_states + self.norm_mlp_post(
            self.mlp(self.norm_mlp_pre(hidden_states))
        )

        return hidden_states, context_hidden_states


class CrossJiTDenoiserConfig(DenoiserConfig):
    norm_position: NormPosition = "sandwich"


class CrossJiT(JiT):
    def __init__(self, config: CrossJiTDenoiserConfig):
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

        # Cross-JiT blocks
        depth = config.depth
        self.blocks = nn.ModuleList(
            [
                CrossJiTBlock(
                    hidden_dim=config.hidden_size,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    attn_dropout=config.attn_dropout,
                    proj_dropout=config.proj_dropout,
                    ffn_dropout=0.0,
                    qkv_bias=True,
                    qk_norm=True,
                    bias=True,
                    eps=1e-6,
                    positional_encoding=config.positional_encoding,
                    norm_type=config.norm_type,
                    norm_position=config.norm_position,
                )
                if i == depth // 2
                else JiTBlock(
                    hidden_dim=config.hidden_size,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    attn_dropout=config.attn_dropout,
                    proj_dropout=config.proj_dropout,
                    ffn_dropout=0.0,
                    qkv_bias=True,
                    qk_norm=True,
                    bias=True,
                    eps=1e-6,
                    positional_encoding=config.positional_encoding,
                    norm_type=config.norm_type,
                )
                for i in range(config.depth)
            ]
        )

        self.final_layer = (
            BottleneckFinalLayer(
                hidden_dim=config.hidden_size,
                bottleneck_dim=config.bottleneck_dim,
                patch_size=config.patch_size,
                out_channels=config.in_channels,
                # Final norm should be RMSNorm for better performance
                norm_type="rms",
            )
            if self.config.use_output_bottleneck
            else FinalLayer(
                hidden_dim=config.hidden_size,
                mlp_ratio=config.mlp_ratio,
                patch_size=config.patch_size,
                out_channels=config.in_channels,
                # Final norm should be RMSNorm for better performance
                norm_type="rms",
            )
        )

        self.gradient_checkpointing = False

    def initialize_weights(self):
        super().initialize_weights()

    def forward_block(
        self,
        block: JiTBlock | CrossJiTBlock,
        image_states: torch.Tensor,
        context_states: torch.Tensor,
        image_rope_freqs: torch.Tensor,
        context_rope_freqs: torch.Tensor,
        image_mask: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.gradient_checkpointing and self.training:
            image_tokens, _context_tokens = checkpoint.checkpoint(  # type: ignore
                block,
                image_states,
                context_states,
                image_rope_freqs,
                context_rope_freqs,
                image_mask,
                context_mask,
            )
        else:
            image_tokens, _context_tokens = block(
                image_states,
                context_states,
                image_rope_freqs=image_rope_freqs,
                context_rope_freqs=context_rope_freqs,
                image_mask=image_mask,
                context_mask=context_mask,
            )

        if self.config.do_context_fuse:
            context_tokens = _context_tokens
        else:
            context_tokens = context_states  # input as is

        return image_tokens, context_tokens

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
        image_freq_cis = freqs_cis[
            :, : patches_len + num_imagesize_tokens + num_time_tokens, :
        ]  # [B, image_seq_len + info, head_dim]
        context_freq_cis = freqs_cis[
            :, -context_len:, :
        ]  # [B, context_seq_len, head_dim]

        # prepare tokens
        image_tokens = torch.cat(
            [
                patches,  # 16x16
                imagesize_embed,  # 6
                time_tokens,  # 4 | 8
            ],
            dim=1,  # cat in seq_len dimension
        )
        context_tokens = context_embed

        # attention mask
        if context_mask is not None:
            patches_mask = torch.ones(batch_size, patches_len, device=image.device)
            imagesize_mask = torch.ones(
                batch_size, num_imagesize_tokens, device=image.device
            )
            time_mask = torch.ones(batch_size, num_time_tokens, device=image.device)
            image_mask = torch.cat(
                [
                    patches_mask,
                    imagesize_mask,
                    time_mask,
                ],
                dim=1,
            )
            context_mask = context_mask.to(device=image.device)
        else:
            # attend all
            image_mask = torch.ones(
                batch_size,
                patches_len + num_imagesize_tokens + num_time_tokens,
                device=image.device,
            )
            context_mask = torch.ones(
                batch_size,
                context_len,
                device=image.device,
            )

        # transformer blocks
        for i, block in enumerate(self.blocks):  # type: ignore
            block: JiTBlock | CrossJiTBlock

            image_tokens, context_tokens = self.forward_block(
                block,
                image_states=image_tokens,
                context_states=context_tokens,
                image_rope_freqs=image_freq_cis,
                context_rope_freqs=context_freq_cis,
                image_mask=image_mask,
                context_mask=context_mask,
            )

        patches = image_tokens[
            :, :patches_len, :
        ]  # only keep patch tokens. remove info tokens
        patches = self.final_layer(patches)

        pred_image = self.unpatchify(
            patches,
            height=height,
            width=width,
        )

        return pred_image


class Denoiser(CrossJiT):
    def __init__(self, config: CrossJiTDenoiserConfig):
        nn.Module.__init__(self)
        CrossJiT.__init__(self, config)


class CrossJiTConfig(JiTConfig):
    denoiser: CrossJiTDenoiserConfig = CrossJiTDenoiserConfig()


class CrossJiTModel(JiTModel):
    denoiser: CrossJiT
    denoiser_class: type[CrossJiT] = Denoiser

    @classmethod
    def new_with_config(
        cls,
        config: CrossJiTConfig,
    ) -> "CrossJiTModel":
        return super().new_with_config(config)  # type: ignore

    @classmethod
    def from_pretrained(
        cls,
        config: CrossJiTConfig,
        checkpoint_path: str,
    ) -> "CrossJiTModel":
        return super().from_pretrained(  # type: ignore
            config,
            checkpoint_path,
        )
