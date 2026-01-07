# internal guidance
# https://arxiv.org/abs/2512.24176


import torch
import torch.nn as nn

from ..denoiser import (
    FinalLayer,
    BottleneckFinalLayer,
    JiT,
    JiTBlock,
)
from ..pipeline import JiTModel
from ..config import DenoiserConfig, JiTConfig


class IGJiTDenoiserConfig(DenoiserConfig):
    intermediate_output_idx: int = 4


class IGJiT(JiT):
    config: IGJiTDenoiserConfig

    def __init__(self, config: IGJiTDenoiserConfig):
        super().__init__(config)

        # internal guidance
        self.intermediate_final_layer = (
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
                eps=1e-6,
                # Final norm should be RMSNorm for better performance
                norm_type="rms",
            )
        )

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

        # prepare RoPE
        freqs_cis = (
            torch.cat(
                [
                    # actually: patches -> imagesize -> time -> context
                    # NOTE: DO NOT embed after concat, embed before concat!
                    # because we don't know the max and min position ids for each part after concat when use Normalized-Pope
                    self.rope_embedder(position_ids=patches_position_ids),
                    self.rope_embedder(position_ids=imagesize_position_ids),
                    self.rope_embedder(position_ids=time_position_ids),
                    self.rope_embedder(position_ids=context_position_ids),
                ],
                dim=1,  # cat in seq_len dimension
            )
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

        for i, block in enumerate(self.blocks):  # type: ignore
            block: JiTBlock

            # fuse context and i == context_start_block,
            # or not fuse context and i >= context_start_block
            if i == self.config.context_start_block or (
                not self.config.do_context_fuse and i >= self.config.context_start_block
            ):
                # add context tokens from this block
                tokens = torch.cat(
                    [
                        tokens,
                        context_embed,  # 32 | 64
                    ],
                    dim=1,  # cat in seq_len dimension
                )

            tokens = self.forward_block(
                block,
                tokens,
                freqs_cis=freqs_cis[:, : tokens.shape[1], :],
                mask=mask[:, : tokens.shape[1]],
            )

            if not self.config.do_context_fuse and i >= self.config.context_start_block:
                # remove context tokens after each block
                tokens = tokens[:, :-context_len, :]

            if i == self.config.intermediate_output_idx:
                # internal guidance
                intermediate_patches = tokens[
                    :, :patches_len, :
                ]  # only keep patch tokens
                intermediate_patches = self.intermediate_final_layer(
                    intermediate_patches
                )
                intermediate_pred_image = self.unpatchify(
                    intermediate_patches,
                    height=height,
                    width=width,
                )

        patches = tokens[:, :patches_len, :]  # only keep patch tokens
        patches = self.final_layer(patches)

        pred_image = self.unpatchify(
            patches,
            height=height,
            width=width,
        )

        return pred_image, intermediate_pred_image

    def initialize_weights(self):
        super().initialize_weights()


class Denoiser(IGJiT):
    def __init__(self, config: IGJiTDenoiserConfig):
        nn.Module.__init__(self)
        IGJiT.__init__(self, config)


class IGJiTConfig(JiTConfig):
    denoiser: IGJiTDenoiserConfig = IGJiTDenoiserConfig()


class IGJiTModel(JiTModel):
    denoiser: IGJiT
    denoiser_class: type[JiT] = Denoiser

    @classmethod
    def new_with_config(
        cls,
        config: IGJiTConfig,
    ) -> "IGJiTModel":
        return super().new_with_config(config)  # type: ignore

    @classmethod
    def from_pretrained(
        cls,
        config: IGJiTConfig,
        checkpoint_path: str,
    ) -> "IGJiTModel":
        return super().from_pretrained(  # type: ignore
            config,
            checkpoint_path,
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        width: int = 256,
        height: int = 256,
        num_inference_steps: int = 20,
        cfg_scale: float = 2.0,
        ig_scale: float = 1.0,
        max_token_length: int = 64,
        seed: int | None = None,
        execution_dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = torch.device("cuda"),
        do_cfg_renorm: bool = False,
        do_dynamic_thresholding: bool = False,
        cfg_time_range: list[float] = [0.0, 1.0],
        ig_time_range: list[float] = [0.0, 1.0],
        # do_offloading: bool = False,
    ):
        # 1. Prepare args
        execution_device: torch.device = (
            torch.device(device) if isinstance(device, str) else device
        )
        do_cfg = cfg_scale > 1.0
        do_ig = ig_scale > 1.0
        timesteps = self.prepare_timesteps(
            num_inference_steps=num_inference_steps,
            device=execution_device,
        )
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        # 3. prepare noise
        noisy_image = self.prepare_noisy_image(
            batch_size=batch_size,
            height=height,
            width=width,
            dtype=execution_dtype,
            device=execution_device,
            seed=seed,
        )

        negative_prompts = [""] if negative_prompt is None else negative_prompt
        negative_prompts = self.normalize_prompts(negative_prompts)
        if len(negative_prompts) != batch_size and len(negative_prompts) == 1:
            negative_prompts = negative_prompts * batch_size

        prompt_embeddings, attention_mask = self.prepare_context_embeddings(
            prompts=prompt,
            negative_prompt=negative_prompts,
            max_token_length=max_token_length,
            do_cfg=do_cfg,
        )
        original_size, target_size, crop_coords = self.prepare_image_size_inputs(
            width=width,
            height=height,
            batch_size=batch_size * 2 if do_cfg else batch_size,
            dtype=execution_dtype,
            device=execution_device,
        )

        # 4. Denoising loop
        with self.progress_bar(total=num_inference_steps) as pbar:
            for i, timestep in enumerate(timesteps[:-1]):
                is_in_cfg_time = (  # cfg interval check
                    cfg_time_range[0] <= float(timestep) <= cfg_time_range[1]
                )
                is_in_ig_time = (  # ig interval check
                    ig_time_range[0] <= float(timestep) <= ig_time_range[1]
                )

                image_input = (
                    torch.cat([noisy_image] * 2)
                    if do_cfg and is_in_cfg_time
                    else noisy_image
                )
                _batch_size = image_input.shape[0]

                # TODO: IG guidance
                model_pred, ig_pred = self.denoiser(
                    image=image_input,
                    timestep=timestep.expand(_batch_size),
                    context=prompt_embeddings[:_batch_size],
                    context_mask=attention_mask[:_batch_size],
                    original_size=original_size[:_batch_size],
                    target_size=target_size[:_batch_size],
                    crop_coords=crop_coords[:_batch_size],
                )

                if do_ig and is_in_ig_time:
                    model_pred = ig_pred + ig_scale * (model_pred - ig_pred)

                if do_cfg and is_in_cfg_time:
                    velocity = self.make_cfg_velocity_pred(
                        model_pred=model_pred,
                        noisy_image=noisy_image,
                        timestep=timestep,
                        cfg_scale=cfg_scale,
                        do_cfg_renorm=do_cfg_renorm,
                        do_dynamic_thresholding=do_dynamic_thresholding,
                    )

                else:
                    velocity = self.make_velocity_pred(
                        model_pred=model_pred,
                        noisy_image=noisy_image,
                        timestep=timestep,
                    )

                # new noisy image
                noisy_image = noisy_image + velocity * (timesteps[i + 1] - timestep)

                pbar.update()

        # now it should be clean
        clean_image = noisy_image

        # to PIL images
        pil_images = self.to_pil_images(clean_image.cpu())

        return pil_images
