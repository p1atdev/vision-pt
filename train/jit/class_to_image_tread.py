import click

import torch
import torch.nn as nn

from src.models.jit import JiTConfig, JiTModel, JiT
from src.models.jit.denoiser import JiTBlock
from src.models.jit.config import DenoiserConfig
from src.trainer.common import Trainer
from src.config import TrainConfig
from src.dataset.square_class_image import SquareClassImageDatasetConfig
from src.dataset.preview.text_to_image import TextToImagePreviewConfig
from src.modules.timestep import TimestepSamplingType


from class_to_image import JiTForClassToImageTraining


class JiTWithTreadDenoiserConfig(DenoiserConfig):
    ### TREAD
    tread_route_rate: float = 0.5  # probability to route tokens
    tread_start_block: int = 2  # start block index to do routing
    tread_end_block: int = 8  # end block index to do routing


class JiTConfigForTreadTraining(JiTConfig):
    checkpoint_path: str | None = None

    max_token_length: int = 64
    noise_scale: float = 1.0
    timestep_eps: float = 0.05

    loss_target: str = "velocity"  # "velocity", "image"
    timestep_sampling: TimestepSamplingType = "scale_shift_sigmoid"  # "uniform"

    train_class_encoder: bool = True

    drop_context_rate: float = 0.1  # for classifier-free guidance

    lowres_loss: list[float] = []  # e.g., [0.5, 0.25] for 1/2 and 1/4 resolutions

    denoiser: JiTWithTreadDenoiserConfig = JiTWithTreadDenoiserConfig()

    @property
    def is_from_scratch(self) -> bool:
        return self.checkpoint_path is None


class JiTWithTread(JiT):
    config: JiTWithTreadDenoiserConfig

    def __init__(self, config: JiTWithTreadDenoiserConfig):
        super().__init__(config)

        if config.context_start_block != 0:
            raise ValueError("JiTWithTread does not support context_start_block != 0")

        assert self.config.tread_start_block < self.config.tread_end_block, (
            "tread_start_block must be less than tread_end_block"
        )
        assert self.config.tread_end_block <= self.config.depth, (
            "tread_end_block must be less than or equal to depth"
        )

        self.use_tread = self.config.tread_route_rate > 0

    def is_start_of_routing(self, block_idx: int) -> bool:
        return block_idx == self.config.tread_start_block

    def is_end_of_routing(self, block_idx: int) -> bool:
        return block_idx == self.config.tread_end_block

    def keep_and_route_tokens(
        self,
        patch_tokens: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        _batch_size, seq_len, _hidden_dim = patch_tokens.shape
        num_keep = int(seq_len * self.config.tread_route_rate)

        perm = torch.randperm(seq_len, device=patch_tokens.device)
        keep_indices = perm[:num_keep]
        route_indices = perm[num_keep:]

        inverse_perm = torch.argsort(perm)

        keep_tokens = patch_tokens[:, keep_indices, :]  # [B, num_keep, hidden_dim]
        route_tokens = patch_tokens[:, route_indices, :]  # [B, num_route, hidden_dim]

        keep_freqs_cis = freqs_cis[:, keep_indices, :]
        route_freqs_cis = freqs_cis[:, route_indices, :]

        keep_mask = mask[:, keep_indices]
        route_mask = mask[:, route_indices]

        return (
            keep_tokens,
            route_tokens,
            keep_freqs_cis,
            route_freqs_cis,
            keep_mask,
            route_mask,
            inverse_perm,
        )

    def compose_tokens(
        self,
        patch: torch.Tensor,
        info: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        tokens = torch.cat(
            [
                patch,
                info,
                context,
            ],
            dim=1,
        )

        return tokens

    def separate_tokens(
        self,
        tokens: torch.Tensor,
        patches_len: int,
        num_imagesize_tokens: int,
        num_time_tokens: int,
        context_len: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        patch_tokens = tokens[:, :patches_len, :]
        info_tokens = tokens[
            :,
            patches_len : patches_len + num_imagesize_tokens + num_time_tokens,
            :,
        ]
        context_tokens = tokens[:, -context_len:, :]

        return patch_tokens, info_tokens, context_tokens

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
            info_mask = torch.ones(
                batch_size, num_imagesize_tokens + num_time_tokens, device=image.device
            )
            context_mask = context_mask.to(image.device)
        else:
            patches_mask = torch.ones(batch_size, patches_len, device=image.device)
            info_mask = torch.ones(
                batch_size, num_imagesize_tokens + num_time_tokens, device=image.device
            )
            context_mask = torch.ones(batch_size, context_len, device=image.device)

        # tokens
        patch_tokens = patches
        info_tokens = torch.cat(
            [
                imagesize_embed,  # 6
                time_tokens,  # 4 | 8
            ],
            dim=1,  # cat in seq_len dimension
        )
        context_tokens = context_embed

        # freq cis
        patch_freqs_cis = freqs_cis[:, :patches_len, :]
        info_freqs_cis = freqs_cis[
            :, patches_len : patches_len + num_imagesize_tokens + num_time_tokens, :
        ]
        context_freqs_cis = freqs_cis[:, -context_len:, :]

        for i, block in enumerate(self.blocks):  # type: ignore
            block: JiTBlock

            # TREAD routing
            if self.training and self.use_tread and self.is_start_of_routing(i):
                (
                    keep_patch_tokens,
                    route_patch_tokens,
                    keep_patch_freqs_cis,
                    route_patch_freqs_cis,
                    keep_patches_mask,
                    route_patches_mask,
                    inverse_perm,
                ) = self.keep_and_route_tokens(
                    patch_tokens=patch_tokens,
                    freqs_cis=patch_freqs_cis,
                    mask=patches_mask,
                )
                patch_tokens = keep_patch_tokens
                patch_freqs_cis = keep_patch_freqs_cis
                patches_mask = keep_patches_mask

            elif self.training and self.use_tread and self.is_end_of_routing(i):
                # combine kept and routed tokens
                patch_tokens = torch.cat(
                    [
                        patch_tokens,
                        route_patch_tokens,
                    ],
                    dim=1,
                )[:, inverse_perm, :]
                patch_freqs_cis = torch.cat(
                    [
                        patch_freqs_cis,
                        route_patch_freqs_cis,
                    ],
                    dim=1,
                )[:, inverse_perm, :]
                patches_mask = torch.cat(
                    [
                        patches_mask,
                        route_patches_mask,
                    ],
                    dim=1,
                )[:, inverse_perm]

            else:
                pass

            # merge tokens
            tokens = self.compose_tokens(
                patch=patch_tokens,
                info=info_tokens,
                context=context_tokens,
            )
            freqs_cis = self.compose_tokens(
                patch=patch_freqs_cis,
                info=info_freqs_cis,
                context=context_freqs_cis,
            )
            mask = self.compose_tokens(
                patch=patches_mask,
                info=info_mask,
                context=context_mask,
            )

            full_tokens = self.forward_block(
                block,
                tokens,
                freqs_cis=freqs_cis,
                mask=mask,
            )

            # separate tokens
            (
                patch_tokens,
                info_tokens,
                _context_tokens,
            ) = self.separate_tokens(
                full_tokens,
                patches_len=patch_tokens.shape[1],  # variable length
                num_imagesize_tokens=num_imagesize_tokens,
                num_time_tokens=num_time_tokens,
                context_len=context_len,
            )
            if self.config.do_context_fuse:
                context_tokens = _context_tokens
            else:
                # reset context tokens
                context_tokens = context_embed

        patches = patch_tokens  # only keep patch tokens
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


class Denoiser(JiTWithTread):
    def __init__(self, config: JiTWithTreadDenoiserConfig):
        nn.Module.__init__(self)
        JiTWithTread.__init__(self, config)


class JiTWithTreadModel(JiTModel):
    denoiser: JiTWithTread
    denoiser_class: type[JiTWithTread] = Denoiser


class JiTForTreadTraining(JiTForClassToImageTraining):
    model: JiTWithTreadModel
    model_class: type[JiTWithTreadModel] = JiTWithTreadModel

    model_config: JiTConfigForTreadTraining
    model_config_class = JiTConfigForTreadTraining


@click.command()
@click.option("--config", type=str, required=True)
def main(config: str):
    _config = TrainConfig.from_config_file(config)

    trainer = Trainer(
        _config,
    )
    trainer.register_train_dataset_class(SquareClassImageDatasetConfig)
    trainer.register_preview_dataset_class(TextToImagePreviewConfig)
    trainer.register_model_class(JiTForTreadTraining)

    trainer.train()


if __name__ == "__main__":
    main()
