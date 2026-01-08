import click
from contextlib import nullcontext
import random

import torch

from src.models.jit.extension.ig import IGJiTModel, IGJiTConfig
from src.trainer.common import Trainer
from src.config import TrainConfig
from src.dataset.square_class_image import SquareClassImageDatasetConfig
from src.dataset.preview.text_to_image import TextToImagePreviewConfig
from src.modules.timestep import TimestepSamplingType
from src.modules.loss.flow_match import (
    prepare_scaled_noised_latents,
)
from src.modules.timestep import sample_timestep


from class_to_image import JiTForClassToImageTraining


class IGJiTConfigForTraining(IGJiTConfig):
    checkpoint_path: str | None = None

    max_token_length: int = 64
    noise_scale: float = 1.0
    timestep_eps: float = 0.05

    loss_target: str = "velocity"  # "velocity", "image"
    timestep_sampling: TimestepSamplingType = "scale_shift_sigmoid"  # "uniform"

    train_class_encoder: bool = True

    drop_context_rate: float = 0.1  # for classifier-free guidance

    # IG
    intermediate_loss_weight: float = 0.5  # weight for internal guidance loss
    ig_scale: float = 0.5  # scale for internal guidance

    @property
    def is_from_scratch(self) -> bool:
        return self.checkpoint_path is None


class IGJiTForClassToImageTraining(JiTForClassToImageTraining):
    model: IGJiTModel
    model_class: type[IGJiTModel] = IGJiTModel

    model_config: IGJiTConfigForTraining
    model_config_class = IGJiTConfigForTraining

    def setup_model(self):
        super().setup_model()

    def train_step(self, batch: dict) -> torch.Tensor:
        images: torch.Tensor = batch["image"]
        caption = batch["caption"]
        dtype = self.model_config.torch_dtype

        drop_context = random.random() < self.model_config.drop_context_rate

        if drop_context:
            caption = [""] * len(caption)

        # 1. Prepare the inputs
        if text_encoder := self.model.text_encoder:
            with torch.no_grad():
                encoding = text_encoder.encode_prompts(
                    caption,
                    max_token_length=self.model_config.max_token_length,
                )
                context = encoding.positive_embeddings
                attention_mask = encoding.positive_attention_mask

        elif class_encoder := self.model.class_encoder:
            grad_block = (
                nullcontext if self.model_config.train_class_encoder else torch.no_grad
            )
            with grad_block():
                context, attention_mask = class_encoder.encode_prompts(
                    caption,
                    max_token_length=self.model_config.max_token_length,
                )
            if drop_context:
                attention_mask = torch.ones_like(attention_mask)
        else:
            raise ValueError("No encoder found in the model.")

        timesteps = sample_timestep(
            latents_shape=images.shape,
            device=self.accelerator.device,
            sampling_type=self.model_config.timestep_sampling,
        )

        # 2. Prepare the noised latents
        noisy_image, _random_noise = prepare_scaled_noised_latents(
            latents=images,
            timestep=timesteps,
            noise_scale=self.model_config.noise_scale,
        )

        image_size_info = torch.tensor(
            [[images.shape[2], images.shape[3]]], device=images.device
        ).repeat(images.shape[0], 1)

        # 3. Predict the noise
        model_pred, intermediate_pred = self.model.denoiser(
            image=noisy_image.to(dtype=dtype),
            timestep=timesteps.to(dtype=dtype),
            context=context.to(dtype=dtype),
            context_mask=attention_mask,
            original_size=image_size_info,
            target_size=image_size_info,
            crop_coords=torch.zeros_like(image_size_info),
        )

        # 4. Calculate the loss
        l2_loss = self.treat_loss(
            model_pred=model_pred,
            noisy_image=noisy_image,
            clean_image=(
                images
                + self.model_config.ig_scale * (model_pred - intermediate_pred).detach()
            ),
            random_noise=_random_noise,  # only for v-pred
            timesteps=timesteps,
        )
        self.log("train/l2_loss", l2_loss, on_step=True, on_epoch=True)

        # intermediate loss
        intermediate_l2_loss = self.treat_loss(
            model_pred=intermediate_pred,
            noisy_image=noisy_image,
            clean_image=images,
            random_noise=_random_noise,  # only for v-pred
            timesteps=timesteps,
        )
        self.log(
            "train/intermediate_l2_loss",
            intermediate_l2_loss,
            on_step=True,
            on_epoch=True,
        )

        total_loss = (
            l2_loss + self.model_config.intermediate_loss_weight * intermediate_l2_loss
        )

        self.log("train/loss", total_loss, on_step=True, on_epoch=True)

        return total_loss


@click.command()
@click.option("--config", type=str, required=True)
def main(config: str):
    _config = TrainConfig.from_config_file(config)

    trainer = Trainer(
        _config,
    )
    trainer.register_train_dataset_class(SquareClassImageDatasetConfig)
    trainer.register_preview_dataset_class(TextToImagePreviewConfig)
    trainer.register_model_class(IGJiTForClassToImageTraining)

    trainer.train()


if __name__ == "__main__":
    main()
