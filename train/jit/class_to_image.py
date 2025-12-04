from PIL.Image import Image
import click
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F


from src.models.jit import JiTConfig, JiTModel
from src.models.for_training import ModelForTraining
from src.trainer.common import Trainer
from src.config import TrainConfig
from src.dataset.square_class_image import SquareClassImageDatasetConfig
from src.dataset.preview.text_to_image import TextToImagePreviewConfig
from src.modules.loss.flow_match import (
    prepare_scaled_noised_latents,
)
from src.modules.timestep import sample_timestep, TimestepSamplingType
from src.modules.peft import get_adapter_parameters
from src.utils.logging import wandb_image


class JiTConfigForTraining(JiTConfig):
    checkpoint_path: str | None = None

    max_token_length: int = 64
    noise_scale: float = 1.0
    timestep_eps: float = 0.05

    loss_target: str = "velocity"  # "velocity", "image"
    timestep_sampling: TimestepSamplingType = "scale_shift_sigmoid"  # "uniform"

    train_class_encoder: bool = True

    @property
    def is_from_scratch(self) -> bool:
        return self.checkpoint_path is None


class JiTForClassToImageTraining(ModelForTraining, nn.Module):
    model: JiTModel

    model_config: JiTConfigForTraining
    model_config_class = JiTConfigForTraining

    def setup_model(self):
        if self.accelerator.is_main_process:
            if self.model_config.is_from_scratch:
                self.model = JiTModel.new_with_config(self.model_config)
                self.model.to(dtype=self.model_config.torch_dtype)
            elif checkpoint := self.model_config.checkpoint_path:
                self.model = JiTModel.from_pretrained(
                    self.model_config,
                    checkpoint,
                )
                self.model.to(dtype=self.model_config.torch_dtype)

    def sanity_check(self):
        batch_size = 2
        noise = self.model.prepare_noisy_image(
            batch_size=batch_size,
            height=256,
            width=256,
            dtype=self.model_config.torch_dtype,
            device=self.accelerator.device,
        )
        prompt = torch.randn(
            batch_size,
            self.model_config.max_token_length,  # max token len
            self.model.config.denoiser.context_dim,
            dtype=self.model_config.torch_dtype,
            device=self.accelerator.device,
        )
        timestep = torch.tensor(
            [0.5] * batch_size,
            dtype=self.model_config.torch_dtype,
            device=self.accelerator.device,
        )

        with self.accelerator.autocast(), torch.no_grad():
            _model_pred = self.model.denoiser(
                image=noise,
                context=prompt,
                timestep=timestep,
            )

    def treat_loss(
        self,
        image_pred: torch.Tensor,
        noisy_image: torch.Tensor,
        clean_image: torch.Tensor,
        timesteps: torch.Tensor,
    ):
        if self.model_config.loss_target == "velocity":
            target_v = self.model.image_to_velocity(
                image=clean_image,  # ground truth image
                noisy=noisy_image,
                timestep=timesteps,
                clamp_eps=self.model_config.timestep_eps,
            )
            pred_v = self.model.image_to_velocity(
                image=image_pred,
                noisy=noisy_image,
                timestep=timesteps,
                clamp_eps=self.model_config.timestep_eps,
            )
            return F.mse_loss(
                pred_v,
                target_v,
                reduction="mean",
            )

        elif self.model_config.loss_target == "image":
            return F.mse_loss(
                image_pred,
                clean_image,
                reduction="mean",
            )

        else:
            raise ValueError(f"Unknown loss target: {self.model_config.loss_target}")

    def train_step(self, batch: dict) -> torch.Tensor:
        images: torch.Tensor = batch["image"]
        caption = batch["caption"]
        dtype = self.model_config.torch_dtype

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

        # 3. Predict the noise
        model_pred = self.model.denoiser(
            image=noisy_image.to(dtype=dtype),
            timestep=timesteps.to(dtype=dtype),
            context=context.to(dtype=dtype),
            context_mask=attention_mask,
        )

        # 4. Calculate the loss
        l2_loss = self.treat_loss(
            image_pred=model_pred,
            noisy_image=noisy_image,
            clean_image=images,
            timesteps=timesteps,
        )

        # TODO: support LPIPS loss?

        total_loss = l2_loss

        self.log("train/loss", total_loss, on_step=True, on_epoch=True)

        return total_loss

    def eval_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    @torch.inference_mode()
    def preview_step(self, batch, preview_index: int) -> list[Image]:
        prompt: str = batch["prompt"]
        # negative_prompt: str | None = batch["negative_prompt"]
        height: int = batch["height"]
        width: int = batch["width"]
        cfg_scale: float = batch["cfg_scale"]
        num_steps: int = batch["num_steps"]
        seed: int = batch["seed"]

        with self.accelerator.autocast():
            image = self.model.generate(
                prompt=prompt,
                # negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_steps,
                cfg_scale=cfg_scale,
                max_token_length=self.model_config.max_token_length,
                seed=seed,
            )[0]

        self.log(
            f"preview/image_{preview_index}",
            wandb_image(image, caption=prompt),
            on_step=True,
            on_epoch=False,
        )

        return [image]

    def after_setup_model(self):
        if self.accelerator.is_main_process:
            if self.config.trainer.gradient_checkpointing:
                self.model.denoiser.set_gradient_checkpointing(True)

        super().after_setup_model()

    def get_state_dict_to_save(
        self,
    ) -> dict[str, torch.Tensor]:
        if not self._is_peft:
            return self.model.state_dict()

        state_dict = get_adapter_parameters(self.model)

        return state_dict

    def before_setup_model(self):
        pass

    def before_eval_step(self):
        pass

    def before_backward(self):
        pass


@click.command()
@click.option("--config", type=str, required=True)
def main(config: str):
    _config = TrainConfig.from_config_file(config)

    trainer = Trainer(
        _config,
    )
    trainer.register_train_dataset_class(SquareClassImageDatasetConfig)
    trainer.register_preview_dataset_class(TextToImagePreviewConfig)
    trainer.register_model_class(JiTForClassToImageTraining)

    trainer.train()


if __name__ == "__main__":
    main()
