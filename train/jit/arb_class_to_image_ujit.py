import click

import torch


from src.models.jit.extension.uvit import UJiTModel, UJiTConfig
from src.trainer.common import Trainer
from src.config import TrainConfig
from src.dataset.text_to_image import TextToImageDatasetConfig
from src.dataset.preview.text_to_image import TextToImagePreviewConfig
from src.modules.timestep import TimestepSamplingType
from src.modules.timestep import sample_timestep
from src.modules.loss.perceptual import (
    PerceptualLossConfig,
    PerceptualLoss,
)

from arb_class_to_image import JiTForClassToImageTraining


class UJiTConfigForTraining(UJiTConfig):
    checkpoint_path: str | None = None

    max_token_length: int = 64
    noise_scale: float = 1.0
    timestep_eps: float = 0.05

    loss_target: str = "velocity"  # "velocity", "image"
    timestep_sampling: TimestepSamplingType = "scale_shift_sigmoid"  # "uniform"

    train_class_encoder: bool = True

    drop_context_rate: float = 0.1  # for classifier-free guidance

    lowres_loss: set[int] = set()  # e.g., [64, 96, 128] for 1/2 and 1/4 resolutions
    perceptual_losses: set[PerceptualLossConfig] = set()

    @property
    def is_from_scratch(self) -> bool:
        return self.checkpoint_path is None

    @property
    def has_additional_losses(self) -> bool:
        return (len(self.lowres_loss) + len(self.perceptual_losses)) > 0


class UJiTForClassToImageTraining(JiTForClassToImageTraining):
    model: UJiTModel
    model_class: type[UJiTModel] = UJiTModel

    model_config: UJiTConfigForTraining
    model_config_class = UJiTConfigForTraining

    def setup_model(self):
        super().setup_model()


@click.command()
@click.option("--config", type=str, required=True)
def main(config: str):
    _config = TrainConfig.from_config_file(config)

    trainer = Trainer(
        _config,
    )
    trainer.register_train_dataset_class(TextToImageDatasetConfig)
    trainer.register_preview_dataset_class(TextToImagePreviewConfig)
    trainer.register_model_class(UJiTForClassToImageTraining)

    trainer.train()


if __name__ == "__main__":
    main()
