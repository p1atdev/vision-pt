import click

from src.models.jit.extension.cross import CrossJiTModel, CrossJiTConfig
from src.trainer.common import Trainer
from src.config import TrainConfig
from src.dataset.square_class_image import SquareClassImageDatasetConfig
from src.dataset.preview.text_to_image import TextToImagePreviewConfig
from src.modules.timestep import TimestepSamplingType


from class_to_image import JiTForClassToImageTraining


class CrossJiTConfigForTraining(CrossJiTConfig):
    checkpoint_path: str | None = None

    max_token_length: int = 64
    noise_scale: float = 1.0
    timestep_eps: float = 0.05

    loss_target: str = "velocity"  # "velocity", "image"
    timestep_sampling: TimestepSamplingType = "scale_shift_sigmoid"  # "uniform"

    train_class_encoder: bool = True

    drop_context_rate: float = 0.1  # for classifier-free guidance

    @property
    def is_from_scratch(self) -> bool:
        return self.checkpoint_path is None


class CrossJiTForClassToImageTraining(JiTForClassToImageTraining):
    model: CrossJiTModel
    model_class: type[CrossJiTModel] = CrossJiTModel

    model_config: CrossJiTConfigForTraining
    model_config_class = CrossJiTConfigForTraining


@click.command()
@click.option("--config", type=str, required=True)
def main(config: str):
    _config = TrainConfig.from_config_file(config)

    trainer = Trainer(
        _config,
    )
    trainer.register_train_dataset_class(SquareClassImageDatasetConfig)
    trainer.register_preview_dataset_class(TextToImagePreviewConfig)
    trainer.register_model_class(CrossJiTForClassToImageTraining)

    trainer.train()


if __name__ == "__main__":
    main()
