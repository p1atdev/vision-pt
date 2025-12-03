from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn

from accelerate import init_empty_weights
from safetensors.torch import load_file

from .denoiser import JiT
from .text_encoder import TextEncoder
from .class_encoder import ClassEncoder
from .config import JiTConfig, ClassContextConfig, TextContextConfig

from ...modules.quant import replace_by_prequantized_weights
from ...modules.timestep.scheduler import get_linear_schedule
from ...modules.timestep.sampling import time_shift_linear
from ...utils import tensor as tensor_utils


class JiTModel(nn.Module):
    denoiser: JiT
    denoiser_class: type[JiT] = JiT

    text_encoder: TextEncoder
    class_encoder: ClassEncoder

    def __init__(
        self,
        config: JiTConfig,
    ):
        super().__init__()

        self.config = config

        self.denoiser = self.denoiser_class(config.denoiser)

        if isinstance(config.context_encoder, ClassContextConfig):
            self.class_encoder = ClassEncoder(
                label2id=config.context_encoder.label2id,
                embedding_dim=config.denoiser.context_embed_dim,
            )
            self.text_encoder = None  # type: ignore
        else:
            self.text_encoder = TextEncoder.from_default()
            self.class_encoder = None  # type: ignore

    def _load_checkpoint(
        self,
        checkpoint_path: str,
        strict: bool = True,
    ):
        state_dict = load_file(checkpoint_path)

        replace_by_prequantized_weights(self, state_dict)

        self.denoiser.load_state_dict(
            {
                key[len("denoiser.") :]: value
                for key, value in state_dict.items()
                if key.startswith("denoiser.")
            },
            strict=strict,
            assign=True,
        )
        if self.class_encoder is not None:
            self.class_encoder.load_state_dict(
                {
                    key[len("class_encoder.") :]: value
                    for key, value in state_dict.items()
                    if key.startswith("class_encoder.")
                },
                strict=strict,
                assign=True,
            )
        if self.text_encoder is not None:
            self.text_encoder.model.load_state_dict(
                {
                    key[len("text_encoder.") :]: value
                    for key, value in state_dict.items()
                    if key.startswith("text_encoder.")
                },
                strict=strict,
                assign=True,
            )

    @classmethod
    def from_pretrained(
        cls,
        config: JiTConfig,
        checkpoint_path: str,
    ) -> "JiTModel":
        with init_empty_weights():
            model = cls(config)

        model._load_checkpoint(checkpoint_path)

        return model

    @classmethod
    def new_with_config(
        cls,
        config: JiTConfig,
    ) -> "JiTModel":
        with init_empty_weights():
            model = cls(config)

        model.denoiser.to_empty(device="cpu")
        model.denoiser.initialize_weights()

        if isinstance(config.context_encoder, ClassContextConfig):
            model.class_encoder.to_empty(device="cpu")
            model.class_encoder.initialize_weights()
        else:
            model.text_encoder = TextEncoder.from_remote(
                repo_id=config.context_encoder.pretrained_model,
            )

        return model
