from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn

from accelerate import init_empty_weights
from safetensors.torch import load_file

from .denoiser import JiT, Denoiser
from .text_encoder import TextEncoder
from .class_encoder import ClassEncoder
from .config import JiTConfig, ClassContextConfig

from ...modules.quant import replace_by_prequantized_weights
from ...utils import tensor as tensor_utils


class JiTModel(nn.Module):
    denoiser: JiT
    denoiser_class: type[JiT] = Denoiser

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
                embedding_dim=config.denoiser.context_dim,
                splitter=config.context_encoder.splitter,
                do_mask_padding=config.context_encoder.do_mask_padding,
            )
            self.text_encoder = None  # type: ignore
        else:
            self.text_encoder = TextEncoder.from_default()
            self.class_encoder = None  # type: ignore

        self.progress_bar = tqdm

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

    def prepare_noisy_image(
        self,
        batch_size: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        seed: int | None = None,
    ):
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
            noise = torch.randn(
                (batch_size, 3, height, width),
                dtype=dtype,
                device=device,
                generator=generator,
            )
        else:
            noise = torch.randn(
                (batch_size, 3, height, width),
                dtype=dtype,
                device=device,
            )

        return noise

    def prepare_timesteps(
        self,
        num_inference_steps: int,
        device: torch.device,
    ):
        timesteps = torch.linspace(
            0.0,
            1.0,
            num_inference_steps + 1,
            device=device,
        )

        return timesteps

    def prepare_context_embeddings(
        self,
        prompts: str | list[str],
        negative_prompt: str | list[str],
        max_token_length: int = 64,
        do_cfg: bool = False,
    ):
        if self.text_encoder is not None:
            encoder_output = self.text_encoder.encode_prompts(
                prompts,
                negative_prompts=negative_prompt,
                use_negative_prompts=do_cfg,
                max_token_length=max_token_length,
            )
            if do_cfg:
                prompt_embeddings = torch.cat(
                    [
                        encoder_output.positive_embeddings,
                        encoder_output.negative_embeddings,
                    ]
                )
                attention_mask = torch.cat(
                    [
                        encoder_output.positive_attention_mask,
                        encoder_output.negative_attention_mask,
                    ]
                )
            else:
                prompt_embeddings = encoder_output.positive_embeddings
                attention_mask = encoder_output.positive_attention_mask

        elif self.class_encoder is not None:
            embeddings, attention_mask = self.class_encoder.encode_prompts(
                prompts,
                max_token_length=max_token_length,
            )
            negative_embeddings, _ = self.class_encoder.encode_prompts(
                negative_prompt,
                max_token_length=max_token_length,
            )
            if do_cfg:
                prompt_embeddings = torch.cat(
                    [
                        embeddings,
                        negative_embeddings,
                    ],
                    dim=0,
                )
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        attention_mask,
                    ],
                    dim=0,
                )
            else:
                prompt_embeddings = embeddings

        return prompt_embeddings, attention_mask

    def prepare_image_size_inputs(
        self,
        width: int,
        height: int,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        original_size = torch.tensor(
            [[height, width]],
            dtype=dtype,
            device=device,
        ).repeat(batch_size, 1)
        target_size = torch.tensor(
            [[height, width]],
            dtype=dtype,
            device=device,
        ).repeat(batch_size, 1)
        crop_coords = torch.tensor(
            [[0, 0]],
            dtype=dtype,
            device=device,
        ).repeat(batch_size, 1)

        return original_size, target_size, crop_coords

    def to_pil_images(self, tensor: torch.Tensor) -> list[Image.Image]:
        return tensor_utils.tensor_to_images(tensor)

    def image_to_velocity(
        self,
        image: torch.Tensor,
        noisy: torch.Tensor,
        timestep: torch.Tensor,
        clamp_eps: float = 1e-5,
    ):
        return (image - noisy) / (1 - timestep.view(-1, 1, 1, 1)).clamp_min_(clamp_eps)

    def renorm_cfg(
        self,
        positive_velocity: torch.Tensor,
        cfg_velocity: torch.Tensor,
    ) -> torch.Tensor:
        positive_norm = torch.norm(positive_velocity, dim=-1, keepdim=True)
        cfg_norm = torch.norm(cfg_velocity, dim=-1, keepdim=True)

        new_cfg_velocity = cfg_velocity * (positive_norm / cfg_norm)

        return new_cfg_velocity

    def dynamic_thresholding(
        self,
        images: torch.Tensor,
        percentile: float = 0.995,
    ) -> torch.Tensor:
        """
        Apply dynamic thresholding to the images.
        Args:
            images (torch.Tensor): The input images tensor.
            percentile (float): The percentile value for thresholding.
        Returns:
            torch.Tensor: The thresholded images tensor.
        """
        batch_size = images.shape[0]
        flattened_images = images.view(batch_size, -1)
        abs_images = torch.abs(flattened_images)

        s = torch.quantile(abs_images, percentile, dim=1, keepdim=True)
        s = torch.clamp(s, min=1.0).view(batch_size, 1, 1, 1)

        thresholded_images = torch.clamp(images, -s, s) / s

        return thresholded_images

    def normalize_prompts(
        self,
        prompt: str | list[str],
    ) -> list[str]:
        return prompt if isinstance(prompt, list) else [prompt]

    def make_velocity_pred(
        self,
        model_pred: torch.Tensor,
        noisy_image: torch.Tensor,
        timestep: torch.Tensor,
    ):
        batch_size = noisy_image.shape[0]

        if self.config.model_pred == "image":
            velocity = self.image_to_velocity(
                image=model_pred[:batch_size],
                noisy=noisy_image,
                timestep=timestep.expand(batch_size),
            ).to(model_pred.dtype)
        elif self.config.model_pred == "velocity":
            velocity = model_pred[:batch_size]
        elif self.config.model_pred == "noise":
            raise NotImplementedError("Noise prediction is not implemented yet.")
        else:
            raise ValueError(f"Unknown model_pred: {self.config.model_pred}")

        return velocity

    def make_cfg_velocity_pred(
        self,
        model_pred: torch.Tensor,
        noisy_image: torch.Tensor,
        timestep: torch.Tensor,
        cfg_scale: float,
        do_cfg_renorm: bool = False,
        do_dynamic_thresholding: bool = False,
    ):
        batch_size = noisy_image.shape[0]

        if self.config.model_pred == "image":
            image_pred_positive, image_pred_negative = model_pred.chunk(2)
            v_pred_positive = self.image_to_velocity(
                image=image_pred_positive,
                noisy=noisy_image,
                timestep=timestep.expand(batch_size),
            ).to(model_pred.dtype)
            v_pred_negative = self.image_to_velocity(
                image=image_pred_negative,
                noisy=noisy_image,
                timestep=timestep.expand(batch_size),
            ).to(model_pred.dtype)
            velocity = v_pred_positive + cfg_scale * (v_pred_positive - v_pred_negative)
        elif self.config.model_pred == "velocity":
            v_pred_positive, v_pred_negative = model_pred.chunk(2)
            velocity = v_pred_positive + cfg_scale * (v_pred_positive - v_pred_negative)
        elif self.config.model_pred == "noise":
            raise NotImplementedError(
                "CFG with noise prediction is not implemented yet."
            )
        else:
            raise ValueError(f"Unknown model_pred: {self.config.model_pred}")

        if do_cfg_renorm:
            velocity = self.renorm_cfg(
                positive_velocity=v_pred_positive,
                cfg_velocity=velocity,
            )
        if do_dynamic_thresholding:
            # re-calculate the image prediction after cfg
            image_pred = noisy_image + velocity * (1 - timestep)
            image_pred = self.dynamic_thresholding(image_pred)
            velocity = self.image_to_velocity(
                image=image_pred,
                noisy=noisy_image,
                timestep=timestep.expand(batch_size),
            )

        return velocity

    @torch.inference_mode()
    def generate(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        width: int = 256,
        height: int = 256,
        num_inference_steps: int = 20,
        cfg_scale: float = 2.0,
        max_token_length: int = 64,
        seed: int | None = None,
        execution_dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = torch.device("cuda"),
        do_cfg_renorm: bool = False,
        do_dynamic_thresholding: bool = False,
        cfg_time_range: list[float] = [0.0, 1.0],
        # do_offloading: bool = False,
    ):
        # 1. Prepare args
        execution_device: torch.device = (
            torch.device(device) if isinstance(device, str) else device
        )
        do_cfg = cfg_scale > 1.0
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

                image_input = (
                    torch.cat([noisy_image] * 2)
                    if do_cfg and is_in_cfg_time
                    else noisy_image
                )
                _batch_size = image_input.shape[0]

                model_pred = self.denoiser(
                    image=image_input,
                    timestep=timestep.expand(_batch_size),
                    context=prompt_embeddings[:_batch_size],
                    context_mask=attention_mask[:_batch_size],
                    original_size=original_size[:_batch_size],
                    target_size=target_size[:_batch_size],
                    crop_coords=crop_coords[:_batch_size],
                )

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
