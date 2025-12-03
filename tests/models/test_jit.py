import torch
import math


from src.modules.timestep.embedding import get_timestep_embedding
from src.models.jit.denoiser import JiT
from src.models.jit.config import JiT_B_16_Config
from src.models.jit.text_encoder import TextEncoder
from src.models.jit.class_encoder import ClassEncoder


def test_timesteps():
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    inputs = torch.arange(0, 10000)

    for input in inputs:
        reference = timestep_embedding(
            t=input.unsqueeze(0),
            dim=256,
        )
        test = get_timestep_embedding(
            timesteps=input.unsqueeze(0),
            embedding_dim=256,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )

        assert torch.equal(reference, test)


def test_position_ids():
    patch_size = 16
    height = 256
    width = 256

    h_patches = height // patch_size
    w_patches = width // patch_size

    h_end_id = h_patches // 2
    h_start_id = (
        h_end_id - h_patches
    )  # even: -h_patches//2 ~ h_patches//2-1, odd: -(h_patches//2)~h_patches//2

    w_end_id = w_patches // 2
    w_start_id = (
        w_end_id - w_patches
    )  # even: -w_patches//2 ~ w_patches//2-1, odd: -(w_patches//2)~w_patches//2

    position_ids = torch.zeros(
        h_patches,
        w_patches,
        3,  # 0: text, 1: height, 2: width
    )

    # text(0) or image(1)
    position_ids[:, :, 0] = 1  # image

    # height
    position_ids[:, :, 1] = (
        torch.arange(
            h_start_id,
            h_end_id,
        )
        .unsqueeze(1)
        .repeat(1, w_patches)
    )  # [[h_start_id, * w_patches], [h_start_id+1, ...], ..., [h_end_id-1, ...]]

    # width
    position_ids[:, :, 2] = (
        torch.arange(
            w_start_id,
            w_end_id,
        )
        .unsqueeze(0)
        .repeat(h_patches, 1)
    )  # [[w_start_id, w_start_id+1, ..., w_end_id-1] * h_patches]

    print(position_ids)


@torch.no_grad()
def test_denoiser_forward():
    config = JiT_B_16_Config(
        context_embed_dim=768,
        hidden_size=768,
        num_heads=12,
        rope_axes_dims=[16, 24, 24],
        rope_axes_lens=[256, 128, 128],
        rope_zero_centered=[False, True, True],
    )
    model = JiT(config)

    batch_size = 2
    height = 256
    width = 256
    in_channels = 3

    image = torch.randn(
        batch_size,
        in_channels,
        height,
        width,
    )

    # uniform [0, 1)
    timestep = torch.rand(batch_size)

    context_len = 128
    context_dim = config.context_embed_dim

    context = torch.randn(
        batch_size,
        context_len,
        context_dim,
    )

    output = model(
        image=image,
        timestep=timestep,
        context=context,
    )

    assert output.shape == image.shape


@torch.no_grad()
def test_text_encoder_forward():
    model = TextEncoder.from_remote()

    prompts = [
        "A beautiful landscape painting of mountains during sunrise.",
        "A futuristic cityscape with flying cars and neon lights.",
    ]
    negative_prompts = [
        "low quality, blurry, distorted",
        "dark, gloomy, night",
    ]

    output = model.encode_prompts(
        prompts=prompts,
        negative_prompts=negative_prompts,
        use_negative_prompts=True,
        max_token_length=128,
    )

    assert output.positive_embeddings.shape[-1] == 2048
    assert output.negative_embeddings.shape[-1] == 2048


@torch.no_grad()
def test_class_encoder_forward():
    label2id = {
        "cat": 0,
        "dog": 1,
        "car": 2,
        "tree": 3,
    }

    model = ClassEncoder(
        label2id=label2id,
        embedding_dim=256,
    )

    class_prompts = [
        "cat dog",
        "car tree dog",
    ]

    embedding, attention_mask = model.encode_prompts(
        prompts=class_prompts,
        max_token_length=4,
    )

    assert embedding.shape == (2, 4, 256)
    assert attention_mask.shape == (2, 4)
