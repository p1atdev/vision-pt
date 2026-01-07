import torch
import math


from src.modules.timestep.embedding import get_timestep_embedding
from src.models.jit.denoiser import JiT, Denoiser as JiTDenoiser
from src.models.jit.config import JiT_B_16_Config, JiTConfig, ClassContextConfig
from src.models.jit.text_encoder import TextEncoder
from src.models.jit.class_encoder import ClassEncoder
from src.models.jit.pipeline import JiTModel
from src.models.jit.extension.uvit import (
    UJiTModel,
    Denoiser as UJiTDenoiser,
    UJiTConfig,
    UJiTDenoiserConfig,
)
from src.models.jit.extension.pope import PopeEmbedder, NormalizedPopeEmbedder
from src.models.jit.extension.cross import (
    CrossJiTDenoiserConfig,
    CrossJiTModel,
    CrossJiTConfig,
    Denoiser as CrossJiTDenoiser,
)
from src.models.jit.extension.ig import (
    IGJiTModel,
    IGJiTDenoiserConfig,
    IGJiTConfig,
    Denoiser as IGJiTDenoiser,
)


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


def test_image_position_id_offset():
    cases = torch.arange(1, 20)

    for case in cases:
        num_patches = case.item()
        start_id = (num_patches // 2) - num_patches
        end_id = num_patches // 2

        assert end_id - start_id == num_patches, (
            f"num_patches: {num_patches}, start_id: {start_id}, end_id: {end_id}"
        )


def test_image_position_ids():
    patch_size = 16
    height = 256
    width = 256

    h_patches = height // patch_size
    w_patches = width // patch_size

    h_start_id = h_patches // 2 - h_patches
    h_end_id = h_patches // 2
    # even: -h_patches//2 ~ h_patches//2-1, odd: -(h_patches//2)~h_patches//2

    w_start_id = w_patches // 2 - w_patches
    w_end_id = w_patches // 2
    # even: -w_patches//2 ~ w_patches//2-1, odd: -(w_patches//2)~w_patches//2

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
        context_dim=768,
        hidden_size=768,
        num_heads=12,
        context_start_block=4,
        rope_axes_dims=[16, 24, 24],
        rope_axes_lens=[256, 128, 128],
        rope_zero_centered=[False, True, True],
    )
    model = JiTDenoiser(config)

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

    context_len = 64
    context_dim = config.context_dim

    context = torch.randn(
        batch_size,
        context_len,
        context_dim,
    )

    original_size = torch.tensor([[height, width]]).repeat(batch_size, 1)
    target_size = original_size.clone()
    crop_coords = torch.tensor([[0, 0]]).repeat(batch_size, 1)

    output = model(
        image=image,
        timestep=timestep,
        context=context,
        context_mask=None,
        original_size=original_size,
        target_size=target_size,
        crop_coords=crop_coords,
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
        "long_label_example": 4,
    }

    model = ClassEncoder(
        label2id=label2id,
        embedding_dim=256,
        splitter=" ",
        do_mask_padding=True,
    )

    class_prompts = [
        "cat dog",
        "car tree dog",
        "long_label_example cat",
    ]

    embedding, attention_mask = model.encode_prompts(
        prompts=class_prompts,
        max_token_length=4,
    )

    assert embedding.shape == (3, 4, 256)
    assert attention_mask.shape == (3, 4)

    print(model.tokenizer.label2id)


@torch.no_grad()
def test_new_jit_pipeline():
    config = JiTConfig(
        denoiser=JiT_B_16_Config(
            context_dim=768,
            hidden_size=768,
            num_heads=12,
            context_start_block=4,
            rope_axes_dims=[16, 24, 24],
            rope_axes_lens=[256, 128, 128],
            rope_zero_centered=[False, True, True],
        ),
        context_encoder=ClassContextConfig(
            label2id_map_path="models/jit-animeface-labels.json"
        ),
    )

    model = JiTModel.new_with_config(
        config=config,
    )

    assert isinstance(model.denoiser, JiTDenoiser)

    batch_size = 2
    height = 64
    width = 64
    in_channels = 3

    image = torch.randn(
        batch_size,
        in_channels,
        height,
        width,
    )

    # uniform [0, 1)
    timestep = torch.rand(batch_size)

    class_prompts = [
        "general 1girl solo looking_at_viewer",
        "sensitive 2girls multiple_girls",
    ]

    embedding, attention_mask = model.class_encoder.encode_prompts(
        prompts=class_prompts,
        max_token_length=32,
    )

    original_size = torch.tensor([[height, width]]).repeat(batch_size, 1)
    target_size = original_size.clone()
    crop_coords = torch.tensor([[0, 0]]).repeat(batch_size, 1)

    output = model.denoiser(
        image=image,
        timestep=timestep,
        context=embedding,
        context_mask=attention_mask,
        original_size=original_size,
        target_size=target_size,
        crop_coords=crop_coords,
    )

    assert output.shape == image.shape

    # generate image
    model.to(device="cpu", dtype=torch.float32)

    pil_images = model.generate(
        prompt="general 1girl solo looking_at_viewer",
        num_inference_steps=20,
        height=160,
        width=144,
        seed=42,
        cfg_scale=2.0,
        device=torch.device("cpu"),
        execution_dtype=torch.float32,
    )

    pil_images[0].save("output/test_jit_pipeline_output.webp")


@torch.no_grad()
def test_new_ujit_pipeline():
    config = UJiTConfig(
        denoiser=UJiTDenoiserConfig(
            context_dim=768,
            hidden_size=768,
            num_heads=12,
            context_start_block=4,
            rope_axes_dims=[16, 24, 24],
            rope_axes_lens=[256, 128, 128],
            rope_zero_centered=[False, True, True],
            do_context_fuse=False,
        ),
        context_encoder=ClassContextConfig(
            label2id_map_path="models/jit-animeface-labels.json"
        ),
    )

    model = UJiTModel.new_with_config(
        config=config,
    )

    assert isinstance(model.denoiser, UJiTDenoiser)

    batch_size = 2
    height = 64
    width = 64
    in_channels = 3

    image = torch.randn(
        batch_size,
        in_channels,
        height,
        width,
    )

    # uniform [0, 1)
    timestep = torch.rand(batch_size)

    class_prompts = [
        "general 1girl solo looking_at_viewer",
        "sensitive 2girls multiple_girls",
    ]

    embedding, attention_mask = model.class_encoder.encode_prompts(
        prompts=class_prompts,
        max_token_length=32,
    )

    original_size = torch.tensor([[height, width]]).repeat(batch_size, 1)
    target_size = original_size.clone()
    crop_coords = torch.tensor([[0, 0]]).repeat(batch_size, 1)

    output = model.denoiser(
        image=image,
        timestep=timestep,
        context=embedding,
        context_mask=attention_mask,
        original_size=original_size,
        target_size=target_size,
        crop_coords=crop_coords,
    )

    assert output.shape == image.shape

    # generate image
    model.to(device="cpu", dtype=torch.float32)

    pil_images = model.generate(
        prompt="general 1girl solo looking_at_viewer",
        num_inference_steps=20,
        height=160,
        width=144,
        seed=42,
        cfg_scale=2.0,
        device=torch.device("cpu"),
        execution_dtype=torch.float32,
    )

    pil_images[0].save("output/test_ujit_pipeline_output.webp")


@torch.no_grad()
def test_pope_embedder():
    embedder = PopeEmbedder(
        pope_theta=256.0,
        axes_dims=[64, 128, 128],
        axes_lens=[256, 128, 128],
        zero_centered=[False, True, True],
    )

    assert embedder.get_offset(0) == 0
    assert embedder.get_offset(1) == 64
    assert embedder.get_offset(2) == 64

    num_axes = len(embedder.axes_dims)
    assert num_axes == 3

    seq_len = 50
    num_patches = 36  # 6 * 6
    height = 6
    width = 6
    num_context = seq_len - num_patches

    context_position_ids = embedder.prepare_context_position_ids(
        seq_len=num_context,
        global_index=0,
    )
    image_position_ids = embedder.prepare_image_position_ids(
        height=height,
        width=width,
        patch_size=1,
        global_index=1,
    )

    position_ids = torch.cat(
        [
            context_position_ids,
            image_position_ids,
        ],
        dim=0,
    )

    freqs_cis = embedder(position_ids=position_ids)

    assert freqs_cis.shape == (1, seq_len, sum(embedder.axes_dims))


@torch.no_grad()
def test_normalized_pope_embedder():
    embedder = NormalizedPopeEmbedder(
        pope_theta=256.0,
        axes_dims=[64, 128, 128],
        axes_lens=[256, 128, 128],
        zero_centered=[False, True, True],
        normalize_by=64.0,
    )

    assert embedder.get_offset(0) == 0
    assert embedder.get_offset(1) == 64
    assert embedder.get_offset(2) == 64

    num_axes = len(embedder.axes_dims)
    assert num_axes == 3

    seq_len = 50
    num_patches = 36  # 6 * 6
    height = 6
    width = 6
    num_context = seq_len - num_patches

    context_position_ids = embedder.prepare_context_position_ids(
        seq_len=num_context,
        global_index=0,
    )

    image_position_ids = embedder.prepare_image_position_ids(
        height=height,
        width=width,
        patch_size=1,
        global_index=1,
    )
    # print("position_ids:", position_ids)

    # do not embed after concat, because we can't know max position id for each group
    freqs_cis = torch.cat(
        [
            embedder(position_ids=context_position_ids),
            embedder(position_ids=image_position_ids),
        ],
        dim=1,
    )

    assert freqs_cis.shape == (1, seq_len, sum(embedder.axes_dims))

    print(freqs_cis)


@torch.no_grad()
def test_cross_jit_pipeline():
    config = CrossJiTConfig(
        denoiser=CrossJiTDenoiserConfig(
            context_dim=768,
            hidden_size=768,
            num_heads=12,
            context_start_block=4,
            rope_axes_dims=[16, 24, 24],
            rope_axes_lens=[256, 128, 128],
            rope_zero_centered=[False, True, True],
        ),
        context_encoder=ClassContextConfig(
            label2id_map_path="models/jit-animeface-labels.json"
        ),
    )

    model = CrossJiTModel.new_with_config(
        config=config,
    )

    assert isinstance(model.denoiser, CrossJiTDenoiser)

    batch_size = 2
    height = 64
    width = 64
    in_channels = 3

    image = torch.randn(
        batch_size,
        in_channels,
        height,
        width,
    )

    # uniform [0, 1)
    timestep = torch.rand(batch_size)

    class_prompts = [
        "general 1girl solo looking_at_viewer",
        "sensitive 2girls multiple_girls",
    ]
    embedding, attention_mask = model.class_encoder.encode_prompts(
        prompts=class_prompts,
        max_token_length=32,
    )
    original_size = torch.tensor([[height, width]]).repeat(batch_size, 1)
    target_size = original_size.clone()
    crop_coords = torch.tensor([[0, 0]]).repeat(batch_size, 1)

    output = model.denoiser(
        image=image,
        timestep=timestep,
        context=embedding,
        context_mask=attention_mask,
        original_size=original_size,
        target_size=target_size,
        crop_coords=crop_coords,
    )

    assert output.shape == image.shape

    # generate image
    model.to(device="cpu", dtype=torch.float32)

    pil_images = model.generate(
        prompt="general 1girl solo looking_at_viewer",
        num_inference_steps=20,
        height=160,
        width=144,
        seed=42,
        cfg_scale=2.0,
        device=torch.device("cpu"),
        execution_dtype=torch.float32,
    )
    pil_images[0].save("output/test_cross_jit_pipeline_output.webp")


@torch.no_grad()
def test_ig_jit_pipeline():
    config = IGJiTConfig(
        denoiser=IGJiTDenoiserConfig(
            context_dim=768,
            hidden_size=768,
            num_heads=12,
            context_start_block=4,
            rope_axes_dims=[16, 24, 24],
            rope_axes_lens=[256, 128, 128],
            rope_zero_centered=[False, True, True],
        ),
        context_encoder=ClassContextConfig(
            label2id_map_path="models/jit-animeface-labels.json"
        ),
    )

    model = IGJiTModel.new_with_config(
        config=config,
    )

    assert isinstance(model.denoiser, IGJiTDenoiser)

    batch_size = 2
    height = 64
    width = 64
    in_channels = 3

    image = torch.randn(
        batch_size,
        in_channels,
        height,
        width,
    )

    # uniform [0, 1)
    timestep = torch.rand(batch_size)

    class_prompts = [
        "general 1girl solo looking_at_viewer",
        "sensitive 2girls multiple_girls",
    ]
    embedding, attention_mask = model.class_encoder.encode_prompts(
        prompts=class_prompts,
        max_token_length=32,
    )
    original_size = torch.tensor([[height, width]]).repeat(batch_size, 1)
    target_size = original_size.clone()
    crop_coords = torch.tensor([[0, 0]]).repeat(batch_size, 1)

    output, intermediate = model.denoiser(
        image=image,
        timestep=timestep,
        context=embedding,
        context_mask=attention_mask,
        original_size=original_size,
        target_size=target_size,
        crop_coords=crop_coords,
    )

    assert output.shape == image.shape
    assert intermediate.shape == image.shape

    # generate image
    model.to(device="cpu", dtype=torch.float32)

    pil_images = model.generate(
        prompt="general 1girl solo looking_at_viewer",
        num_inference_steps=20,
        height=160,
        width=144,
        seed=42,
        cfg_scale=2.0,
        device=torch.device("cpu"),
        execution_dtype=torch.float32,
    )

    pil_images[0].save("output/test_ig_jit_pipeline_output.webp")
