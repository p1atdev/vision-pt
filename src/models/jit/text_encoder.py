import torch
import torch.nn as nn

from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen3VLTextModel,
    Qwen3VLTextConfig,
)

from ..utils import PromptType, TextEncodingOutput

DEFAULT_TEXT_ENCPDER_CONFIG = {
    "architectures": ["Qwen3VLTextModel"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "dtype": "bfloat16",
    "eos_token_id": 151645,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": 6144,
    "max_position_embeddings": 262144,
    "model_type": "qwen3_vl_text",
    "num_attention_heads": 16,
    "num_hidden_layers": 28,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-06,
    "rope_scaling": {
        "mrope_interleaved": True,
        "mrope_section": [24, 20, 20],
        "rope_type": "default",
    },
    "rope_theta": 5000000,
    "tie_word_embeddings": True,
    "use_cache": True,
    "vocab_size": 151936,
}
DEFAULT_TEXT_ENCODER_CLASS = Qwen3VLTextModel
DEFAULT_TEXT_ENCODER_CONFIG_CLASS = Qwen3VLTextConfig
TEXT_ENCODER_PREFIX = "text_encoder."
DEFAULT_MAX_TOKEN_LENGTH = 128

DEFAULT_REPO = "p1atdev/Qwen3-VL-2B-Instruct-Text-Only"


class TextEncoder(nn.Module):
    tokenizer: PreTrainedTokenizerBase
    model: PreTrainedModel

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase):
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def from_default(cls):
        config = DEFAULT_TEXT_ENCODER_CONFIG_CLASS(**DEFAULT_TEXT_ENCPDER_CONFIG)
        model = DEFAULT_TEXT_ENCODER_CLASS(config)

        tokenizer = AutoTokenizer.from_pretrained(
            DEFAULT_REPO,
        )

        return cls(model, tokenizer)

    @classmethod
    def from_remote(cls, repo_id: str = DEFAULT_REPO):
        model = DEFAULT_TEXT_ENCODER_CLASS.from_pretrained(
            repo_id,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            repo_id,
        )

        return cls(model, tokenizer)

    def normalize_prompts(
        self,
        prompts: PromptType,
        negative_prompts: PromptType | None = None,
        use_negative_prompts: bool = True,
    ) -> tuple[list[str], list[str]]:
        _prompts: list[str] = prompts if isinstance(prompts, list) else [prompts]
        if use_negative_prompts:
            if negative_prompts is not None:
                _negative_prompts: list[str] = (
                    negative_prompts
                    if isinstance(negative_prompts, list)
                    else [negative_prompts]
                )
                if len(_negative_prompts) == 1 and len(_prompts) > 1:
                    _negative_prompts = _negative_prompts * len(_prompts)
            else:
                _negative_prompts = [""] * len(_prompts)
        else:
            _negative_prompts = []

        return _prompts, _negative_prompts

    def encode_prompts(
        self,
        prompts: PromptType,
        negative_prompts: PromptType | None = None,
        use_negative_prompts: bool = False,
        max_token_length: int = DEFAULT_MAX_TOKEN_LENGTH,
    ):
        # 1. Normalize prompts
        _prompts, _negative_prompts = self.normalize_prompts(
            prompts,
            negative_prompts,
            use_negative_prompts,
        )
        prompts_len = len(_prompts)

        # 2. Tokenize prompts
        text_inputs = self.tokenizer(
            _prompts + _negative_prompts,
            max_length=max_token_length,
            padding="longest",  # not use max length
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        # 3. Move input_ids to model device
        input_ids = text_inputs.input_ids.to(self.model.device)
        attention_mask = text_inputs.attention_mask.to(self.model.device)

        # 5. Encode prompts
        prompt_encodings = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        ).hidden_states[-2]  # penultimate hidden state

        # 6. Split prompts and negative prompts
        positive_embeddings = prompt_encodings[:prompts_len]
        negative_embeddings = prompt_encodings[prompts_len:]

        return TextEncodingOutput(
            positive_embeddings=positive_embeddings,
            positive_attention_mask=torch.ones_like(input_ids),
            negative_embeddings=negative_embeddings,
            negative_attention_mask=torch.ones_like(input_ids),
        )
