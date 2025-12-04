import warnings
from typing import NamedTuple

import torch
import torch.nn as nn

from ..utils import PromptType


class ClassTokenizerOutput(NamedTuple):
    class_ids: torch.Tensor
    attention_mask: torch.Tensor


class ClassTokenizer:
    def __init__(
        self,
        label2id: dict[str, int],
        splitter: str = " ",
    ) -> None:
        self.label2id = label2id
        self.id2label = {v: k for k, v in label2id.items()}
        self.splitter = splitter

        self.pad_token_id = len(label2id)

        assert all([id < len(label2id) for id in label2id.values()]), (
            "All label IDs must be less than the number of classes."
        )

    def normalize_prompts(
        self,
        class_names: PromptType,
    ) -> list[str]:
        _class_names: list[str] = (
            class_names if isinstance(class_names, list) else [class_names]
        )
        return _class_names

    def tokenize(
        self,
        prompts: PromptType,
        max_length: int = 32,
    ) -> ClassTokenizerOutput:
        # 1. Normalize class names
        _prompts = self.normalize_prompts(prompts)

        # 2. Convert to IDs
        class_ids = []
        masks = []
        for text in _prompts:
            ids = []

            for label in text.split(self.splitter):
                id = self.label2id.get(label.strip())
                if id is not None:  # 0 is OK
                    ids.append(id)
                    masks.append(1)
                else:
                    warnings.warn(f"Label '{label}' not found in label2id mapping.")
            class_ids.append(ids)

        # 3. Pad to max_length
        padded_class_ids = []
        padded_masks = []

        for _i, ids in enumerate(class_ids):
            if len(ids) < max_length:
                mask = [1] * len(ids) + [0] * (max_length - len(ids))
                ids = ids + [self.pad_token_id] * (max_length - len(ids))  # padding idx
            else:
                mask = [1] * max_length
                ids = ids[:max_length]

            padded_class_ids.append(ids)
            padded_masks.append(mask)

        return ClassTokenizerOutput(
            class_ids=torch.tensor(padded_class_ids, dtype=torch.long),
            attention_mask=torch.tensor(padded_masks, dtype=torch.long),
        )


class ClassEncoderOutput(NamedTuple):
    embeddings: torch.Tensor
    attention_mask: torch.Tensor


class ClassEncoder(nn.Module):
    def __init__(
        self,
        label2id: dict[str, int],
        embedding_dim: int,
    ):
        super().__init__()

        self.num_classes = len(label2id)

        self.pad_token_id = self.num_classes  # padding idx

        self.embedding = nn.Embedding(
            self.num_classes + 1,  # +1 for padding idx
            embedding_dim,
            padding_idx=self.num_classes,
        )

        self.tokenizer = ClassTokenizer(label2id)

    def initialize_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def encode_prompts(
        self,
        prompts: PromptType,
        max_token_length: int = 32,
    ):
        # 1. Tokenize prompts
        class_ids, attention_mask = self.tokenizer.tokenize(
            prompts,
            max_length=max_token_length,
        )

        # 3. Get embeddings
        embeddings = self.embedding(class_ids.to(self.embedding.weight.device))

        return ClassEncoderOutput(
            embeddings=embeddings,
            attention_mask=attention_mask,
        )
