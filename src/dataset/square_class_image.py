import json
import os
from tqdm import tqdm
from pathlib import Path

# import imagesize
from PIL import Image
from functools import reduce
from typing import Sequence, Iterator

import torch
import torchvision.transforms.v2 as v2

from .text_to_image import TextToImageBucket, TextToImageDatasetConfig, ImageCaptionPair


class SquareClassImagePair(ImageCaptionPair):
    @property
    def should_skip(self) -> bool:
        if m := self.metadata:
            return not m.exists()

        return True

    def read_caption(self) -> str:
        if m := self.metadata:
            with open(m, "r") as f:
                metadata = json.load(f)

            rating: str = metadata.get("rating", "general")
            character: list[str] = metadata.get("character_tags", {}).keys()
            general: list[str] = metadata.get("general_tags", {}).keys()

            caption = " ".join([rating, *character, *general])

            return caption
        else:
            raise ValueError("No metadata found for image.")


class SquareClassImageBucket(TextToImageBucket):
    def __init__(
        self,
        image_size: int,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )

        self.resize_transform = v2.Compose(
            [
                v2.PILToTensor(),
                v2.Resize(
                    size=None,
                    max_size=image_size,
                ),
                v2.CenterCrop(size=(image_size, image_size)),
                v2.ToDtype(torch.float16, scale=True),  # 0~255 -> 0~1
                v2.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),  # 0~1 -> -1 ~ 1
            ]
        )

    def __getitem__(self, idx: int | slice):
        # the __len__ is multiplied by num_repeats,
        # so the provided idx may be larger than the length of the dataset.
        # we need to get the real index by modulo operation.
        local_idx = self.to_local_idx(idx)
        batch: dict[str, Sequence | torch.Tensor] = self.items[local_idx]

        # transform image
        if "image" in batch:
            # this is a list of image paths
            image_paths: list[str] = batch["image"]  # type: ignore
            # _images = [io.decode_image(image_path) for image_path in image_paths]
            _pil_images = [Image.open(image_path) for image_path in image_paths]
            #  convert to tensor and apply transforms
            _images = [self.resize_transform(image) for image in _pil_images]

            images: list[torch.Tensor] = []
            for image in _images:
                resized_image = self.resize_transform(image)
                images.append(resized_image)

            batch["image"] = torch.stack(images)

        if "caption" in batch:
            captions: list[str] = batch["caption"]  # type: ignore
            assert isinstance(captions, list)
            # apply all caption processors
            captions = [
                reduce(
                    lambda c, processor: processor(c),
                    self.caption_processors,
                    caption,
                )  # type: ignore
                for caption in captions
            ]
            batch["caption"] = captions

        return batch

    def _generate_ds_from_pairs(self, pairs: list[ImageCaptionPair]) -> Iterator:
        for pair in pairs:
            image = str(pair.image)
            caption = pair.read_caption()

            yield {
                "image": image,
                "caption": caption,
                "width": pair.width,
                "height": pair.height,
            }


class SquareClassImageDatasetConfig(TextToImageDatasetConfig):
    tags_folder: str
    image_size: int = 256

    def _retrive_images(self):
        pairs: list[ImageCaptionPair] = []

        tags_folder_path = Path(self.tags_folder)

        for root, _, files in os.walk(self.folder):
            for file in tqdm(files):
                if any([file.endswith(ext) for ext in self.supported_extensions]):
                    image_path = Path(root) / file  # hogehoge.png
                    metadata_path = (tags_folder_path / (file)).with_suffix(
                        self.metadata_extension
                    )

                    width, height = self.image_size, self.image_size

                    if metadata_path is not None:
                        pair = SquareClassImagePair(
                            image=image_path,
                            width=width,
                            height=height,
                            caption=None,
                            metadata=metadata_path,
                        )
                        if pair.should_skip:
                            continue
                        pairs.append(pair)
                    else:
                        raise FileNotFoundError(
                            f"Metadata file {metadata_path} \
                            not found for image {image_path}"
                        )

        return pairs

    def generate_buckets(self) -> list[TextToImageBucket]:  # type: ignore
        pairs = self._retrive_images()
        bucket = TextToImageBucket(
            items=pairs,
            batch_size=self.batch_size,
            width=self.image_size,
            height=self.image_size,
            do_upscale=self.do_upscale,
            num_repeats=self.num_repeats,
            caption_processors=self.caption_processors,
        )

        return [bucket]
