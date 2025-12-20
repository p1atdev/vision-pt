import os
import imagesize
import random
from pathlib import Path
from PIL import Image
from pydantic import BaseModel
import warnings
import json
from functools import reduce
from collections import defaultdict
from typing import Sequence, Iterator, NamedTuple
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import polars as pl

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms.v2 as v2
import torchvision.transforms.v2.functional as TF

from datasets import Dataset


from .transform import ObjectCoverResize
from .bucket import (
    BucketDataset,
)
from .aspect_ratio_bucket import (
    AspectRatioBucketConfig,
    AspectRatioBucketManager,
    AspectRatioBucket,
    print_arb_info,
)
from .caption import CaptionProcessorList
from .tags import format_general_character_tags, map_replace_underscore


class ImageCaptionPair(BaseModel):
    image: Path
    width: int
    height: int
    caption: Path | None
    metadata: Path | None = None

    def read_caption(self) -> str:
        if self.metadata is not None:
            with open(self.metadata, "r") as f:
                metadata = json.load(f)

            if "tag_string" in metadata:
                return format_general_character_tags(
                    general=map_replace_underscore(
                        metadata.get("tag_string_general", "").split(" ")
                    ),
                    character=map_replace_underscore(
                        metadata.get("tag_string_copyright", "").split(" ")
                        + metadata.get("tag_string_character", "").split(" ")
                    ),
                    rating=metadata.get("rating", "general"),
                    separator=", ",
                    group_separator="|||",
                )

            # wd-tagger-rs format
            if "tagger" in metadata:
                return format_general_character_tags(
                    general=metadata["tagger"].get("general", []),
                    character=metadata["tagger"].get("character", []),
                    rating=metadata.get("rating", "general"),
                    separator=", ",
                    group_separator="|||",
                )

            if "tags" in metadata:
                return metadata["tags"]

            if "caption" in metadata:
                return metadata["caption"]

            if "captions" in metadata:
                return random.choice(metadata["captions"])

            raise ValueError(
                f"Caption not found in metadata {self.metadata}. Available keys: {', '.join(metadata.keys())}"
            )

        assert self.caption is not None
        return self.caption.read_text()

    @property
    def should_skip(self) -> bool:
        if self.metadata is None:
            return False

        # if skip parameter is set and it is true, skip this image
        with open(self.metadata, "r") as f:
            metadata = json.load(f)
        if "skip" in metadata and metadata["skip"]:
            return True

        return False


class RandomCropOutput(NamedTuple):
    image: torch.Tensor

    top: int
    left: int
    crop_height: int
    crop_width: int

    original_height: int
    original_width: int


class TextToImageBucket(AspectRatioBucket):
    """
    Bucket for Text to Image dataset.
    Each image is classified into a bucket based on its aspect ratio.
    """

    items: Dataset
    caption_processors: CaptionProcessorList

    def __init__(
        self,
        items: list[ImageCaptionPair],
        batch_size: int,
        width: int,
        height: int,
        do_upscale: bool,
        num_repeats: int,
        caption_processors: CaptionProcessorList = [],
    ):
        ds = Dataset.from_generator(
            self._generate_ds_from_pairs,
            gen_kwargs={"pairs": items},
            cache_dir="cache",
        )

        super().__init__(
            # (Dataset is compatible with Sequence)
            items=ds,  # type: ignore
            batch_size=batch_size,
        )

        # random crop
        self.resize_transform = v2.Compose(
            [
                v2.PILToTensor(),  # PIL -> Tensor
                v2.ToDtype(torch.float16, scale=True),  # 0~255 -> 0~1
                v2.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),  # 0~1 -> -1~1
                ObjectCoverResize(
                    width,
                    height,
                    do_upscale=do_upscale,
                ),
            ]
        )
        self.width = width
        self.height = height
        self.do_upscale = do_upscale
        self.num_repeats = num_repeats
        self.caption_processors = caption_processors

    def random_crop(self, image: torch.Tensor) -> RandomCropOutput:
        top, left, crop_height, crop_width = v2.RandomCrop.get_params(
            image, (self.height, self.width)
        )
        cropped_img = TF.crop(image, top, left, crop_height, crop_width)
        return RandomCropOutput(
            image=cropped_img,
            top=top,
            left=left,
            crop_height=crop_height,
            crop_width=crop_width,
            original_height=image.shape[1],
            original_width=image.shape[2],
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

            images: list[torch.Tensor] = []
            original_size: list[torch.Tensor] = []
            target_size: list[torch.Tensor] = []
            crop_coords_top_left: list[torch.Tensor] = []

            # Process one image at a time to reduce memory usage
            for image_path in image_paths:
                pil_image = Image.open(image_path)
                transformed = self.resize_transform(pil_image)
                pil_image.close()  # Release PIL image immediately

                crop_image, top, left, crop_height, crop_width, height, width = (
                    self.random_crop(transformed)
                )
                images.append(crop_image)
                original_size.append(torch.tensor([height, width]))
                target_size.append(torch.tensor([crop_height, crop_width]))
                crop_coords_top_left.append(torch.tensor([top, left]))

            batch["image"] = torch.stack(images)
            batch["original_size"] = torch.stack(original_size)
            batch["target_size"] = torch.stack(target_size)
            batch["crop_coords_top_left"] = torch.stack(crop_coords_top_left)

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

    def _yield_pair(self, pair: ImageCaptionPair) -> dict:
        image = str(pair.image)
        caption = pair.read_caption()

        return {
            "image": image,
            "caption": caption,
            "width": pair.width,
            "height": pair.height,
        }

    def _generate_ds_from_pairs(self, pairs: list[ImageCaptionPair]) -> Iterator:
        with ThreadPoolExecutor() as executor:
            results = executor.map(
                self._yield_pair,
                pairs,
                chunksize=1000,
            )
            for result in results:
                yield result


class TextToImageDatasetConfig(AspectRatioBucketConfig):
    supported_extensions: list[str] = [".png", ".jpg", ".jpeg", ".webp", ".avif"]
    caption_extension: str = ".txt"
    metadata_extension: str = ".json"

    has_skip_metadata: bool = False

    folder: str

    do_upscale: bool = False
    num_repeats: int = 1

    # shuffle, setting prefix, dropping tags, etc.
    caption_processors: CaptionProcessorList = []

    # cache settings
    imagesize_cache_path: str | None = None

    def _get_imagesize_cache_path(self) -> Path | None:
        if self.imagesize_cache_path is None:
            return None

        return Path(self.imagesize_cache_path)

    def _has_imagesize_cache(self) -> bool:
        """Check if a valid imagesize cache exists."""
        cache_path = self._get_imagesize_cache_path()
        if cache_path is None:
            return False

        return cache_path.exists() and cache_path.stat().st_size > 0

    def _scan_imagesize_cache(self) -> pl.LazyFrame | None:
        if self.imagesize_cache_path is None:
            return None

        if not Path(self.imagesize_cache_path).exists():
            # not yet cached
            return None

        if self.imagesize_cache_path.endswith(".parquet"):
            lf = pl.scan_parquet(self.imagesize_cache_path)
            return lf

        if self.imagesize_cache_path.endswith(".jsonl"):
            lf = pl.scan_ndjson(self.imagesize_cache_path)
            return lf

        raise ValueError(
            f"Unsupported imagesize_cache_path format: {self.imagesize_cache_path}. Supported formats are .parquet, .jsonl"
        )

    def _save_imagesize_cache(self, pairs: list[ImageCaptionPair]) -> None:
        """Save bucket data to cache."""
        cache_path = self._get_imagesize_cache_path()
        if cache_path is None:
            return

        assert cache_path.suffix == ".jsonl", (
            "Only .jsonl format is supported for imagesize cache."
        )

        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)

        # Convert ImageCaptionPair to serializable format
        with open(cache_path, "wb") as f:
            for p in pairs:
                f.write(
                    json.dumps(
                        {
                            "image": str(p.image),
                            "width": p.width,
                            "height": p.height,
                            "caption": str(p.caption) if p.caption else None,
                            "metadata": str(p.metadata) if p.metadata else None,
                        },
                        ensure_ascii=False,
                        indent=None,
                    ).encode("utf-8")
                    + b"\n"
                )

        print(f"Imagesize cache saved to {cache_path}")

    def _load_imagesize_cache(self) -> Iterator[ImageCaptionPair]:
        """Load imagesize data from cache if available.

        Yields:
            ImageCaptionPair instances from cache.
            If cache doesn't exist or fails to load, yields nothing.
        """
        lf = self._scan_imagesize_cache()
        if lf is None:
            return

        try:
            count = 0
            for row in lf.collect().iter_rows():
                pair = ImageCaptionPair(
                    image=Path(row[0]),
                    width=row[1],
                    height=row[2],
                    caption=Path(row[3]) if row[3] is not None else None,
                    metadata=Path(row[4]) if row[4] is not None else None,
                )
                count += 1
                yield pair

            if count > 0:
                print(f"Loaded {count} entries from imagesize cache")

        except Exception as e:
            print(f"Failed to load cache: {e}")
            return

    def _process_single_entry(
        self,
        entry: tuple[Path, Path | None, Path | None],
    ) -> ImageCaptionPair | None:
        image_path, caption_path, metadata_path = entry

        try:
            width, height = imagesize.get(image_path)
        except Exception:
            return None

        assert isinstance(width, int) and isinstance(height, int)

        pair = ImageCaptionPair(
            image=image_path,
            width=width,
            height=height,
            caption=caption_path,
            metadata=metadata_path,
        )

        if self.has_skip_metadata:
            if pair.should_skip:  # very slow operation!
                return None

        return pair

    def _yield_tasks(self) -> Iterator[tuple]:
        for root, _, files in os.walk(self.folder):
            files_set = set(files)
            root_path = Path(root)

            for file_name in tqdm(files, desc=f"Scanning {root}"):
                # 文字列判定で高速フィルタリング
                if not any(
                    file_name.endswith(ext) for ext in self.supported_extensions
                ):
                    continue

                # パス生成
                file_path = root_path / file_name
                stem = file_path.stem

                # setを使った高速存在確認
                caption_name = stem + self.caption_extension
                caption_path = (
                    root_path / caption_name if caption_name in files_set else None
                )

                metadata_name = stem + self.metadata_extension
                metadata_path = (
                    root_path / metadata_name if metadata_name in files_set else None
                )

                if caption_path is None and metadata_path is None:
                    continue

                # リストに追加せず、ここで yield する
                yield (file_path, caption_path, metadata_path)

    def _retrieve_images(self) -> Iterator[ImageCaptionPair]:
        tasks = list(self._yield_tasks())
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = executor.map(
                self._process_single_entry,
                tasks,
                chunksize=100,
            )
            for pair in tqdm(
                results,
                total=len(tasks),
                desc="Processing images",
                mininterval=0.5,
            ):
                if pair is not None:
                    yield pair

    def generate_buckets(self) -> list[TextToImageBucket]:  # type: ignore
        # aspect ratio buckets
        ar_buckets: np.ndarray = self.buckets
        arb_manager = AspectRatioBucketManager(ar_buckets)

        # Check if cache exists before loading
        has_cache = self._has_imagesize_cache()

        # Use cache if available, otherwise retrieve images
        if has_cache:
            pairs_iterator = self._load_imagesize_cache()
        else:
            pairs_iterator = self._retrieve_images()

        # If cache miss, generate bucket subsets
        bucket_subsets: dict[int, list[ImageCaptionPair]] = defaultdict(list)

        # classify images into buckets
        for pair in pairs_iterator:
            try:
                # TODO: current is only the behavior of (not do_upscale)
                bucket_idx = arb_manager.find_nearest(pair.width, pair.height)
                bucket_subsets[bucket_idx].append(pair)
                # TODO: implement upscale
            except Exception as e:
                warnings.warn(
                    f"Image size {pair.width}x{pair.height} is too small, and `do_upscale` is set False. Skipping... \n{e}",
                    UserWarning,
                )
                continue

        # Save to cache if it wasn't loaded and cache path is configured
        if self.imagesize_cache_path is not None and not has_cache:
            all_pairs = [pair for pairs in bucket_subsets.values() for pair in pairs]
            self._save_imagesize_cache(all_pairs)

        # create buckets
        buckets = []
        for bucket_idx, pairs in tqdm(bucket_subsets.items(), desc="Creating buckets"):
            if len(pairs) == 0:
                continue

            width, height = ar_buckets[bucket_idx]

            bucket = TextToImageBucket(
                items=pairs,
                batch_size=self.batch_size,
                width=width,
                height=height,
                do_upscale=self.do_upscale,
                num_repeats=self.num_repeats,
                caption_processors=self.caption_processors,
            )
            buckets.append(bucket)

        return buckets

    def get_dataset(self) -> data.Dataset:
        buckets = self.generate_buckets()
        print_arb_info(buckets)

        return data.ConcatDataset([BucketDataset(bucket) for bucket in buckets])
