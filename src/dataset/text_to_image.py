import os
import imagesize
import random
import hashlib
import pickle
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
    cache_dir: str | None = "cache/buckets"
    use_cache: bool = True

    def _get_cache_key(self) -> str:
        """Generate a cache key based on config and folder state."""
        # Get folder stats for cache invalidation
        file_count = 0
        total_size = 0

        for root, _, files in os.walk(self.folder):
            for f in files:
                if any(f.endswith(ext) for ext in self.supported_extensions):
                    file_count += 1
                    try:
                        total_size += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass

        config_str = f"{self.folder}_fc{file_count}_ts{total_size}"

        hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]

        # print(f"Cache config hash: {hash}")

        return hash

    def _get_cache_path(self) -> Path | None:
        """Get the cache file path."""
        if self.cache_dir is None:
            return None
        cache_key = self._get_cache_key()
        folder_name = Path(self.folder).name
        return Path(self.cache_dir) / f"{folder_name}_{cache_key}.pkl"

    def _save_bucket_cache(
        self, bucket_subsets: dict[int, list[ImageCaptionPair]]
    ) -> None:
        """Save bucket data to cache."""
        cache_path = self._get_cache_path()
        if cache_path is None:
            return

        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert ImageCaptionPair to serializable format
        cache_data = {}
        for bucket_idx, pairs in bucket_subsets.items():
            cache_data[bucket_idx] = [
                {
                    "image": str(p.image),
                    "width": p.width,
                    "height": p.height,
                    "caption": str(p.caption) if p.caption else None,
                    "metadata": str(p.metadata) if p.metadata else None,
                }
                for p in pairs
            ]

        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f)

        print(f"Bucket cache saved to {cache_path}")

    def _load_bucket_cache(self) -> dict[int, list[ImageCaptionPair]] | None:
        """Load bucket data from cache if available."""
        cache_path = self._get_cache_path()

        if cache_path is None or not cache_path.exists():
            print("No bucket cache found.")
            return None

        try:
            with open(cache_path, "rb") as f:
                cache_data = pickle.load(f)

            # Convert back to ImageCaptionPair
            bucket_subsets: dict[int, list[ImageCaptionPair]] = {}
            for bucket_idx, pairs_data in cache_data.items():
                pairs = []
                for p in pairs_data:
                    pair = ImageCaptionPair(
                        image=Path(p["image"]),
                        width=p["width"],
                        height=p["height"],
                        caption=Path(p["caption"]) if p["caption"] else None,
                        metadata=Path(p["metadata"]) if p["metadata"] else None,
                    )
                    # Verify the image still exists
                    if not pair.image.exists():
                        print(f"Cache invalidated: {pair.image} no longer exists")
                        return None
                    pairs.append(pair)
                bucket_subsets[bucket_idx] = pairs

            print(f"Loaded bucket cache from {cache_path}")
            return bucket_subsets

        except Exception as e:
            print(f"Failed to load cache: {e}")
            return None

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

    def _retrive_images(self) -> Iterator[ImageCaptionPair]:
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

        # Try to load from cache first
        bucket_subsets: dict[int, list[ImageCaptionPair]] | None = None
        if self.use_cache:
            bucket_subsets = self._load_bucket_cache()

        # If cache miss, generate bucket subsets
        if bucket_subsets is None:
            bucket_subsets = defaultdict(list)

            # classify images into buckets
            for pair in self._retrive_images():
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

            # Save to cache
            if self.use_cache:
                self._save_bucket_cache(dict(bucket_subsets))

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
