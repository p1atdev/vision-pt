import os
from pathlib import Path
import polars as pl
import json
from tqdm import tqdm
import hashlib
from concurrent.futures import ThreadPoolExecutor
import click
from tqdm import tqdm
from typing import Iterator
from itertools import islice
import pyarrow.parquet as pq
import imagesize


def get_num_images(input_path: str, supported_extensions: list[str]) -> int:
    count = 0
    for root, _, files in os.walk(input_path):
        for file_name in tqdm(files, desc=f"Counting images in {root}"):
            if any(file_name.endswith(ext) for ext in supported_extensions):
                count += 1
    return count


def yield_tasks(
    input_path: str,
    supported_extensions: list[str],
    caption_extension: str,
    metadata_extension: str,
):
    for root, _, files in os.walk(input_path):
        files_set = set(files)
        root_path = Path(root)

        for file_name in tqdm(files, desc=f"Scanning {root}"):
            # 文字列判定で高速フィルタリング
            if not any(file_name.endswith(ext) for ext in supported_extensions):
                continue

            # パス生成
            file_path = root_path / file_name
            stem = file_path.stem

            # setを使った高速存在確認
            caption_name = stem + caption_extension
            caption_path = (
                root_path / caption_name if caption_name in files_set else None
            )

            metadata_name = stem + metadata_extension
            metadata_path = (
                root_path / metadata_name if metadata_name in files_set else None
            )

            if caption_path is None and metadata_path is None:
                continue

            # リストに追加せず、ここで yield する
            yield (file_path, caption_path, metadata_path)


def process_single_entry(
    entry: tuple[Path, Path | None, Path | None],
) -> dict | None:
    image_path, caption_path, metadata_path = entry

    try:
        width, height = imagesize.get(image_path)
    except Exception:
        return None

    assert isinstance(width, int) and isinstance(height, int)

    pair = {
        "image": image_path.as_posix(),
        "width": width,
        "height": height,
        "caption": caption_path.as_posix() if caption_path is not None else None,
        "metadata": metadata_path.as_posix() if metadata_path is not None else None,
    }

    return pair


def chunked(iterable: Iterator, n: int) -> Iterator[list]:
    """イテレータを n 個ずつのリストに分割する"""
    it = iter(iterable)
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            return
        yield chunk


def iterate_pairs(
    input_path: str,
    supported_extensions: list[str],
    caption_extension: str,
    metadata_extension: str,
    num_workers: int = os.cpu_count() or 16,
):
    tasks = list(
        yield_tasks(
            input_path,
            supported_extensions,
            caption_extension,
            metadata_extension,
        )
    )

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(
            process_single_entry,
            tasks,
            chunksize=100,
        )
        for pair in results:
            if pair is not None:
                yield pair


@click.command()
@click.option(
    "--input-path",
    "-i",
    type=str,
    required=True,
    help="Path to the input image directory",
)
@click.option(
    "--output-path",
    "-o",
    type=str,
    required=True,
    help="Path to the output parquet file",
)
@click.option(
    "--num-workers",
    "-j",
    type=int,
    default=os.cpu_count() or 16,
    help="Number of worker threads",
)
@click.option(
    "--supported-extensions",
    "-e",
    type=str,
    multiple=True,
    default=[
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    ],
    help="Supported image file extensions",
)
@click.option(
    "--caption-extension",
    "-c",
    type=str,
    default=".txt",
    help="Caption file extension",
)
@click.option(
    "--metadata-extension",
    "-m",
    type=str,
    default=".json",
    help="Metadata file extension",
)
def main(
    input_path: str,
    output_path: str,
    num_workers: int = os.cpu_count() or 16,
    supported_extensions: list[str] = [
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    ],
    caption_extension: str = ".txt",
    metadata_extension: str = ".json",
):
    num_images = get_num_images(input_path, supported_extensions)
    print(f"Found {num_images} images in {input_path}")

    pairs = iterate_pairs(
        input_path,
        supported_extensions,
        caption_extension,
        metadata_extension,
        num_workers,
    )

    schema = {
        "image": pl.String,
        "width": pl.Int32,
        "height": pl.Int32,
        "caption": pl.String,
        "metadata": pl.String,
    }

    writer = pq.ParquetWriter(
        output_path,
        schema=pl.DataFrame(
            None,
            schema=schema,
        )
        .to_arrow()
        .schema,
    )

    for i, batch in tqdm(
        enumerate(chunked(pairs, 1000)),
        total=(num_images + 999) // 1000,
        desc="Writing parquet files",
    ):
        df = pl.DataFrame(batch, schema=schema)

        # 書き込む
        table = df.to_arrow()
        writer.write_table(table)

    writer.close()
    print(f"Saved parquet file to {output_path}")


if __name__ == "__main__":
    main()
