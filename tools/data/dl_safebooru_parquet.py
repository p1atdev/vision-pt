from tqdm.auto import tqdm
from itertools import islice
from PIL import Image
from pathlib import Path
import json
import os
from datetime import time, datetime, date, timezone

import polars as pl

import click

from cheesechaser.datapool import Danbooru2024WebpDataPool
from cheesechaser.pipe import SimpleImagePipe, PipeItem


tqdm.pandas()  # register progress_apply


@click.command()
@click.option(
    "--parquet-path",
    "-i",
    type=Path,
    default=Path("./parquets/danbooru2024-webp-4Mpixel.parquet"),
)
@click.option(
    "--outptut-path",
    "-o",
    type=Path,
    required=True,
)
@click.option("--start-date", type=str, default="2020-01-01")
@click.option("--end-date", type=str, default="2025-12-31")
@click.option("--limit", "-l", type=int, default=1000)
def main(
    parquet_path: Path,
    outptut_path: Path,
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
    limit: int = 1000,
):
    outptut_path.mkdir(parents=True, exist_ok=True)
    cache_parquet_path = outptut_path / "cache.parquet"

    if not cache_parquet_path.exists():
        # read parquet file
        full_lf = pl.scan_parquet(parquet_path)
        print(full_lf.head().collect())  # check the first 5 rows

        lf = full_lf.with_columns(
            pl.col("created_at").str.to_datetime(time_zone="UTC")
        )  # convert to datetime
        lf.columns

        datetime_start = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
        datetime_end = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
        print(f"Filtering from {datetime_start} to {datetime_end}")

        lf = (
            lf.filter(
                pl.col("created_at").is_between(
                    datetime_start,
                    datetime_end,
                    closed="left",
                ),
            )
            .with_columns(
                tag_list_meta=pl.col("tag_string_meta").str.split(" "),
            )
            .filter(
                (~pl.col("tag_list_meta").list.contains("animated"))
                & (~pl.col("tag_list_meta").list.contains("duplicate"))
                & (~pl.col("tag_list_meta").list.contains("pixel-perfect_duplicate"))
                & (~pl.col("tag_list_meta").list.contains("lowres"))
                & (~pl.col("tag_list_meta").list.contains("watermark"))
            )
            .drop("tag_list_meta")
            .limit(limit)
        )

        print(f"Writing filtered parquet to {cache_parquet_path}")
        lf.sink_parquet(cache_parquet_path)

    print(f"Reading cached parquet from {cache_parquet_path}")
    df = pl.read_parquet(cache_parquet_path)

    print(f"Total images to process: {df.height}")

    images_dir = outptut_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    ids = set(df.select("id").to_series().to_list())

    # skip duplicates
    print("Checking existing images to skip...")
    existing_ids = {
        int(entry.name[:-5])  # remove ".webp"
        for entry in os.scandir(images_dir)
        if entry.name.endswith(".webp") and entry.is_file()
    }
    ids -= existing_ids
    print(f"Skipped {len(existing_ids)} existing images")

    # save all json
    for row in tqdm(
        df.iter_rows(named=True),
        desc="Saving JSON metadata",
        total=df.height,
    ):
        image_id = row["id"]
        if int(image_id) in existing_ids:
            continue
        with open(images_dir / f"{image_id}.json", "w") as f:
            json.dump(row, f, indent=2, ensure_ascii=False, default=str)

    pool = Danbooru2024WebpDataPool()
    pipe = SimpleImagePipe(pool)

    print(f"Downloading {len(ids)} images to {images_dir}")

    with pipe.batch_retrieve(ids, silent=True) as session:
        for item in tqdm(session, total=len(ids), desc="Downloading images"):
            item.data.save(images_dir / f"{item.id}.webp")


if __name__ == "__main__":
    main()
