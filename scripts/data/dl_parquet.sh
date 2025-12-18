#!/bin/bash

uv run hf download "deepghs/danbooru2024-webp-4Mpixel" metadata.parquet \
    --repo-type dataset \
    --local-dir ./data/parquets

mv ./data/parquets/metadata.parquet ./data/parquets/danbooru2024-webp-4Mpixel.parquet

