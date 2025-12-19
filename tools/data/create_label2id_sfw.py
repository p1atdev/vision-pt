from pathlib import Path
import os
from collections import defaultdict
import orjson
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
import click

from src.dataset.tags import map_replace_underscore


def load_json_file(filepath: str) -> dict | None:
    """Load a single JSON file."""
    try:
        with open(filepath, "r") as f:
            return orjson.loads(f.read())
    except Exception:
        return None


@click.command()
@click.option(
    "--input",
    "-i",
    type=Path,
    required=True,
)
@click.option(
    "--output",
    "-o",
    type=Path,
    required=True,
)
@click.option(
    "--character_threshold",
    "-ch",
    type=int,
    default=10,
    help="Minimum number of occurrences for a character tag to be included.",
)
@click.option(
    "--coopyright_threshold",
    "-cp",
    type=int,
    default=10,
    help="Minimum number of occurrences for a copyright tag to be included.",
)
@click.option(
    "--general_threshold",
    "-g",
    type=int,
    default=100,
    help="Minimum number of occurrences for a general tag to be included.",
)
@click.option(
    "--meta_threshold",
    "-m",
    type=int,
    default=10,
    help="Minimum number of occurrences for a meta tag to be included.",
)
@click.option(
    "--special_tags",
    "-s",
    multiple=True,
    default=[
        "masterpiece",
        "best_quality",
        "high_quality",
        "low_quality",
        "worst_quality",
    ],
    help="List of special tags to always include.",
)
def main(
    input: Path,
    output: Path,
    character_threshold: int = 10,
    coopyright_threshold: int = 10,
    general_threshold: int = 100,
    meta_threshold: int = 100,
    special_tags: list[str] = [
        "masterpiece",
        "best_quality",
        "high_quality",
        "low_quality",
        "worst_quality",
    ],
):
    label2id = {}

    ratings: set[str] = set()
    character_count = defaultdict(int)
    copyright_count = defaultdict(int)
    general_count = defaultdict(int)
    meta_count = defaultdict(int)

    num_tags_in_data = []

    def process_data(data):
        rating = data.get("rating", "g")
        character_tags = data.get("tag_string_character", "").split(" ")
        copyright_tags = data.get("tag_string_copyright", "").split(" ")
        general_tags = data.get("tag_string_general", "").split(" ")
        meta_tags = data.get("tag_string_meta", "").split(" ")

        ratings.add(rating)

        for tag in character_tags:
            if tag.strip() == "":
                continue
            character_count[tag] += 1

        for tag in copyright_tags:
            if tag.strip() == "":
                continue
            copyright_count[tag] += 1

        for tag in general_tags:
            if tag.strip() == "":
                continue
            general_count[tag] += 1

        for tag in meta_tags:
            if tag.strip() == "":
                continue
            meta_count[tag] += 1

        num_tags_in_data.append(
            len(character_tags)
            + len(general_tags)
            + len(copyright_tags)
            + len(meta_tags)
        )

    # Collect all JSON file paths first
    json_files = []
    for root, _dirs, files in os.walk(input):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))

    print(f"Found {len(json_files)} JSON files")

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(load_json_file, f): f for f in json_files}
        for future in tqdm(as_completed(futures), total=len(json_files)):
            data = future.result()
            if data is not None:
                process_data(data)

    print(
        f"Found {len(ratings)} ratings, {len(character_count)} characters, {len(general_count)} general tags."
    )
    avg_num_tags = sum(num_tags_in_data) / len(num_tags_in_data)
    max_num_tags = max(num_tags_in_data)
    min_num_tags = min(num_tags_in_data)
    print(f"Average number of tags per data: {avg_num_tags:.2f}")
    print(f"Max number of tags per data: {max_num_tags}")
    print(f"Min number of tags per data: {min_num_tags}")

    popular_general_tags = {
        tag for tag, count in general_count.items() if count >= general_threshold
    }
    print(
        f"Filtered to {len(popular_general_tags)} popular general tags. (threshold: {general_threshold})"
    )

    popular_character_tags = {
        tag for tag, count in character_count.items() if count >= character_threshold
    }
    print(
        f"Filtered to {len(popular_character_tags)} popular character tags. (threshold: {character_threshold})"
    )

    popular_copyright_tags = {
        tag for tag, count in copyright_count.items() if count >= coopyright_threshold
    }
    print(
        f"Filtered to {len(popular_copyright_tags)} popular copyright tags. (threshold: {coopyright_threshold})"
    )

    # filter meta tags
    for tag, _count in list(meta_count.items()):
        if any(
            word in tag
            for word in [
                "request",
                "comment",
                "bad",
                "source",
                "translat",  # translate | translation
                "commission",
                "scan",
                "account",
                "version",
                "md5",
                "mismatch",
                "revision",
                "link",
                "upload",
                "spoilter",
                "variant",
                "artist",
                "available",
                "reward",
                "language",
                "annotate",
                "sample",
                "check",
                "corrupted",
                "metadata",
                "waifu2x",
                "topic",
                "text",
                "trace",
                "issue",
                "edit",
                # useless medium
                "photoshop",
                "studio",
                "krita",
                "procreate",
                "paint.net",
                "gimp",
                "painttool",  # sai
            ]
        ):
            del meta_count[tag]
            continue

    popular_meta_tags = {
        tag for tag, count in meta_count.items() if count >= meta_threshold
    }
    print(
        f"Filtered to {len(popular_meta_tags)} popular meta tags. (threshold: {meta_threshold})"
    )

    # rename rating tags
    rating_rename_map = {
        "g": "general",
        "s": "sensitive",
        "q": "questionable",
        "e": "explicit",
    }
    ratings = {rating_rename_map.get(r, r) for r in ratings}

    all_labels = (
        list(special_tags)
        + sorted(list(ratings))
        + sorted(list(popular_copyright_tags))
        + sorted(list(popular_character_tags))
        + sorted(list(popular_general_tags))
        + sorted(list(popular_meta_tags))
    )
    all_labels = map_replace_underscore(all_labels)  # escape underscores except kaomoji

    label2id = {label: idx for idx, label in enumerate(all_labels)}
    counts = {
        "special": special_tags,
        "ratings": len(ratings),
        "copyrights": copyright_count,
        "characters": character_count,
        "general": general_count,
        "meta": meta_count,
        "total": len(all_labels),
    }

    with open(output, "w") as f:
        json.dump(label2id, f, indent=4, ensure_ascii=False)
    with open(output.with_suffix(".count.json"), "w") as f:
        json.dump(counts, f, indent=4, ensure_ascii=False)

    print(f"Saved label2id mapping to {output}")


if __name__ == "__main__":
    main()
