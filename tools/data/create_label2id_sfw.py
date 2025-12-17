from pathlib import Path
import os
from collections import defaultdict
import json

from tqdm import tqdm
import click

from src.dataset.tags import map_replace_underscore


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

    all_data = []

    for root, _dirs, files in os.walk(input):
        for file in tqdm(files):
            if not file.endswith(".json"):
                continue

            file = os.path.join(root, file)
            with open(file, "r") as f:
                data = json.load(f)

            all_data.append(data)

    ratings = set()
    character_count = defaultdict(int)
    copyright_count = defaultdict(int)
    general_count = defaultdict(int)
    meta_count = defaultdict(int)

    num_tags_in_data = []

    for data in all_data:
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

    print(
        f"Found {len(ratings)} ratings, {len(character_count)} characters, {len(general_count)} general tags."
    )
    avg_num_tags = sum(num_tags_in_data) / len(num_tags_in_data)
    max_num_tags = max(num_tags_in_data)
    min_num_tags = min(num_tags_in_data)
    print(f"Average number of tags per data: {avg_num_tags:.2f}")
    print(f"Max number of tags per data: {max_num_tags}")
    print(f"Min number of tags per data: {min_num_tags}")

    # # top 100 general tags
    # print(dict(sorted(general_count.items(), key=lambda x: x[1], reverse=True)[:10]))
    # # bottom 10 general tags
    # print(dict(sorted(general_count.items(), key=lambda x: x[1])[:10]))
    # return

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
        if any(word in tag for word in ["request", "comment", "bad", "source", ""]):
            del meta_count[tag]
            continue

    popular_meta_tags = {
        tag for tag, count in meta_count.items() if count >= meta_threshold
    }
    print(
        f"Filtered to {len(popular_meta_tags)} popular meta tags. (threshold: {meta_threshold})"
    )

    all_labels = (
        list(special_tags)
        # + sorted(list(ratings)) # all sfw
        + sorted(list(popular_copyright_tags))
        + sorted(list(popular_character_tags))
        + sorted(list(popular_general_tags))
    )
    all_labels = map_replace_underscore(all_labels)  # escape underscores except kaomoji

    label2id = {label: idx for idx, label in enumerate(all_labels)}
    counts = {
        "special": special_tags,
        "ratings": len(ratings),
        "copyrights": copyright_count,
        "characters": character_count,
        "general": general_count,
        "total": len(all_labels),
    }

    with open(output, "w") as f:
        json.dump(label2id, f, indent=4, ensure_ascii=False)
    with open(output.with_suffix(".count.json"), "w") as f:
        json.dump(counts, f, indent=4, ensure_ascii=False)

    print(f"Saved label2id mapping to {output}")


if __name__ == "__main__":
    main()
