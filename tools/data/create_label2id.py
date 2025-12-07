from pathlib import Path
import os
from collections import defaultdict
import json

from tqdm import tqdm
import click


@click.command()
@click.option(
    "--tags_dir",
    "-t",
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
    "-c",
    type=int,
    default=10,
    help="Minimum number of occurrences for a character tag to be included.",
)
@click.option(
    "--general_threshold",
    "-g",
    type=int,
    default=100,
    help="Minimum number of occurrences for a general tag to be included.",
)
def main(
    tags_dir: Path,
    output: Path,
    character_threshold: int = 10,
    general_threshold: int = 100,
):
    label2id = {}

    all_data = []

    for root, _dirs, files in os.walk(tags_dir):
        for file in tqdm(files):
            if not file.endswith(".json"):
                continue

            file = os.path.join(root, file)
            with open(file, "r") as f:
                data = json.load(f)

            all_data.append(data)

    ratings = set()
    character_count = defaultdict(int)
    general_count = defaultdict(int)

    num_tags_in_data = []

    for data in all_data:
        rating = data.get("rating", "general")
        character_tags = data.get("character_tags", {}).keys()
        general_tags = data.get("general_tags", {}).keys()

        ratings.add(rating)

        for tag in character_tags:
            character_count[tag] += 1

        for tag in general_tags:
            general_count[tag] += 1

        num_tags_in_data.append(len(character_tags) + len(general_tags))

    print(
        f"Found {len(ratings)} ratings, {len(character_count)} characters, {len(general_count)} general tags."
    )
    print(
        f"Average number of tags per data point: {sum(num_tags_in_data) / len(num_tags_in_data):.2f}"
    )

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

    all_labels = (
        sorted(list(ratings))
        + sorted(list(character_count.keys()))
        + sorted(list(popular_general_tags))
    )

    label2id = {label: idx for idx, label in enumerate(all_labels)}
    counts = {
        "ratings": len(ratings),
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
