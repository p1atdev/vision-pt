from pathlib import Path
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
def main(
    tags_dir: Path,
    output: Path,
):
    label2id = {}

    ratings = set()
    characters = set()
    general_count = defaultdict(int)

    for file in tqdm(list(tags_dir.glob("*.json"))):
        with open(file, "r") as f:
            data = json.load(f)

            rating = data.get("rating", "general")
            character_tags = data.get("character_tags", {}).keys()
            general_tags = data.get("general_tags", {}).keys()

            ratings.add(rating)
            characters.update(character_tags)

            for tag in general_tags:
                general_count[tag] += 1

    print(
        f"Found {len(ratings)} ratings, {len(characters)} characters, {len(general_count)} general tags."
    )

    all_labels = (
        sorted(list(ratings))
        + sorted(list(characters))
        + sorted(list(general_count.keys()))
    )

    label2id = {label: idx for idx, label in enumerate(all_labels)}

    with open(output, "w") as f:
        json.dump(label2id, f, indent=4, ensure_ascii=False)

    print(f"Saved label2id mapping to {output}")


if __name__ == "__main__":
    main()
