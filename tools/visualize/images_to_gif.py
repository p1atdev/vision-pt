import os
import re
import argparse
from PIL import Image
from pathlib import Path


def create_gif_from_path_pattern(
    input_path: str,
    output_path: str | None = None,
    duration: int = 200,
):
    input_dir = Path(input_path)
    directory = input_dir.parent
    filename = input_dir.name

    # Pattern: {epoch}e_{steps}s_{id}.webp
    # We use regex to extract epoch, steps, and id from the filename
    pattern = r"(\d+)e_(\d+)s_(.+)\.webp"
    match = re.match(pattern, filename)
    if not match:
        print(
            f"Error: Filename '{filename}' does not match the pattern '{{epoch}}e_{{steps}}s_{{id}}.webp'"
        )
        return

    target_id = match.group(3)

    # Find all files in the same directory that match the pattern and have the same id
    files = []
    for f in directory.glob(f"*_*s_{target_id}.webp"):
        m = re.match(pattern, f.name)
        if m and m.group(3) == target_id:
            epoch = int(m.group(1))
            steps = int(m.group(2))
            files.append((epoch, steps, f))

    if not files:
        print(f"No files found for id: {target_id}")
        return

    # Sort by epoch first, then by steps to ensure correct chronological order
    files.sort(key=lambda x: (x[0], x[1]))

    print(f"Found {len(files)} images for id '{target_id}'. Creating GIF...")

    images = []
    for _, _, f_path in files:
        try:
            img = Image.open(f_path)
            # Ensure all images are in the same mode (e.g., RGB) for GIF creation
            if img.mode != "RGB":
                img = img.convert("RGB")
            images.append(img)
        except Exception as e:
            print(f"Warning: Could not load image {f_path}: {e}")

    if not images:
        print("No valid images loaded.")
        return

    if not output_path:
        output_path = str(directory / f"{target_id}.gif")
    else:
        output_path = str(Path(output_path))

    # Save as GIF
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
        optimize=True,
    )

    print(f"Successfully created GIF: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a GIF from images with the same ID, sorted by steps."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to one of the images (e.g., path/to/10e_1000s_abc123.webp)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output GIF path (optional, defaults to {id}.gif in the same directory)",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=int,
        default=200,
        help="Duration of each frame in ms (default: 200)",
    )

    args = parser.parse_args()
    create_gif_from_path_pattern(args.input_path, args.output, args.duration)
