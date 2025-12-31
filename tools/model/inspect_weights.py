"""
Safetensors checkpoint statistics analyzer.
Outputs mean, variance, and other statistics for each weight tensor.
"""

import argparse
from pathlib import Path

from safetensors import safe_open


def analyze_checkpoint(
    checkpoint_path: str, verbose: bool = False, markdown: bool = False
) -> None:
    """Analyze safetensors checkpoint and print statistics."""

    if markdown:
        _analyze_markdown(checkpoint_path, verbose)
    else:
        _analyze_plain(checkpoint_path, verbose)


def _analyze_plain(checkpoint_path: str, verbose: bool = False) -> None:
    """Output in plain text format."""
    print(f"Loading checkpoint: {checkpoint_path}")
    print("=" * 80)

    with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
        keys = f.keys()

        total_params = 0

        for key in keys:
            tensor = f.get_tensor(key)

            # Calculate statistics
            numel = tensor.numel()
            total_params += numel

            mean = tensor.float().mean().item()
            var = tensor.float().var().item()
            std = tensor.float().std().item()
            min_val = tensor.min().item()
            max_val = tensor.max().item()

            print(f"\n{key}")
            print(f"  Shape: {list(tensor.shape)}, dtype: {tensor.dtype}")
            print(f"  Parameters: {numel:,}")
            print(f"  Mean: {mean:.6e}")
            print(f"  Variance: {var:.6e}")
            print(f"  Std: {std:.6e}")
            print(f"  Min: {min_val:.6e}, Max: {max_val:.6e}")

            if verbose:
                # Additional statistics for verbose mode
                abs_mean = tensor.float().abs().mean().item()
                print(f"  Abs Mean: {abs_mean:.6e}")

                # Check for NaN/Inf
                nan_count = tensor.isnan().sum().item()
                inf_count = tensor.isinf().sum().item()
                if nan_count > 0 or inf_count > 0:
                    print(f"  ⚠️  NaN: {nan_count}, Inf: {inf_count}")

        print("\n" + "=" * 80)
        print(f"Total parameters: {total_params:,}")
        print(f"Total tensors: {len(keys)}")


def _analyze_markdown(checkpoint_path: str, verbose: bool = False) -> None:
    """Output in markdown table format."""
    print("# Checkpoint Statistics\n")
    print(f"**File:** `{checkpoint_path}`\n")

    with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())

        total_params = 0

        # Build table header
        if verbose:
            print(
                "| Name | Shape | dtype | Params | Mean | Variance | Std | Min | Max | Abs Mean | NaN | Inf |"
            )
            print(
                "|------|-------|-------|--------|------|----------|-----|-----|-----|----------|-----|-----|"
            )
        else:
            print(
                "| Name | Shape | dtype | Params | Mean | Variance | Std | Min | Max |"
            )
            print(
                "|------|-------|-------|--------|------|----------|-----|-----|-----|"
            )

        for key in keys:
            tensor = f.get_tensor(key)

            # Calculate statistics
            numel = tensor.numel()
            total_params += numel

            mean = tensor.float().mean().item()
            var = tensor.float().var().item()
            std = tensor.float().std().item()
            min_val = tensor.min().item()
            max_val = tensor.max().item()

            shape_str = str(list(tensor.shape))

            if verbose:
                abs_mean = tensor.float().abs().mean().item()
                nan_count = tensor.isnan().sum().item()
                inf_count = tensor.isinf().sum().item()
                print(
                    f"| `{key}` | {shape_str} | {tensor.dtype} | {numel:,} | "
                    f"{mean:.4e} | {var:.4e} | {std:.4e} | {min_val:.4e} | {max_val:.4e} | "
                    f"{abs_mean:.4e} | {nan_count} | {inf_count} |"
                )
            else:
                print(
                    f"| `{key}` | {shape_str} | {tensor.dtype} | {numel:,} | "
                    f"{mean:.4e} | {var:.4e} | {std:.4e} | {min_val:.4e} | {max_val:.4e} |"
                )

        print("\n## Summary\n")
        print(f"- **Total parameters:** {total_params:,}")
        print(f"- **Total tensors:** {len(keys)}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze safetensors checkpoint statistics"
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to the safetensors checkpoint file",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show additional statistics (abs mean, NaN/Inf check)",
    )
    parser.add_argument(
        "-m",
        "--markdown",
        action="store_true",
        help="Output in markdown table format",
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        return

    if not checkpoint_path.suffix == ".safetensors":
        print("Warning: File does not have .safetensors extension")

    analyze_checkpoint(
        str(checkpoint_path), verbose=args.verbose, markdown=args.markdown
    )


if __name__ == "__main__":
    main()
