import click
import torch

from safetensors.torch import load_file, save_file
import torch.nn.functional as F


EMBED_WEIGHT_KEY = "denoiser.patch_embedder.proj_1.weight"
FINAL_WEIGHT_KEY = "denoiser.final_layer.linear.weight"
FINAL_BIAS_KEY = "denoiser.final_layer.linear.bias"


def resize_tensor(
    tensor: torch.Tensor,
    size: tuple[int, int],
    mode: str = "bicubic",
) -> torch.Tensor:
    """
    テンソルを指定サイズにリサイズする。

    Args:
        tensor: リサイズ対象のテンソル [Batch, Channel, H, W] 形式
        size: 出力サイズ (H, W)
        mode: 補間モード ("bicubic", "bilinear", "nearest", "nearest-exact")

    Returns:
        リサイズされたテンソル
    """
    return F.interpolate(
        tensor,
        size=size,
        mode=mode,
        align_corners=False if mode in ["bicubic", "bilinear"] else None,
    )


def resize_patch_embed_weight(
    weight: torch.Tensor,
    target_size: tuple[int, int],
    mode: str = "bicubic",
) -> torch.Tensor:
    """
    Patch embedding の重みをリサイズする。

    Args:
        weight: 重みテンソル [Out, In, H, W] 形式
        target_size: 出力サイズ (H, W)
        mode: 補間モード

    Returns:
        リサイズされた重みテンソル
    """
    # F.interpolate は [Batch, Channel, H, W] を期待するため、
    # [Out, In, H, W] をそのまま渡しても機能しますが、
    # 厳密には "Out個のフィルタ" をそれぞれリサイズしたいので、次元の意味的には合致します。
    new_weight = resize_tensor(weight, target_size, mode)

    # # energyのスケーリング
    new_weight = (
        new_weight
        * (target_size[0] * target_size[1])
        / (weight.shape[2] * weight.shape[3])
    )
    return new_weight


def resize_final_layer_weight(
    weight: torch.Tensor,
    old_patch_size: int,
    target_size: tuple[int, int],
    mode: str = "bicubic",
    out_channels: int = 3,
) -> torch.Tensor:
    """
    Final layer の重みをリサイズする。

    unpatchify での並びを参照:
        patches = patches.view(B, h_patches, w_patches, patch_size, patch_size, out_channels)
    つまり、Linear の出力は [patch_size, patch_size, out_channels] の順で Flatten されている。

    Args:
        weight: 重みテンソル [Out, In] = [H_old * W_old * C, hidden_size] 形式
        old_patch_size: 元のパッチサイズ
        target_size: 出力サイズ (H, W)
        mode: 補間モード
        out_channels: 出力チャンネル数

    Returns:
        リサイズされた重みテンソル [H_new * W_new * C, hidden_size]
    """
    out_dim, hidden_size = weight.shape

    # (A) 空間構造にReshape
    # Linear出力は [H, W, C] の順で Flatten されている
    # Shape: [H_old, W_old, C, hidden_size] に変形
    w_spatial = weight.view(old_patch_size, old_patch_size, out_channels, hidden_size)

    # (B) InterpolateのためにPermute
    # interpolate は (Batch, Channel, H, W) を期待
    # hidden_size を Batch、C を Channel と見なして空間をリサイズ
    # [H_old, W_old, C, hidden_size] -> [hidden_size, C, H_old, W_old]
    w_permuted = w_spatial.permute(3, 2, 0, 1)

    # (C) リサイズ実行
    new_w_permuted = resize_tensor(w_permuted, target_size, mode)
    # -> [hidden_size, C, H_new, W_new]

    # (D) 元のLinearの形状に戻す
    # Permute back: [H_new, W_new, C, hidden_size]
    new_w_spatial = new_w_permuted.permute(2, 3, 1, 0).contiguous()
    # Flatten: [H_new * W_new * C, hidden_size]
    return new_w_spatial.reshape(-1, hidden_size)


def resize_final_layer_bias(
    bias: torch.Tensor,
    old_patch_size: int,
    target_size: tuple[int, int],
    mode: str = "bicubic",
    out_channels: int = 3,
) -> torch.Tensor:
    """
    Final layer のバイアスをリサイズする。

    Args:
        bias: バイアステンソル [C * H * W] 形式 (Flattenされた状態)
        old_patch_size: 元のパッチサイズ
        target_size: 出力サイズ (H, W)
        mode: 補間モード
        out_channels: 出力チャンネル数

    Returns:
        リサイズされたバイアステンソル [C * H_new * W_new]
    """
    # (A) Reshape: [1, C, H_old, W_old] (Batch次元をダミーで追加)
    b_spatial = bias.view(
        1,
        old_patch_size,
        old_patch_size,
        out_channels,
    ).permute(0, 3, 1, 2)  # -> [1, C, H_old, W_old]

    # (B) Interpolate
    new_b_spatial = resize_tensor(b_spatial, target_size, mode)

    # (C) Permute back & Flatten
    # [1, C, H, W] -> [1, H, W, C] -> Flatten
    new_b_spatial = new_b_spatial.permute(0, 2, 3, 1).flatten()

    return new_b_spatial


@click.command()
@click.option(
    "--input",
    "-i",
    type=str,
    required=True,
    help="Path to the input patch embedding file (safetensors format).",
)
@click.option(
    "--output",
    "-o",
    type=str,
    required=True,
    help="Path to save the expanded patch embedding file (safetensors format).",
)
@click.option(
    "--patch_size",
    "-p",
    type=int,
    default=32,
    help="Size of each patch (default: 32).",
)
@click.option(
    "--mode",
    "-m",
    type=click.Choice(
        ["bicubic", "bilinear", "nearest", "nearest-exact"], case_sensitive=False
    ),
    default="bicubic",
    help="Interpolation mode (default: bicubic).",
)
def main(
    input: str,
    output: str,
    patch_size: int = 32,
    mode: str = "bicubic",
):
    state_dict = load_file(input)
    target_size = (patch_size, patch_size)

    # 1. patch embedding
    assert EMBED_WEIGHT_KEY in state_dict, (
        f"Key '{EMBED_WEIGHT_KEY}' not found in the input file."
    )
    embed_weight = state_dict[EMBED_WEIGHT_KEY]
    _, _, height, width = embed_weight.shape
    old_patch_size = height

    print(
        f"Resizing '{EMBED_WEIGHT_KEY}': {height}x{width} -> {patch_size}x{patch_size} ({mode})"
    )
    state_dict[EMBED_WEIGHT_KEY] = resize_patch_embed_weight(
        embed_weight, target_size, mode
    )

    # 2. final layer weight
    assert FINAL_WEIGHT_KEY in state_dict, (
        f"Key '{FINAL_WEIGHT_KEY}' not found in the input file."
    )
    print(
        f"Resizing '{FINAL_WEIGHT_KEY}': {old_patch_size}x{old_patch_size} -> {patch_size}x{patch_size} ({mode})"
    )
    state_dict[FINAL_WEIGHT_KEY] = resize_final_layer_weight(
        state_dict[FINAL_WEIGHT_KEY], old_patch_size, target_size, mode
    )

    # 3. final layer bias
    assert FINAL_BIAS_KEY in state_dict, (
        f"Key '{FINAL_BIAS_KEY}' not found in the input file."
    )
    state_dict[FINAL_BIAS_KEY] = resize_final_layer_bias(
        state_dict[FINAL_BIAS_KEY], old_patch_size, target_size, mode
    )

    save_file(state_dict, output)
    print(f"Saved expanded patch embedding to '{output}'.")


if __name__ == "__main__":
    main()
