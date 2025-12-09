import torch
import torchvision.utils as vutils
from PIL import Image

from .tensor import images_to_tensor


@torch.no_grad()
def images_to_grid_image(
    images: list[Image.Image] | torch.Tensor,
    padding: int = 2,
) -> Image.Image:
    if isinstance(images, list):
        tensor_images = images_to_tensor(
            images,
            dtype=torch.float16,
            device=torch.device("cpu"),
        )

    grid = vutils.make_grid(
        tensor_images,
        nrow=int(len(tensor_images) ** 0.5),
        padding=padding,
        normalize=True,
    )

    # TensorをPIL画像に変換して保存
    # (C, H, W) -> (H, W, C) への変換なども自動化できる
    image = Image.fromarray(
        grid.mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .to("cpu", torch.uint8)
        .numpy()
    )

    return image
