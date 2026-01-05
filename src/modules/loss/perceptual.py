from abc import ABC
from typing import Literal
from pydantic import BaseModel

import torch
import torch.nn as nn

from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    StructuralSimilarityIndexMeasure,
)

PerceptualLossType = Literal["lpips", "dists"]


class AbstractPerceptualLossConfig(BaseModel, ABC):
    type: PerceptualLossType
    weight: float = 1.0

    # to make hashable
    def __hash__(self) -> int:
        return hash(self.type)


class LPIPSLossConfig(AbstractPerceptualLossConfig):
    type: Literal["lpips"] = "lpips"
    model: Literal["alex", "vgg", "squeeze"] = "alex"


class DISTSConfig(AbstractPerceptualLossConfig):
    type: Literal["dists"] = "dists"


PerceptualLossConfig = LPIPSLossConfig | DISTSConfig


class PerceptualLoss(nn.Module):
    def __init__(
        self,
        loss_configs: list[PerceptualLossConfig],
        convert_zero_to_one: bool = True,
    ):
        super().__init__()

        self.loss_configs = loss_configs
        self.convert_zero_to_one = convert_zero_to_one

        self.metrics: nn.ModuleDict = nn.ModuleDict()

        for loss_config in loss_configs:
            if isinstance(loss_config, LPIPSLossConfig):
                self.metrics["lpips"] = LearnedPerceptualImagePatchSimilarity(
                    net_type=loss_config.model,
                    reduction="mean",
                    normalize=True,  # 0 ~ 1 input
                )
            elif isinstance(loss_config, DISTSConfig):
                self.metrics["dists"] = StructuralSimilarityIndexMeasure(
                    reduction="elementwise_mean",
                )

    @torch.compile
    def forward(
        self,
        pred: torch.Tensor,  # may be in [-1, 1]
        target: torch.Tensor,  # may be in [-1, 1]
    ) -> dict[PerceptualLossType, torch.Tensor]:
        if self.convert_zero_to_one:
            pred = (pred + 1.0) / 2.0
            target = (target + 1.0) / 2.0

        pred = pred.clamp(0.0, 1.0)
        target = target.clamp(0.0, 1.0)

        losses: dict[PerceptualLossType, torch.Tensor] = {}

        for loss_config in self.loss_configs:
            loss_type = loss_config.type
            weight = loss_config.weight

            metric = self.metrics[loss_type]
            loss = metric(pred, target)
            losses[loss_type] = loss * weight

        return losses
