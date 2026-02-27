import torch
import torch.nn as nn

from .dice import DiceLoss


class BCELoss(nn.Module):
    """Binary cross-entropy with logits."""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce(logits, targets)


class BCEDiceLoss(nn.Module):
    """
    Weighted sum of BCE and Dice losses.
    total = bce_weight * BCE + dice_weight * Dice
    """

    def __init__(
        self,
        smooth: float = 1.0,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
    ):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.bce_weight * self.bce(logits, targets)
            + self.dice_weight * self.dice(logits, targets)
        )
