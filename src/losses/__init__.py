import torch.nn as nn
from .dice import DiceLoss
from .bce import BCELoss, BCEDiceLoss
from .topology import SoftclDiceLoss


class TopologyAwareLoss(nn.Module):
    """
    Combined loss: alpha * (BCE + Dice) + beta * SoftclDice.
    Used for topology-preserving vessel segmentation.
    """
    def __init__(
        self,
        smooth: float = 1.0,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        cldice_weight: float = 1.0,
        iters: int = 3,
    ):
        super().__init__()
        self.bce_dice = BCEDiceLoss(smooth=smooth, bce_weight=bce_weight, dice_weight=dice_weight)
        self.cldice = SoftclDiceLoss(iters=iters, smooth=smooth)
        self.cldice_weight = cldice_weight

    def forward(self, logits, targets):
        return self.bce_dice(logits, targets) + self.cldice_weight * self.cldice(logits, targets)


def build_loss(cfg: dict):
    name = cfg.get("loss", "bce_dice").lower()
    smooth = cfg.get("dice_smooth", 1.0)
    bce_w = cfg.get("bce_weight", 1.0)
    dice_w = cfg.get("dice_weight", 1.0)
    cldice_w = cfg.get("cldice_weight", 1.0)
    iters = cfg.get("cldice_iters", 3)

    if name == "dice":
        return DiceLoss(smooth=smooth)
    elif name == "bce":
        return BCELoss()
    elif name == "bce_dice":
        return BCEDiceLoss(smooth=smooth, bce_weight=bce_w, dice_weight=dice_w)
    elif name == "cldice":
        return SoftclDiceLoss(iters=iters, smooth=smooth)
    elif name == "topology_aware" or name == "cldice_bce_dice":
        return TopologyAwareLoss(
            smooth=smooth,
            bce_weight=bce_w,
            dice_weight=dice_w,
            cldice_weight=cldice_w,
            iters=iters
        )
    else:
        raise ValueError(f"Unknown loss: {name}")
