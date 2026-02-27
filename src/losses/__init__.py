from .dice import DiceLoss
from .bce import BCELoss, BCEDiceLoss


def build_loss(cfg: dict):
    name = cfg.get("loss", "bce_dice").lower()
    smooth = cfg.get("dice_smooth", 1.0)
    bce_w = cfg.get("bce_weight", 1.0)
    dice_w = cfg.get("dice_weight", 1.0)

    if name == "dice":
        return DiceLoss(smooth=smooth)
    elif name == "bce":
        return BCELoss()
    elif name == "bce_dice":
        return BCEDiceLoss(smooth=smooth, bce_weight=bce_w, dice_weight=dice_w)
    else:
        raise ValueError(f"Unknown loss: {name}")
