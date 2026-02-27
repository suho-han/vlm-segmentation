from .swinunetr import SwinUNETR2D
from .unetpp import UNetPlusPlus


def build_model(cfg: dict):
    name = cfg.get("model", "swinunetr").lower()
    in_ch = cfg.get("in_channels", 1)
    out_ch = cfg.get("out_channels", 1)

    if name == "swinunetr":
        mc = cfg.get("swinunetr", {})
        # img_size: swinunetr.img_size > top-level image_size > 400
        img_size = mc.get("img_size") or cfg.get("image_size", 400)
        return SwinUNETR2D(
            img_size=img_size,
            in_channels=in_ch,
            out_channels=out_ch,
            feature_size=mc.get("feature_size", 48),
            drop_rate=mc.get("drop_rate", 0.0),
            attn_drop_rate=mc.get("attn_drop_rate", 0.0),
            use_checkpoint=mc.get("use_checkpoint", False),
        )
    elif name in ("unetpp", "unet++"):
        mc = cfg.get("unetpp", {})
        return UNetPlusPlus(
            in_channels=in_ch,
            out_channels=out_ch,
            base_channels=mc.get("base_channels", 64),
            depth=mc.get("depth", 4),
            deep_supervision=mc.get("deep_supervision", False),
        )
    else:
        raise ValueError(f"Unknown model: {name}")
