from .nnunet_2d import NNUNet2D
from .segformer_b2 import SegFormerB2Seg
from .swinunetr import SwinUNETR2D
from .swinunetr_vlm import SwinUNETR2DVLM
from .swinunetr_vlm_v1 import SwinUNETR2DVLMV1
from .transunet import TransUNetSeg
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
    elif name == "swinunetr_vlm":
        mc = cfg.get("swinunetr", {})
        img_size = mc.get("img_size") or cfg.get("image_size", 400)
        return SwinUNETR2DVLM(
            img_size=img_size,
            in_channels=in_ch,
            out_channels=out_ch,
            feature_size=mc.get("feature_size", 48),
            text_dim=cfg.get("vlm", {}).get("text_dim", 512),
            drop_rate=mc.get("drop_rate", 0.0),
            attn_drop_rate=mc.get("attn_drop_rate", 0.0),
            use_checkpoint=mc.get("use_checkpoint", False),
        )
    elif name == "swinunetr_vlm_v1":
        mc      = cfg.get("swinunetr", {})
        vlm_cfg = cfg.get("vlm", {})
        img_size = mc.get("img_size") or cfg.get("image_size", 400)
        return SwinUNETR2DVLMV1(
            img_size=img_size,
            in_channels=in_ch,
            out_channels=out_ch,
            feature_size=mc.get("feature_size", 48),
            text_dim=vlm_cfg.get("text_dim", 512),
            vlm_dim=vlm_cfg.get("vlm_dim", 512),
            vlm_mode=vlm_cfg.get("mode", "image"),
            alpha_init=vlm_cfg.get("alpha_init", 0.1),
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
    elif name in ("nnunet_2d", "nnunet"):
        mc = cfg.get("nnunet", {})
        return NNUNet2D(
            in_channels=in_ch,
            out_channels=out_ch,
            base_channels=mc.get("base_channels", 32),
            n_pool=mc.get("n_pool", 5),
            max_channels=mc.get("max_channels", 320),
            deep_supervision=mc.get("deep_supervision", False),
        )
    elif name == "segformer_b2":
        mc      = cfg.get("segformer", {})
        vlm_cfg = cfg.get("vlm", {})
        img_size = cfg.get("image_size", 384)
        return SegFormerB2Seg(
            in_channels=in_ch,
            out_channels=out_ch,
            img_size=img_size,
            pretrained=mc.get("pretrained", False),
            decoder_dim=mc.get("decoder_dim", 256),
            dropout=mc.get("dropout", 0.1),
            vlm_dim=vlm_cfg.get("vlm_dim", 512),
            text_dim=vlm_cfg.get("text_dim", 512),
        )
    elif name == "transunet":
        mc      = cfg.get("transunet", {})
        vlm_cfg = cfg.get("vlm", {})
        img_size = cfg.get("image_size", 384)
        return TransUNetSeg(
            in_channels=in_ch,
            out_channels=out_ch,
            img_size=img_size,
            timm_model=mc.get("timm_model", "vit_base_patch16_384"),
            pretrained=mc.get("pretrained", False),
            decoder_chs=mc.get("decoder_chs", [256, 128, 64, 16]),
            vlm_dim=vlm_cfg.get("vlm_dim", 512),
            text_dim=vlm_cfg.get("text_dim", 512),
        )
    else:
        raise ValueError(f"Unknown model: {name}")
