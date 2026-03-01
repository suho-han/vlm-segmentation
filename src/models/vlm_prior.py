"""
VLM prior — BiomedCLIP with CLIP ViT-B/16 fallback.

Provides two types of feature extraction from a frozen VLM:

1. Text embedding (V0):
    prior.get_text_embed()  →  (1, 512)  — L2-normalised, cached after first call

2. Image spatial features (V1):
    prior.get_image_features(x)  →  (B, 512, 14, 14)
    Input: (B, C, H, W) in [0, 1] range (grayscale or RGB).
    Internally: resize to 224x224, expand to 3ch, CLIP-normalise, run frozen ViT,
    extract patch tokens, reshape to spatial grid.

Usage:
    prior = VLMPrior(dataset="OCTA500-6M", device=device)
    text_embed = prior.get_text_embed()        # (1, 512) on device
    img_feat   = prior.get_image_features(x)  # (B, 512, 14, 14) on device
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# CLIP preprocessing constants (BiomedCLIP uses standard CLIP normalisation)
_CLIP_MEAN = (0.48145466, 0.4578275,  0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

# Fixed prompts per dataset
PROMPTS: dict[str, str] = {
    "OCTA500-6M": "retinal vessel segmentation in OCTA fundus image",
    "DRIVE":      "retinal blood vessel segmentation in fundus photograph",
}

# BiomedCLIP model hub ID (open_clip hf-hub format)
_BIOMEDCLIP_HUB = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
_CLIP_FALLBACK  = ("ViT-B-16", "openai")


class VLMPrior:
    """
    Frozen VLM text encoder.

    Tries BiomedCLIP first; falls back to CLIP ViT-B/16 if loading fails.
    The embedding is computed once and cached on `device`.

    Args:
        dataset:  Key for PROMPTS dict (e.g. "OCTA500-6M", "DRIVE").
        device:   torch.device where the cached embed will live.
        prompt:   Override the fixed prompt (optional).
    """

    def __init__(
        self,
        dataset: str,
        device: torch.device,
        prompt: Optional[str] = None,
    ):
        self.device = device
        self.prompt = prompt or PROMPTS.get(dataset, PROMPTS["OCTA500-6M"])
        self.backbone_name: str = "unknown"
        self._embed: Optional[torch.Tensor] = None  # lazy cache

        self._clip_model = None
        self._tokenizer  = None
        self._vlm_patch_size: int = 16   # ViT-B/16 default
        self._vlm_grid_size:  int = 14   # 224 / 16
        self._load()

    # ------------------------------------------------------------------
    # Internal loader
    # ------------------------------------------------------------------

    def _load(self) -> None:
        import open_clip  # imported here so module-level import not required

        # ── Try BiomedCLIP ───────────────────────────────────────────
        try:
            logger.info(f"Loading BiomedCLIP from {_BIOMEDCLIP_HUB} …")
            model, _, _ = open_clip.create_model_and_transforms(_BIOMEDCLIP_HUB)
            tokenizer   = open_clip.get_tokenizer(_BIOMEDCLIP_HUB)
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            self._clip_model  = model.to(self.device)
            self._tokenizer   = tokenizer
            self.backbone_name = "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            logger.info(f"VLMPrior backbone: {self.backbone_name}")
            self._detect_grid_size()
            return
        except Exception as e:
            logger.warning(f"BiomedCLIP load failed ({e}); falling back to CLIP ViT-B/16")

        # ── Fallback: CLIP ViT-B/16 ──────────────────────────────────
        model_name, pretrained = _CLIP_FALLBACK
        logger.info(f"Loading CLIP {model_name} ({pretrained}) …")
        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self._clip_model  = model.to(self.device)
        self._tokenizer   = tokenizer
        self.backbone_name = f"CLIP-{model_name}-{pretrained}"
        logger.info(f"VLMPrior backbone: {self.backbone_name}")
        self._detect_grid_size()

    def _detect_grid_size(self) -> None:
        """Infer patch size and grid size from the visual encoder's patch embedding."""
        try:
            visual = self._clip_model.visual
            if hasattr(visual, "conv1"):
                # Standard open_clip VisionTransformer
                ps = visual.conv1.kernel_size
            elif hasattr(visual, "trunk") and hasattr(visual.trunk, "patch_embed"):
                # TimmModel (BiomedCLIP)
                ps = visual.trunk.patch_embed.proj.kernel_size
            else:
                ps = (16, 16)  # default ViT-B/16
            self._vlm_patch_size = ps[0] if isinstance(ps, tuple) else int(ps)
            self._vlm_grid_size  = 224 // self._vlm_patch_size
            logger.info(
                f"VLMPrior grid: patch_size={self._vlm_patch_size}, "
                f"grid={self._vlm_grid_size}x{self._vlm_grid_size}"
            )
        except Exception as e:
            logger.warning(f"Could not detect grid size ({e}); using defaults 16/14")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_text_embed(self) -> torch.Tensor:
        """
        Returns the cached text embedding tensor of shape (1, 512).

        The embedding is L2-normalised and lives on self.device.
        Computed once on the first call; subsequent calls return the cached value.
        """
        if self._embed is not None:
            return self._embed

        tokens = self._tokenizer([self.prompt]).to(self.device)  # (1, seq_len)
        feats = self._clip_model.encode_text(tokens)             # (1, D)
        feats = F.normalize(feats, dim=-1)                       # L2 normalise
        self._embed = feats
        logger.info(
            f"VLMPrior text embed cached | shape={self._embed.shape} "
            f"| prompt='{self.prompt}'"
        )
        return self._embed

    # ------------------------------------------------------------------
    # Image spatial feature extraction (V1)
    # ------------------------------------------------------------------

    def _preprocess_for_vlm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prepare image tensor for CLIP/BiomedCLIP image encoder.

        Args:
            x: (B, C, H, W) in [0, 1] range on any device.
        Returns:
            (B, 3, 224, 224) float32 tensor on self.device, CLIP-normalised.
        """
        x = x.to(self.device)
        # Resize to 224x224 if needed
        if x.shape[-2:] != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        # Expand single-channel to RGB by replication
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        elif x.shape[1] != 3:
            x = x[:, :3]  # take first 3 channels if more than 3
        # CLIP normalisation
        mean = x.new_tensor(_CLIP_MEAN).view(1, 3, 1, 1)
        std  = x.new_tensor(_CLIP_STD).view(1, 3, 1, 1)
        return (x - mean) / std

    @torch.no_grad()
    def get_image_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract per-image spatial patch features from the frozen ViT image encoder.

        Dispatches to the appropriate backend:
          - TimmModel  (BiomedCLIP): uses trunk.forward_features()
          - open_clip VisionTransformer: manually extracts patch tokens

        Args:
            x: (B, C, H, W) image tensor in [0, 1] range.
        Returns:
            (B, D, grid, grid) spatial feature map on self.device
            where D=512, grid=14 for ViT-B/16.
        """
        x = self._preprocess_for_vlm(x)     # (B, 3, 224, 224) on self.device
        visual = self._clip_model.visual

        if hasattr(visual, "trunk"):
            # ── TimmModel (BiomedCLIP) ────────────────────────────────
            return self._image_features_timm(x, visual)
        else:
            # ── Standard open_clip VisionTransformer ──────────────────
            return self._image_features_openclip(x, visual)

    def _image_features_timm(
        self, x: torch.Tensor, visual
    ) -> torch.Tensor:
        """
        Extract patch features from a TimmModel wrapper (BiomedCLIP).

        trunk.forward_features() returns (B, N+1, hidden_dim) with CLS at [0].
        visual.head is Sequential(drop, Linear(hidden_dim → out_dim)).
        """
        trunk = visual.trunk
        gh = gw = self._vlm_grid_size

        # All tokens: (B, 197, 768) — CLS at position 0
        all_tokens = trunk.forward_features(x)     # (B, N+1, hidden)
        B, N_total, hidden = all_tokens.shape
        patches = all_tokens[:, 1:]                # (B, N, hidden)  N=196

        # Apply projection head element-wise (works on any leading dims)
        head = getattr(visual, "head", None)
        if head is not None:
            # Sequential(drop, Linear) — Linear works on (..., in_features)
            patches = head(patches)                # (B, N, out_dim=512)

        # Reshape to spatial grid
        D = patches.shape[-1]
        patches = patches.permute(0, 2, 1).reshape(B, D, gh, gw)
        logger.debug(f"VLMPrior image features (timm) | shape={patches.shape}")
        return patches

    def _image_features_openclip(
        self, x: torch.Tensor, visual
    ) -> torch.Tensor:
        """
        Extract patch features from a standard open_clip VisionTransformer.

        Manually replicates the forward pass to capture patch tokens before
        global pooling.
        """
        B  = x.shape[0]
        gh = gw = self._vlm_grid_size

        # ── Patch embedding ──────────────────────────────────────────
        x = visual.conv1(x)                            # (B, width, gh, gw)
        C_vit = x.shape[1]
        x = x.reshape(B, C_vit, -1).permute(0, 2, 1)  # (B, N, width)

        # ── CLS token + positional embedding ─────────────────────────
        cls_tok = visual.class_embedding.to(x.dtype)
        cls_tok = cls_tok + torch.zeros(B, 1, C_vit, dtype=x.dtype, device=x.device)
        x = torch.cat([cls_tok, x], dim=1)             # (B, N+1, width)
        x = x + visual.positional_embedding.to(x.dtype)

        # ── Pre-norm + patch dropout (identity in eval) ───────────────
        if hasattr(visual, "patch_dropout"):
            x = visual.patch_dropout(x)
        x = visual.ln_pre(x)

        # ── Transformer (open_clip passes in LND format) ──────────────
        x = x.permute(1, 0, 2)      # NLD → LND
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)      # LND → NLD  (B, N+1, width)

        # ── Extract patch tokens (skip CLS) + post-norm ───────────────
        patches = x[:, 1:]           # (B, N, width)
        patches = visual.ln_post(patches)

        # ── Project to output dim ─────────────────────────────────────
        if visual.proj is not None:
            patches = patches @ visual.proj   # (B, N, out_dim)

        # ── Reshape to spatial grid ───────────────────────────────────
        D = patches.shape[-1]
        patches = patches.permute(0, 2, 1).reshape(B, D, gh, gw)
        logger.debug(f"VLMPrior image features (openclip) | shape={patches.shape}")
        return patches
