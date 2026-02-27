"""
VLM text-embedding prior — BiomedCLIP with CLIP ViT-B/16 fallback.

Returns a frozen, normalised 512-dim text embedding for a fixed prompt.
The embedding is computed once and cached on the target device.

Usage:
    prior = VLMPrior(dataset="OCTA500-6M", device=device)
    text_embed = prior.get_text_embed()  # (1, 512) on device
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

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
            self._clip_model  = model
            self._tokenizer   = tokenizer
            self.backbone_name = "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            logger.info(f"VLMPrior backbone: {self.backbone_name}")
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
        self._clip_model  = model
        self._tokenizer   = tokenizer
        self.backbone_name = f"CLIP-{model_name}-{pretrained}"
        logger.info(f"VLMPrior backbone: {self.backbone_name}")

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

        tokens = self._tokenizer([self.prompt])  # (1, seq_len) int tensor
        # Encode on CPU, move to device afterwards
        feats = self._clip_model.encode_text(tokens)      # (1, D)
        feats = F.normalize(feats, dim=-1)                # L2 normalise
        self._embed = feats.to(self.device)
        logger.info(
            f"VLMPrior embed cached | shape={self._embed.shape} "
            f"| prompt='{self.prompt}'"
        )
        return self._embed
