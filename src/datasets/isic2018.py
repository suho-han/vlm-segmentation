"""
ISIC 2018 skin lesion segmentation dataset loader.

Directory layout expected under <root>/:
    train/
        images/  ISIC_XXXXXXX.jpg          (RGB, variable size)
        labels/  ISIC_XXXXXXX_segmentation.png  (L mode, values {0,255})
    val/
        images/  ISIC_XXXXXXX.jpg
        labels/  ISIC_XXXXXXX_segmentation.png
    test/
        images/  ISIC_XXXXXXX.jpg
        labels/  ISIC_XXXXXXX_segmentation.png  (ground truth)

Splits are pre-defined by the dataset organisers (2594 / 100 / 1000).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .transforms import (
    ColorJitter, Compose, Normalize, RandomHorizontalFlip, RandomVerticalFlip,
    Resize, ToTensor,
)

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]
from .discovery import find_dataset_root


class ISIC2018Dataset(Dataset):
    """
    ISIC 2018 Task 1 lesion segmentation dataset.

    Args:
        root_dir:   Path to the dataset root (contains train/val/test subdirs).
        split:      One of "train", "val", "test".
        image_size: Square resize size for model input.
        transform:  Paired (img, mask) transform.  If None a default is built.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_size: int = 256,
        transform=None,
    ):
        self.root = Path(root_dir)
        self.image_size = image_size
        self.images_dir = self.root / split / "images"
        self.labels_dir = self.root / split / "labels"

        self.pairs = _collect_pairs(self.images_dir, self.labels_dir)
        if not self.pairs:
            raise FileNotFoundError(
                f"No image/label pairs found under {self.images_dir} / {self.labels_dir}"
            )
        self.pairs.sort(key=lambda p: p[0].name)
        self.transform = transform or _build_transforms(split, image_size)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, lbl_path = self.pairs[idx]
        img = Image.open(img_path).convert("RGB")
        lbl = Image.open(lbl_path).convert("L")   # already binary {0,255}
        img, lbl = self.transform(img, lbl)
        lbl = (lbl > 0.5).float()
        return img, lbl


# ── helpers ───────────────────────────────────────────────────────────────────

def _collect_pairs(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    """Match .jpg images to _segmentation.png labels by stem."""
    pairs = []
    for img_path in sorted(images_dir.glob("*.jpg")):
        lbl_path = labels_dir / (img_path.stem + "_segmentation.png")
        if lbl_path.exists():
            pairs.append((img_path, lbl_path))
    return pairs


def _build_transforms(split: str, image_size: int):
    if split == "train":
        return Compose([
            Resize(image_size),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            ToTensor(),
            Normalize(_MEAN, _STD),
        ])
    else:
        return Compose([
            ToTensor(),
            Normalize(_MEAN, _STD),
        ])


# ── loader factory ────────────────────────────────────────────────────────────

def get_isic2018_loaders(cfg: dict):
    """Return (train_loader, val_loader, test_loader) for ISIC2018."""
    data_root = cfg.get("data_root", "./data")
    dataset_root = find_dataset_root(data_root, "ISIC2018")
    if dataset_root is None:
        raise FileNotFoundError(
            "ISIC2018 dataset not found. "
            "Expected structure: <data_root>/ISIC2018/{train,val,test}/{images,labels}/"
        )

    image_size  = cfg.get("image_size", 256)
    batch_size  = cfg.get("batch_size", 8)
    num_workers = cfg.get("num_workers", 4)
    pin_memory  = cfg.get("pin_memory", True)

    train_ds = ISIC2018Dataset(str(dataset_root), "train", image_size)
    val_ds   = ISIC2018Dataset(str(dataset_root), "val",   image_size,
                               transform=_build_transforms("val", image_size))
    test_ds  = ISIC2018Dataset(str(dataset_root), "test",  image_size,
                               transform=_build_transforms("test", image_size))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader
