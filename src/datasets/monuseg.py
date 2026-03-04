"""
MoNuSeg dataset loader.

Directory layout expected under <root>/:
    train/
        images/  TCGA-*.tif   (RGB, 1000×1000)
        labels/  TCGA-*.npy   (uint8, shape (H,W,1), values {0,255})
    test/
        images/  TCGA-*.tif
        labels/  TCGA-*.npy

Split strategy: 37 train images → deterministic 80/20 train/val split (seed from cfg).
Test set is fixed (14 images).
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from .transforms import (
    ColorJitter, Compose, Normalize, RandomHorizontalFlip, RandomRotation,
    RandomVerticalFlip, Resize, ToTensor,
)

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]
from .discovery import find_dataset_root


class MoNuSegDataset(Dataset):
    """
    MoNuSeg nuclei segmentation dataset.

    Args:
        root_dir:   Path to the dataset root (contains train/ and test/ subdirs).
        split:      One of "train", "val", "test".  "val" shares the train/ folder;
                    the caller is responsible for passing the correct subset indices.
        image_size: Square resize / crop size for model input.
        transform:  Paired (img, mask) transform.  If None, a default is built.
        indices:    Optional list of indices to restrict this Dataset to (used to
                    implement train/val split without duplicating the file list).
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_size: int = 256,
        transform=None,
        indices: list[int] | None = None,
    ):
        self.root = Path(root_dir)
        self.image_size = image_size
        folder = "test" if split == "test" else "train"
        self.images_dir = self.root / folder / "images"
        self.labels_dir = self.root / folder / "labels"

        pairs = _collect_pairs(self.images_dir, self.labels_dir)
        if not pairs:
            raise FileNotFoundError(
                f"No image/label pairs found under {self.images_dir} / {self.labels_dir}"
            )
        pairs.sort(key=lambda p: p[0].name)

        if indices is not None:
            pairs = [pairs[i] for i in indices]

        self.pairs = pairs
        self.transform = transform or _build_transforms(split, image_size)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, lbl_path = self.pairs[idx]

        img = Image.open(img_path).convert("RGB")

        # Labels are stored as uint8 numpy arrays, shape (H, W, 1), values {0, 255}
        lbl_arr = np.load(lbl_path)
        lbl_arr = lbl_arr.squeeze()                          # (H, W)
        lbl_arr = (lbl_arr > 127).astype(np.uint8) * 255    # normalise to {0,255}
        lbl = Image.fromarray(lbl_arr, mode="L")

        img, lbl = self.transform(img, lbl)
        lbl = (lbl > 0.5).float()
        return img, lbl


# ── helpers ───────────────────────────────────────────────────────────────────

def _collect_pairs(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    """Match .tif images to .npy labels by stem."""
    pairs = []
    for img_path in sorted(images_dir.glob("*.tif")):
        lbl_path = labels_dir / (img_path.stem + ".npy")
        if lbl_path.exists():
            pairs.append((img_path, lbl_path))
    return pairs


def _build_transforms(split: str, image_size: int):
    if split == "train":
        return Compose([
            Resize(image_size),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomRotation(90),
            # Histology: colour augmentation on all channels including saturation/hue
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

def get_monuseg_loaders(cfg: dict):
    """Return (train_loader, val_loader, test_loader) for MoNuSeg."""
    data_root = cfg.get("data_root", "./data")
    dataset_root = find_dataset_root(data_root, "MoNuSeg")
    if dataset_root is None:
        raise FileNotFoundError(
            "MoNuSeg dataset not found. "
            "Expected structure: <data_root>/MoNuSeg/{train,test}/{images,labels}/"
        )

    image_size  = cfg.get("image_size", 256)
    batch_size  = cfg.get("batch_size", 8)
    num_workers = cfg.get("num_workers", 4)
    pin_memory  = cfg.get("pin_memory", True)
    seed        = cfg.get("seed", 42)

    # Build full train dataset to get the file list
    full_train = MoNuSegDataset(str(dataset_root), split="train",
                                image_size=image_size,
                                transform=_build_transforms("train", image_size))
    n = len(full_train)
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)
    split_idx = max(1, int(0.8 * n))
    train_indices = indices[:split_idx]
    val_indices   = indices[split_idx:]

    # Separate dataset with val transforms for the val subset
    val_base = MoNuSegDataset(str(dataset_root), split="train",
                              image_size=image_size,
                              transform=_build_transforms("val", image_size))

    train_ds = Subset(full_train, train_indices)
    val_ds   = Subset(val_base,   val_indices)
    test_ds  = MoNuSegDataset(str(dataset_root), split="test",
                              image_size=image_size,
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
