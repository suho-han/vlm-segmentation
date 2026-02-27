"""
DRIVE retinal vessel segmentation dataset.

Expected directory layout:
  <dataset_root>/
    train/
      images/  *.tif
      masks/   *.gif  (binary vessel ground truth)
    test/
      images/  *.tif
      masks/   *.gif

Note: DRIVE has no pre-made val split; we use a deterministic 20% of train as val.
Image extensions are auto-detected; the default on the public DRIVE dataset is
images=.tif and masks=.gif.
"""

import os
import random

import numpy as np
from natsort import natsorted
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from .discovery import IMAGE_EXTS, find_dataset_root
from .transforms import (
    ColorJitter,
    Compose,
    RandomCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    Resize,
    ToTensor,
)


def _build_transforms(split: str, image_size: int):
    if split == "train":
        return Compose([
            Resize(image_size),
            RandomCrop(image_size),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            ColorJitter(brightness=0.3, contrast=0.3, saturation=0.0, hue=0.0),
            ToTensor(),
        ])
    else:
        return Compose([
            Resize(image_size),
            ToTensor(),
        ])


def _detect_ext(directory: str) -> str:
    """Return most common image extension found in directory."""
    if not os.path.exists(directory):
        return ".tif"
    counts: dict[str, int] = {}
    for f in os.listdir(directory):
        ext = os.path.splitext(f)[1].lower()
        if ext in IMAGE_EXTS:
            counts[ext] = counts.get(ext, 0) + 1
    return max(counts, key=counts.get) if counts else ".tif"


class DRIVEDataset(Dataset):
    """DRIVE binary vessel segmentation dataset."""

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_size: int = 512,
        image_ext: str | None = None,  # None → auto-detect
        label_ext: str | None = None,  # None → auto-detect (.gif on official DRIVE)
        transform=None,
    ):
        self.images_dir = os.path.join(root_dir, split, "images")
        # DRIVE stores labels in 'masks/' subdirectory
        self.labels_dir = os.path.join(root_dir, split, "masks")
        self.image_size = image_size

        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Images dir not found: {self.images_dir}")
        if not os.path.exists(self.labels_dir):
            raise FileNotFoundError(f"Labels dir not found: {self.labels_dir}")

        self.image_ext = image_ext or _detect_ext(self.images_dir)
        self.label_ext = label_ext or _detect_ext(self.labels_dir)
        self.transform = transform or _build_transforms(split, image_size)

        self.image_files = natsorted([
            f for f in os.listdir(self.images_dir)
            if os.path.splitext(f)[1].lower() == self.image_ext
        ])
        if len(self.image_files) == 0:
            raise ValueError(f"No {self.image_ext} images found in {self.images_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fname = self.image_files[idx]
        stem = os.path.splitext(fname)[0]

        img = Image.open(os.path.join(self.images_dir, fname)).convert("L")

        lbl_path = os.path.join(self.labels_dir, stem + self.label_ext)
        if not os.path.exists(lbl_path):
            raise FileNotFoundError(f"Label not found: {lbl_path}")
        lbl = Image.open(lbl_path).convert("L")

        img, lbl = self.transform(img, lbl)
        lbl = (lbl > 0.5).float()
        return img, lbl


def get_drive_loaders(cfg: dict):
    data_root = cfg.get("data_root", "./data")

    # Auto-discover the dataset folder
    dataset_root = find_dataset_root(data_root, "DRIVE")
    if dataset_root is None:
        raise FileNotFoundError(
            f"Could not locate DRIVE under '{data_root}'. "
            f"See runs/_debug/ for scan report."
        )

    image_size = cfg.get("image_size", 512)
    batch_size = cfg.get("batch_size", 2)
    num_workers = cfg.get("num_workers", 4)
    pin_memory = cfg.get("pin_memory", True)
    seed = cfg.get("seed", 42)
    # Extensions: None → auto-detect per split dir
    image_ext = cfg.get("image_ext", None)
    label_ext = cfg.get("label_ext", None)

    full_train = DRIVEDataset(str(dataset_root), "train", image_size, image_ext, label_ext)
    test_ds    = DRIVEDataset(str(dataset_root), "test",  image_size, image_ext, label_ext,
                              transform=_build_transforms("test", image_size))

    # Deterministic 80/20 train/val split (logged in config snapshot)
    n = len(full_train)
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)
    split_idx = max(1, int(0.8 * n))
    train_indices = indices[:split_idx]
    val_indices   = indices[split_idx:]

    # Separate Dataset for val with val transforms
    val_base = DRIVEDataset(str(dataset_root), "train", image_size, image_ext, label_ext,
                            transform=_build_transforms("val", image_size))
    train_ds = Subset(full_train, train_indices)
    val_ds   = Subset(val_base,   val_indices)

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
