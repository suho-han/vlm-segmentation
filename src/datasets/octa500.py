"""
OCTA500 dataset loader (6M or 3M).

Expected directory layout (train/val/test split pre-created):
  <dataset_root>/
    train/
      images/  *.bmp (or auto-detected extension)
      labels/  *.bmp
    val/
      images/  *.bmp
      labels/  *.bmp
    test/
      images/  *.bmp
      labels/  *.bmp

Auto-discovery: call get_octa500_loaders(cfg) — it will search for the
dataset folder under cfg["data_root"] and auto-detect image extensions.
"""

import os

import numpy as np
from natsort import natsorted
from PIL import Image
from torch.utils.data import DataLoader, Dataset

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
            RandomCrop(image_size),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            ColorJitter(brightness=0.4, contrast=0.4, saturation=0.0, hue=0.0),
            ToTensor(),
        ])
    else:
        return Compose([
            Resize(image_size),
            ToTensor(),
        ])


def _detect_ext(directory: str) -> str:
    """Return the most common image extension in a directory."""
    if not os.path.exists(directory):
        return ".bmp"
    counts: dict[str, int] = {}
    for f in os.listdir(directory):
        ext = os.path.splitext(f)[1].lower()
        if ext in IMAGE_EXTS:
            counts[ext] = counts.get(ext, 0) + 1
    return max(counts, key=counts.get) if counts else ".bmp"


class OCTA500Dataset(Dataset):
    """OCTA500 binary vessel segmentation dataset."""

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_size: int = 400,
        image_ext: str | None = None,   # None → auto-detect
        label_ext: str | None = None,   # None → auto-detect
        transform=None,
    ):
        self.images_dir = os.path.join(root_dir, split, "images")
        self.labels_dir = os.path.join(root_dir, split, "labels")
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

        # Pad if smaller than crop size (needed for RandomCrop)
        w, h = img.size
        if w < self.image_size or h < self.image_size:
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            lbl = lbl.resize((self.image_size, self.image_size), Image.NEAREST)

        img, lbl = self.transform(img, lbl)
        lbl = (lbl > 0.5).float()
        return img, lbl


def get_octa500_loaders(cfg: dict):
    dataset_name = cfg.get("dataset", "OCTA500-6M")
    data_root = cfg.get("data_root", "./data")

    # Auto-discover the dataset folder
    dataset_root = find_dataset_root(data_root, dataset_name)
    if dataset_root is None:
        raise FileNotFoundError(
            f"Could not locate {dataset_name} under '{data_root}'. "
            f"See runs/_debug/ for scan report."
        )

    image_size = cfg.get("image_size", 400)
    batch_size = cfg.get("batch_size", 4)
    num_workers = cfg.get("num_workers", 4)
    pin_memory = cfg.get("pin_memory", True)
    # Extensions: None means auto-detect per split
    image_ext = cfg.get("image_ext", None)
    label_ext = cfg.get("label_ext", None)

    train_ds = OCTA500Dataset(str(dataset_root), "train", image_size, image_ext, label_ext)
    val_ds   = OCTA500Dataset(str(dataset_root), "val",   image_size, image_ext, label_ext)
    test_ds  = OCTA500Dataset(str(dataset_root), "test",  image_size, image_ext, label_ext)

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
