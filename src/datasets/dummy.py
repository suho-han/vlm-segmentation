"""
Dummy dataset for smoke-testing train/eval without real data.

Returns random (image, mask) pairs of the configured image_size.
Activated via --dummy flag or dataset="dummy" in config.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class DummyDataset(Dataset):
    """Generates random images + binary masks on the fly."""

    def __init__(self, n: int = 32, image_size: int = 64, in_channels: int = 1, seed: int = 42):
        self.n = n
        self.image_size = image_size
        self.in_channels = in_channels
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        size = self.image_size
        img = torch.from_numpy(
            self.rng.random((self.in_channels, size, size), dtype=np.float32)
        )
        mask = torch.from_numpy(
            (self.rng.random((1, size, size)) > 0.5).astype(np.float32)
        )
        return img, mask


def get_dummy_loaders(cfg: dict):
    image_size = cfg.get("image_size", 64)
    in_channels = cfg.get("in_channels", 1)
    batch_size = cfg.get("batch_size", 4)
    num_workers = cfg.get("num_workers", 0)  # 0 for dummy — avoids fork overhead
    seed = cfg.get("seed", 42)

    train_ds = DummyDataset(n=32, image_size=image_size, in_channels=in_channels, seed=seed)
    val_ds   = DummyDataset(n=8,  image_size=image_size, in_channels=in_channels, seed=seed + 1)
    test_ds  = DummyDataset(n=8,  image_size=image_size, in_channels=in_channels, seed=seed + 2)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader
