from .octa500 import OCTA500Dataset, get_octa500_loaders
from .drive import DRIVEDataset, get_drive_loaders
from .dummy import DummyDataset, get_dummy_loaders
from .monuseg import MoNuSegDataset, get_monuseg_loaders
from .isic2018 import ISIC2018Dataset, get_isic2018_loaders


def get_loaders(cfg):
    """Return (train_loader, val_loader, test_loader) for the configured dataset.

    Special values:
        dataset="dummy"  — random tensors, no files required (for smoke tests)
    """
    name = cfg.get("dataset", "OCTA500-6M")
    if name == "dummy":
        return get_dummy_loaders(cfg)
    elif name in ("OCTA500-6M", "OCTA500-3M"):
        return get_octa500_loaders(cfg)
    elif name == "DRIVE":
        return get_drive_loaders(cfg)
    elif name == "MoNuSeg":
        return get_monuseg_loaders(cfg)
    elif name == "ISIC2018":
        return get_isic2018_loaders(cfg)
    else:
        raise ValueError(
            f"Unknown dataset: {name!r}. "
            "Valid: dummy, OCTA500-6M, OCTA500-3M, DRIVE, MoNuSeg, ISIC2018"
        )
