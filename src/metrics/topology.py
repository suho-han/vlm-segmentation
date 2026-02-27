"""
Topology metrics: Betti number error (β0, β1).

β0 = number of connected components.
β1 = number of holes (cycles) in 2D.

Uses scikit-image label/regionprops for component counting.
Full persistent homology (gudhi/ripser) is NOT required; this is a lightweight proxy.

Interface is fixed so higher-quality implementations can be dropped in later.
"""

import numpy as np

try:
    from skimage.measure import label as sk_label
    _SKIMAGE_AVAILABLE = True
except ImportError:
    _SKIMAGE_AVAILABLE = False


def _count_components(mask: np.ndarray) -> int:
    """Count connected components (β0) using 8-connectivity."""
    if not _SKIMAGE_AVAILABLE:
        raise ImportError(
            "scikit-image is required for topology metrics. "
            "Install with: pip install scikit-image"
        )
    if not mask.any():
        return 0
    labeled = sk_label(mask.astype(bool), connectivity=2)
    return int(labeled.max())


def _count_holes(mask: np.ndarray) -> int:
    """
    Count holes (β1) in 2D binary mask using the Euler characteristic proxy.
    β1 = β0(complement) - 1  (subtracting the infinite background component)
    This is an approximation; for exact β1 use persistent homology.
    """
    if not _SKIMAGE_AVAILABLE:
        raise ImportError(
            "scikit-image is required for topology metrics."
        )
    complement = ~mask.astype(bool)
    if not complement.any():
        return 0
    labeled = sk_label(complement, connectivity=1)
    n_complement_components = int(labeled.max())
    # Background is 1 component; the rest are holes
    return max(0, n_complement_components - 1)


def betti_error(pred: np.ndarray, target: np.ndarray) -> dict:
    """
    Compute Betti number errors between prediction and ground truth.

    Args:
        pred:   Binary prediction array (H, W).
        target: Binary ground-truth array (H, W).

    Returns:
        dict with keys:
            b0_pred, b0_target, b0_error   (connected components)
            b1_pred, b1_target, b1_error   (holes)
    """
    pred = pred.astype(bool)
    target = target.astype(bool)

    b0_pred = _count_components(pred)
    b0_tgt = _count_components(target)
    b1_pred = _count_holes(pred)
    b1_tgt = _count_holes(target)

    return {
        "b0_pred": b0_pred,
        "b0_target": b0_tgt,
        "b0_error": abs(b0_pred - b0_tgt),
        "b1_pred": b1_pred,
        "b1_target": b1_tgt,
        "b1_error": abs(b1_pred - b1_tgt),
    }
