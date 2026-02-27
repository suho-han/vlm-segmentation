"""
Hausdorff Distance 95th percentile (HD95).

Uses scipy.ndimage for distance transform.
Returns np.inf if either mask is empty.
"""

import numpy as np

try:
    from scipy.ndimage import distance_transform_edt
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


def hd95(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute the 95th-percentile Hausdorff Distance between two binary masks.

    Args:
        pred:   Binary prediction array (H, W), dtype bool or uint8.
        target: Binary ground-truth array (H, W).

    Returns:
        HD95 in pixels. Returns np.inf if either mask is empty.
    """
    if not _SCIPY_AVAILABLE:
        raise ImportError("scipy is required for hd95. Install with: pip install scipy")

    pred = pred.astype(bool)
    target = target.astype(bool)

    if not pred.any() or not target.any():
        return float("inf")

    # Surface voxels
    pred_border = pred ^ _erode(pred)
    target_border = target ^ _erode(target)

    # Distance from each surface voxel to the other surface
    dist_pred_to_tgt = distance_transform_edt(~target_border)[pred_border]
    dist_tgt_to_pred = distance_transform_edt(~pred_border)[target_border]

    all_distances = np.concatenate([dist_pred_to_tgt, dist_tgt_to_pred])
    return float(np.percentile(all_distances, 95))


def _erode(mask: np.ndarray) -> np.ndarray:
    """Simple 4-connected binary erosion without scipy.ndimage.binary_erosion."""
    eroded = mask.copy()
    eroded[1:, :] &= mask[:-1, :]
    eroded[:-1, :] &= mask[1:, :]
    eroded[:, 1:] &= mask[:, :-1]
    eroded[:, :-1] &= mask[:, 1:]
    return eroded
