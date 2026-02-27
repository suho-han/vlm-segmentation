"""
Dice coefficient and IoU (Jaccard) for binary segmentation.

All functions operate on numpy arrays (H, W) or (N, H, W) with values in {0, 1}.
"""

import numpy as np


def dice_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Dice coefficient: 2 * |P ∩ T| / (|P| + |T|).

    Args:
        pred:   Binary prediction array, shape (H, W) or (N, H, W).
        target: Binary ground-truth array, same shape as pred.
        smooth: Laplace smoothing to avoid 0/0.

    Returns:
        Scalar Dice score in [0, 1].
    """
    pred = pred.astype(bool).ravel()
    target = target.astype(bool).ravel()
    intersection = (pred & target).sum()
    return float((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))


def iou_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Intersection over Union (Jaccard index).

    Args:
        pred:   Binary prediction array.
        target: Binary ground-truth array.
        smooth: Laplace smoothing.

    Returns:
        Scalar IoU score in [0, 1].
    """
    pred = pred.astype(bool).ravel()
    target = target.astype(bool).ravel()
    intersection = (pred & target).sum()
    union = (pred | target).sum()
    return float((intersection + smooth) / (union + smooth))


def compute_metrics(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5):
    """
    Compute a standard set of binary segmentation metrics.

    Args:
        pred:      Probability/logit array (float) or binary array.
        target:    Binary ground-truth array.
        threshold: Binarisation threshold (applied if pred is float).

    Returns:
        dict with keys: dice, iou, precision, recall, f1, accuracy
    """
    if pred.dtype != bool and pred.max() > 1.0:
        # logits → sigmoid → threshold
        pred = 1.0 / (1.0 + np.exp(-pred))
    binary_pred = (pred >= threshold).astype(bool).ravel()
    binary_tgt = target.astype(bool).ravel()

    tp = (binary_pred & binary_tgt).sum()
    fp = (binary_pred & ~binary_tgt).sum()
    fn = (~binary_pred & binary_tgt).sum()
    tn = (~binary_pred & ~binary_tgt).sum()

    eps = 1e-6
    precision = float((tp + eps) / (tp + fp + eps))
    recall = float((tp + eps) / (tp + fn + eps))
    f1 = float(2 * precision * recall / (precision + recall + eps))
    accuracy = float((tp + tn) / (tp + fp + fn + tn + eps))

    dice = dice_score(binary_pred.reshape(pred.shape), binary_tgt.reshape(target.shape))
    iou = iou_score(binary_pred.reshape(pred.shape), binary_tgt.reshape(target.shape))

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }
