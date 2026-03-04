import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_erosion(img):
    if len(img.shape) == 4:
        p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
        return torch.min(p1, p2)
    elif len(img.shape) == 5:
        p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
        p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
        p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
        return torch.min(torch.min(p1, p2), p3)
    return img


def soft_dilation(img):
    if len(img.shape) == 4:
        p1 = F.max_pool2d(img, (3, 1), (1, 1), (1, 0))
        p2 = F.max_pool2d(img, (1, 3), (1, 1), (0, 1))
        return torch.max(p1, p2)
    elif len(img.shape) == 5:
        p1 = F.max_pool3d(img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
        p2 = F.max_pool3d(img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
        p3 = F.max_pool3d(img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
        return torch.max(torch.max(p1, p2), p3)
    return img


def soft_open(img):
    return soft_dilation(soft_erosion(img))


def soft_skel(img, iters=3):
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = soft_erosion(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


class SoftclDiceLoss(nn.Module):
    """
    Soft clDice loss: Centerline Dice loss for topology preservation.
    Paper: clDice - A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation.
    """
    def __init__(self, iters=3, smooth=1.0):
        super(SoftclDiceLoss, self).__init__()
        self.iters = iters
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: (B, C, H, W) or (B, C, D, H, W) - Logits or Sigmoid. 
                    If values are outside [0, 1], sigmoid will be applied.
            y_true: (B, C, H, W) or (B, C, D, H, W) - Ground truth (0 or 1).
        """
        if y_pred.min() < 0 or y_pred.max() > 1:
            y_pred = torch.sigmoid(y_pred)

        skel_pred = soft_skel(y_pred, self.iters)
        skel_true = soft_skel(y_true, self.iters)

        tprec = (torch.sum(torch.multiply(skel_pred, y_true)) + self.smooth) / \
                (torch.sum(skel_pred) + self.smooth)
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)) + self.smooth) / \
                (torch.sum(skel_true) + self.smooth)
        
        cl_dice = 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)
        return cl_dice
