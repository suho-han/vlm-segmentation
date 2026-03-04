"""
Paired image-mask transforms (PIL-based, ported from JiT/util/transforms.py).
All transforms receive (img: PIL.Image, mask: PIL.Image) and return the same.
"""

import collections
import numbers
import random

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

Sequence = collections.abc.Sequence
Iterable = collections.abc.Iterable


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class ToTensor:
    def __call__(self, img, mask):
        img = torch.from_numpy(np.array(img)).float()
        mask = torch.from_numpy(np.array(mask)).float()

        if img.ndim == 2:
            img = img.unsqueeze(0)
        elif img.ndim == 3:
            img = img.permute(2, 0, 1)

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        elif mask.ndim == 3:
            mask = mask.permute(2, 0, 1)

        img = img / 255.0
        mask = mask / 255.0
        return img, mask


class Resize:
    def __init__(self, size):
        if isinstance(size, int):
            size = (size, size)
        self.size = size  # (H, W)

    def __call__(self, img, mask):
        img = F.resize(img, self.size)
        mask = F.resize(mask, self.size, interpolation=Image.NEAREST)
        return img, mask


class RandomCrop:
    def __init__(self, size):
        if isinstance(size, int):
            size = (size, size)
        self.size = size

    def __call__(self, img, mask):
        i, j, h, w = self._get_params(img)
        img = F.crop(img, i, j, h, w)
        mask = F.crop(mask, i, j, h, w)
        return img, mask

    def _get_params(self, img):
        w, h = img.size
        th, tw = self.size
        if w < tw or h < th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            img = F.hflip(img)
            mask = F.hflip(mask)
        return img, mask


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            img = F.vflip(img)
            mask = F.vflip(mask)
        return img, mask


class RandomRotation:
    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            degrees = (-degrees, degrees)
        self.degrees = degrees

    def __call__(self, img, mask):
        angle = random.uniform(*self.degrees)
        img = F.rotate(img, angle)
        mask = F.rotate(mask, angle)
        return img, mask


class ColorJitter:
    """Only applied to the image, not the mask."""
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, img, mask):
        if self.brightness > 0:
            factor = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            img = F.adjust_brightness(img, factor)
        if self.contrast > 0:
            factor = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            img = F.adjust_contrast(img, factor)
        if self.saturation > 0 and img.mode != 'L':
            factor = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
            img = F.adjust_saturation(img, factor)
        if self.hue > 0 and img.mode != 'L':
            factor = random.uniform(-self.hue, self.hue)
            img = F.adjust_hue(img, factor)
        return img, mask


class Normalize:
    """Normalise tensor image (not mask). Applied after ToTensor."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, mask):
        # Handle grayscale images by taking the mean of the stats if necessary
        c = img.shape[0]
        m, s = self.mean, self.std
        if c == 1 and len(m) == 3:
            m = [sum(m) / 3.0]
            s = [sum(s) / 3.0]
        
        img = F.normalize(img, m, s)
        return img, mask
