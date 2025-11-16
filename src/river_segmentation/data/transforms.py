"""Data augmentation and preprocessing transforms."""

import numpy as np
import torch
from typing import Tuple


class Compose:
    """Compose multiple transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for transform in self.transforms:
            image, mask = transform(image, mask)
        return image, mask


class ToTensor:
    """Convert numpy arrays to PyTorch tensors."""

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        # Ensure channel-first format (C, H, W)
        if image.ndim == 2:
            image = image[np.newaxis, ...]
        if mask.ndim == 2:
            mask = mask[np.newaxis, ...]

        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()

        return image, mask


class RandomHorizontalFlip:
    """Randomly flip image and mask horizontally."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.rand() < self.p:
            image = np.flip(image, axis=2).copy()  # Flip along width
            mask = np.flip(mask, axis=2).copy()
        return image, mask


class RandomVerticalFlip:
    """Randomly flip image and mask vertically."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.rand() < self.p:
            image = np.flip(image, axis=1).copy()  # Flip along height
            mask = np.flip(mask, axis=1).copy()
        return image, mask


class RandomRotate90:
    """Randomly rotate image and mask by 90 degrees (0, 90, 180, or 270)."""

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        k = np.random.randint(0, 4)  # 0, 1, 2, or 3 rotations
        if k > 0:
            image = np.rot90(image, k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k, axes=(1, 2)).copy()
        return image, mask


class Normalize:
    """Normalize image with mean and std."""

    def __init__(self, mean: list, std: list):
        self.mean = np.array(mean).reshape(-1, 1, 1)
        self.std = np.array(std).reshape(-1, 1, 1)

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        image = (image - self.mean) / self.std
        return image, mask


class RandomBrightnessContrast:
    """Randomly adjust brightness and contrast."""

    def __init__(self, brightness_limit: float = 0.2, contrast_limit: float = 0.2, p: float = 0.5):
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.p = p

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.rand() < self.p:
            # Random brightness factor
            alpha = 1.0 + np.random.uniform(-self.brightness_limit, self.brightness_limit)
            # Random contrast factor
            beta = np.random.uniform(-self.contrast_limit, self.contrast_limit)

            image = np.clip(image * alpha + beta, 0, 1)

        return image, mask


def get_train_transforms() -> Compose:
    """
    Get training transforms with data augmentation.

    Returns:
        Composed transforms for training
    """
    return Compose([
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotate90(),
        RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        ToTensor(),
    ])


def get_val_transforms() -> Compose:
    """
    Get validation transforms (no augmentation).

    Returns:
        Composed transforms for validation
    """
    return Compose([
        ToTensor(),
    ])
