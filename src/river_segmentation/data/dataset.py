"""Dataset class for river segmentation from Sentinel-2 imagery."""

import os
from pathlib import Path
from typing import Optional, Tuple, Callable

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


class RiverSegmentationDataset(Dataset):
    """
    Dataset for river segmentation from Sentinel-2 imagery.

    Expects:
    - Input images: Multi-band GeoTIFF (e.g., RGB + NIR + SWIR bands)
    - Masks: Binary GeoTIFF (0=water/river, 1=land) from GEE script

    Args:
        image_dir: Directory containing input satellite images
        mask_dir: Directory containing binary water masks
        transform: Optional transforms to apply to both image and mask
        bands: List of band indices to use (None = all bands)
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        transform: Optional[Callable] = None,
        bands: Optional[list] = None,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.bands = bands

        # Get all image files
        self.image_files = sorted(list(self.image_dir.glob("*.tif")) +
                                  list(self.image_dir.glob("*.tiff")))

        if len(self.image_files) == 0:
            raise ValueError(f"No GeoTIFF files found in {image_dir}")

        print(f"Found {len(self.image_files)} images in {image_dir}")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load image and mask pair.

        Returns:
            image: Tensor of shape (C, H, W) - normalized satellite bands
            mask: Tensor of shape (1, H, W) - binary mask (0=water, 1=land)
        """
        # Load image
        image_path = self.image_files[idx]
        image = self._load_image(image_path)

        # Load corresponding mask
        mask_filename = image_path.stem + ".tif"
        mask_path = self.mask_dir / mask_filename

        if not mask_path.exists():
            # Try .tiff extension
            mask_path = self.mask_dir / (image_path.stem + ".tiff")

        if not mask_path.exists():
            raise FileNotFoundError(
                f"Mask not found for image {image_path.name}. "
                f"Expected: {mask_path}"
            )

        mask = self._load_mask(mask_path)

        # Apply transforms
        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask

    def _load_image(self, path: Path) -> np.ndarray:
        """
        Load multi-band satellite image from GeoTIFF.

        Returns:
            Array of shape (C, H, W) with pixel values normalized to [0, 1]
        """
        with rasterio.open(path) as src:
            if self.bands is not None:
                # Load specific bands (1-indexed in rasterio)
                image = src.read([b + 1 for b in self.bands])
            else:
                # Load all bands
                image = src.read()

            # Convert to float32 and normalize
            image = image.astype(np.float32)

            # Sentinel-2 data is typically scaled 0-10000
            # Normalize to [0, 1]
            if image.max() > 100:  # Check if data needs scaling
                image = image / 10000.0

            # Clip values to [0, 1] range
            image = np.clip(image, 0, 1)

        return image

    def _load_mask(self, path: Path) -> np.ndarray:
        """
        Load binary mask from GeoTIFF.

        Returns:
            Array of shape (1, H, W) with values {0, 1}
            0 = water/river, 1 = land
        """
        with rasterio.open(path) as src:
            mask = src.read(1)  # Read first band

            # Ensure binary
            mask = mask.astype(np.float32)
            mask = np.clip(mask, 0, 1)

            # Add channel dimension
            mask = mask[np.newaxis, ...]

        return mask

    def get_image_path(self, idx: int) -> Path:
        """Get the file path for an image by index."""
        return self.image_files[idx]
