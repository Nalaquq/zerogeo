"""
Create training patches from a large GeoTIFF image.

This script splits a large image (and its corresponding mask) into smaller patches
suitable for training a U-Net model.
"""

import argparse
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm


def create_patches(
    image_path: Path,
    mask_path: Path,
    output_dir: Path,
    patch_size: int = 256,
    overlap: int = 0,
    min_water_percent: float = 0.01,
):
    """
    Extract patches from large image and mask.

    Args:
        image_path: Path to input image
        mask_path: Path to input mask (same size as image)
        output_dir: Directory to save patches
        patch_size: Size of square patches
        overlap: Overlap between patches in pixels
        min_water_percent: Minimum percentage of water pixels to include patch
    """
    # Create output directories
    image_out_dir = output_dir / "images"
    mask_out_dir = output_dir / "masks"
    image_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating patches of size {patch_size}x{patch_size}")
    print(f"Overlap: {overlap} pixels")
    print(f"Minimum water percentage: {min_water_percent * 100}%")

    # Open image and mask
    with rasterio.open(image_path) as img_src, rasterio.open(mask_path) as mask_src:
        # Verify dimensions match
        if img_src.shape != mask_src.shape:
            raise ValueError(
                f"Image shape {img_src.shape} doesn't match mask shape {mask_src.shape}"
            )

        height, width = img_src.shape
        stride = patch_size - overlap

        print(f"Input image size: {height} x {width}")
        print(f"Number of bands: {img_src.count}")

        # Calculate number of patches
        n_patches_h = (height - patch_size) // stride + 1
        n_patches_w = (width - patch_size) // stride + 1
        total_patches = n_patches_h * n_patches_w

        print(f"Potential patches: {total_patches}")

        patch_count = 0
        skipped_count = 0

        # Iterate over patches
        progress_bar = tqdm(total=total_patches, desc="Extracting patches")

        for i in range(n_patches_h):
            for j in range(n_patches_w):
                # Calculate window
                row_start = i * stride
                col_start = j * stride

                # Ensure we don't go out of bounds
                if row_start + patch_size > height:
                    row_start = height - patch_size
                if col_start + patch_size > width:
                    col_start = width - patch_size

                window = Window(col_start, row_start, patch_size, patch_size)

                # Read image and mask patches
                img_patch = img_src.read(window=window)
                mask_patch = mask_src.read(1, window=window)

                # Check if patch has enough water content
                water_pixels = (mask_patch == 0).sum()
                water_percent = water_pixels / (patch_size * patch_size)

                if water_percent < min_water_percent:
                    skipped_count += 1
                    progress_bar.update(1)
                    continue

                # Create patch filename
                patch_name = f"patch_{i:04d}_{j:04d}.tif"

                # Save image patch
                img_meta = img_src.meta.copy()
                img_meta.update({
                    'height': patch_size,
                    'width': patch_size,
                    'transform': rasterio.windows.transform(window, img_src.transform)
                })

                with rasterio.open(image_out_dir / patch_name, 'w', **img_meta) as dst:
                    dst.write(img_patch)

                # Save mask patch
                mask_meta = mask_src.meta.copy()
                mask_meta.update({
                    'height': patch_size,
                    'width': patch_size,
                    'transform': rasterio.windows.transform(window, mask_src.transform)
                })

                with rasterio.open(mask_out_dir / patch_name, 'w', **mask_meta) as dst:
                    dst.write(mask_patch, 1)

                patch_count += 1
                progress_bar.update(1)

        progress_bar.close()

    print(f"\n{'='*60}")
    print(f"Patch extraction complete!")
    print(f"  Total patches created: {patch_count}")
    print(f"  Patches skipped (low water content): {skipped_count}")
    print(f"  Images saved to: {image_out_dir}")
    print(f"  Masks saved to: {mask_out_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Create training patches from large GeoTIFF images"
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to input image (GeoTIFF)"
    )
    parser.add_argument(
        "--mask",
        type=Path,
        required=True,
        help="Path to input mask (GeoTIFF, same size as image)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save patches"
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
        help="Size of square patches (default: 256)"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=32,
        help="Overlap between patches in pixels (default: 32)"
    )
    parser.add_argument(
        "--min-water-percent",
        type=float,
        default=0.01,
        help="Minimum percentage of water pixels to include patch (default: 0.01)"
    )

    args = parser.parse_args()

    # Verify files exist
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not args.mask.exists():
        raise FileNotFoundError(f"Mask not found: {args.mask}")

    create_patches(
        args.image,
        args.mask,
        args.output_dir,
        args.patch_size,
        args.overlap,
        args.min_water_percent,
    )


if __name__ == "__main__":
    main()
