"""
Visualize image and mask overlay to verify annotations.

This script helps verify manual annotations by creating overlay visualizations.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import rasterio


def visualize_annotation(
    image_path: Path,
    mask_path: Path,
    output_path: Path = None,
    sample_region: tuple = None,
):
    """
    Create visualization of image with mask overlay.

    Args:
        image_path: Path to input image
        mask_path: Path to mask
        output_path: Path to save visualization
        sample_region: Tuple of (row_start, row_end, col_start, col_end) for zoomed view
    """
    print("Creating annotation visualization...")

    with rasterio.open(image_path) as img_src, rasterio.open(mask_path) as mask_src:
        # Read data
        if sample_region:
            row_start, row_end, col_start, col_end = sample_region
            from rasterio.windows import Window
            window = Window(col_start, row_start, col_end - col_start, row_end - row_start)
            image = img_src.read(1, window=window)
            mask = mask_src.read(1, window=window)
        else:
            # Downsample for large images
            if img_src.height > 2000 or img_src.width > 2000:
                scale_factor = min(2000 / img_src.height, 2000 / img_src.width)
                out_shape = (
                    int(img_src.height * scale_factor),
                    int(img_src.width * scale_factor)
                )
                print(f"Downsampling from {img_src.shape} to {out_shape}")

                from rasterio.enums import Resampling
                image = img_src.read(
                    1,
                    out_shape=out_shape,
                    resampling=Resampling.bilinear
                )
                mask = mask_src.read(
                    1,
                    out_shape=out_shape,
                    resampling=Resampling.nearest
                )
            else:
                image = img_src.read(1)
                mask = mask_src.read(1)

        # Normalize image for display
        image_display = np.clip(image, np.percentile(image, 2), np.percentile(image, 98))
        image_display = (image_display - image_display.min()) / (image_display.max() - image_display.min())

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Original image
        axes[0, 0].imshow(image_display, cmap='gray')
        axes[0, 0].set_title('Original Image (NIR Band)', fontsize=14)
        axes[0, 0].axis('off')

        # Mask
        axes[0, 1].imshow(mask, cmap='RdYlGn', vmin=0, vmax=1)
        axes[0, 1].set_title('Mask (0=Water, 1=Land)', fontsize=14)
        axes[0, 1].axis('off')

        # Overlay: Image with water highlighted
        axes[1, 0].imshow(image_display, cmap='gray')
        water_mask = mask == 0
        # Create colored overlay for water
        overlay = np.zeros((*mask.shape, 4))
        overlay[water_mask] = [0, 0.5, 1, 0.5]  # Blue with transparency
        axes[1, 0].imshow(overlay)
        axes[1, 0].set_title('Overlay: Water Highlighted (Blue)', fontsize=14)
        axes[1, 0].axis('off')

        # Statistics
        water_pixels = (mask == 0).sum()
        land_pixels = (mask == 1).sum()
        total_pixels = mask.size
        water_percent = (water_pixels / total_pixels) * 100

        stats_text = (
            f"Mask Statistics:\n\n"
            f"Total pixels: {total_pixels:,}\n"
            f"Water pixels (0): {water_pixels:,}\n"
            f"Land pixels (1): {land_pixels:,}\n\n"
            f"Water coverage: {water_percent:.2f}%\n"
            f"Land coverage: {100-water_percent:.2f}%\n\n"
            f"Image value range:\n"
            f"  Min: {image.min():.4f}\n"
            f"  Max: {image.max():.4f}\n"
            f"  Mean: {image.mean():.4f}\n"
        )

        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                       fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Statistics', fontsize=14)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {output_path}")
        else:
            plt.show()

        plt.close()

        # Create histogram comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram of water vs land pixels
        water_values = image[mask == 0]
        land_values = image[mask == 1]

        axes[0].hist(water_values.flatten(), bins=50, alpha=0.7, label='Water', color='blue', density=True)
        axes[0].hist(land_values.flatten(), bins=50, alpha=0.7, label='Land', color='green', density=True)
        axes[0].set_xlabel('Pixel Value', fontsize=12)
        axes[0].set_ylabel('Density', fontsize=12)
        axes[0].set_title('Value Distribution: Water vs Land', fontsize=14)
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Box plot
        axes[1].boxplot([water_values.flatten(), land_values.flatten()],
                       labels=['Water (0)', 'Land (1)'])
        axes[1].set_ylabel('Pixel Value', fontsize=12)
        axes[1].set_title('Value Distribution Box Plot', fontsize=14)
        axes[1].grid(alpha=0.3)

        plt.tight_layout()

        if output_path:
            hist_path = output_path.parent / (output_path.stem + '_histogram.png')
            plt.savefig(hist_path, dpi=150, bbox_inches='tight')
            print(f"Histogram saved to: {hist_path}")
        else:
            plt.show()

        plt.close()

        print("\nValidation checks:")
        print(f"  ✓ Mask has correct values (0 and 1): {set(np.unique(mask)) == {0, 1}}")
        print(f"  ✓ Image and mask have same shape: {image.shape == mask.shape}")
        print(f"  ✓ Water coverage reasonable (1-50%): {1 < water_percent < 50}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize image and mask annotations"
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--mask",
        type=Path,
        required=True,
        help="Path to mask"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save visualization (default: display)"
    )
    parser.add_argument(
        "--region",
        type=int,
        nargs=4,
        default=None,
        metavar=('ROW_START', 'ROW_END', 'COL_START', 'COL_END'),
        help="Sample region for zoomed view"
    )

    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not args.mask.exists():
        raise FileNotFoundError(f"Mask not found: {args.mask}")

    visualize_annotation(
        args.image,
        args.mask,
        args.output,
        args.region,
    )


if __name__ == "__main__":
    main()
