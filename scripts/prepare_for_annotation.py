"""
Prepare image for manual annotation by creating an initial mask template.

This script helps create a starting point for manual annotation by:
1. Creating a binary mask using simple thresholding (as a baseline)
2. Exporting the image in a format suitable for annotation tools
"""

import argparse
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling


def create_mask_template(
    image_path: Path,
    output_mask_path: Path,
    threshold: float = None,
    auto_threshold: bool = True,
):
    """
    Create initial mask template for manual annotation.

    Args:
        image_path: Path to input image
        output_mask_path: Path to save mask template
        threshold: Manual threshold value (for single-band images)
        auto_threshold: Use Otsu's method for automatic thresholding
    """
    print("=" * 60)
    print("Creating Mask Template for Manual Annotation")
    print("=" * 60)

    with rasterio.open(image_path) as src:
        # Read image
        image = src.read(1)  # Read first band

        print(f"Image shape: {image.shape}")
        print(f"Value range: [{image.min():.4f}, {image.max():.4f}]")

        # Create initial mask using thresholding
        if auto_threshold:
            # Use Otsu's method for automatic thresholding
            from skimage.filters import threshold_otsu
            try:
                threshold_value = threshold_otsu(image[image > 0])  # Ignore zeros
                print(f"Auto-detected threshold (Otsu): {threshold_value:.4f}")
            except:
                # Fallback to simple percentile-based threshold
                threshold_value = np.percentile(image[image > 0], 30)
                print(f"Auto-detected threshold (30th percentile): {threshold_value:.4f}")
        elif threshold is not None:
            threshold_value = threshold
            print(f"Using manual threshold: {threshold_value:.4f}")
        else:
            # Default: use mean
            threshold_value = image.mean()
            print(f"Using mean as threshold: {threshold_value:.4f}")

        # Create binary mask
        # Low values = water (0), high values = land (1)
        mask = (image > threshold_value).astype(np.uint8)

        # Calculate statistics
        water_pixels = (mask == 0).sum()
        land_pixels = (mask == 1).sum()
        total_pixels = mask.size
        water_percent = (water_pixels / total_pixels) * 100

        print(f"\nMask statistics:")
        print(f"  Water pixels (0): {water_pixels:,} ({water_percent:.2f}%)")
        print(f"  Land pixels (1): {land_pixels:,} ({100-water_percent:.2f}%)")

        # Save mask
        mask_meta = src.meta.copy()
        mask_meta.update({
            'count': 1,
            'dtype': 'uint8',
            'compress': 'lzw'
        })

        output_mask_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(output_mask_path, 'w', **mask_meta) as dst:
            dst.write(mask, 1)

        print(f"\nMask template saved to: {output_mask_path}")
        print("\nNext steps:")
        print("1. Open the mask in QGIS or another GIS tool")
        print("2. Manually refine the mask by:")
        print("   - Removing non-river water bodies (lakes, ponds)")
        print("   - Correcting misclassified areas")
        print("   - Adding missing river sections")
        print("3. Save the edited mask with the same filename")
        print("4. Use create_patches.py to create training patches")
        print("=" * 60)


def export_for_qgis(image_path: Path, output_dir: Path):
    """
    Export image and create a QGIS-friendly mask template.

    Args:
        image_path: Path to input image
        output_dir: Directory to save outputs
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy image to output directory
    import shutil
    output_image = output_dir / "image_to_annotate.tif"
    shutil.copy(image_path, output_image)

    # Create mask template
    output_mask = output_dir / "mask_template.tif"
    create_mask_template(image_path, output_mask, auto_threshold=True)

    # Create a QGIS project file helper
    qgis_instructions = output_dir / "QGIS_INSTRUCTIONS.txt"
    with open(qgis_instructions, 'w') as f:
        f.write("Instructions for Manual Annotation in QGIS\n")
        f.write("=" * 60 + "\n\n")
        f.write("1. Open QGIS and create a new project\n\n")
        f.write("2. Add the image layer:\n")
        f.write(f"   - Layer > Add Layer > Add Raster Layer\n")
        f.write(f"   - Select: {output_image.name}\n\n")
        f.write("3. Add the mask template layer:\n")
        f.write(f"   - Layer > Add Layer > Add Raster Layer\n")
        f.write(f"   - Select: {output_mask.name}\n\n")
        f.write("4. Edit the mask:\n")
        f.write("   - Raster > Raster Calculator (or use vector digitizing)\n")
        f.write("   - Or use plugins like 'Serval' for raster editing\n")
        f.write("   - Make sure mask values are:\n")
        f.write("     * 0 = Water/River (black)\n")
        f.write("     * 1 = Land (white)\n\n")
        f.write("5. Save the edited mask as:\n")
        f.write(f"   - {output_dir}/mask_final.tif\n")
        f.write("   - Use GeoTIFF format\n")
        f.write("   - Maintain same dimensions and CRS as input image\n\n")
        f.write("6. After annotation, run:\n")
        f.write("   python scripts/create_patches.py \\\n")
        f.write(f"     --image {output_image} \\\n")
        f.write(f"     --mask {output_dir}/mask_final.tif \\\n")
        f.write("     --output-dir data/patches\n")

    print(f"\nQGIS instructions saved to: {qgis_instructions}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare image for manual annotation"
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to input image (GeoTIFF)"
    )
    parser.add_argument(
        "--output-mask",
        type=Path,
        required=True,
        help="Path to save mask template"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Manual threshold value (default: auto-detect)"
    )
    parser.add_argument(
        "--export-for-qgis",
        action="store_true",
        help="Export files ready for QGIS annotation"
    )
    parser.add_argument(
        "--qgis-dir",
        type=Path,
        default=Path("data/annotation"),
        help="Directory for QGIS annotation files"
    )

    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    if args.export_for_qgis:
        export_for_qgis(args.image, args.qgis_dir)
    else:
        create_mask_template(
            args.image,
            args.output_mask,
            threshold=args.threshold,
            auto_threshold=(args.threshold is None)
        )


if __name__ == "__main__":
    main()
