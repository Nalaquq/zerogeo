#!/usr/bin/env python3
"""
Reassemble tile masks into a complete GeoTIFF raster.

This script combines individual tile masks back into a full-size raster that can be
viewed in GIS software like QGIS or ArcGIS. It handles overlapping tiles with
weighted blending for seamless results.

Usage:
    python scripts/reassemble_masks.py --run-dir output/PROJECT_TIMESTAMP/
    python scripts/reassemble_masks.py --masks-dir path/to/masks/ --original-image input/image.tif

Examples:
    # Reassemble reviewed masks from a run directory
    python scripts/reassemble_masks.py --run-dir output/rivers_20251113_143022/

    # Reassemble raw masks instead of reviewed
    python scripts/reassemble_masks.py --run-dir output/rivers_20251113_143022/ --use-raw

    # Specify custom output location
    python scripts/reassemble_masks.py \
        --run-dir output/rivers_20251113_143022/ \
        --output final_results/river_mask.tif

    # Legacy: Manual paths
    python scripts/reassemble_masks.py \
        --masks-dir data/masks/ \
        --original-image input/original.tif \
        --output output.tif
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

# Add parent directory and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from river_segmentation.annotation.tiler import TileManager
from civic.utils.path_manager import get_run_paths


def find_tile_metadata(tiles_dir: Path) -> Optional[Path]:
    """Find tile metadata JSON file in directory."""
    # Try standard name first
    metadata_path = tiles_dir / "tile_metadata.json"
    if metadata_path.exists():
        return metadata_path

    # Try with any prefix
    metadata_files = list(tiles_dir.glob("*_metadata.json"))
    if metadata_files:
        return metadata_files[0]

    return None


def create_weight_map(height: int, width: int, overlap: int) -> np.ndarray:
    """
    Create distance-based weight map for blending overlapping tiles.

    Weights are higher in the center and taper towards edges in overlap regions.

    Args:
        height: Tile height
        width: Tile width
        overlap: Overlap size in pixels

    Returns:
        Weight map of shape (height, width)
    """
    def create_1d_weights(size, overlap):
        weights = np.ones(size)
        if overlap > 0:
            # Linear ramp in overlap regions
            ramp = np.linspace(0, 1, overlap)
            weights[:overlap] = ramp
            weights[-overlap:] = ramp[::-1]
        return weights

    # Create 2D weight map
    row_weights = create_1d_weights(height, overlap)
    col_weights = create_1d_weights(width, overlap)

    weight_map = row_weights[:, np.newaxis] * col_weights[np.newaxis, :]

    return weight_map


def get_original_image_info(run_dir: Path) -> Tuple[Optional[Path], Optional[dict]]:
    """
    Try to find original image path and metadata from run directory.

    Args:
        run_dir: Run directory path

    Returns:
        Tuple of (original_image_path, run_metadata)
    """
    # Try to load run metadata
    metadata_file = run_dir / "run_metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            run_metadata = json.load(f)

        # Some metadata files might have the original image path
        if "input_image" in run_metadata:
            img_path = Path(run_metadata["input_image"])
            if img_path.exists():
                return img_path, run_metadata

    return None, None


def reassemble_masks(
    masks_dir: Path,
    tiles_dir: Path,
    original_image: Path,
    output_path: Path,
    overlap: int = 64,
    blend_overlap: bool = True,
    binary_threshold: float = 0.5
):
    """
    Reassemble tile masks into a full-size GeoTIFF raster.

    Args:
        masks_dir: Directory containing mask tiles
        tiles_dir: Directory containing original tiles (for metadata)
        original_image: Path to original input image (for size and CRS)
        output_path: Where to save the reassembled raster
        overlap: Overlap used during tiling (default: 64)
        blend_overlap: Use weighted blending in overlap regions (default: True)
        binary_threshold: Threshold for converting to binary (0-1, default: 0.5)
    """
    print(f"Reassembling masks from: {masks_dir}")
    print(f"Original image: {original_image}")
    print(f"Output: {output_path}")
    print()

    # Load tile metadata
    metadata_path = find_tile_metadata(tiles_dir)
    if not metadata_path:
        raise FileNotFoundError(f"Tile metadata not found in {tiles_dir}")

    print(f"Loading tile metadata from: {metadata_path}")
    tile_manager = TileManager.load(metadata_path)
    print(f"Found {len(tile_manager)} tiles")
    print()

    # Open original image for size and metadata
    with rasterio.open(original_image) as src:
        profile = src.profile.copy()
        height, width = src.height, src.width
        crs = src.crs
        transform = src.transform

        print(f"Output dimensions: {width} x {height} pixels")
        print(f"CRS: {crs}")
        print(f"Blend overlap: {blend_overlap}")
        print()

        # Create accumulation arrays
        accumulated = np.zeros((height, width), dtype=np.float32)
        weights = np.zeros((height, width), dtype=np.float32)

        # Process each tile
        print("Reassembling tiles...")
        tiles_found = 0
        tiles_missing = 0

        for tile_info in tqdm(tile_manager, desc="Processing tiles"):
            # Find mask file for this tile - prefer .tif files
            mask_files = list(masks_dir.glob(f"{tile_info.tile_id}_mask.tif"))
            if not mask_files:
                # Fallback to .tiff extension
                mask_files = list(masks_dir.glob(f"{tile_info.tile_id}_mask.tiff"))

            if not mask_files:
                tiles_missing += 1
                continue

            mask_path = mask_files[0]
            tiles_found += 1

            # Load mask
            try:
                with rasterio.open(mask_path) as mask_src:
                    mask_data = mask_src.read(1)  # Read first band
            except Exception as e:
                print(f"\nWarning: Could not read {mask_path}: {e}")
                continue

            # Get window coordinates
            window = tile_info.window
            row_start = int(window.row_off)
            row_end = int(window.row_off + window.height)
            col_start = int(window.col_off)
            col_end = int(window.col_off + window.width)

            # Ensure mask matches window size
            if mask_data.shape != (int(window.height), int(window.width)):
                print(f"\nWarning: Mask {tile_info.tile_id} size mismatch. Skipping.")
                continue

            # Create weight map for this tile
            if blend_overlap:
                tile_weights = create_weight_map(
                    int(window.height),
                    int(window.width),
                    overlap
                )
            else:
                # Uniform weights
                tile_weights = np.ones((int(window.height), int(window.width)))

            # Convert mask to float if needed
            if mask_data.dtype == np.uint8:
                # Check if mask is already binary (0,1) or needs normalization (0,255)
                max_val = mask_data.max()
                if max_val > 1:
                    # Scale from 0-255 to 0-1
                    mask_data = mask_data.astype(np.float32) / 255.0
                else:
                    # Already binary (0,1)
                    mask_data = mask_data.astype(np.float32)
            else:
                mask_data = mask_data.astype(np.float32)

            # Accumulate weighted mask
            accumulated[row_start:row_end, col_start:col_end] += (
                mask_data * tile_weights
            )
            weights[row_start:row_end, col_start:col_end] += tile_weights

        print(f"\nProcessed {tiles_found} tiles ({tiles_missing} missing)")
        print()

        # Normalize by weights
        print("Normalizing and creating final raster...")
        weights = np.maximum(weights, 1e-8)  # Avoid division by zero
        reassembled = accumulated / weights

        # Convert to binary if threshold provided
        if binary_threshold is not None:
            reassembled_binary = (reassembled >= binary_threshold).astype(np.uint8)
            # Scale to 0-255 for better compatibility with ArcGIS and other GIS software
            reassembled_binary = reassembled_binary * 255
        else:
            # Keep as float
            reassembled_binary = reassembled

        # Update profile for output
        profile.update({
            'height': height,
            'width': width,
            'count': 1,
            'dtype': 'uint8' if binary_threshold is not None else 'float32',
            'compress': 'lzw',
            'nodata': None  # Don't set nodata to avoid confusion with 0 values
        })

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save reassembled raster
        print(f"Writing output to: {output_path}")
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(reassembled_binary, 1)

        # Calculate statistics
        coverage = (reassembled_binary > 0).sum() / (height * width) * 100

        print()
        print("=" * 60)
        print("SUCCESS")
        print("=" * 60)
        print(f"Reassembled raster saved to: {output_path}")
        print(f"Dimensions: {width} x {height} pixels")
        print(f"Coverage: {coverage:.2f}% of image")
        print(f"Data type: {profile['dtype']}")
        print(f"CRS: {crs}")
        print()
        print("You can now open this file in:")
        print("  - QGIS")
        print("  - ArcGIS")
        print("  - Any GIS software that supports GeoTIFF")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Reassemble tile masks into a complete GeoTIFF raster",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Reassemble reviewed masks from run directory
  %(prog)s --run-dir output/rivers_20251113_143022/

  # Use raw masks instead of reviewed
  %(prog)s --run-dir output/rivers_20251113_143022/ --use-raw

  # Custom output path
  %(prog)s --run-dir output/rivers_20251113_143022/ --output final_river_mask.tif

  # Manual paths (legacy)
  %(prog)s --masks-dir data/masks/ --original-image input/image.tif --output output.tif
        """
    )

    # Run directory workflow (recommended)
    parser.add_argument(
        "--run-dir",
        type=str,
        help="Run directory containing tiles/ and masks/"
    )

    parser.add_argument(
        "--use-raw",
        action="store_true",
        help="Use raw_masks/ instead of reviewed_masks/ (only with --run-dir)"
    )

    # Manual paths (legacy)
    parser.add_argument(
        "--masks-dir",
        type=str,
        help="Directory containing mask tiles [legacy]"
    )

    parser.add_argument(
        "--tiles-dir",
        type=str,
        help="Directory containing original tiles (for metadata) [legacy]"
    )

    parser.add_argument(
        "--original-image",
        type=str,
        help="Path to original input image (for size and CRS) [legacy]"
    )

    # Output options
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for reassembled raster (default: run_dir/final_mask.tif or reassembled.tif)"
    )

    # Processing options
    parser.add_argument(
        "--overlap",
        type=int,
        default=64,
        help="Overlap used during tiling in pixels (default: 64)"
    )

    parser.add_argument(
        "--no-blend",
        action="store_true",
        help="Disable weighted blending in overlap regions"
    )

    parser.add_argument(
        "--binary-threshold",
        type=float,
        default=0.5,
        help="Threshold for binary output (0-1, default: 0.5). Set to -1 to keep float values"
    )

    args = parser.parse_args()

    # Determine paths based on workflow
    if args.run_dir:
        # Run directory workflow (recommended)
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            print(f"Error: Run directory not found: {run_dir}", file=sys.stderr)
            return 1

        # Get standard paths
        run_paths = get_run_paths(run_dir)
        tiles_dir = run_paths["tiles"]

        # Choose masks directory
        if args.use_raw:
            masks_dir = run_paths["raw_masks"]
            print("Using raw masks (auto-generated, not reviewed)")
        else:
            masks_dir = run_paths["reviewed_masks"]
            print("Using reviewed masks")
        print()

        if not masks_dir.exists() or not list(masks_dir.glob("*_mask.*")):
            print(f"Error: No mask files found in {masks_dir}", file=sys.stderr)
            if not args.use_raw:
                print("Hint: Try --use-raw to use raw_masks/ instead", file=sys.stderr)
            return 1

        # Try to find original image
        original_image, run_metadata = get_original_image_info(run_dir)

        if original_image is None:
            # Try to infer from tile metadata
            metadata_path = find_tile_metadata(tiles_dir)
            if metadata_path:
                # We'll use the first tile's dimensions and CRS
                # User should provide --original-image if this doesn't work
                print("Warning: Could not find original image path in metadata.")
                print("Will attempt to reconstruct dimensions from tile metadata.")
                print("For best results, provide --original-image")
                print()

                # Create a temporary reference - we'll use tile info to reconstruct size
                # This will be handled in reassemble_masks by calculating from tiles

        # Default output path
        if args.output:
            output_path = Path(args.output)
        else:
            mask_type = "raw" if args.use_raw else "reviewed"
            output_path = run_dir / f"final_mask_{mask_type}.tif"

    elif args.masks_dir and args.original_image:
        # Legacy workflow
        masks_dir = Path(args.masks_dir)
        tiles_dir = Path(args.tiles_dir) if args.tiles_dir else masks_dir.parent / "tiles"
        original_image = Path(args.original_image)

        if not masks_dir.exists():
            print(f"Error: Masks directory not found: {masks_dir}", file=sys.stderr)
            return 1

        if not original_image.exists():
            print(f"Error: Original image not found: {original_image}", file=sys.stderr)
            return 1

        # Default output path
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = Path("reassembled.tif")

    else:
        print("Error: Must provide either --run-dir or (--masks-dir and --original-image)", file=sys.stderr)
        return 1

    # Validate tiles directory
    if not tiles_dir.exists():
        print(f"Error: Tiles directory not found: {tiles_dir}", file=sys.stderr)
        return 1

    # Binary threshold (-1 means keep float)
    binary_threshold = None if args.binary_threshold < 0 else args.binary_threshold

    try:
        # Handle case where original_image might not be set
        if original_image is None or not original_image.exists():
            print("Error: Original image required for reassembly", file=sys.stderr)
            print("Please provide --original-image", file=sys.stderr)
            return 1

        reassemble_masks(
            masks_dir=masks_dir,
            tiles_dir=tiles_dir,
            original_image=original_image,
            output_path=output_path,
            overlap=args.overlap,
            blend_overlap=not args.no_blend,
            binary_threshold=binary_threshold
        )

        return 0

    except Exception as e:
        print(f"\nError during reassembly: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
