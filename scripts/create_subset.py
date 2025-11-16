#!/usr/bin/env python3
"""
Create a subset of a large orthomosaic for testing.

Usage:
    python scripts/create_subset.py data/Quinhagak-Orthomosaic.tiff \
        --output data/quinhagak_subset_4096x4096.tif \
        --size 4096 \
        --center
"""

import argparse
import sys
from pathlib import Path

import rasterio
from rasterio.windows import Window

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.river_segmentation.utils.image_utils import get_image_info


def create_subset(
    input_path: Path,
    output_path: Path,
    size: int = 4096,
    offset_x: int = None,
    offset_y: int = None,
    center: bool = False
):
    """
    Extract a subset from a large image.

    Args:
        input_path: Path to input image
        output_path: Path to output subset
        size: Size of subset (width and height)
        offset_x: X offset (if None, uses center or 0)
        offset_y: Y offset (if None, uses center or 0)
        center: Extract from center of image
    """
    print(f"Creating subset from: {input_path}")

    # Get input image info
    info = get_image_info(input_path)
    print(f"  Input size: {info.width} x {info.height} pixels")
    print(f"  Bands: {info.bands}")
    print(f"  Size: {info.size_mb:.2f} MB")

    # Calculate offsets
    if center:
        offset_x = (info.width - size) // 2
        offset_y = (info.height - size) // 2
        print(f"  Using center extraction")
    else:
        offset_x = offset_x or 0
        offset_y = offset_y or 0

    # Validate
    if offset_x < 0 or offset_y < 0:
        raise ValueError(f"Offsets must be non-negative: ({offset_x}, {offset_y})")

    if offset_x + size > info.width or offset_y + size > info.height:
        raise ValueError(
            f"Subset extends beyond image bounds: "
            f"({offset_x}, {offset_y}) + {size} > ({info.width}, {info.height})"
        )

    print(f"  Subset offset: ({offset_x}, {offset_y})")
    print(f"  Subset size: {size} x {size}")

    # Extract subset
    with rasterio.open(input_path) as src:
        # Create window
        window = Window(offset_x, offset_y, size, size)

        # Read subset
        print(f"  Reading subset...")
        subset = src.read(window=window)

        # Update profile for output
        profile = src.profile.copy()
        profile.update({
            'width': size,
            'height': size,
            'transform': src.window_transform(window)
        })

        # Write output
        print(f"  Writing to: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(subset)

    # Get output info
    output_info = get_image_info(output_path)
    print(f"\nSubset created successfully!")
    print(f"  Output size: {output_info.width} x {output_info.height}")
    print(f"  File size: {output_info.size_mb:.2f} MB")
    print(f"  Georeferenced: {output_info.is_georeferenced}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract a subset from a large orthomosaic for testing"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to input image (e.g., data/Quinhagak-Orthomosaic.tiff)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Path to output subset (default: input_subset_SIZExSIZE.tif)"
    )
    parser.add_argument(
        "--size", "-s",
        type=int,
        default=4096,
        help="Size of subset (width and height in pixels, default: 4096)"
    )
    parser.add_argument(
        "--offset-x",
        type=int,
        default=None,
        help="X offset (default: center if --center, else 0)"
    )
    parser.add_argument(
        "--offset-y",
        type=int,
        default=None,
        help="Y offset (default: center if --center, else 0)"
    )
    parser.add_argument(
        "--center", "-c",
        action="store_true",
        help="Extract from center of image (default: top-left)"
    )

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Default output path
    if args.output is None:
        stem = args.input.stem
        suffix = args.input.suffix
        args.output = args.input.parent / f"{stem}_subset_{args.size}x{args.size}{suffix}"

    # Create subset
    try:
        create_subset(
            input_path=args.input,
            output_path=args.output,
            size=args.size,
            offset_x=args.offset_x,
            offset_y=args.offset_y,
            center=args.center
        )
    except Exception as e:
        print(f"\nError creating subset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
