#!/usr/bin/env python3
"""
Command-line tool for tiling large GeoTIFF images.

Usage:
    python scripts/tile_image.py input.tif --project PROJECT_NAME [options]

Examples:
    # Basic usage with defaults (512x512 tiles, 64px overlap)
    python scripts/tile_image.py input/Sentinel2_1_bands.tif --project quinhagak

    # Custom tile size and overlap
    python scripts/tile_image.py input/large_image.tif --project myproject --tile-size 256 --overlap 32

    # Large tiles with high overlap for river detection
    python scripts/tile_image.py input/river_scene.tif --project river_01 --tile-size 1024 --overlap 128

    # Custom output directory
    python scripts/tile_image.py input/scene.tif --project test --output-base custom_output/
"""

import argparse
import sys
from pathlib import Path

# Add parent directory and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from river_segmentation.annotation.tiler import ImageTiler
from civic.utils.path_manager import RunDirectoryManager


def main():
    parser = argparse.ArgumentParser(
        description="Tile large GeoTIFF images with overlap",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input/image.tif --project myproject
  %(prog)s input/image.tif --project test --tile-size 256 --overlap 32
  %(prog)s input/image.tif --project river_01 --tile-size 1024 --overlap 128
        """
    )

    parser.add_argument(
        "input_image",
        type=str,
        help="Path to input GeoTIFF file (typically in input/ directory)"
    )

    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="Project name for this run (used in output directory naming)"
    )

    parser.add_argument(
        "--output-base",
        type=str,
        default="output",
        help="Base output directory (default: output)"
    )

    parser.add_argument(
        "--tile-size",
        type=int,
        default=512,
        help="Size of tiles in pixels (default: 512)"
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=64,
        help="Overlap between tiles in pixels (default: 64)"
    )

    parser.add_argument(
        "--min-tile-size",
        type=int,
        default=None,
        help="Minimum tile size to keep (default: tile_size/2)"
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="tile",
        help="Prefix for tile filenames (default: 'tile')"
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save tiles to disk (dry run)"
    )

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input_image)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    if not input_path.suffix.lower() in ['.tif', '.tiff']:
        print(f"Warning: Input file may not be a GeoTIFF: {input_path}", file=sys.stderr)

    # Create run directory structure
    run_manager = RunDirectoryManager(output_base=args.output_base)

    if not args.no_save:
        run_dir = run_manager.create_run_directory(
            project_name=args.project,
            input_image=str(input_path.absolute()),
            metadata_extra={
                "tile_size": args.tile_size,
                "overlap": args.overlap,
                "prefix": args.prefix
            }
        )
        tiles_dir = run_dir / "tiles"
        logs_dir = run_dir / "logs"
        print(f"Created run directory: {run_dir}")
        print()
    else:
        # Dry run - just create a temp path
        run_dir = Path(args.output_base) / f"{args.project}_dryrun"
        tiles_dir = run_dir / "tiles"
        logs_dir = run_dir / "logs"

    # Create tiler
    try:
        tiler = ImageTiler(
            tile_size=args.tile_size,
            overlap=args.overlap,
            min_tile_size=args.min_tile_size,
            log_dir=logs_dir if not args.no_save else None
        )
    except ValueError as e:
        print(f"Error: Invalid parameters - {e}", file=sys.stderr)
        return 1

    # Tile the image
    print(f"Tiling image: {input_path}")
    print(f"Run directory: {run_dir}")
    print(f"Tiles directory: {tiles_dir}")
    print(f"Tile size: {args.tile_size}x{args.tile_size} pixels")
    print(f"Overlap: {args.overlap} pixels")
    print(f"Stride: {tiler.stride} pixels")
    print()

    try:
        tile_manager = tiler.tile_image(
            input_path=input_path,
            output_dir=tiles_dir,
            prefix=args.prefix,
            save_tiles=not args.no_save
        )

        print()
        print("=" * 60)
        print("SUCCESS")
        print("=" * 60)
        print(f"Created {len(tile_manager)} tiles")

        if not args.no_save:
            print(f"Run directory: {run_dir}")
            print(f"Tiles saved to: {tiles_dir}")
            print(f"Metadata saved to: {tiles_dir / f'{args.prefix}_metadata.json'}")
        else:
            print("(Dry run - no files saved)")

        print()
        print("Next steps:")
        print(f"  1. Review tiles with: ls -lh {tiles_dir}")
        print(f"  2. Run annotation: python scripts/annotate_tiles.py --run-dir {run_dir}")
        print()

        return 0

    except Exception as e:
        print(f"\nError during tiling: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
