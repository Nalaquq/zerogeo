#!/usr/bin/env python3
"""
Annotate tiles using zero-shot models (Grounding DINO + SAM).

Usage:
    python scripts/annotate_tiles.py --config config.yaml
    python scripts/annotate_tiles.py --tiles data/tiles/ --prompts "river" "stream" --output data/masks/

Examples:
    # Using config file (recommended)
    python scripts/annotate_tiles.py --config config/river_annotation_example.yaml

    # Direct usage without config
    python scripts/annotate_tiles.py \\
        --tiles data/tiles/ \\
        --prompts "river" "stream" "waterway" \\
        --output data/masks/ \\
        --dino-config model_configs/GroundingDINO_SwinT_OGC.py \\
        --dino-checkpoint weights/groundingdino_swint_ogc.pth \\
        --sam-checkpoint weights/sam_vit_h_4b8939.pth
"""

import argparse
import sys
from pathlib import Path

# Add parent directory and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from river_segmentation.annotation.zero_shot_annotator import (
    ZeroShotAnnotator,
    check_dependencies,
    print_dependency_status
)
from river_segmentation.annotation.batch_annotator import BatchAnnotator, annotate_from_config
from river_segmentation.annotation.tiler import TileManager
from civic.utils.path_manager import get_run_paths


def main():
    parser = argparse.ArgumentParser(
        description="Annotate tiles using zero-shot models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using config file (recommended)
  %(prog)s --config config/river_annotation_example.yaml

  # Check dependencies
  %(prog)s --check-deps

  # Direct usage
  %(prog)s --tiles data/tiles/ --prompts "river" "stream" --output data/masks/
        """
    )

    # Config-based workflow
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration YAML file"
    )

    # Run directory workflow (recommended)
    parser.add_argument(
        "--run-dir",
        type=str,
        help="Run directory containing tiles/ subdirectory (output/PROJECT_TIMESTAMP/)"
    )

    # Direct usage options (legacy)
    parser.add_argument(
        "--tiles",
        type=str,
        help="Directory containing tiles (with tile_metadata.json)"
    )

    parser.add_argument(
        "--prompts",
        nargs='+',
        help="Text prompts for detection (e.g., 'river' 'stream')"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for masks"
    )

    # Model paths
    parser.add_argument(
        "--dino-config",
        type=str,
        default="model_configs/GroundingDINO_SwinT_OGC.py",
        help="Path to Grounding DINO config file"
    )

    parser.add_argument(
        "--dino-checkpoint",
        type=str,
        default="weights/groundingdino_swint_ogc.pth",
        help="Path to Grounding DINO checkpoint"
    )

    parser.add_argument(
        "--sam-checkpoint",
        type=str,
        default="weights/sam_vit_h_4b8939.pth",
        help="Path to SAM checkpoint"
    )

    parser.add_argument(
        "--sam-model",
        type=str,
        default="vit_h",
        choices=["vit_h", "vit_l", "vit_b"],
        help="SAM model variant"
    )

    # Thresholds
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.35,
        help="Detection confidence threshold (default: 0.35)"
    )

    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.25,
        help="Text similarity threshold (default: 0.25)"
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Device to run on (default: cuda)"
    )

    # Other options
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=None,
        help="Maximum number of tiles to process (for testing)"
    )

    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check annotation dependencies and exit"
    )

    args = parser.parse_args()

    # Check dependencies
    if args.check_deps:
        print_dependency_status()
        deps = check_dependencies()
        return 0 if all(deps.values()) else 1

    # Config-based workflow
    if args.config:
        print("Running annotation from config file...")
        print()

        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}", file=sys.stderr)
            return 1

        try:
            annotate_from_config(config_path)
            return 0
        except Exception as e:
            print(f"\nError during annotation: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return 1

    # Run directory workflow (recommended)
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            print(f"Error: Run directory not found: {run_dir}", file=sys.stderr)
            return 1

        # Get standard paths from run directory
        run_paths = get_run_paths(run_dir)
        tiles_dir = run_paths["tiles"]
        output_dir = run_paths["raw_masks"]
        logs_dir = run_paths["logs"]

        if not tiles_dir.exists():
            print(f"Error: Tiles directory not found: {tiles_dir}", file=sys.stderr)
            print(f"Run tiling first: python scripts/tile_image.py <input> --project <name>", file=sys.stderr)
            return 1

        if not args.prompts:
            print("Error: --prompts required for annotation", file=sys.stderr)
            return 1

    # Direct usage workflow (legacy)
    elif args.tiles and args.output:
        tiles_dir = Path(args.tiles)
        if not tiles_dir.exists():
            print(f"Error: Tiles directory not found: {tiles_dir}", file=sys.stderr)
            return 1

        output_dir = Path(args.output)
        logs_dir = None  # No dedicated logs directory in legacy mode

        if not args.prompts:
            print("Error: --prompts required for annotation", file=sys.stderr)
            return 1

    else:
        print("Error: Must provide either --run-dir or (--tiles and --output)", file=sys.stderr)
        print("Or use --config to run from config file", file=sys.stderr)
        return 1

    # Validate tile metadata
    metadata_path = tiles_dir / "tile_metadata.json"
    if not metadata_path.exists():
        # Try with prefix
        metadata_files = list(tiles_dir.glob("*_metadata.json"))
        if metadata_files:
            metadata_path = metadata_files[0]
        else:
            print(f"Error: Tile metadata not found in: {tiles_dir}", file=sys.stderr)
            print("Run tiling first: python scripts/tile_image.py <input> --project <name>", file=sys.stderr)
            return 1

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check dependencies
    deps = check_dependencies()
    if not all(deps.values()):
        print_dependency_status()
        print("\nError: Missing required dependencies", file=sys.stderr)
        return 1

    # Load tiles
    print(f"Loading tiles from {tiles_dir}...")
    tile_manager = TileManager.load(metadata_path)
    print(f"Found {len(tile_manager)} tiles")
    print()

    # Initialize annotator
    print("Initializing zero-shot annotator...")
    print(f"  Grounding DINO config: {args.dino_config}")
    print(f"  Grounding DINO checkpoint: {args.dino_checkpoint}")
    print(f"  SAM checkpoint: {args.sam_checkpoint}")
    print(f"  SAM model: {args.sam_model}")
    print(f"  Device: {args.device}")
    print()

    try:
        annotator = ZeroShotAnnotator(
            grounding_dino_config=args.dino_config,
            grounding_dino_checkpoint=args.dino_checkpoint,
            sam_checkpoint=args.sam_checkpoint,
            sam_model_type=args.sam_model,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            device=args.device,
        )
    except Exception as e:
        print(f"\nError initializing models: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    # Create batch annotator
    batch_annotator = BatchAnnotator(
        annotator=annotator,
        output_dir=output_dir,
        save_format="geotiff"
    )

    # Run annotation
    try:
        batch_annotator.annotate_tiles(
            tile_manager=tile_manager,
            prompts=args.prompts,
            merge_masks=True,
            max_tiles=args.max_tiles
        )

        print("\nAnnotation complete!")
        print(f"\nNext step: Review annotations with:")
        if args.run_dir:
            print(f"  python scripts/launch_reviewer.py --run-dir {run_dir}")
        else:
            print(f"  python scripts/launch_reviewer.py --tiles {tiles_dir} --masks {output_dir}")
        print()

        return 0

    except Exception as e:
        print(f"\nError during annotation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
