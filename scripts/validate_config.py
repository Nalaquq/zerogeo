#!/usr/bin/env python3
"""
Configuration validation and inspection tool.

Usage:
    python scripts/validate_config.py config.yaml
    python scripts/validate_config.py config.yaml --show-summary
    python scripts/validate_config.py config.yaml --create-dirs

Examples:
    # Validate config
    python scripts/validate_config.py config/river_annotation_example.yaml

    # Validate and show summary
    python scripts/validate_config.py config/minimal_config.yaml --show-summary

    # Validate and create output directories
    python scripts/validate_config.py config/minimal_config.yaml --create-dirs
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from river_segmentation.config import load_config, validate_config


def main():
    parser = argparse.ArgumentParser(
        description="Validate annotation pipeline configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s config.yaml
  %(prog)s config.yaml --show-summary
  %(prog)s config.yaml --create-dirs
        """
    )

    parser.add_argument(
        "config_file",
        type=str,
        help="Path to configuration YAML file"
    )

    parser.add_argument(
        "--show-summary",
        action="store_true",
        help="Show configuration summary"
    )

    parser.add_argument(
        "--create-dirs",
        action="store_true",
        help="Create output directories"
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress output (only show errors)"
    )

    args = parser.parse_args()

    config_path = Path(args.config_file)

    # Validate
    if not args.quiet:
        print(f"Validating configuration: {config_path}")
        print()

    is_valid, error = validate_config(config_path)

    if not is_valid:
        print(f"❌ Configuration is INVALID", file=sys.stderr)
        print(f"\nError: {error}", file=sys.stderr)
        return 1

    if not args.quiet:
        print("✓ Configuration is valid")
        print()

    # Load config
    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}", file=sys.stderr)
        return 1

    # Show summary
    if args.show_summary:
        print(config.get_summary())
        print()

    # Create directories
    if args.create_dirs:
        if not args.quiet:
            print("Creating output directories...")

        try:
            config.create_directories()

            if not args.quiet:
                print(f"✓ Created directories in: {config.project.output_dir}")
                print(f"  - Tiles: {config.tiles_dir}")
                print(f"  - Raw masks: {config.raw_masks_dir}")
                print(f"  - Reviewed masks: {config.reviewed_masks_dir}")
                print(f"  - Training data: {config.training_data_dir}")
                print(f"  - Logs: {config.logs_dir}")
                print()

        except Exception as e:
            print(f"❌ Failed to create directories: {e}", file=sys.stderr)
            return 1

    # Show quick info
    if not args.show_summary and not args.quiet:
        print("Configuration details:")
        print(f"  Project: {config.project.name}")
        print(f"  Input: {config.project.input_image}")
        print(f"  Output: {config.project.output_dir}")
        print(f"  Tile size: {config.tiling.tile_size}x{config.tiling.tile_size}")
        print(f"  Overlap: {config.tiling.overlap} pixels")
        print(f"  Prompts: {', '.join(config.prompts)}")
        print()
        print("Use --show-summary for full details")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
