#!/usr/bin/env python3
"""
CIVIC - Complete Integrated Versatile Image Classification pipeline.

Main command-line interface for running the complete annotation workflow:
  1. Tile large images
  2. Run zero-shot annotation
  3. Launch interactive reviewer
  4. Reassemble masks into complete raster

Usage:
    python scripts/civic.py run config/my_project.yaml
    python scripts/civic.py tile config/my_project.yaml
    python scripts/civic.py annotate config/my_project.yaml
    python scripts/civic.py review config/my_project.yaml
    python scripts/civic.py reassemble config/my_project.yaml

Examples:
    # Run complete pipeline (tile -> annotate -> review -> reassemble)
    python scripts/civic.py run config/river_annotation.yaml

    # Run with auto-reassemble (skip interactive review)
    python scripts/civic.py run config/river_annotation.yaml --auto

    # Skip completed steps
    python scripts/civic.py run config/river_annotation.yaml --skip-existing

    # Run individual steps
    python scripts/civic.py tile config/my_project.yaml
    python scripts/civic.py annotate config/my_project.yaml
    python scripts/civic.py review config/my_project.yaml
    python scripts/civic.py reassemble config/my_project.yaml

    # Dry run (show what would be done)
    python scripts/civic.py run config/my_project.yaml --dry-run
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import yaml
import json
import shutil

from river_segmentation.annotation.tiler import ImageTiler
from river_segmentation.annotation.batch_annotator import annotate_from_config
from river_segmentation.annotation import ReviewSession
from civic.utils.path_manager import RunDirectoryManager, get_run_paths


def load_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config


def run_tiling(config: dict, run_dir: Path, dry_run: bool = False) -> bool:
    """
    Run tiling step.

    Returns:
        True if successful, False otherwise
    """
    print("\n" + "=" * 80)
    print("STEP 1: TILING")
    print("=" * 80)

    # Get configuration
    project = config.get('project', {})
    tiling = config.get('tiling', {})

    input_image = Path(project.get('input_image'))
    if not input_image.exists():
        print(f"Error: Input image not found: {input_image}", file=sys.stderr)
        return False

    tile_size = tiling.get('tile_size', 512)
    overlap = tiling.get('overlap', 64)
    min_tile_size = tiling.get('min_tile_size', None)
    prefix = tiling.get('prefix', 'tile')

    print(f"Input image: {input_image}")
    print(f"Output directory: {run_dir}")
    print(f"Tile size: {tile_size}x{tile_size}")
    print(f"Overlap: {overlap} pixels")
    print(f"Prefix: {prefix}")

    if dry_run:
        print("\n[DRY RUN] Would create tiles in: {run_dir}/tiles/")
        return True

    # Create tiler
    try:
        tiles_dir = run_dir / "tiles"
        logs_dir = run_dir / "logs"

        tiler = ImageTiler(
            tile_size=tile_size,
            overlap=overlap,
            min_tile_size=min_tile_size,
            log_dir=logs_dir
        )

        print("\nTiling image...")
        tile_manager = tiler.tile_image(
            input_path=input_image,
            output_dir=tiles_dir,
            prefix=prefix,
            save_tiles=True
        )

        print(f"\n✓ Created {len(tile_manager)} tiles")
        print(f"  Tiles: {tiles_dir}")
        print(f"  Metadata: {tiles_dir}/{prefix}_metadata.json")
        return True

    except Exception as e:
        print(f"\n✗ Error during tiling: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


def run_annotation(config: dict, run_dir: Path, dry_run: bool = False) -> bool:
    """
    Run annotation step.

    Returns:
        True if successful, False otherwise
    """
    print("\n" + "=" * 80)
    print("STEP 2: ANNOTATION")
    print("=" * 80)

    # Get configuration
    prompts = config.get('prompts', [])
    if not prompts:
        print("Error: No prompts specified in config", file=sys.stderr)
        return False

    print(f"Prompts: {', '.join(prompts)}")
    print(f"Run directory: {run_dir}")

    if dry_run:
        print("\n[DRY RUN] Would annotate tiles with prompts:", prompts)
        return True

    # Create temporary config for annotation
    # Update paths to use run directory (convert new format to old format)
    import copy
    temp_config = copy.deepcopy(config)

    # Remove 'output_base' if present (new format) and set 'output_dir' (old format)
    if 'output_base' in temp_config['project']:
        del temp_config['project']['output_base']
    temp_config['project']['output_dir'] = str(run_dir)

    # Remove fields that are not valid parameters for their respective dataclasses
    # These are either computed properties or extra fields not defined in schema
    fields_to_remove = {
        'tiling': ['stride'],  # stride is a computed property
        'review': ['require_manual_review'],  # not in ReviewConfig
    }

    for section, fields in fields_to_remove.items():
        if section in temp_config:
            for field in fields:
                if field in temp_config[section]:
                    del temp_config[section][field]

    # tiles_dir will be automatically computed from output_dir by AnnotationConfig

    # Save temp config
    temp_config_path = run_dir / "annotation_config.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(temp_config, f)

    print("\nRunning annotation...")
    print("This may take a while depending on the number of tiles and GPU speed...")

    try:
        annotate_from_config(temp_config_path)
        print("\n✓ Annotation complete")
        print(f"  Masks: {run_dir}/raw_masks/")
        return True

    except Exception as e:
        print(f"\n✗ Error during annotation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


def run_review(config: dict, run_dir: Path, auto: bool = False, dry_run: bool = False) -> bool:
    """
    Run review step.

    Returns:
        True if successful, False otherwise
    """
    print("\n" + "=" * 80)
    print("STEP 3: REVIEW")
    print("=" * 80)

    if auto:
        print("Auto mode: Skipping interactive review")
        print("All masks will be automatically accepted")
        if not dry_run:
            # Auto-accept all masks by copying to reviewed_masks
            raw_masks = run_dir / "raw_masks"
            reviewed_masks = run_dir / "reviewed_masks"

            # Ensure reviewed_masks directory exists
            reviewed_masks.mkdir(parents=True, exist_ok=True)

            if raw_masks.exists():
                print(f"\nCopying masks from {raw_masks} to {reviewed_masks}...")
                for mask_file in raw_masks.glob("*_mask.*"):
                    shutil.copy(mask_file, reviewed_masks / mask_file.name)
                mask_count = len(list(reviewed_masks.glob("*_mask.*")))
                print(f"✓ Auto-accepted {mask_count} masks")
            else:
                print(f"Warning: No raw masks found in {raw_masks}")
        return True

    # Get review configuration
    review_config = config.get('review', {})
    auto_accept_threshold = review_config.get('auto_accept_threshold', 0.9)

    print(f"Run directory: {run_dir}")
    print(f"Auto-accept threshold: {auto_accept_threshold}")

    if dry_run:
        print("\n[DRY RUN] Would launch interactive reviewer")
        return True

    # Get paths
    run_paths = get_run_paths(run_dir)
    tiles_dir = run_paths["tiles"]
    masks_dir = run_paths["raw_masks"]
    session_dir = run_paths["reviewed_masks"]

    # Initialize review session
    try:
        session = ReviewSession(
            session_dir=session_dir,
            tiles_dir=tiles_dir,
            masks_dir=masks_dir,
            auto_accept_threshold=auto_accept_threshold
        )

        stats = session.get_statistics()
        print(f"\nReview Statistics:")
        print(f"  Total tiles: {stats['counts']['total']}")
        print(f"  Pending: {stats['counts']['pending']}")
        print(f"  Accepted: {stats['counts']['accepted']}")
        print(f"  Rejected: {stats['counts']['rejected']}")

        # Launch web reviewer
        try:
            from river_segmentation.annotation.reviewer_webapp import launch_web_reviewer
            print("\nLaunching web-based reviewer...")
            print("The reviewer will open in your browser at http://127.0.0.1:5000")
            print("Press Ctrl+C when finished reviewing to continue to reassembly\n")

            launch_web_reviewer(session, host="127.0.0.1", port=5000)
            print("\n✓ Review complete")
            return True

        except ImportError:
            print("Error: Flask not installed. Install with: pip install flask", file=sys.stderr)
            return False

    except Exception as e:
        print(f"\n✗ Error during review: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


def run_reassembly(config: dict, run_dir: Path, dry_run: bool = False) -> bool:
    """
    Run reassembly step.

    Returns:
        True if successful, False otherwise
    """
    print("\n" + "=" * 80)
    print("STEP 4: REASSEMBLY")
    print("=" * 80)

    print(f"Run directory: {run_dir}")

    if dry_run:
        print("\n[DRY RUN] Would reassemble masks into complete GeoTIFF")
        return True

    # Import reassembly function
    from scripts.reassemble_masks import reassemble_masks

    # Get paths
    run_paths = get_run_paths(run_dir)
    tiles_dir = run_paths["tiles"]
    masks_dir = run_paths["reviewed_masks"]

    # Check if reviewed masks exist
    if not masks_dir.exists() or not list(masks_dir.glob("*_mask.*")):
        print("Warning: No reviewed masks found, using raw masks instead")
        masks_dir = run_paths["raw_masks"]

    # Get original image path from run metadata
    metadata_file = run_dir / "run_metadata.json"
    original_image = None

    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
            if 'input_image' in metadata:
                original_image = Path(metadata['input_image'])

    if original_image is None or not original_image.exists():
        # Try to get from config
        project = config.get('project', {})
        original_image = Path(project.get('input_image'))

    if not original_image or not original_image.exists():
        print(f"Error: Original image not found", file=sys.stderr)
        return False

    # Get tiling parameters
    tiling = config.get('tiling', {})
    overlap = tiling.get('overlap', 64)

    # Output path
    output_path = run_dir / "final_mask_complete.tif"

    print(f"Reassembling masks...")
    print(f"  Tiles: {tiles_dir}")
    print(f"  Masks: {masks_dir}")
    print(f"  Original: {original_image}")
    print(f"  Output: {output_path}")

    try:
        reassemble_masks(
            masks_dir=masks_dir,
            tiles_dir=tiles_dir,
            original_image=original_image,
            output_path=output_path,
            overlap=overlap,
            blend_overlap=True,
            binary_threshold=0.5
        )

        print(f"\n✓ Reassembly complete")
        print(f"  Final raster: {output_path}")
        print(f"\nYou can now open this file in:")
        print(f"  - QGIS: qgis {output_path}")
        print(f"  - ArcGIS or any GIS software")
        return True

    except Exception as e:
        print(f"\n✗ Error during reassembly: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


def run_complete_pipeline(
    config_path: Path,
    skip_existing: bool = False,
    auto: bool = False,
    dry_run: bool = False
) -> int:
    """
    Run the complete annotation pipeline.

    Returns:
        0 if successful, 1 otherwise
    """
    print("\n" + "=" * 80)
    print("CIVIC ANNOTATION PIPELINE")
    print("=" * 80)
    print(f"Config: {config_path}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if dry_run:
        print("Mode: DRY RUN (no changes will be made)")
    if auto:
        print("Mode: AUTO (skip interactive review)")
    if skip_existing:
        print("Mode: SKIP EXISTING (resume from previous run)")
    print("=" * 80)

    # Load config
    config = load_config(config_path)

    # Get project info
    project = config.get('project', {})
    project_name = project.get('name', 'unnamed_project')
    output_base = project.get('output_base', 'output')

    # Create or find run directory
    run_manager = RunDirectoryManager(output_base=output_base)

    if skip_existing:
        # Try to find existing run
        latest_run = run_manager.get_latest_run(project_name=project_name)
        if latest_run:
            run_dir = latest_run
            print(f"\nUsing existing run directory: {run_dir}")
        else:
            # Create new run
            if dry_run:
                run_dir = Path(output_base) / f"{project_name}_dryrun"
            else:
                input_image = project.get('input_image')
                tiling = config.get('tiling', {})
                run_dir = run_manager.create_run_directory(
                    project_name=project_name,
                    input_image=input_image,
                    metadata_extra={
                        'tile_size': tiling.get('tile_size', 512),
                        'overlap': tiling.get('overlap', 64),
                        'prompts': config.get('prompts', [])
                    }
                )
                print(f"\nCreated new run directory: {run_dir}")
    else:
        # Always create new run
        if dry_run:
            run_dir = Path(output_base) / f"{project_name}_dryrun"
        else:
            input_image = project.get('input_image')
            tiling = config.get('tiling', {})
            run_dir = run_manager.create_run_directory(
                project_name=project_name,
                input_image=input_image,
                metadata_extra={
                    'tile_size': tiling.get('tile_size', 512),
                    'overlap': tiling.get('overlap', 64),
                    'prompts': config.get('prompts', [])
                }
            )
            print(f"\nCreated new run directory: {run_dir}")

    # Check which steps to run
    run_paths = get_run_paths(run_dir)
    tiles_exist = (run_paths["tiles"] / "tile_metadata.json").exists() or \
                  any(run_paths["tiles"].glob("*_metadata.json"))
    masks_exist = run_paths["raw_masks"].exists() and \
                  any(run_paths["raw_masks"].glob("*_mask.*"))
    reviewed_exist = run_paths["reviewed_masks"].exists() and \
                     any(run_paths["reviewed_masks"].glob("*_mask.*"))
    final_exists = (run_dir / "final_mask_complete.tif").exists()

    # Run pipeline steps
    # Step 1: Tiling
    if skip_existing and tiles_exist:
        print("\n✓ Tiles already exist, skipping tiling step")
    else:
        if not run_tiling(config, run_dir, dry_run):
            return 1

    # Step 2: Annotation
    if skip_existing and masks_exist:
        print("\n✓ Masks already exist, skipping annotation step")
    else:
        if not run_annotation(config, run_dir, dry_run):
            return 1

    # Step 3: Review
    if skip_existing and reviewed_exist:
        print("\n✓ Reviewed masks already exist, skipping review step")
    else:
        if not run_review(config, run_dir, auto=auto, dry_run=dry_run):
            return 1

    # Step 4: Reassembly
    if skip_existing and final_exists:
        print("\n✓ Final raster already exists, skipping reassembly step")
    else:
        if not run_reassembly(config, run_dir, dry_run):
            return 1

    # Success!
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"Run directory: {run_dir}")
    print(f"\nOutputs:")
    print(f"  Tiles: {run_paths['tiles']}")
    print(f"  Raw masks: {run_paths['raw_masks']}")
    print(f"  Reviewed masks: {run_paths['reviewed_masks']}")
    print(f"  Final raster: {run_dir}/final_mask_complete.tif")
    print(f"  Logs: {run_paths['logs']}")
    print("\n" + "=" * 80)

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="CIVIC - Complete annotation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  %(prog)s run config/my_project.yaml

  # Run with auto mode (no interactive review)
  %(prog)s run config/my_project.yaml --auto

  # Skip completed steps
  %(prog)s run config/my_project.yaml --skip-existing

  # Dry run (show what would happen)
  %(prog)s run config/my_project.yaml --dry-run

  # Run individual steps
  %(prog)s tile config/my_project.yaml
  %(prog)s annotate config/my_project.yaml
  %(prog)s review config/my_project.yaml
  %(prog)s reassemble config/my_project.yaml
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Run command (complete pipeline)
    run_parser = subparsers.add_parser('run', help='Run complete pipeline')
    run_parser.add_argument('config', type=str, help='Path to config file')
    run_parser.add_argument('--skip-existing', action='store_true',
                            help='Skip steps that have already been completed')
    run_parser.add_argument('--auto', action='store_true',
                            help='Auto mode: skip interactive review, accept all masks')
    run_parser.add_argument('--dry-run', action='store_true',
                            help='Show what would be done without executing')

    # Individual step commands
    for cmd in ['tile', 'annotate', 'review', 'reassemble']:
        cmd_parser = subparsers.add_parser(cmd, help=f'Run {cmd} step only')
        cmd_parser.add_argument('config', type=str, help='Path to config file')
        if cmd in ['tile', 'annotate', 'reassemble']:
            cmd_parser.add_argument('--dry-run', action='store_true',
                                    help='Show what would be done without executing')
        if cmd == 'review':
            cmd_parser.add_argument('--auto', action='store_true',
                                    help='Auto mode: accept all masks without review')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    config_path = Path(args.config)

    # Load config
    config = load_config(config_path)
    project = config.get('project', {})
    project_name = project.get('name', 'unnamed_project')
    output_base = project.get('output_base', 'output')

    # Get or create run directory
    run_manager = RunDirectoryManager(output_base=output_base)

    if args.command == 'run':
        return run_complete_pipeline(
            config_path,
            skip_existing=args.skip_existing,
            auto=args.auto,
            dry_run=args.dry_run
        )

    else:
        # Individual step commands
        # Try to find latest run directory
        latest_run = run_manager.get_latest_run(project_name=project_name)

        if latest_run:
            run_dir = latest_run
            print(f"Using existing run directory: {run_dir}")
        else:
            # Create new run directory
            input_image = project.get('input_image')
            tiling = config.get('tiling', {})
            run_dir = run_manager.create_run_directory(
                project_name=project_name,
                input_image=input_image,
                metadata_extra={
                    'tile_size': tiling.get('tile_size', 512),
                    'overlap': tiling.get('overlap', 64),
                    'prompts': config.get('prompts', [])
                }
            )
            print(f"Created new run directory: {run_dir}")

        # Run the requested step
        dry_run = getattr(args, 'dry_run', False)
        auto = getattr(args, 'auto', False)

        if args.command == 'tile':
            success = run_tiling(config, run_dir, dry_run)
        elif args.command == 'annotate':
            success = run_annotation(config, run_dir, dry_run)
        elif args.command == 'review':
            success = run_review(config, run_dir, auto=auto)
        elif args.command == 'reassemble':
            success = run_reassembly(config, run_dir, dry_run)

        return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
