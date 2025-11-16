#!/usr/bin/env python3
"""
Launch the annotation reviewer UI.

Usage:
    python scripts/launch_reviewer.py <tiles_dir> [masks_dir] [--session-dir SESSION_DIR]

Examples:
    # Review tiles without masks
    python scripts/launch_reviewer.py data/tiles/

    # Review tiles with masks
    python scripts/launch_reviewer.py data/tiles/ data/masks/

    # Specify custom session directory
    python scripts/launch_reviewer.py data/tiles/ data/masks/ --session-dir data/review_session/
"""

import argparse
import sys
from pathlib import Path

# Add parent directory and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from river_segmentation.annotation import ReviewSession
from civic.utils.path_manager import get_run_paths

# Try to import both UI options
try:
    from river_segmentation.annotation.reviewer_ui import launch_reviewer
    HAS_QT_UI = True
except ImportError:
    HAS_QT_UI = False

try:
    from river_segmentation.annotation.reviewer_webapp import launch_web_reviewer
    HAS_WEB_UI = True
except ImportError:
    HAS_WEB_UI = False


def main():
    parser = argparse.ArgumentParser(
        description="Launch annotation reviewer UI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Review tiles only
  %(prog)s data/tiles/

  # Review tiles with masks
  %(prog)s data/tiles/ data/masks/

  # Custom session directory
  %(prog)s data/tiles/ data/masks/ --session-dir data/my_review/
        """
    )

    # Run directory workflow (recommended)
    parser.add_argument(
        "--run-dir",
        type=str,
        help="Run directory containing tiles/ and raw_masks/ subdirectories"
    )

    # Legacy positional arguments
    parser.add_argument(
        "tiles_dir",
        type=str,
        nargs='?',
        default=None,
        help="Directory containing tile images (*.tif) [legacy]"
    )

    parser.add_argument(
        "masks_dir",
        type=str,
        nargs='?',
        default=None,
        help="Directory containing masks (optional) [legacy]"
    )

    parser.add_argument(
        "--session-dir",
        type=str,
        default=None,
        help="Directory for review session data (default: tiles_dir/../review_session)"
    )

    parser.add_argument(
        "--auto-accept-threshold",
        type=float,
        default=0.9,
        help="Confidence threshold for auto-accepting masks (default: 0.9)"
    )

    parser.add_argument(
        "--web",
        action="store_true",
        help="Use web-based UI instead of Qt (works in WSL)"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address for web UI (default: 127.0.0.1)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port number for web UI (default: 5000)"
    )

    args = parser.parse_args()

    # Determine paths based on workflow
    if args.run_dir:
        # Run directory workflow (recommended)
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            print(f"Error: Run directory not found: {run_dir}", file=sys.stderr)
            return 1

        # Get standard paths from run directory
        run_paths = get_run_paths(run_dir)
        tiles_dir = run_paths["tiles"]
        masks_dir = run_paths["raw_masks"]

        # Session directory for reviewed masks
        if args.session_dir:
            session_dir = Path(args.session_dir)
        else:
            session_dir = run_paths["reviewed_masks"]

    elif args.tiles_dir:
        # Legacy workflow
        tiles_dir = Path(args.tiles_dir)
        if not tiles_dir.exists():
            print(f"Error: Tiles directory not found: {tiles_dir}", file=sys.stderr)
            return 1

        # Masks directory (optional)
        masks_dir = Path(args.masks_dir) if args.masks_dir else None
        if masks_dir and not masks_dir.exists():
            print(f"Warning: Masks directory not found: {masks_dir}")
            masks_dir = None

        # Session directory
        if args.session_dir:
            session_dir = Path(args.session_dir)
        else:
            session_dir = tiles_dir.parent / "review_session"

    else:
        print("Error: Must provide either --run-dir or tiles_dir", file=sys.stderr)
        return 1

    # Validate tiles directory
    if not tiles_dir.exists():
        print(f"Error: Tiles directory not found: {tiles_dir}", file=sys.stderr)
        return 1

    # Check for tile files
    tile_files = list(tiles_dir.glob("*.tif"))
    if not tile_files:
        print(f"Error: No .tif files found in {tiles_dir}", file=sys.stderr)
        return 1

    print(f"Found {len(tile_files)} tile(s) in {tiles_dir}")
    print(f"Masks directory: {masks_dir}")
    print(f"Session directory: {session_dir}")

    # Create session
    print("\nInitializing review session...")
    session = ReviewSession(
        session_dir=session_dir,
        tiles_dir=tiles_dir,
        masks_dir=masks_dir,
        auto_accept_threshold=args.auto_accept_threshold,
    )

    # Show statistics
    stats = session.get_statistics()
    print(f"\nReview Statistics:")
    print(f"  Total tiles: {stats['counts']['total']}")
    print(f"  Pending: {stats['counts']['pending']}")
    print(f"  Accepted: {stats['counts']['accepted']}")
    print(f"  Rejected: {stats['counts']['rejected']}")
    print(f"  Progress: {stats['progress_percent']:.1f}%")
    print()

    # Default to web UI (works everywhere including WSL)
    # Qt UI is deprecated and only used if explicitly requested with --qt flag
    use_web = not args.web  # Web is now default, --web flag kept for backward compatibility

    if use_web:
        if not HAS_WEB_UI:
            print("Error: Flask not installed. Install with: pip install flask")
            return 1

        print("Launching web-based reviewer...")
        launch_web_reviewer(session, host=args.host, port=args.port)
    else:
        # Qt UI path (deprecated)
        if not HAS_QT_UI:
            print("Error: PyQt5 not installed.")
            print("\nThe Qt UI has been deprecated in favor of the web UI.")
            print("Please remove the --web flag to use the default web UI.")
            return 1

        print("Launching Qt reviewer UI...")
        print("\nKeyboard shortcuts:")
        print("  A - Accept tile")
        print("  R - Reject tile")
        print("  S - Skip tile")
        print("  E - Toggle editing")
        print("  Left/Right arrows - Navigate tiles")
        print()

        launch_reviewer(session)

    return 0


if __name__ == "__main__":
    sys.exit(main())
