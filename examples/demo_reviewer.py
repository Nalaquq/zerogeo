"""
Demo script for the annotation reviewer UI.

This script:
1. Creates sample tiles from the test image
2. Generates dummy masks for testing
3. Launches the reviewer UI

Usage:
    python examples/demo_reviewer.py
"""

import sys
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import Affine

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from river_segmentation.annotation import ImageTiler, ReviewSession
from river_segmentation.annotation.reviewer_ui import launch_reviewer


def create_demo_tiles():
    """Create demo tiles from sample data."""
    print("Creating demo tiles...")

    input_image = Path("data/Sentinel2_1_bands.tif")
    if not input_image.exists():
        print(f"Error: Sample data not found: {input_image}")
        print("Please ensure you have the sample data file.")
        return None

    output_dir = Path("data/demo_review/tiles")

    # Tile the image (small tiles for demo)
    tiler = ImageTiler(tile_size=256, overlap=32)
    tile_manager = tiler.tile_image(
        input_path=input_image,
        output_dir=output_dir,
        prefix="demo_tile",
        save_tiles=True
    )

    print(f"Created {len(tile_manager)} demo tiles in {output_dir}")
    return output_dir, tile_manager


def create_dummy_masks(tile_manager, masks_dir):
    """Create dummy masks for testing."""
    print("\nCreating dummy masks...")

    masks_dir = Path(masks_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)

    for i, tile in enumerate(tile_manager.tiles[:10]):  # Only first 10 for demo
        # Load tile to get dimensions
        with rasterio.open(tile.file_path) as src:
            width, height = src.width, src.height
            profile = src.profile.copy()

        # Create dummy mask with some random shapes
        mask = np.zeros((height, width), dtype=np.uint8)

        # Add some random "detections"
        num_shapes = np.random.randint(1, 5)
        for _ in range(num_shapes):
            # Random circle
            cx = np.random.randint(50, width - 50)
            cy = np.random.randint(50, height - 50)
            radius = np.random.randint(20, 60)

            y, x = np.ogrid[:height, :width]
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            mask[dist <= radius] = 1

        # Save mask
        mask_path = masks_dir / f"{tile.tile_id}_mask.tif"

        profile.update({
            'count': 1,
            'dtype': 'uint8',
            'compress': 'lzw',
        })

        with rasterio.open(mask_path, 'w', **profile) as dst:
            dst.write(mask, 1)

    print(f"Created dummy masks in {masks_dir}")
    return masks_dir


def main():
    """Run the demo."""
    print("=" * 60)
    print("ANNOTATION REVIEWER DEMO")
    print("=" * 60)
    print()

    # Create demo data
    result = create_demo_tiles()
    if result is None:
        return 1

    tiles_dir, tile_manager = result

    # Create dummy masks
    masks_dir = Path("data/demo_review/masks")
    create_dummy_masks(tile_manager, masks_dir)

    # Create review session
    print("\nInitializing review session...")
    session_dir = Path("data/demo_review/session")
    session = ReviewSession(
        session_dir=session_dir,
        tiles_dir=tiles_dir,
        masks_dir=masks_dir,
        auto_accept_threshold=0.9,
    )

    print(f"Session initialized with {len(session.reviews)} tiles")
    print()

    print("=" * 60)
    print("LAUNCHING REVIEWER UI")
    print("=" * 60)
    print()
    print("Instructions:")
    print("  - Use keyboard shortcuts (A/R/S) for quick review")
    print("  - Enable editing (E key) to draw/erase masks")
    print("  - Adjust overlay transparency with slider")
    print("  - Add notes for any tiles that need attention")
    print()
    print("Keyboard shortcuts:")
    print("  A - Accept tile")
    print("  R - Reject tile")
    print("  S - Skip tile")
    print("  E - Toggle editing mode")
    print("  Left/Right arrows - Navigate tiles")
    print()

    # Launch reviewer
    launch_reviewer(session)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nDemo cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
