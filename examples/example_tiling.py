"""
Example script demonstrating the tiling functionality.

This script shows how to:
1. Tile a large GeoTIFF image with overlap
2. Save and load tile metadata
3. Access individual tiles
4. Find neighboring tiles
5. Reconstruct the full image from tiles
"""

from pathlib import Path
from river_segmentation.annotation.tiler import ImageTiler, TileManager


def main():
    """Main example workflow."""

    # Configuration
    input_image = "data/Sentinel2_1_bands.tif"
    output_dir = "data/tiles/example"
    tile_size = 512
    overlap = 64

    print("=" * 60)
    print("TILING EXAMPLE")
    print("=" * 60)
    print()

    # Step 1: Create tiler
    print("Step 1: Initialize ImageTiler")
    print(f"  Tile size: {tile_size}x{tile_size} pixels")
    print(f"  Overlap: {overlap} pixels")
    print(f"  Stride: {tile_size - overlap} pixels")
    print()

    tiler = ImageTiler(
        tile_size=tile_size,
        overlap=overlap,
        min_tile_size=tile_size // 2  # Keep tiles at least half the standard size
    )

    # Step 2: Tile the image
    print("Step 2: Tile the input image")
    print(f"  Input: {input_image}")
    print(f"  Output: {output_dir}")
    print()

    tile_manager = tiler.tile_image(
        input_path=input_image,
        output_dir=output_dir,
        prefix="tile",
        save_tiles=True
    )

    print()
    print(f"Created {len(tile_manager)} tiles")
    print()

    # Step 3: Explore tile metadata
    print("Step 3: Explore tile metadata")
    print()

    # Show first few tiles
    print("First 5 tiles:")
    for i, tile in enumerate(tile_manager.tiles[:5]):
        print(f"  {tile.tile_id}:")
        print(f"    Position: row={tile.row}, col={tile.col}")
        print(f"    Window: {tile.window}")
        print(f"    Bounds: {tile.bounds}")
        print(f"    File: {Path(tile.file_path).name}")
        print()

    # Step 4: Find neighbors
    print("Step 4: Find neighboring tiles")
    print()

    # Get a tile from the middle
    middle_idx = len(tile_manager) // 2
    center_tile = tile_manager.tiles[middle_idx]

    print(f"Center tile: {center_tile.tile_id} (row={center_tile.row}, col={center_tile.col})")

    # Find orthogonal neighbors
    neighbors = tile_manager.get_neighbors(center_tile.tile_id, include_diagonal=False)
    print(f"  Orthogonal neighbors ({len(neighbors)}):")
    for neighbor in neighbors:
        print(f"    {neighbor.tile_id} (row={neighbor.row}, col={neighbor.col})")

    # Find all neighbors including diagonals
    all_neighbors = tile_manager.get_neighbors(center_tile.tile_id, include_diagonal=True)
    print(f"  All neighbors ({len(all_neighbors)}):")
    for neighbor in all_neighbors:
        print(f"    {neighbor.tile_id} (row={neighbor.row}, col={neighbor.col})")
    print()

    # Step 5: Load metadata from file
    print("Step 5: Load tile metadata from file")
    print()

    metadata_path = Path(output_dir) / "tile_metadata.json"
    print(f"Loading from: {metadata_path}")

    loaded_manager = TileManager.load(metadata_path)
    print(f"Loaded {len(loaded_manager)} tiles")
    print()

    # Step 6: Demonstrate reconstruction (optional)
    print("Step 6: Reconstruct full image from tiles (optional)")
    print()

    reconstructed_path = Path(output_dir) / "reconstructed.tif"
    print(f"This would reconstruct the full image at: {reconstructed_path}")
    print("Skipping actual reconstruction to save time...")
    print()

    # Uncomment to actually run reconstruction:
    # tiler.reconstruct_from_tiles(
    #     tile_manager=tile_manager,
    #     output_path=reconstructed_path,
    #     original_image_path=input_image,
    #     blend_overlap=True
    # )

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total tiles created: {len(tile_manager)}")
    print(f"Output directory: {output_dir}")
    print(f"Metadata file: {metadata_path}")
    print()
    print("Next steps:")
    print("  1. Run zero-shot annotation on these tiles")
    print("  2. Review and validate annotations")
    print("  3. Export to training dataset")
    print("  4. Train U-Net model")
    print()


def tile_with_config():
    """
    Alternative example using custom configuration.

    This demonstrates how to use different tile sizes and overlaps
    for different use cases.
    """

    print("\nAlternative configuration examples:")
    print()

    # Small tiles for detailed annotation
    print("1. Small tiles (256x256) for detailed annotation:")
    small_tiler = ImageTiler(tile_size=256, overlap=32)
    print(f"   Tile size: 256, Overlap: 32, Stride: {small_tiler.stride}")

    # Large tiles for faster processing
    print("\n2. Large tiles (1024x1024) for faster processing:")
    large_tiler = ImageTiler(tile_size=1024, overlap=128)
    print(f"   Tile size: 1024, Overlap: 128, Stride: {large_tiler.stride}")

    # High overlap for better boundary handling
    print("\n3. High overlap (25%) for better boundary handling:")
    overlap_tiler = ImageTiler(tile_size=512, overlap=128)
    print(f"   Tile size: 512, Overlap: 128 (25%), Stride: {overlap_tiler.stride}")

    print()


if __name__ == "__main__":
    # Check if input file exists
    input_file = Path("data/Sentinel2_1_bands.tif")

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        print()
        print("Please ensure you have a GeoTIFF file in the data directory.")
        print("You can use the sample file or provide your own.")
        exit(1)

    # Run main example
    main()

    # Show alternative configurations
    tile_with_config()
