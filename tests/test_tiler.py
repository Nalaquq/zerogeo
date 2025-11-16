"""
Tests for the tiling module.
"""

import pytest
import numpy as np
import rasterio
from rasterio.transform import Affine
from pathlib import Path
import tempfile
import shutil

from river_segmentation.annotation.tiler import ImageTiler, TileManager, TileInfo


@pytest.fixture
def sample_geotiff(tmp_path):
    """Create a sample GeoTIFF for testing."""
    # Create a simple test image (1000x800, 4 bands)
    width, height = 1000, 800
    num_bands = 4

    data = np.random.randint(0, 255, size=(num_bands, height, width), dtype=np.uint8)

    # Create simple affine transform
    transform = Affine.translation(0, 0) * Affine.scale(10, -10)

    filepath = tmp_path / "test_image.tif"

    with rasterio.open(
        filepath,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=num_bands,
        dtype=data.dtype,
        crs='EPSG:4326',
        transform=transform
    ) as dst:
        dst.write(data)

    return filepath


def test_tiler_initialization():
    """Test ImageTiler initialization."""
    tiler = ImageTiler(tile_size=512, overlap=64)
    assert tiler.tile_size == 512
    assert tiler.overlap == 64
    assert tiler.stride == 448  # 512 - 64

    # Test invalid parameters
    with pytest.raises(ValueError):
        ImageTiler(tile_size=100, overlap=100)  # overlap == tile_size

    with pytest.raises(ValueError):
        ImageTiler(tile_size=100, overlap=150)  # overlap > tile_size

    with pytest.raises(ValueError):
        ImageTiler(tile_size=100, overlap=-10)  # negative overlap


def test_calculate_grid():
    """Test grid calculation."""
    tiler = ImageTiler(tile_size=512, overlap=64)

    # Test simple case
    num_cols, num_rows, windows = tiler.calculate_grid(1000, 800)

    assert num_cols > 0
    assert num_rows > 0
    assert len(windows) > 0

    # Check that all windows are within bounds
    for row_off, col_off, tile_width, tile_height in windows:
        assert row_off >= 0
        assert col_off >= 0
        assert row_off + tile_height <= 800
        assert col_off + tile_width <= 1000


def test_tile_image(sample_geotiff, tmp_path):
    """Test tiling a GeoTIFF."""
    tiler = ImageTiler(tile_size=256, overlap=32)
    output_dir = tmp_path / "tiles"

    tile_manager = tiler.tile_image(
        input_path=sample_geotiff,
        output_dir=output_dir,
        prefix="test_tile",
        save_tiles=True
    )

    # Check that tiles were created
    assert len(tile_manager) > 0

    # Check that files exist
    tile_files = list(output_dir.glob("test_tile_*.tif"))
    assert len(tile_files) == len(tile_manager)

    # Check metadata file
    metadata_file = output_dir / "test_tile_metadata.json"
    assert metadata_file.exists()

    # Load a tile and verify it has correct properties
    first_tile = tile_manager.tiles[0]
    assert first_tile.file_path is not None

    with rasterio.open(first_tile.file_path) as src:
        assert src.width <= 256
        assert src.height <= 256
        assert src.count == 4  # Same number of bands
        assert src.crs is not None


def test_tile_manager_serialization(tmp_path):
    """Test TileManager save/load."""
    from rasterio.windows import Window
    from rasterio.transform import Affine

    # Create some sample tiles
    tiles = []
    for i in range(3):
        tile = TileInfo(
            tile_id=f"tile_{i}",
            row=i,
            col=0,
            window=Window(0, i * 100, 100, 100),
            transform=Affine.translation(0, i * 100),
            bounds=(0, i * 100, 100, (i + 1) * 100),
            file_path=f"/tmp/tile_{i}.tif"
        )
        tiles.append(tile)

    manager = TileManager(tiles)

    # Save
    save_path = tmp_path / "tile_metadata.json"
    manager.save(save_path)
    assert save_path.exists()

    # Load
    loaded_manager = TileManager.load(save_path)
    assert len(loaded_manager) == len(manager)

    # Verify content
    for orig, loaded in zip(manager.tiles, loaded_manager.tiles):
        assert orig.tile_id == loaded.tile_id
        assert orig.row == loaded.row
        assert orig.col == loaded.col
        assert orig.bounds == loaded.bounds


def test_get_neighbors():
    """Test getting neighboring tiles."""
    from rasterio.windows import Window
    from rasterio.transform import Affine

    # Create a 3x3 grid
    tiles = []
    for row in range(3):
        for col in range(3):
            tile = TileInfo(
                tile_id=f"tile_{row}_{col}",
                row=row,
                col=col,
                window=Window(col * 100, row * 100, 100, 100),
                transform=Affine.translation(col * 100, row * 100),
                bounds=(col * 100, row * 100, (col + 1) * 100, (row + 1) * 100)
            )
            tiles.append(tile)

    manager = TileManager(tiles)

    # Test center tile (1,1) - should have 4 orthogonal neighbors
    neighbors = manager.get_neighbors("tile_1_1", include_diagonal=False)
    assert len(neighbors) == 4

    neighbor_ids = {n.tile_id for n in neighbors}
    expected = {"tile_0_1", "tile_2_1", "tile_1_0", "tile_1_2"}
    assert neighbor_ids == expected

    # Test with diagonals - should have 8 neighbors
    neighbors_diag = manager.get_neighbors("tile_1_1", include_diagonal=True)
    assert len(neighbors_diag) == 8

    # Test corner tile (0,0) - should have 2 orthogonal neighbors
    corner_neighbors = manager.get_neighbors("tile_0_0", include_diagonal=False)
    assert len(corner_neighbors) == 2


def test_overlap_handling(sample_geotiff, tmp_path):
    """Test that overlap is correctly handled."""
    tiler = ImageTiler(tile_size=256, overlap=64)
    output_dir = tmp_path / "tiles"

    tile_manager = tiler.tile_image(
        input_path=sample_geotiff,
        output_dir=output_dir,
        save_tiles=True
    )

    # Find two adjacent tiles (same row, consecutive columns)
    adjacent_tiles = []
    for tile in tile_manager.tiles:
        if tile.row == 0 and tile.col < 2:
            adjacent_tiles.append(tile)

    if len(adjacent_tiles) >= 2:
        tile1, tile2 = sorted(adjacent_tiles, key=lambda t: t.col)[:2]

        # Check that tiles overlap
        # tile2's left edge should be before tile1's right edge
        assert tile2.window.col_off < tile1.window.col_off + tile1.window.width

        # The overlap should be approximately the specified overlap
        actual_overlap = (tile1.window.col_off + tile1.window.width) - tile2.window.col_off
        assert abs(actual_overlap - 64) < 10  # Allow small variance


def test_weight_map_creation():
    """Test weight map generation for blending."""
    tiler = ImageTiler(tile_size=256, overlap=64)

    weight_map = tiler._create_weight_map(height=256, width=256, overlap=64)

    assert weight_map.shape == (256, 256)

    # Check that center has weight 1.0
    center_weight = weight_map[128, 128]
    assert abs(center_weight - 1.0) < 0.01

    # Check that edges taper down
    edge_weight = weight_map[0, 128]  # Top edge
    assert edge_weight < 1.0

    # Check that weights are in valid range
    assert np.all(weight_map >= 0)
    assert np.all(weight_map <= 1)


def test_reconstruct_from_tiles(sample_geotiff, tmp_path):
    """Test reconstruction from tiles."""
    tiler = ImageTiler(tile_size=256, overlap=64)
    tiles_dir = tmp_path / "tiles"

    # Create tiles
    tile_manager = tiler.tile_image(
        input_path=sample_geotiff,
        output_dir=tiles_dir,
        save_tiles=True
    )

    # Reconstruct
    output_path = tmp_path / "reconstructed.tif"
    tiler.reconstruct_from_tiles(
        tile_manager=tile_manager,
        output_path=output_path,
        original_image_path=sample_geotiff,
        blend_overlap=True
    )

    assert output_path.exists()

    # Compare dimensions
    with rasterio.open(sample_geotiff) as src:
        orig_width, orig_height = src.width, src.height
        orig_bands = src.count

    with rasterio.open(output_path) as dst:
        assert dst.width == orig_width
        assert dst.height == orig_height
        assert dst.count == orig_bands


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
