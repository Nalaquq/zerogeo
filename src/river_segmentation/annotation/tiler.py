"""
Grid-based tiling for large geospatial images.

This module provides tools for splitting large GeoTIFF images into smaller tiles
with configurable overlap, while preserving geospatial metadata.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import logging

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
from tqdm import tqdm


class TileInfo:
    """Metadata for a single tile."""

    def __init__(
        self,
        tile_id: str,
        row: int,
        col: int,
        window: Window,
        transform: Affine,
        bounds: Tuple[float, float, float, float],
        file_path: Optional[str] = None
    ):
        """
        Initialize tile metadata.

        Args:
            tile_id: Unique identifier (e.g., "tile_0_0")
            row: Row index in grid
            col: Column index in grid
            window: Rasterio window object
            transform: Affine transform for this tile
            bounds: Geographic bounds (minx, miny, maxx, maxy)
            file_path: Path to saved tile file
        """
        self.tile_id = tile_id
        self.row = row
        self.col = col
        self.window = window
        self.transform = transform
        self.bounds = bounds
        self.file_path = file_path

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "tile_id": self.tile_id,
            "row": self.row,
            "col": self.col,
            "window": {
                "col_off": self.window.col_off,
                "row_off": self.window.row_off,
                "width": self.window.width,
                "height": self.window.height
            },
            "transform": list(self.transform)[:6],  # First 6 elements of affine
            "bounds": self.bounds,
            "file_path": self.file_path
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TileInfo":
        """Create from dictionary."""
        window = Window(
            col_off=data["window"]["col_off"],
            row_off=data["window"]["row_off"],
            width=data["window"]["width"],
            height=data["window"]["height"]
        )
        transform = Affine(*data["transform"])

        return cls(
            tile_id=data["tile_id"],
            row=data["row"],
            col=data["col"],
            window=window,
            transform=transform,
            bounds=tuple(data["bounds"]),
            file_path=data.get("file_path")
        )


class TileManager:
    """Manages collection of tiles and their metadata."""

    def __init__(self, tiles: Optional[List[TileInfo]] = None):
        """
        Initialize tile manager.

        Args:
            tiles: List of TileInfo objects
        """
        self.tiles = tiles or []
        self._tile_dict = {tile.tile_id: tile for tile in self.tiles}

    def add_tile(self, tile: TileInfo):
        """Add a tile to the collection."""
        self.tiles.append(tile)
        self._tile_dict[tile.tile_id] = tile

    def get_tile(self, tile_id: str) -> Optional[TileInfo]:
        """Get tile by ID."""
        return self._tile_dict.get(tile_id)

    def get_neighbors(self, tile_id: str, include_diagonal: bool = False) -> List[TileInfo]:
        """
        Get neighboring tiles.

        Args:
            tile_id: ID of the center tile
            include_diagonal: Include diagonal neighbors

        Returns:
            List of neighboring TileInfo objects
        """
        tile = self.get_tile(tile_id)
        if tile is None:
            return []

        neighbors = []
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Top, bottom, left, right

        if include_diagonal:
            offsets.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])

        for dr, dc in offsets:
            neighbor_row = tile.row + dr
            neighbor_col = tile.col + dc

            # Find tile with this row/col
            for t in self.tiles:
                if t.row == neighbor_row and t.col == neighbor_col:
                    neighbors.append(t)
                    break

        return neighbors

    def save(self, filepath: Union[str, Path]):
        """Save tile metadata to JSON file."""
        data = {
            "tiles": [tile.to_dict() for tile in self.tiles],
            "num_tiles": len(self.tiles)
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "TileManager":
        """Load tile metadata from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        tiles = [TileInfo.from_dict(tile_data) for tile_data in data["tiles"]]
        return cls(tiles)

    def __len__(self) -> int:
        return len(self.tiles)

    def __iter__(self):
        return iter(self.tiles)


class ImageTiler:
    """
    Splits large GeoTIFF images into tiles with overlap.

    Handles geospatial metadata preservation and edge cases.
    """

    def __init__(
        self,
        tile_size: int = 512,
        overlap: int = 64,
        min_tile_size: Optional[int] = None,
        log_dir: Optional[Path] = None
    ):
        """
        Initialize tiler.

        Args:
            tile_size: Size of tiles in pixels (width and height)
            overlap: Overlap between adjacent tiles in pixels
            min_tile_size: Minimum tile size to keep (default: tile_size // 2)
            log_dir: Directory for log files (optional)
        """
        # Setup logger
        from ..utils.logging_utils import setup_logger
        self.logger = setup_logger(
            name=__name__,
            log_dir=log_dir,
            level=logging.INFO,
            console=True
        )

        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
        self.min_tile_size = min_tile_size or (tile_size // 2)

        self.logger.info("Initializing ImageTiler")
        self.logger.info(f"  Tile size: {tile_size}x{tile_size}")
        self.logger.info(f"  Overlap: {overlap} pixels")
        self.logger.info(f"  Stride: {self.stride} pixels")
        self.logger.info(f"  Min tile size: {self.min_tile_size} pixels")

        if self.overlap >= self.tile_size:
            self.logger.error(f"Overlap ({overlap}) must be less than tile_size ({tile_size})")
            raise ValueError(f"Overlap ({overlap}) must be less than tile_size ({tile_size})")

        if self.overlap < 0:
            self.logger.error(f"Overlap must be non-negative, got {overlap}")
            raise ValueError(f"Overlap must be non-negative, got {overlap}")

    def calculate_grid(
        self,
        width: int,
        height: int
    ) -> Tuple[int, int, List[Tuple[int, int, int, int]]]:
        """
        Calculate tile grid positions.

        Args:
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            Tuple of (num_cols, num_rows, tile_windows)
            where tile_windows is list of (row_off, col_off, tile_width, tile_height)
        """
        tile_windows = []

        # Calculate number of tiles
        num_cols = max(1, (width - self.overlap) // self.stride)
        num_rows = max(1, (height - self.overlap) // self.stride)

        for row_idx in range(num_rows):
            row_off = row_idx * self.stride
            tile_height = min(self.tile_size, height - row_off)

            # Skip if tile is too small
            if tile_height < self.min_tile_size:
                continue

            for col_idx in range(num_cols):
                col_off = col_idx * self.stride
                tile_width = min(self.tile_size, width - col_off)

                # Skip if tile is too small
                if tile_width < self.min_tile_size:
                    continue

                tile_windows.append((row_off, col_off, tile_width, tile_height))

        return num_cols, num_rows, tile_windows

    def tile_image(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        prefix: str = "tile",
        save_tiles: bool = True
    ) -> TileManager:
        """
        Tile a large image (GeoTIFF or standard format).

        Supports:
        - GeoTIFF with geospatial metadata
        - Standard formats (JPEG, PNG, BMP) - saved as non-georeferenced TIFF tiles

        Args:
            input_path: Path to input image
            output_dir: Directory to save tiles
            prefix: Prefix for tile filenames
            save_tiles: Whether to save tiles to disk

        Returns:
            TileManager with metadata for all tiles
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)

        if save_tiles:
            output_dir.mkdir(parents=True, exist_ok=True)

        tile_manager = TileManager()

        self.logger.info(f"Tiling image: {input_path}")
        self.logger.info(f"  Output directory: {output_dir}")
        self.logger.info(f"  Prefix: {prefix}")
        self.logger.info(f"  Save tiles: {save_tiles}")

        # Try to open as GeoTIFF first
        is_georeferenced = False
        try:
            with rasterio.open(input_path) as src:
                is_georeferenced = src.crs is not None

            if is_georeferenced:
                self.logger.info("  Image type: GeoTIFF (georeferenced)")
                return self._tile_geotiff(input_path, output_dir, prefix, save_tiles, tile_manager)
            else:
                self.logger.info("  Image type: TIFF (non-georeferenced)")
                return self._tile_standard(input_path, output_dir, prefix, save_tiles, tile_manager)

        except Exception:
            # Not a TIFF, try as standard image
            self.logger.info("  Image type: Standard format (JPEG/PNG/etc)")
            return self._tile_standard(input_path, output_dir, prefix, save_tiles, tile_manager)

    def _tile_geotiff(
        self,
        input_path: Path,
        output_dir: Path,
        prefix: str,
        save_tiles: bool,
        tile_manager: TileManager
    ) -> TileManager:
        """Tile a georeferenced GeoTIFF image."""
        with rasterio.open(input_path) as src:
            # Calculate grid
            width, height = src.width, src.height
            num_cols, num_rows, tile_windows = self.calculate_grid(width, height)

            self.logger.info(f"Input image: {width}x{height} pixels")
            self.logger.info(f"  CRS: {src.crs}")
            self.logger.info(f"  Bands: {src.count}")
            self.logger.info(f"  Data type: {src.dtypes[0]}")
            self.logger.info(f"Tile grid: {num_cols}x{num_rows} = {len(tile_windows)} tiles")

            # Process each tile
            for idx, (row_off, col_off, tile_width, tile_height) in enumerate(tqdm(
                tile_windows,
                desc="Creating tiles"
            )):
                # Calculate grid position
                row_idx = row_off // self.stride
                col_idx = col_off // self.stride

                # Create window
                window = Window(col_off, row_off, tile_width, tile_height)

                # Get transform for this tile
                transform = src.window_transform(window)

                # Get bounds
                bounds = rasterio.windows.bounds(window, src.transform)

                # Create tile ID
                tile_id = f"{prefix}_{row_idx}_{col_idx}"

                # File path
                tile_filename = f"{tile_id}.tif" if save_tiles else None
                tile_path = str(output_dir / tile_filename) if save_tiles else None

                # Create tile info
                tile_info = TileInfo(
                    tile_id=tile_id,
                    row=row_idx,
                    col=col_idx,
                    window=window,
                    transform=transform,
                    bounds=bounds,
                    file_path=tile_path
                )

                tile_manager.add_tile(tile_info)

                # Save tile to disk
                if save_tiles:
                    self._save_tile(src, window, transform, tile_path)

            # Save metadata
            if save_tiles:
                metadata_path = output_dir / f"{prefix}_metadata.json"
                tile_manager.save(metadata_path)
                self.logger.info(f"Saved {len(tile_manager)} tiles to {output_dir}")
                self.logger.info(f"Metadata saved to {metadata_path}")

        self.logger.info("Tiling complete")
        return tile_manager

    def _tile_standard(
        self,
        input_path: Path,
        output_dir: Path,
        prefix: str,
        save_tiles: bool,
        tile_manager: TileManager
    ) -> TileManager:
        """Tile a non-georeferenced image (JPEG, PNG, etc.)."""
        from ..utils.image_utils import load_image_rgb, get_image_info

        # Get image info
        img_info = get_image_info(input_path, logger=self.logger)

        self.logger.info(f"Input image: {img_info.width}x{img_info.height} pixels")
        self.logger.info(f"  Bands: {img_info.bands}")
        self.logger.info(f"  Data type: {img_info.dtype}")

        # Load full image
        self.logger.debug("Loading full image into memory...")
        image = load_image_rgb(input_path, logger=self.logger)

        height, width = image.shape[:2]
        num_cols, num_rows, tile_windows = self.calculate_grid(width, height)

        self.logger.info(f"Tile grid: {num_cols}x{num_rows} = {len(tile_windows)} tiles")

        # Process each tile
        for idx, (row_off, col_off, tile_width, tile_height) in enumerate(tqdm(
            tile_windows,
            desc="Creating tiles"
        )):
            # Calculate grid position
            row_idx = row_off // self.stride
            col_idx = col_off // self.stride

            # Create window
            window = Window(col_off, row_off, tile_width, tile_height)

            # No geospatial transform for standard images
            # Use identity transform (pixel coordinates)
            from rasterio.transform import Affine
            transform = Affine.identity()

            # Bounds are just pixel coordinates
            bounds = (col_off, row_off, col_off + tile_width, row_off + tile_height)

            # Create tile ID
            tile_id = f"{prefix}_{row_idx}_{col_idx}"

            # File path
            tile_filename = f"{tile_id}.tif" if save_tiles else None
            tile_path = str(output_dir / tile_filename) if save_tiles else None

            # Create tile info
            tile_info = TileInfo(
                tile_id=tile_id,
                row=row_idx,
                col=col_idx,
                window=window,
                transform=transform,
                bounds=bounds,
                file_path=tile_path
            )

            tile_manager.add_tile(tile_info)

            # Save tile to disk
            if save_tiles:
                # Extract tile from image
                tile_data = image[row_off:row_off+tile_height, col_off:col_off+tile_width]
                self._save_standard_tile(tile_data, tile_path)

        # Save metadata
        if save_tiles:
            metadata_path = output_dir / f"{prefix}_metadata.json"
            tile_manager.save(metadata_path)
            self.logger.info(f"Saved {len(tile_manager)} tiles to {output_dir}")
            self.logger.info(f"Metadata saved to {metadata_path}")

        self.logger.info("Tiling complete")
        return tile_manager

    def _save_standard_tile(self, tile_data: np.ndarray, output_path: str):
        """Save a non-georeferenced tile as TIFF."""
        # Transpose to (bands, height, width) for rasterio
        if tile_data.ndim == 3:
            tile_data = np.transpose(tile_data, (2, 0, 1))
        else:
            tile_data = tile_data[np.newaxis, :, :]

        height, width = tile_data.shape[1:]
        bands = tile_data.shape[0]

        # Save as TIFF without geospatial metadata
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=bands,
            dtype=tile_data.dtype,
            compress='lzw'
        ) as dst:
            dst.write(tile_data)

    def _save_tile(
        self,
        src: rasterio.DatasetReader,
        window: Window,
        transform: Affine,
        output_path: str
    ):
        """
        Save a single tile to disk.

        Args:
            src: Source rasterio dataset
            window: Window to read
            transform: Affine transform for the tile
            output_path: Output file path
        """
        # Read data
        data = src.read(window=window)

        # Copy metadata
        profile = src.profile.copy()
        profile.update({
            'height': window.height,
            'width': window.width,
            'transform': transform
        })

        # Write tile
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data)

    def reconstruct_from_tiles(
        self,
        tile_manager: TileManager,
        output_path: Union[str, Path],
        original_image_path: Union[str, Path],
        blend_overlap: bool = True
    ):
        """
        Reconstruct full image from tiles (useful for merging predictions).

        Args:
            tile_manager: TileManager with tile metadata
            output_path: Path for output reconstructed image
            original_image_path: Path to original image (for metadata)
            blend_overlap: Use weighted blending in overlap regions
        """
        output_path = Path(output_path)

        # Open original for metadata
        with rasterio.open(original_image_path) as src:
            profile = src.profile.copy()
            height, width = src.height, src.width
            num_bands = src.count

            # Create accumulation arrays
            accumulated = np.zeros((num_bands, height, width), dtype=np.float32)
            weights = np.zeros((height, width), dtype=np.float32)

            print(f"Reconstructing {width}x{height} image from {len(tile_manager)} tiles")

            # Process each tile
            for tile_info in tqdm(tile_manager, desc="Merging tiles"):
                if tile_info.file_path is None or not Path(tile_info.file_path).exists():
                    print(f"Warning: Tile {tile_info.tile_id} not found, skipping")
                    continue

                # Load tile
                with rasterio.open(tile_info.file_path) as tile_src:
                    tile_data = tile_src.read()

                # Get window
                window = tile_info.window
                row_start = int(window.row_off)
                row_end = int(window.row_off + window.height)
                col_start = int(window.col_off)
                col_end = int(window.col_off + window.width)

                if blend_overlap:
                    # Create distance-based weight map
                    tile_weights = self._create_weight_map(
                        int(window.height),
                        int(window.width),
                        self.overlap
                    )
                else:
                    # Uniform weights
                    tile_weights = np.ones((int(window.height), int(window.width)))

                # Accumulate
                accumulated[:, row_start:row_end, col_start:col_end] += (
                    tile_data * tile_weights
                )
                weights[row_start:row_end, col_start:col_end] += tile_weights

            # Normalize by weights
            weights = np.maximum(weights, 1e-8)  # Avoid division by zero
            reconstructed = accumulated / weights

            # Convert back to original dtype
            if profile['dtype'] in ['uint8', 'uint16', 'int16']:
                reconstructed = np.clip(reconstructed, 0, np.iinfo(profile['dtype']).max)
                reconstructed = reconstructed.astype(profile['dtype'])

            # Save
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(reconstructed)

            print(f"Saved reconstructed image to {output_path}")

    def _create_weight_map(
        self,
        height: int,
        width: int,
        overlap: int
    ) -> np.ndarray:
        """
        Create distance-based weight map for blending.

        Weights are higher in the center and taper towards edges
        in overlap regions.

        Args:
            height: Tile height
            width: Tile width
            overlap: Overlap size

        Returns:
            Weight map of shape (height, width)
        """
        # Create 1D weight profiles
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


def create_tiles_cli(
    input_image: str,
    output_dir: str,
    tile_size: int = 512,
    overlap: int = 64,
    prefix: str = "tile"
):
    """
    Command-line interface for tiling.

    Args:
        input_image: Path to input GeoTIFF
        output_dir: Output directory for tiles
        tile_size: Tile size in pixels
        overlap: Overlap in pixels
        prefix: Filename prefix
    """
    tiler = ImageTiler(tile_size=tile_size, overlap=overlap)
    tile_manager = tiler.tile_image(
        input_path=input_image,
        output_dir=output_dir,
        prefix=prefix,
        save_tiles=True
    )

    print(f"\nTiling complete!")
    print(f"Created {len(tile_manager)} tiles")
    print(f"Output directory: {output_dir}")

    return tile_manager


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tile large GeoTIFF images")
    parser.add_argument("input_image", help="Path to input GeoTIFF")
    parser.add_argument("output_dir", help="Output directory for tiles")
    parser.add_argument("--tile-size", type=int, default=512, help="Tile size (default: 512)")
    parser.add_argument("--overlap", type=int, default=64, help="Overlap size (default: 64)")
    parser.add_argument("--prefix", default="tile", help="Tile filename prefix")

    args = parser.parse_args()

    create_tiles_cli(
        input_image=args.input_image,
        output_dir=args.output_dir,
        tile_size=args.tile_size,
        overlap=args.overlap,
        prefix=args.prefix
    )
