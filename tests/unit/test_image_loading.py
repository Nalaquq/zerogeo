"""
Unit tests for image loading functionality.

Tests the fix for 32-bit TIFF loading using rasterio.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import rasterio
from rasterio.transform import Affine

from src.river_segmentation.annotation.zero_shot_annotator import load_image_rgb


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def create_test_tiff():
    """Factory fixture to create test TIFF files."""
    def _create_tiff(path, dtype=np.float32, shape=(512, 512), bands=1):
        """Create a test TIFF file with specified parameters."""
        # Create random data
        if bands == 1:
            data = np.random.rand(*shape).astype(dtype)
        else:
            data = np.random.rand(bands, *shape).astype(dtype)

        # Write to file
        profile = {
            'driver': 'GTiff',
            'height': shape[0],
            'width': shape[1],
            'count': bands,
            'dtype': dtype,
            'crs': 'EPSG:4326',
            'transform': Affine.identity(),
        }

        with rasterio.open(path, 'w', **profile) as dst:
            if bands == 1:
                dst.write(data, 1)
            else:
                for i in range(bands):
                    dst.write(data[i], i + 1)

        return path

    return _create_tiff


class TestLoadImageRGB:
    """Test suite for load_image_rgb function."""

    def test_load_float32_single_band(self, temp_dir, create_test_tiff):
        """Test loading single-band float32 TIFF."""
        tiff_path = temp_dir / "test_float32_single.tif"
        create_test_tiff(tiff_path, dtype=np.float32, bands=1)

        # Load image
        image = load_image_rgb(str(tiff_path))

        # Verify output
        assert image.shape == (512, 512, 3), "Should be RGB format"
        assert image.dtype == np.uint8, "Should be uint8"
        assert image.min() >= 0 and image.max() <= 255, "Should be in 0-255 range"

        # All channels should be identical (grayscale)
        assert np.array_equal(image[:, :, 0], image[:, :, 1])
        assert np.array_equal(image[:, :, 1], image[:, :, 2])

    def test_load_float32_multi_band(self, temp_dir, create_test_tiff):
        """Test loading multi-band float32 TIFF."""
        tiff_path = temp_dir / "test_float32_multi.tif"
        create_test_tiff(tiff_path, dtype=np.float32, bands=3)

        # Load image
        image = load_image_rgb(str(tiff_path))

        # Verify output
        assert image.shape == (512, 512, 3), "Should be RGB format"
        assert image.dtype == np.uint8, "Should be uint8"
        assert image.min() >= 0 and image.max() <= 255, "Should be in 0-255 range"

    def test_load_float64_tiff(self, temp_dir, create_test_tiff):
        """Test loading float64 TIFF."""
        tiff_path = temp_dir / "test_float64.tif"
        create_test_tiff(tiff_path, dtype=np.float64, bands=1)

        # Load image
        image = load_image_rgb(str(tiff_path))

        # Verify output
        assert image.shape == (512, 512, 3), "Should be RGB format"
        assert image.dtype == np.uint8, "Should be uint8"

    def test_load_uint16_tiff(self, temp_dir, create_test_tiff):
        """Test loading uint16 TIFF."""
        tiff_path = temp_dir / "test_uint16.tif"
        create_test_tiff(tiff_path, dtype=np.uint16, bands=1)

        # Load image
        image = load_image_rgb(str(tiff_path))

        # Verify output
        assert image.shape == (512, 512, 3), "Should be RGB format"
        assert image.dtype == np.uint8, "Should be uint8"

    def test_load_uint8_tiff(self, temp_dir, create_test_tiff):
        """Test loading uint8 TIFF."""
        tiff_path = temp_dir / "test_uint8.tif"
        create_test_tiff(tiff_path, dtype=np.uint8, bands=3)

        # Load image
        image = load_image_rgb(str(tiff_path))

        # Verify output
        assert image.shape == (512, 512, 3), "Should be RGB format"
        assert image.dtype == np.uint8, "Should be uint8"

    def test_load_four_band_tiff(self, temp_dir, create_test_tiff):
        """Test loading 4-band TIFF (RGBA) - should extract RGB."""
        tiff_path = temp_dir / "test_rgba.tif"
        create_test_tiff(tiff_path, dtype=np.uint8, bands=4)

        # Load image
        image = load_image_rgb(str(tiff_path))

        # Verify output
        assert image.shape == (512, 512, 3), "Should extract RGB from RGBA"
        assert image.dtype == np.uint8, "Should be uint8"

    def test_normalization_preserves_relative_values(self, temp_dir, create_test_tiff):
        """Test that normalization preserves relative values."""
        tiff_path = temp_dir / "test_norm.tif"

        # Create image with known values
        data = np.array([[0.0, 0.5, 1.0]] * 512, dtype=np.float32)

        profile = {
            'driver': 'GTiff',
            'height': 512,
            'width': 3,
            'count': 1,
            'dtype': np.float32,
        }

        with rasterio.open(tiff_path, 'w', **profile) as dst:
            dst.write(data, 1)

        # Load and check normalization
        image = load_image_rgb(str(tiff_path))

        # First column should be darkest, last brightest
        assert image[0, 0, 0] < image[0, 1, 0] < image[0, 2, 0]
        assert image[0, 2, 0] == 255  # Max value should be 255
        assert image[0, 0, 0] == 0    # Min value should be 0

    def test_load_nonexistent_file_raises_error(self):
        """Test that loading non-existent file raises error."""
        with pytest.raises((ValueError, FileNotFoundError, rasterio.errors.RasterioIOError)):
            load_image_rgb("nonexistent_file.tif")

    def test_handles_constant_value_image(self, temp_dir):
        """Test handling of image with all same values."""
        tiff_path = temp_dir / "test_constant.tif"

        # Create image with constant value
        data = np.ones((512, 512), dtype=np.float32) * 0.5

        profile = {
            'driver': 'GTiff',
            'height': 512,
            'width': 512,
            'count': 1,
            'dtype': np.float32,
        }

        with rasterio.open(tiff_path, 'w', **profile) as dst:
            dst.write(data, 1)

        # Should not crash and return valid image
        image = load_image_rgb(str(tiff_path))

        assert image.shape == (512, 512, 3)
        assert image.dtype == np.uint8
        # Constant value images should result in zeros (since max == min)
        assert np.all(image == 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
