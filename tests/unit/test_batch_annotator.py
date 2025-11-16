"""
Unit tests for batch annotator.

Tests the fix for mask dimension handling and GeoTIFF saving.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import rasterio
from rasterio.transform import Affine

from src.river_segmentation.annotation.batch_annotator import BatchAnnotator
from src.river_segmentation.annotation.zero_shot_annotator import SegmentationResult
from src.river_segmentation.annotation.tiler import TileInfo


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_annotator():
    """Create a mock ZeroShotAnnotator."""
    annotator = Mock()
    return annotator


@pytest.fixture
def sample_tile_info(temp_dir):
    """Create a sample TileInfo object with a real TIFF file."""
    # Create a real tile file
    tile_path = temp_dir / "test_tile.tif"
    data = np.random.randint(0, 255, (512, 512), dtype=np.uint8)

    profile = {
        'driver': 'GTiff',
        'height': 512,
        'width': 512,
        'count': 1,
        'dtype': np.uint8,
        'crs': 'EPSG:4326',
        'transform': Affine.identity(),
    }

    with rasterio.open(tile_path, 'w', **profile) as dst:
        dst.write(data, 1)

    # Create TileInfo with correct parameters
    from rasterio.windows import Window
    window = Window(col_off=0, row_off=0, width=512, height=512)
    bounds = (0, 0, 512, 512)  # minx, miny, maxx, maxy

    tile_info = TileInfo(
        tile_id="tile_0_0",
        row=0,
        col=0,
        window=window,
        transform=Affine.identity(),
        bounds=bounds,
        file_path=str(tile_path)
    )

    return tile_info


class TestBatchAnnotator:
    """Test suite for BatchAnnotator."""

    def test_mask_dimension_squeezing_4d(self, mock_annotator, temp_dir, sample_tile_info):
        """Test that 4D masks (1, 1, H, W) are properly squeezed to 2D."""
        batch_annotator = BatchAnnotator(
            annotator=mock_annotator,
            output_dir=temp_dir / "masks",
            save_format="geotiff"
        )

        # Create result with 4D mask (the bug we fixed)
        mask_4d = np.random.randint(0, 2, (1, 1, 512, 512), dtype=bool)
        result = SegmentationResult(
            masks=mask_4d,
            boxes=np.array([[10, 20, 100, 200]]),
            scores=np.array([0.9]),
            labels=['river']
        )

        # Save the result (should not crash)
        batch_annotator._save_result(sample_tile_info, result)

        # Verify mask file was created
        mask_file = temp_dir / "masks" / "tile_0_0_mask.tif"
        assert mask_file.exists()

        # Verify mask is 2D
        with rasterio.open(mask_file) as src:
            saved_mask = src.read(1)
            assert saved_mask.ndim == 2, "Saved mask should be 2D"
            assert saved_mask.shape == (512, 512)
            assert saved_mask.dtype == np.uint8

    def test_mask_dimension_squeezing_3d(self, mock_annotator, temp_dir, sample_tile_info):
        """Test that 3D masks (1, H, W) are properly squeezed to 2D."""
        batch_annotator = BatchAnnotator(
            annotator=mock_annotator,
            output_dir=temp_dir / "masks",
            save_format="geotiff"
        )

        # Create result with 3D mask
        mask_3d = np.random.randint(0, 2, (1, 512, 512), dtype=bool)
        result = SegmentationResult(
            masks=mask_3d,
            boxes=np.array([[10, 20, 100, 200]]),
            scores=np.array([0.9]),
            labels=['river']
        )

        # Save the result
        batch_annotator._save_result(sample_tile_info, result)

        # Verify mask is 2D
        mask_file = temp_dir / "masks" / "tile_0_0_mask.tif"
        with rasterio.open(mask_file) as src:
            saved_mask = src.read(1)
            assert saved_mask.ndim == 2
            assert saved_mask.shape == (512, 512)

    def test_mask_merging_multiple_detections(self, mock_annotator, temp_dir, sample_tile_info):
        """Test that multiple masks are properly merged."""
        batch_annotator = BatchAnnotator(
            annotator=mock_annotator,
            output_dir=temp_dir / "masks",
            save_format="geotiff"
        )

        # Create result with multiple masks
        mask1 = np.zeros((512, 512), dtype=bool)
        mask1[100:200, 100:200] = True

        mask2 = np.zeros((512, 512), dtype=bool)
        mask2[150:250, 150:250] = True

        masks = np.stack([mask1, mask2], axis=0)

        result = SegmentationResult(
            masks=masks,
            boxes=np.array([[100, 100, 200, 200], [150, 150, 250, 250]]),
            scores=np.array([0.9, 0.85]),
            labels=['river', 'river']
        )

        # Save the result
        batch_annotator._save_result(sample_tile_info, result)

        # Verify merged mask
        mask_file = temp_dir / "masks" / "tile_0_0_mask.tif"
        with rasterio.open(mask_file) as src:
            saved_mask = src.read(1)

            # Check overlap region is merged (union)
            assert saved_mask[175, 175] == 1  # Overlap region
            assert saved_mask[120, 120] == 1  # Only mask1
            assert saved_mask[220, 220] == 1  # Only mask2
            assert saved_mask[50, 50] == 0    # Neither mask

    def test_empty_mask_handling(self, mock_annotator, temp_dir, sample_tile_info):
        """Test handling of tiles with no detections."""
        batch_annotator = BatchAnnotator(
            annotator=mock_annotator,
            output_dir=temp_dir / "masks",
            save_format="geotiff"
        )

        # Create result with no masks
        result = SegmentationResult(
            masks=np.array([]),
            boxes=np.array([]),
            scores=np.array([]),
            labels=[]
        )

        # Save the result (should create empty mask)
        batch_annotator._save_result(sample_tile_info, result)

        # Verify empty mask file was created
        mask_file = temp_dir / "masks" / "tile_0_0_mask.tif"
        assert mask_file.exists()

        with rasterio.open(mask_file) as src:
            saved_mask = src.read(1)
            assert saved_mask.shape == (512, 512)
            assert np.all(saved_mask == 0), "Empty mask should be all zeros"

    def test_numpy_format_saving(self, mock_annotator, temp_dir, sample_tile_info):
        """Test saving in numpy format."""
        batch_annotator = BatchAnnotator(
            annotator=mock_annotator,
            output_dir=temp_dir / "masks",
            save_format="numpy"
        )

        mask = np.random.randint(0, 2, (512, 512), dtype=bool)
        boxes = np.array([[10, 20, 100, 200]])
        scores = np.array([0.9])
        labels = ['river']

        result = SegmentationResult(
            masks=mask[np.newaxis, :, :],
            boxes=boxes,
            scores=scores,
            labels=labels
        )

        # Save the result
        batch_annotator._save_result(sample_tile_info, result)

        # Verify npz file was created
        npz_file = temp_dir / "masks" / "tile_0_0_mask.npz"
        assert npz_file.exists()

        # Load and verify contents
        data = np.load(npz_file)
        assert 'masks' in data  # Changed from 'mask' to 'masks' (plural)
        assert 'boxes' in data
        assert 'scores' in data
        assert 'labels' in data
        assert np.array_equal(data['boxes'], boxes)
        assert np.array_equal(data['scores'], scores)

    def test_both_format_saving(self, mock_annotator, temp_dir, sample_tile_info):
        """Test saving in both GeoTIFF and numpy formats."""
        batch_annotator = BatchAnnotator(
            annotator=mock_annotator,
            output_dir=temp_dir / "masks",
            save_format="both"
        )

        mask = np.random.randint(0, 2, (512, 512), dtype=bool)
        result = SegmentationResult(
            masks=mask[np.newaxis, :, :],
            boxes=np.array([[10, 20, 100, 200]]),
            scores=np.array([0.9]),
            labels=['river']
        )

        # Save the result
        batch_annotator._save_result(sample_tile_info, result)

        # Verify both files were created
        tif_file = temp_dir / "masks" / "tile_0_0_mask.tif"
        npz_file = temp_dir / "masks" / "tile_0_0_mask.npz"
        assert tif_file.exists()
        assert npz_file.exists()

    def test_calculate_coverage(self):
        """Test mask coverage calculation."""
        # Test empty masks
        empty_masks = np.zeros((0, 512, 512))
        assert BatchAnnotator._calculate_coverage(empty_masks) == 0.0

        # Test single mask with 25% coverage
        mask = np.zeros((1, 512, 512), dtype=bool)
        mask[0, :256, :256] = True
        coverage = BatchAnnotator._calculate_coverage(mask)
        assert coverage == pytest.approx(0.25, abs=0.01)

        # Test multiple overlapping masks
        mask1 = np.zeros((512, 512), dtype=bool)
        mask1[:256, :] = True  # Top half

        mask2 = np.zeros((512, 512), dtype=bool)
        mask2[:, :256] = True  # Left half

        masks = np.stack([mask1, mask2], axis=0)
        coverage = BatchAnnotator._calculate_coverage(masks)
        # Union should be 75% (top half + left bottom quarter)
        assert coverage == pytest.approx(0.75, abs=0.01)

    def test_georeferencing_preservation(self, mock_annotator, temp_dir, sample_tile_info):
        """Test that georeferencing is preserved in saved masks."""
        batch_annotator = BatchAnnotator(
            annotator=mock_annotator,
            output_dir=temp_dir / "masks",
            save_format="geotiff"
        )

        mask = np.random.randint(0, 2, (512, 512), dtype=bool)
        result = SegmentationResult(
            masks=mask[np.newaxis, :, :],
            boxes=np.array([[10, 20, 100, 200]]),
            scores=np.array([0.9]),
            labels=['river']
        )

        # Save the result
        batch_annotator._save_result(sample_tile_info, result)

        # Verify georeferencing
        mask_file = temp_dir / "masks" / "tile_0_0_mask.tif"
        with rasterio.open(mask_file) as src:
            assert src.crs is not None  # Should have CRS
            assert src.transform is not None  # Should have transform


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
