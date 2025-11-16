"""Basic tests for river segmentation package."""

import river_segmentation


def test_import():
    """Test that the package can be imported."""
    assert river_segmentation is not None


def test_version():
    """Test that version is defined."""
    assert hasattr(river_segmentation, "__version__")
    assert isinstance(river_segmentation.__version__, str)
