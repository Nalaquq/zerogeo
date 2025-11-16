"""
Pytest configuration and shared fixtures.

This file is automatically loaded by pytest and provides
fixtures available to all test files.
"""

import pytest
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def project_root_dir():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def sample_image():
    """Create a sample RGB image for testing."""
    import numpy as np
    return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)


@pytest.fixture
def sample_mask():
    """Create a sample binary mask for testing."""
    import numpy as np
    mask = np.zeros((512, 512), dtype=bool)
    mask[100:400, 100:400] = True
    return mask


# Mark tests that require GPU
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "models: mark test as requiring model weights"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


# Skip GPU tests if CUDA not available
def pytest_collection_modifyitems(config, items):
    """Modify test collection based on available resources."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False

    if not cuda_available:
        skip_gpu = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
