"""
Unit tests for Flask reviewer web application.

Tests all API endpoints and functionality.
"""

import pytest
import json
import tempfile
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import Affine
from unittest.mock import Mock, patch

from src.river_segmentation.annotation.reviewer_webapp import ReviewerWebApp
from src.river_segmentation.annotation.review_session import ReviewSession


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def create_test_tiles(temp_dir):
    """Create test tile files."""
    tiles_dir = temp_dir / "tiles"
    tiles_dir.mkdir()

    for i in range(3):
        tile_path = tiles_dir / f"tile_{i}.tif"
        data = np.random.randint(0, 255, (512, 512), dtype=np.uint8)

        profile = {
            'driver': 'GTiff',
            'height': 512,
            'width': 512,
            'count': 1,
            'dtype': np.uint8,
        }

        with rasterio.open(tile_path, 'w', **profile) as dst:
            dst.write(data, 1)

    return tiles_dir


@pytest.fixture
def create_test_masks(temp_dir):
    """Create test mask files."""
    masks_dir = temp_dir / "masks"
    masks_dir.mkdir()

    for i in range(3):
        mask_path = masks_dir / f"tile_{i}_mask.tif"
        mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8)

        profile = {
            'driver': 'GTiff',
            'height': 512,
            'width': 512,
            'count': 1,
            'dtype': np.uint8,
        }

        with rasterio.open(mask_path, 'w', **profile) as dst:
            dst.write(mask, 1)

    return masks_dir


@pytest.fixture
def review_session(temp_dir, create_test_tiles, create_test_masks):
    """Create a review session with test data."""
    session_dir = temp_dir / "session"
    session = ReviewSession(
        session_dir=session_dir,
        tiles_dir=create_test_tiles,
        masks_dir=create_test_masks
    )
    return session


@pytest.fixture
def flask_app(review_session):
    """Create Flask app for testing."""
    webapp = ReviewerWebApp(review_session, host='127.0.0.1', port=5555)
    webapp.app.config['TESTING'] = True
    return webapp.app


@pytest.fixture
def client(flask_app):
    """Create test client."""
    return flask_app.test_client()


class TestFlaskReviewerAPI:
    """Test suite for Flask reviewer API endpoints."""

    def test_index_route(self, client):
        """Test that index route returns HTML."""
        response = client.get('/')
        assert response.status_code == 200
        assert b'<!DOCTYPE html>' in response.data or b'<html' in response.data

    def test_get_status(self, client):
        """Test GET /api/status endpoint."""
        response = client.get('/api/status')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['success'] is True
        assert 'current_tile' in data
        assert 'current_index' in data
        assert 'statistics' in data

        stats = data['statistics']
        assert 'counts' in stats
        assert stats['counts']['total'] == 3

    def test_get_current_tile(self, client):
        """Test GET /api/tile/current endpoint."""
        response = client.get('/api/tile/current')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['success'] is True
        assert 'tile_id' in data
        assert 'index' in data
        assert 'total' in data
        assert 'status' in data
        assert data['total'] == 3

    def test_get_tile_image(self, client):
        """Test GET /api/image/tile endpoint."""
        response = client.get('/api/image/tile')
        assert response.status_code == 200
        assert response.content_type == 'image/png'
        assert len(response.data) > 0

    def test_get_mask_image(self, client):
        """Test GET /api/image/mask endpoint."""
        response = client.get('/api/image/mask')
        assert response.status_code == 200
        assert response.content_type == 'image/png'
        assert len(response.data) > 0

    def test_accept_tile(self, client):
        """Test POST /api/tile/accept endpoint."""
        response = client.post(
            '/api/tile/accept',
            data=json.dumps({'notes': 'Test note'}),
            content_type='application/json'
        )
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['success'] is True
        assert 'message' in data

        # Verify status updated
        status_response = client.get('/api/status')
        status_data = json.loads(status_response.data)
        assert status_data['statistics']['counts']['accepted'] == 1

    def test_reject_tile(self, client):
        """Test POST /api/tile/reject endpoint."""
        response = client.post(
            '/api/tile/reject',
            data=json.dumps({'notes': 'Bad annotation'}),
            content_type='application/json'
        )
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['success'] is True

        # Verify status updated
        status_response = client.get('/api/status')
        status_data = json.loads(status_response.data)
        assert status_data['statistics']['counts']['rejected'] == 1

    def test_skip_tile(self, client):
        """Test POST /api/tile/skip endpoint."""
        response = client.post('/api/tile/skip')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['success'] is True

        # Verify moved to next tile
        status_response = client.get('/api/status')
        status_data = json.loads(status_response.data)
        assert status_data['current_index'] == 1

    def test_navigate_next(self, client):
        """Test POST /api/tile/navigate with direction=next."""
        response = client.post(
            '/api/tile/navigate',
            data=json.dumps({'direction': 'next'}),
            content_type='application/json'
        )
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['success'] is True

        # Verify index increased
        status_response = client.get('/api/status')
        status_data = json.loads(status_response.data)
        assert status_data['current_index'] == 1

    def test_navigate_previous(self, client):
        """Test POST /api/tile/navigate with direction=previous."""
        # First move forward
        client.post(
            '/api/tile/navigate',
            data=json.dumps({'direction': 'next'}),
            content_type='application/json'
        )

        # Then move back
        response = client.post(
            '/api/tile/navigate',
            data=json.dumps({'direction': 'previous'}),
            content_type='application/json'
        )
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['success'] is True

        # Verify back to start
        status_response = client.get('/api/status')
        status_data = json.loads(status_response.data)
        assert status_data['current_index'] == 0

    def test_navigate_invalid_direction(self, client):
        """Test POST /api/tile/navigate with invalid direction."""
        response = client.post(
            '/api/tile/navigate',
            data=json.dumps({'direction': 'invalid'}),
            content_type='application/json'
        )
        assert response.status_code == 400

        data = json.loads(response.data)
        assert data['success'] is False

    def test_update_notes(self, client):
        """Test POST /api/tile/update_notes endpoint."""
        test_notes = "This is a test note"

        response = client.post(
            '/api/tile/update_notes',
            data=json.dumps({'notes': test_notes}),
            content_type='application/json'
        )
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['success'] is True

        # Verify notes were saved
        tile_response = client.get('/api/tile/current')
        tile_data = json.loads(tile_response.data)
        assert tile_data['notes'] == test_notes

    def test_accept_reject_workflow(self, client):
        """Test complete workflow: accept, reject, navigate."""
        # Accept first tile
        client.post(
            '/api/tile/accept',
            data=json.dumps({'notes': 'Good'}),
            content_type='application/json'
        )

        # Should auto-navigate to next tile
        status = json.loads(client.get('/api/status').data)
        assert status['current_index'] == 1
        assert status['statistics']['counts']['accepted'] == 1

        # Reject second tile
        client.post(
            '/api/tile/reject',
            data=json.dumps({'notes': 'Bad'}),
            content_type='application/json'
        )

        # Verify counts
        status = json.loads(client.get('/api/status').data)
        assert status['current_index'] == 2
        assert status['statistics']['counts']['accepted'] == 1
        assert status['statistics']['counts']['rejected'] == 1
        assert status['statistics']['counts']['pending'] == 1

    def test_statistics_calculation(self, client):
        """Test that statistics are calculated correctly."""
        # Accept all tiles
        for i in range(3):
            client.post('/api/tile/accept')

        # Check final statistics
        status = json.loads(client.get('/api/status').data)
        stats = status['statistics']

        assert stats['counts']['total'] == 3
        assert stats['counts']['accepted'] == 3
        assert stats['counts']['rejected'] == 0
        assert stats['counts']['pending'] == 0
        assert stats['progress_percent'] == 100.0

    def test_missing_mask_handling(self, client, temp_dir, create_test_tiles):
        """Test handling of tiles without masks."""
        # Create session without masks
        session_dir = temp_dir / "session_no_masks"
        session = ReviewSession(
            session_dir=session_dir,
            tiles_dir=create_test_tiles,
            masks_dir=None
        )

        webapp = ReviewerWebApp(session)
        webapp.app.config['TESTING'] = True
        test_client = webapp.app.test_client()

        # Should still work
        response = test_client.get('/api/tile/current')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['success'] is True
        assert data['has_mask'] is False

        # Mask image should return empty/transparent
        mask_response = test_client.get('/api/image/mask')
        assert mask_response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
