"""
Unit tests for zero-shot annotator.

Tests the fixes for Grounding DINO tuple unpacking and class_id handling.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from collections import namedtuple

from src.river_segmentation.annotation.zero_shot_annotator import (
    GroundingDINOWrapper,
    SAMWrapper,
    ZeroShotAnnotator,
    DetectionResult,
    SegmentationResult,
)


# Mock Detections class (from supervision library)
MockDetections = namedtuple('Detections', ['xyxy', 'confidence', 'class_id', 'mask', 'tracker_id'])


class TestGroundingDINOWrapper:
    """Test suite for GroundingDINOWrapper."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock Grounding DINO model."""
        model = Mock()
        return model

    @pytest.fixture
    def wrapper_with_mock(self, mock_model):
        """Create wrapper with mocked model."""
        with patch('src.river_segmentation.annotation.zero_shot_annotator.GroundingDINOModel', return_value=mock_model):
            wrapper = GroundingDINOWrapper(
                model_config="dummy_config.py",
                model_checkpoint="dummy_checkpoint.pth",
                device="cpu"
            )
            wrapper.model = mock_model
            return wrapper

    def test_predict_with_tuple_result(self, wrapper_with_mock):
        """Test handling of tuple result from predict_with_caption (new API)."""
        # Mock the new API that returns (Detections, phrases) tuple
        mock_detections = MockDetections(
            xyxy=np.array([[10, 20, 100, 200], [30, 40, 150, 250]]),
            confidence=np.array([0.9, 0.8]),
            class_id=None,  # New API returns None
            mask=None,
            tracker_id=None
        )
        detected_phrases = ['river', 'river']

        wrapper_with_mock.model.predict_with_caption.return_value = (mock_detections, detected_phrases)

        # Create test image
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

        # Run prediction
        result = wrapper_with_mock.predict(image, prompts=['river'])

        # Verify results
        assert isinstance(result, DetectionResult)
        assert len(result.boxes) == 2
        assert len(result.scores) == 2
        assert len(result.labels) == 2
        assert len(result.phrases) == 2
        assert result.phrases == ['river', 'river']
        assert np.array_equal(result.scores, np.array([0.9, 0.8]))

    def test_predict_with_detections_only(self, wrapper_with_mock):
        """Test handling of Detections object only (old API compatibility)."""
        # Note: The current API returns tuple, but we test that isinstance check works
        # If someone has old API that returns only Detections, this should still work
        # However, since we can't actually force that without patching isinstance,
        # we'll test the tuple format with empty phrases list instead
        mock_detections = MockDetections(
            xyxy=np.array([[10, 20, 100, 200]]),
            confidence=np.array([0.95]),
            class_id=np.array([0]),
            mask=None,
            tracker_id=None
        )

        # Return tuple with empty phrases to simulate API with no phrases
        wrapper_with_mock.model.predict_with_caption.return_value = (mock_detections, [])

        # Create test image
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

        # Run prediction
        result = wrapper_with_mock.predict(image, prompts=['river', 'stream'])

        # Verify results - should fall back to mapping with prompts
        assert isinstance(result, DetectionResult)
        assert len(result.boxes) == 1
        assert result.phrases[0] == 'river'  # Should map to first prompt via class_id

    def test_predict_with_no_detections(self, wrapper_with_mock):
        """Test handling of empty detections."""
        # Mock empty detections
        mock_detections = MockDetections(
            xyxy=np.array([]).reshape(0, 4),
            confidence=np.array([]),
            class_id=None,
            mask=None,
            tracker_id=None
        )

        wrapper_with_mock.model.predict_with_caption.return_value = (mock_detections, [])

        # Create test image
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

        # Run prediction
        result = wrapper_with_mock.predict(image, prompts=['river'])

        # Verify empty results
        assert isinstance(result, DetectionResult)
        assert len(result.boxes) == 0
        assert len(result.scores) == 0
        assert len(result.labels) == 0
        assert len(result.phrases) == 0

    def test_predict_with_class_id_none(self, wrapper_with_mock):
        """Test handling when class_id is None (common in new API)."""
        # Mock detections with class_id = None
        mock_detections = MockDetections(
            xyxy=np.array([[10, 20, 100, 200], [30, 40, 150, 250], [50, 60, 180, 280]]),
            confidence=np.array([0.9, 0.85, 0.92]),
            class_id=None,
            mask=None,
            tracker_id=None
        )
        detected_phrases = ['river', 'river', 'stream']

        wrapper_with_mock.model.predict_with_caption.return_value = (mock_detections, detected_phrases)

        # Create test image
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

        # Run prediction
        result = wrapper_with_mock.predict(image, prompts=['river', 'stream'])

        # Verify labels are created sequentially
        assert len(result.labels) == 3
        assert np.array_equal(result.labels, np.array([0, 1, 2]))
        assert result.phrases == ['river', 'river', 'stream']

    def test_prompt_concatenation(self, wrapper_with_mock):
        """Test that prompts are properly concatenated."""
        mock_detections = MockDetections(
            xyxy=np.array([]).reshape(0, 4),
            confidence=np.array([]),
            class_id=None,
            mask=None,
            tracker_id=None
        )

        wrapper_with_mock.model.predict_with_caption.return_value = (mock_detections, [])

        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

        # Call with multiple prompts
        wrapper_with_mock.predict(image, prompts=['river', 'stream', 'creek'])

        # Verify the caption was constructed correctly
        call_args = wrapper_with_mock.model.predict_with_caption.call_args
        assert call_args[1]['caption'] == 'river. stream. creek.'

    def test_custom_thresholds(self, wrapper_with_mock):
        """Test that custom thresholds override defaults."""
        mock_detections = MockDetections(
            xyxy=np.array([]).reshape(0, 4),
            confidence=np.array([]),
            class_id=None,
            mask=None,
            tracker_id=None
        )

        wrapper_with_mock.model.predict_with_caption.return_value = (mock_detections, [])

        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

        # Call with custom thresholds
        wrapper_with_mock.predict(image, prompts=['river'], box_threshold=0.5, text_threshold=0.3)

        # Verify thresholds were passed
        call_args = wrapper_with_mock.model.predict_with_caption.call_args
        assert call_args[1]['box_threshold'] == 0.5
        assert call_args[1]['text_threshold'] == 0.3


class TestDetectionResult:
    """Test suite for DetectionResult dataclass."""

    def test_detection_result_creation(self):
        """Test creating DetectionResult."""
        boxes = np.array([[10, 20, 100, 200]])
        scores = np.array([0.95])
        labels = np.array([0])
        phrases = ['river']

        result = DetectionResult(
            boxes=boxes,
            scores=scores,
            labels=labels,
            phrases=phrases
        )

        assert np.array_equal(result.boxes, boxes)
        assert np.array_equal(result.scores, scores)
        assert np.array_equal(result.labels, labels)
        assert result.phrases == phrases


class TestSegmentationResult:
    """Test suite for SegmentationResult dataclass."""

    def test_segmentation_result_creation(self):
        """Test creating SegmentationResult."""
        masks = np.random.randint(0, 2, (3, 512, 512), dtype=bool)
        boxes = np.array([[10, 20, 100, 200], [30, 40, 150, 250], [50, 60, 180, 280]])
        scores = np.array([0.9, 0.85, 0.92])
        labels = ['river', 'river', 'stream']

        result = SegmentationResult(
            masks=masks,
            boxes=boxes,
            scores=scores,
            labels=labels
        )

        assert np.array_equal(result.masks, masks)
        assert np.array_equal(result.boxes, boxes)
        assert np.array_equal(result.scores, scores)
        assert result.labels == labels


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
