"""
Zero-shot annotation using Grounding DINO + SAM.

This module provides wrappers for Grounding DINO and SAM models
for automated annotation of satellite imagery.
"""

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from dataclasses import dataclass
import warnings
import logging

# Check for optional dependencies
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    warnings.warn("opencv-python not installed. Install with: pip install opencv-python")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    warnings.warn("PyTorch not installed. Required for annotation pipeline.")

# Try to import Grounding DINO
try:
    from groundingdino.util.inference import Model as GroundingDINOModel
    HAS_GROUNDING_DINO = True
except ImportError:
    HAS_GROUNDING_DINO = False
    GroundingDINOModel = None

# Try to import SAM
try:
    from segment_anything import sam_model_registry, SamPredictor
    HAS_SAM = True
except ImportError:
    HAS_SAM = False
    sam_model_registry = None
    SamPredictor = None


@dataclass
class DetectionResult:
    """Result from object detection."""

    boxes: np.ndarray  # (N, 4) bounding boxes in xyxy format
    scores: np.ndarray  # (N,) confidence scores
    labels: np.ndarray  # (N,) label indices
    phrases: List[str]  # (N,) text phrases


@dataclass
class SegmentationResult:
    """Result from segmentation."""

    masks: np.ndarray  # (N, H, W) binary masks
    boxes: np.ndarray  # (N, 4) bounding boxes
    scores: np.ndarray  # (N,) confidence scores
    labels: List[str]  # (N,) text labels


def check_dependencies() -> Dict[str, bool]:
    """
    Check availability of annotation dependencies.

    Returns:
        Dictionary with dependency status
    """
    return {
        "opencv": HAS_OPENCV,
        "torch": HAS_TORCH,
        "grounding_dino": HAS_GROUNDING_DINO,
        "sam": HAS_SAM,
    }


def print_dependency_status():
    """Print status of annotation dependencies."""
    deps = check_dependencies()

    print("Annotation Pipeline Dependencies:")
    print(f"  OpenCV: {'✓' if deps['opencv'] else '✗'}")
    print(f"  PyTorch: {'✓' if deps['torch'] else '✗'}")
    print(f"  Grounding DINO: {'✓' if deps['grounding_dino'] else '✗'}")
    print(f"  SAM: {'✓' if deps['sam'] else '✗'}")

    if not all(deps.values()):
        print("\nTo install missing dependencies:")
        print("  pip install -r requirements-annotation.txt")
        print("\nSee docs/model_setup.md for detailed instructions.")


class GroundingDINOWrapper:
    """
    Wrapper for Grounding DINO model.

    Provides text-based object detection.
    """

    def __init__(
        self,
        model_config: str,
        model_checkpoint: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        device: str = "cuda"
    ):
        """
        Initialize Grounding DINO model.

        Args:
            model_config: Path to model config file
            model_checkpoint: Path to model checkpoint
            box_threshold: Confidence threshold for boxes
            text_threshold: Text similarity threshold
            device: Device to run on ('cuda' or 'cpu')
        """
        if not HAS_GROUNDING_DINO:
            raise ImportError(
                "Grounding DINO not installed. Install with:\n"
                "  pip install groundingdino-py\n"
                "or from source:\n"
                "  git clone https://github.com/IDEA-Research/GroundingDINO.git\n"
                "  cd GroundingDINO && pip install -e ."
            )

        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device

        # Load model
        try:
            self.model = GroundingDINOModel(
                model_config_path=model_config,
                model_checkpoint_path=model_checkpoint,
                device=device
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Grounding DINO model: {e}")

    def predict(
        self,
        image: np.ndarray,
        prompts: List[str],
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None
    ) -> DetectionResult:
        """
        Detect objects in image using text prompts.

        Args:
            image: Image array (H, W, 3) in RGB format
            prompts: List of text prompts (e.g., ["river", "stream"])
            box_threshold: Override default box threshold
            text_threshold: Override default text threshold

        Returns:
            DetectionResult with boxes, scores, labels, and phrases
        """
        box_thresh = box_threshold or self.box_threshold
        text_thresh = text_threshold or self.text_threshold

        # Combine prompts into single query
        caption = ". ".join(prompts) + "."

        # Run detection
        # Note: predict_with_caption returns (Detections, phrases) tuple
        result = self.model.predict_with_caption(
            image=image,
            caption=caption,
            box_threshold=box_thresh,
            text_threshold=text_thresh
        )

        # Unpack the tuple
        if isinstance(result, tuple):
            detections, detected_phrases = result
        else:
            # Fallback for older API versions
            detections = result
            detected_phrases = []

        # Extract results
        boxes = detections.xyxy  # (N, 4)
        scores = detections.confidence  # (N,)

        # class_id may be None, so use detected phrases or create labels from count
        if detections.class_id is not None:
            labels = detections.class_id  # (N,)
        else:
            # Create sequential labels if class_id is None
            labels = np.arange(len(boxes))

        # Use detected phrases if available, otherwise map to prompts
        if detected_phrases and len(detected_phrases) > 0:
            phrases = detected_phrases
        else:
            phrases = [prompts[int(label)] if int(label) < len(prompts) else "unknown"
                      for label in labels]

        return DetectionResult(
            boxes=boxes,
            scores=scores,
            labels=labels,
            phrases=phrases
        )


class SAMWrapper:
    """
    Wrapper for Segment Anything Model (SAM).

    Provides high-quality segmentation from prompts.
    """

    def __init__(
        self,
        model_type: str = "vit_h",
        checkpoint: str = "sam_vit_h_4b8939.pth",
        device: str = "cuda"
    ):
        """
        Initialize SAM model.

        Args:
            model_type: Model variant ('vit_h', 'vit_l', or 'vit_b')
            checkpoint: Path to model checkpoint
            device: Device to run on ('cuda' or 'cpu')
        """
        if not HAS_SAM:
            raise ImportError(
                "Segment Anything not installed. Install with:\n"
                "  pip install segment-anything\n"
                "or from source:\n"
                "  pip install git+https://github.com/facebookresearch/segment-anything.git"
            )

        self.device = device
        self.model_type = model_type

        # Load model
        try:
            sam = sam_model_registry[model_type](checkpoint=checkpoint)
            sam.to(device=device)
            self.predictor = SamPredictor(sam)
        except Exception as e:
            raise RuntimeError(f"Failed to load SAM model: {e}")

    def set_image(self, image: np.ndarray):
        """
        Preprocess image for segmentation.

        Args:
            image: Image array (H, W, 3) in RGB format
        """
        self.predictor.set_image(image)

    def predict_from_boxes(
        self,
        boxes: np.ndarray,
        multimask: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate masks from bounding boxes.

        Args:
            boxes: Bounding boxes (N, 4) in xyxy format
            multimask: Whether to return multiple masks per box

        Returns:
            Tuple of (masks, scores, logits)
            - masks: (N, H, W) binary masks
            - scores: (N,) confidence scores
            - logits: (N, H, W) mask logits
        """
        if len(boxes) == 0:
            return np.array([]), np.array([]), np.array([])

        # Convert to SAM format (N, 4) xyxy
        transformed_boxes = self.predictor.transform.apply_boxes_torch(
            torch.tensor(boxes, device=self.predictor.device),
            self.predictor.original_size
        )

        # Predict masks
        masks_list = []
        scores_list = []

        for box in transformed_boxes:
            mask, score, logit = self.predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=box.unsqueeze(0),
                multimask_output=multimask,
            )

            # Take best mask if multiple
            if multimask:
                best_idx = score.argmax()
                mask = mask[best_idx:best_idx+1]
                score = score[best_idx:best_idx+1]

            masks_list.append(mask.cpu().numpy())
            scores_list.append(score.cpu().numpy())

        masks = np.concatenate(masks_list, axis=0)  # (N, H, W)
        scores = np.concatenate(scores_list, axis=0)  # (N,)

        return masks, scores, None


class ZeroShotAnnotator:
    """
    Complete zero-shot annotation pipeline.

    Combines Grounding DINO (detection) + SAM (segmentation) for
    automated annotation from text prompts.
    """

    def __init__(
        self,
        grounding_dino_config: str,
        grounding_dino_checkpoint: str,
        sam_checkpoint: str,
        sam_model_type: str = "vit_h",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        device: str = "cuda",
        log_dir: Optional[Path] = None
    ):
        """
        Initialize zero-shot annotator.

        Args:
            grounding_dino_config: Path to Grounding DINO config
            grounding_dino_checkpoint: Path to Grounding DINO checkpoint
            sam_checkpoint: Path to SAM checkpoint
            sam_model_type: SAM model variant ('vit_h', 'vit_l', 'vit_b')
            box_threshold: Detection confidence threshold
            text_threshold: Text similarity threshold
            device: Device to run on
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

        # Check dependencies
        self.logger.info("Checking dependencies for zero-shot annotation...")
        deps = check_dependencies()
        if not all(deps.values()):
            self.logger.error("Missing required dependencies")
            print_dependency_status()
            raise RuntimeError("Missing required dependencies for annotation pipeline")
        self.logger.info("All dependencies available")

        # Initialize models
        self.logger.info("Loading Grounding DINO model...")
        self.logger.info(f"  Config: {grounding_dino_config}")
        self.logger.info(f"  Checkpoint: {grounding_dino_checkpoint}")
        self.logger.info(f"  Box threshold: {box_threshold}")
        self.logger.info(f"  Text threshold: {text_threshold}")
        self.logger.info(f"  Device: {device}")

        self.dino = GroundingDINOWrapper(
            model_config=grounding_dino_config,
            model_checkpoint=grounding_dino_checkpoint,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device
        )
        self.logger.info("Grounding DINO loaded successfully")

        self.logger.info("Loading SAM model...")
        self.logger.info(f"  Model type: {sam_model_type}")
        self.logger.info(f"  Checkpoint: {sam_checkpoint}")

        self.sam = SAMWrapper(
            model_type=sam_model_type,
            checkpoint=sam_checkpoint,
            device=device
        )
        self.logger.info("SAM loaded successfully")

        self.logger.info("Zero-shot annotator initialized")

    def annotate(
        self,
        image: np.ndarray,
        prompts: List[str],
        merge_masks: bool = True
    ) -> SegmentationResult:
        """
        Annotate image with text prompts.

        Args:
            image: Image array (H, W, 3) in RGB format
            prompts: Text prompts (e.g., ["river", "stream"])
            merge_masks: Merge overlapping masks into single mask

        Returns:
            SegmentationResult with masks, boxes, scores, and labels
        """
        # Step 1: Detect objects with Grounding DINO
        self.logger.debug(f"Running Grounding DINO detection with prompts: {prompts}")
        detections = self.dino.predict(image, prompts)
        self.logger.debug(f"Grounding DINO found {len(detections.boxes)} detections")

        if len(detections.boxes) == 0:
            self.logger.debug("No detections found, returning empty result")
            # No detections
            h, w = image.shape[:2]
            return SegmentationResult(
                masks=np.zeros((0, h, w), dtype=bool),
                boxes=np.array([]),
                scores=np.array([]),
                labels=[]
            )

        # Log detection details
        for i, (box, score, phrase) in enumerate(zip(detections.boxes, detections.scores, detections.phrases)):
            self.logger.debug(f"  Detection {i}: {phrase} (score: {score:.3f})")

        # Step 2: Segment with SAM
        self.logger.debug("Running SAM segmentation...")
        self.sam.set_image(image)
        masks, sam_scores, _ = self.sam.predict_from_boxes(detections.boxes)
        self.logger.debug(f"SAM generated {len(masks)} masks")

        # Combine scores (DINO confidence * SAM quality)
        combined_scores = detections.scores * sam_scores

        # Convert masks to binary
        masks = (masks > 0.5).astype(bool)

        # Merge overlapping masks if requested
        if merge_masks and len(masks) > 0:
            self.logger.debug("Merging overlapping masks...")
            merged_mask = self._merge_masks(masks)
            self.logger.debug(f"Merged into single mask")
            return SegmentationResult(
                masks=merged_mask[np.newaxis, :, :],  # (1, H, W)
                boxes=detections.boxes,
                scores=combined_scores,
                labels=detections.phrases
            )

        self.logger.debug(f"Returning {len(masks)} individual masks (multi-class)")
        return SegmentationResult(
            masks=masks,
            boxes=detections.boxes,
            scores=combined_scores,
            labels=detections.phrases
        )

    @staticmethod
    def _merge_masks(masks: np.ndarray) -> np.ndarray:
        """
        Merge multiple masks into single mask.

        Args:
            masks: Array of masks (N, H, W)

        Returns:
            Single merged mask (H, W)
        """
        # Simple union of all masks
        return np.any(masks, axis=0)


def load_image_rgb(image_path: str, logger: Optional[logging.Logger] = None) -> np.ndarray:
    """
    Load image in RGB format for annotation.

    DEPRECATED: Use image_utils.load_image_rgb() instead for better functionality.

    Supports both standard images (via OpenCV) and geospatial TIFFs (via rasterio).
    Automatically handles float32 TIFF files and converts them to RGB uint8.

    Args:
        image_path: Path to image file
        logger: Optional logger (for backwards compatibility)

    Returns:
        Image array (H, W, 3) in RGB format (uint8)
    """
    # Use new utility function if available
    try:
        from ..utils.image_utils import load_image_rgb as new_load_rgb
        return new_load_rgb(Path(image_path), logger=logger)
    except ImportError:
        # Fall back to old implementation
        pass
    image_path = str(image_path)

    # Try rasterio first for TIFF/GeoTIFF files (better format support)
    if image_path.lower().endswith(('.tif', '.tiff')):
        try:
            import rasterio
            from rasterio.errors import RasterioIOError

            with rasterio.open(image_path) as src:
                # Read all bands
                data = src.read()  # Shape: (bands, height, width)

                # Transpose to (height, width, bands)
                if data.ndim == 3:
                    data = np.transpose(data, (1, 2, 0))
                else:
                    # Single band - convert to 3-channel grayscale
                    data = data[0]  # Remove band dimension
                    data = np.stack([data, data, data], axis=-1)

                # Normalize to 0-255 range if needed
                if data.dtype in [np.float32, np.float64]:
                    # Assume data is in 0-1 range or needs normalization
                    data_min = np.nanmin(data)
                    data_max = np.nanmax(data)

                    if data_max > data_min:
                        # Normalize to 0-255
                        data = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
                    else:
                        # All same value
                        data = np.zeros_like(data, dtype=np.uint8)
                elif data.dtype == np.uint16:
                    # Convert 16-bit to 8-bit
                    data = (data / 256).astype(np.uint8)
                elif data.dtype != np.uint8:
                    # Convert other types to uint8
                    data = data.astype(np.uint8)

                # Ensure 3 channels
                if data.shape[-1] == 1:
                    data = np.repeat(data, 3, axis=-1)
                elif data.shape[-1] == 4:
                    # RGBA -> RGB
                    data = data[..., :3]
                elif data.shape[-1] != 3:
                    raise ValueError(f"Unexpected number of channels: {data.shape[-1]}")

                return data

        except (ImportError, RasterioIOError) as e:
            # Fall back to OpenCV if rasterio fails
            print(f"Warning: rasterio failed ({e}), trying OpenCV...")

    # Fallback: Use OpenCV for standard image formats
    if not HAS_OPENCV:
        raise ImportError("opencv-python required. Install with: pip install opencv-python")

    # Load with OpenCV (BGR)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image
