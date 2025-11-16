"""
Annotation module for zero-shot segmentation and manual review.

This module provides tools for:
- Grid-based tiling of large GeoTIFF images
- Zero-shot annotation using Grounding DINO + SAM
- Manual review UI for annotation quality control
- Export to training-ready datasets
"""

from .tiler import ImageTiler, TileManager
from .review_session import ReviewSession, TileReview, ReviewStatus

# UI imports are optional (require PyQt5)
try:
    from .reviewer_ui import launch_reviewer
    _HAS_UI = True
except ImportError:
    _HAS_UI = False

# Web UI imports are optional (require Flask)
try:
    from .reviewer_webapp import launch_web_reviewer
    _HAS_WEB_UI = True
except ImportError:
    _HAS_WEB_UI = False

# Annotation imports are optional (require Grounding DINO + SAM)
try:
    from .zero_shot_annotator import (
        ZeroShotAnnotator,
        GroundingDINOWrapper,
        SAMWrapper,
        DetectionResult,
        SegmentationResult,
        check_dependencies,
        print_dependency_status,
    )
    from .batch_annotator import BatchAnnotator, annotate_from_config
    _HAS_ANNOTATION = True
except ImportError:
    _HAS_ANNOTATION = False

# Build __all__ based on available modules
__all__ = [
    "ImageTiler",
    "TileManager",
    "ReviewSession",
    "TileReview",
    "ReviewStatus",
]

if _HAS_UI:
    __all__.append("launch_reviewer")

if _HAS_WEB_UI:
    __all__.append("launch_web_reviewer")

if _HAS_ANNOTATION:
    __all__.extend([
        "ZeroShotAnnotator",
        "GroundingDINOWrapper",
        "SAMWrapper",
        "DetectionResult",
        "SegmentationResult",
        "check_dependencies",
        "print_dependency_status",
        "BatchAnnotator",
        "annotate_from_config",
    ])
