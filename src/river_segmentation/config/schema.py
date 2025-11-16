"""
Configuration schema using dataclasses.

Defines the structure and validation for annotation pipeline configuration.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import os


@dataclass
class ProjectConfig:
    """Project-level configuration."""

    name: str
    """Project name"""

    input_image: str
    """Path to input GeoTIFF image"""

    output_dir: str
    """Output directory for all results"""

    description: Optional[str] = None
    """Optional project description"""

    def __post_init__(self):
        """Validate paths."""
        input_path = Path(self.input_image)
        if not input_path.exists():
            raise ValueError(f"Input image not found: {self.input_image}")

        # Expand paths
        self.input_image = str(input_path.resolve())
        self.output_dir = str(Path(self.output_dir).resolve())


@dataclass
class TilingConfig:
    """Tiling configuration."""

    tile_size: int = 512
    """Tile size in pixels (width and height)"""

    overlap: int = 64
    """Overlap between tiles in pixels"""

    min_tile_size: Optional[int] = None
    """Minimum tile size to keep (default: tile_size // 2)"""

    prefix: str = "tile"
    """Prefix for tile filenames"""

    def __post_init__(self):
        """Validate tiling parameters."""
        if self.tile_size <= 0:
            raise ValueError(f"tile_size must be positive, got {self.tile_size}")

        if self.overlap < 0:
            raise ValueError(f"overlap must be non-negative, got {self.overlap}")

        if self.overlap >= self.tile_size:
            raise ValueError(
                f"overlap ({self.overlap}) must be less than tile_size ({self.tile_size})"
            )

        if self.min_tile_size is None:
            self.min_tile_size = self.tile_size // 2

        if self.min_tile_size <= 0:
            raise ValueError(f"min_tile_size must be positive, got {self.min_tile_size}")

    @property
    def stride(self) -> int:
        """Stride between tiles."""
        return self.tile_size - self.overlap


@dataclass
class GroundingDINOConfig:
    """Grounding DINO model configuration."""

    model_checkpoint: str = "groundingdino_swint_ogc.pth"
    """Path to model checkpoint"""

    config_file: str = "GroundingDINO_SwinT_OGC.py"
    """Path to model config file"""

    box_threshold: float = 0.35
    """Confidence threshold for bounding boxes"""

    text_threshold: float = 0.25
    """Text similarity threshold"""

    device: str = "cuda"
    """Device to run on (cuda/cpu)"""

    def __post_init__(self):
        """Validate thresholds."""
        if not 0.0 <= self.box_threshold <= 1.0:
            raise ValueError(f"box_threshold must be in [0, 1], got {self.box_threshold}")

        if not 0.0 <= self.text_threshold <= 1.0:
            raise ValueError(f"text_threshold must be in [0, 1], got {self.text_threshold}")

        if self.device not in ["cuda", "cpu", "mps"]:
            raise ValueError(f"device must be 'cuda', 'cpu', or 'mps', got {self.device}")


@dataclass
class SAMConfig:
    """SAM (Segment Anything Model) configuration."""

    model_type: str = "vit_h"
    """Model variant: vit_h (huge), vit_l (large), or vit_b (base)"""

    checkpoint: str = "sam_vit_h_4b8939.pth"
    """Path to SAM checkpoint"""

    device: str = "cuda"
    """Device to run on (cuda/cpu)"""

    points_per_side: Optional[int] = None
    """Number of points per side for automatic mask generation"""

    pred_iou_thresh: float = 0.88
    """IoU threshold for mask prediction"""

    stability_score_thresh: float = 0.95
    """Stability score threshold for masks"""

    def __post_init__(self):
        """Validate SAM config."""
        valid_types = ["vit_h", "vit_l", "vit_b"]
        if self.model_type not in valid_types:
            raise ValueError(
                f"model_type must be one of {valid_types}, got {self.model_type}"
            )

        if self.device not in ["cuda", "cpu", "mps"]:
            raise ValueError(f"device must be 'cuda', 'cpu', or 'mps', got {self.device}")

        if not 0.0 <= self.pred_iou_thresh <= 1.0:
            raise ValueError(
                f"pred_iou_thresh must be in [0, 1], got {self.pred_iou_thresh}"
            )

        if not 0.0 <= self.stability_score_thresh <= 1.0:
            raise ValueError(
                f"stability_score_thresh must be in [0, 1], got {self.stability_score_thresh}"
            )


@dataclass
class ReviewConfig:
    """Manual review UI configuration."""

    auto_accept_threshold: float = 0.9
    """Auto-accept masks with confidence above this threshold"""

    show_tiles_per_page: int = 16
    """Number of tiles to show per page in grid view"""

    default_view: str = "grid"
    """Default view mode: 'grid' or 'single'"""

    enable_editing: bool = True
    """Enable mask editing in UI"""

    brush_size: int = 10
    """Default brush size for editing"""

    confidence_colors: Dict[str, Tuple[int, int, int]] = field(default_factory=lambda: {
        "high": (0, 255, 0),      # Green
        "medium": (255, 255, 0),  # Yellow
        "low": (255, 0, 0)        # Red
    })
    """RGB colors for different confidence levels"""

    def __post_init__(self):
        """Validate review config."""
        if not 0.0 <= self.auto_accept_threshold <= 1.0:
            raise ValueError(
                f"auto_accept_threshold must be in [0, 1], got {self.auto_accept_threshold}"
            )

        if self.show_tiles_per_page <= 0:
            raise ValueError(
                f"show_tiles_per_page must be positive, got {self.show_tiles_per_page}"
            )

        if self.default_view not in ["grid", "single"]:
            raise ValueError(
                f"default_view must be 'grid' or 'single', got {self.default_view}"
            )

        if self.brush_size <= 0:
            raise ValueError(f"brush_size must be positive, got {self.brush_size}")


@dataclass
class ExportConfig:
    """Export configuration for training dataset."""

    format: str = "geotiff"
    """Output format: 'geotiff', 'numpy', or 'png'"""

    split_ratio: Dict[str, float] = field(default_factory=lambda: {
        "train": 0.7,
        "val": 0.2,
        "test": 0.1
    })
    """Train/val/test split ratios"""

    seed: int = 42
    """Random seed for splitting"""

    create_patches: bool = True
    """Create patches for training"""

    patch_size: int = 256
    """Patch size if create_patches is True"""

    min_mask_coverage: float = 0.01
    """Minimum mask coverage to include patch (fraction of pixels)"""

    def __post_init__(self):
        """Validate export config."""
        valid_formats = ["geotiff", "numpy", "png"]
        if self.format not in valid_formats:
            raise ValueError(
                f"format must be one of {valid_formats}, got {self.format}"
            )

        # Validate split ratios
        if not isinstance(self.split_ratio, dict):
            raise ValueError("split_ratio must be a dictionary")

        required_keys = {"train", "val", "test"}
        if not required_keys.issubset(self.split_ratio.keys()):
            raise ValueError(f"split_ratio must contain keys: {required_keys}")

        total = sum(self.split_ratio.values())
        if not 0.99 <= total <= 1.01:  # Allow small floating point error
            raise ValueError(
                f"split_ratio values must sum to 1.0, got {total}"
            )

        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {self.patch_size}")

        if not 0.0 <= self.min_mask_coverage <= 1.0:
            raise ValueError(
                f"min_mask_coverage must be in [0, 1], got {self.min_mask_coverage}"
            )


@dataclass
class AnnotationConfig:
    """Complete annotation pipeline configuration."""

    project: ProjectConfig
    """Project configuration"""

    tiling: TilingConfig = field(default_factory=TilingConfig)
    """Tiling configuration"""

    grounding_dino: GroundingDINOConfig = field(default_factory=GroundingDINOConfig)
    """Grounding DINO configuration"""

    sam: SAMConfig = field(default_factory=SAMConfig)
    """SAM configuration"""

    prompts: List[str] = field(default_factory=list)
    """Text prompts for detection"""

    review: ReviewConfig = field(default_factory=ReviewConfig)
    """Review UI configuration"""

    export: ExportConfig = field(default_factory=ExportConfig)
    """Export configuration"""

    def __post_init__(self):
        """Validate complete config."""
        if not self.prompts:
            raise ValueError("At least one prompt is required")

        # Ensure output directories
        self._setup_directories()

    def _setup_directories(self):
        """Create output directory structure."""
        base_dir = Path(self.project.output_dir)

        # Create subdirectories
        self.tiles_dir = base_dir / "tiles"
        self.raw_masks_dir = base_dir / "raw_masks"
        self.reviewed_masks_dir = base_dir / "reviewed_masks"
        self.training_data_dir = base_dir / "training_data"
        self.logs_dir = base_dir / "logs"

    def create_directories(self, exist_ok: bool = True):
        """Create all output directories."""
        dirs = [
            self.tiles_dir,
            self.raw_masks_dir,
            self.reviewed_masks_dir,
            self.training_data_dir,
            self.logs_dir,
        ]

        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=exist_ok)

    def get_summary(self) -> str:
        """Get a human-readable summary of the configuration."""
        summary = []
        summary.append("=" * 60)
        summary.append("ANNOTATION PIPELINE CONFIGURATION")
        summary.append("=" * 60)
        summary.append(f"\nProject: {self.project.name}")
        summary.append(f"Input: {self.project.input_image}")
        summary.append(f"Output: {self.project.output_dir}")

        summary.append(f"\nTiling:")
        summary.append(f"  Size: {self.tiling.tile_size}x{self.tiling.tile_size}")
        summary.append(f"  Overlap: {self.tiling.overlap} pixels")
        summary.append(f"  Stride: {self.tiling.stride} pixels")

        summary.append(f"\nGrounding DINO:")
        summary.append(f"  Checkpoint: {self.grounding_dino.model_checkpoint}")
        summary.append(f"  Box threshold: {self.grounding_dino.box_threshold}")
        summary.append(f"  Text threshold: {self.grounding_dino.text_threshold}")

        summary.append(f"\nSAM:")
        summary.append(f"  Model: {self.sam.model_type}")
        summary.append(f"  Checkpoint: {self.sam.checkpoint}")

        summary.append(f"\nPrompts:")
        for prompt in self.prompts:
            summary.append(f"  - {prompt}")

        summary.append(f"\nReview:")
        summary.append(f"  Auto-accept threshold: {self.review.auto_accept_threshold}")
        summary.append(f"  Tiles per page: {self.review.show_tiles_per_page}")

        summary.append(f"\nExport:")
        summary.append(f"  Format: {self.export.format}")
        summary.append(f"  Split: train={self.export.split_ratio['train']}, "
                      f"val={self.export.split_ratio['val']}, "
                      f"test={self.export.split_ratio['test']}")

        summary.append("=" * 60)

        return "\n".join(summary)
