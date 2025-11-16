"""
Review session management for manual annotation review.

Manages the state of the review process, including:
- Loading tiles and masks
- Tracking review status (pending/accepted/rejected/edited)
- Saving reviewed masks
- Session statistics
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from enum import Enum
import numpy as np
import rasterio
from datetime import datetime
import logging


class ReviewStatus(Enum):
    """Status of a tile's review."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    EDITED = "edited"
    SKIPPED = "skipped"


@dataclass
class TileReview:
    """Review information for a single tile."""

    tile_id: str
    """Unique tile identifier"""

    tile_path: str
    """Path to tile image"""

    mask_path: Optional[str] = None
    """Path to mask (if exists)"""

    status: ReviewStatus = ReviewStatus.PENDING
    """Current review status"""

    confidence: float = 0.0
    """Confidence score from zero-shot detection"""

    edited: bool = False
    """Whether mask was manually edited"""

    notes: str = ""
    """Optional review notes"""

    timestamp: Optional[str] = None
    """Timestamp of last review action"""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "tile_id": self.tile_id,
            "tile_path": self.tile_path,
            "mask_path": self.mask_path,
            "status": self.status.value,
            "confidence": self.confidence,
            "edited": self.edited,
            "notes": self.notes,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TileReview":
        """Create from dictionary."""
        return cls(
            tile_id=data["tile_id"],
            tile_path=data["tile_path"],
            mask_path=data.get("mask_path"),
            status=ReviewStatus(data["status"]),
            confidence=data.get("confidence", 0.0),
            edited=data.get("edited", False),
            notes=data.get("notes", ""),
            timestamp=data.get("timestamp"),
        )


class ReviewSession:
    """
    Manages a review session for manual annotation.

    Handles loading tiles, tracking review status, and saving results.
    """

    def __init__(
        self,
        session_dir: Path,
        tiles_dir: Path,
        masks_dir: Optional[Path] = None,
        auto_accept_threshold: float = 0.9,
        log_dir: Optional[Path] = None
    ):
        """
        Initialize review session.

        Args:
            session_dir: Directory to save session data
            tiles_dir: Directory containing tile images
            masks_dir: Directory containing masks (optional)
            auto_accept_threshold: Confidence threshold for auto-accepting masks
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

        self.session_dir = Path(session_dir)
        self.tiles_dir = Path(tiles_dir)
        self.masks_dir = Path(masks_dir) if masks_dir else None
        self.auto_accept_threshold = auto_accept_threshold

        self.logger.info("Initializing review session")
        self.logger.info(f"  Session directory: {self.session_dir}")
        self.logger.info(f"  Tiles directory: {self.tiles_dir}")
        self.logger.info(f"  Masks directory: {self.masks_dir}")
        self.logger.info(f"  Auto-accept threshold: {auto_accept_threshold}")

        # Create session directory
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Storage
        self.reviews: Dict[str, TileReview] = {}
        self.current_index: int = 0

        # Session metadata
        self.session_file = self.session_dir / "session.json"

        # If session_dir already ends with "reviewed_masks", use it directly
        # Otherwise create a reviewed_masks subdirectory
        if self.session_dir.name == "reviewed_masks":
            self.reviewed_masks_dir = self.session_dir
        else:
            self.reviewed_masks_dir = self.session_dir / "reviewed_masks"
            self.reviewed_masks_dir.mkdir(exist_ok=True)

        self.logger.debug(f"Reviewed masks directory: {self.reviewed_masks_dir}")

        # Load or initialize
        if self.session_file.exists():
            self.logger.info(f"Loading existing session from {self.session_file}")
            self.load_session()
        else:
            self.logger.info("Creating new review session")
            self.initialize_session()

    def initialize_session(self):
        """Initialize a new session by scanning for tiles."""
        self.logger.info(f"Scanning for tiles in {self.tiles_dir}")

        # Find all tile files
        tile_files = sorted(self.tiles_dir.glob("*.tif"))

        if not tile_files:
            self.logger.error(f"No tile files found in {self.tiles_dir}")
            raise ValueError(f"No tile files found in {self.tiles_dir}")

        self.logger.info(f"Found {len(tile_files)} tile files")

        # Create review entries
        masks_found = 0
        for tile_path in tile_files:
            tile_id = tile_path.stem

            # Look for corresponding mask
            # Prefer .npz (multi-class) over .tif (merged)
            mask_path = None
            if self.masks_dir:
                # First try .npz format (multi-class with labels)
                potential_mask = self.masks_dir / f"{tile_id}_mask.npz"
                if potential_mask.exists():
                    mask_path = str(potential_mask)
                    masks_found += 1
                    self.logger.debug(f"  {tile_id}: Found NPZ mask (multi-class)")
                else:
                    # Fall back to .tif format (merged mask)
                    potential_mask = self.masks_dir / f"{tile_id}_mask.tif"
                    if potential_mask.exists():
                        mask_path = str(potential_mask)
                        masks_found += 1
                        self.logger.debug(f"  {tile_id}: Found GeoTIFF mask")
                    else:
                        self.logger.debug(f"  {tile_id}: No mask found")

            review = TileReview(
                tile_id=tile_id,
                tile_path=str(tile_path),
                mask_path=mask_path,
            )

            self.reviews[tile_id] = review

        self.logger.info(f"Created {len(self.reviews)} review entries ({masks_found} with masks)")
        self.save_session()

    def load_session(self):
        """Load existing session from disk."""
        self.logger.info(f"Loading session from {self.session_file}")

        with open(self.session_file, 'r') as f:
            data = json.load(f)

        self.current_index = data.get("current_index", 0)

        # Load reviews
        self.reviews = {}
        for review_data in data["reviews"]:
            review = TileReview.from_dict(review_data)
            self.reviews[review.tile_id] = review

        # Log stats
        stats = self.get_statistics()
        self.logger.info(f"Loaded {len(self.reviews)} tile reviews")
        self.logger.info(f"  Pending: {stats['counts']['pending']}")
        self.logger.info(f"  Accepted: {stats['counts']['accepted']}")
        self.logger.info(f"  Rejected: {stats['counts']['rejected']}")
        self.logger.info(f"  Progress: {stats['progress_percent']:.1f}%")

    def save_session(self):
        """Save session to disk."""
        data = {
            "session_dir": str(self.session_dir),
            "tiles_dir": str(self.tiles_dir),
            "masks_dir": str(self.masks_dir) if self.masks_dir else None,
            "current_index": self.current_index,
            "auto_accept_threshold": self.auto_accept_threshold,
            "timestamp": datetime.now().isoformat(),
            "reviews": [review.to_dict() for review in self.reviews.values()],
        }

        with open(self.session_file, 'w') as f:
            json.dump(data, f, indent=2)

    def get_tile_ids(self) -> List[str]:
        """Get list of all tile IDs."""
        return list(self.reviews.keys())

    def get_review(self, tile_id: str) -> Optional[TileReview]:
        """Get review for a specific tile."""
        return self.reviews.get(tile_id)

    def get_current_review(self) -> Optional[TileReview]:
        """Get current tile review."""
        tile_ids = self.get_tile_ids()
        if 0 <= self.current_index < len(tile_ids):
            return self.reviews[tile_ids[self.current_index]]
        return None

    def set_status(
        self,
        tile_id: str,
        status: ReviewStatus,
        notes: str = "",
        edited: bool = False
    ):
        """
        Set review status for a tile.

        Args:
            tile_id: Tile identifier
            status: New status
            notes: Optional notes
            edited: Whether mask was edited
        """
        if tile_id not in self.reviews:
            raise ValueError(f"Unknown tile: {tile_id}")

        review = self.reviews[tile_id]
        review.status = status
        review.notes = notes
        review.edited = edited or review.edited
        review.timestamp = datetime.now().isoformat()

        self.save_session()

    def accept_current(self, notes: str = ""):
        """Accept current tile."""
        current = self.get_current_review()
        if current:
            self.logger.info(f"Accepting tile: {current.tile_id}")
            if notes:
                self.logger.debug(f"  Notes: {notes}")
            self.set_status(current.tile_id, ReviewStatus.ACCEPTED, notes)
            self.next_tile()

    def reject_current(self, notes: str = ""):
        """Reject current tile."""
        current = self.get_current_review()
        if current:
            self.logger.info(f"Rejecting tile: {current.tile_id}")
            if notes:
                self.logger.debug(f"  Notes: {notes}")
            self.set_status(current.tile_id, ReviewStatus.REJECTED, notes)
            self.next_tile()

    def skip_current(self):
        """Skip current tile."""
        current = self.get_current_review()
        if current:
            self.logger.info(f"Skipping tile: {current.tile_id}")
            self.set_status(current.tile_id, ReviewStatus.SKIPPED)
            self.next_tile()

    def save_edited_mask(self, tile_id: str, mask: np.ndarray):
        """
        Save an edited mask.

        Args:
            tile_id: Tile identifier
            mask: Binary mask (H, W) or (1, H, W)
        """
        if tile_id not in self.reviews:
            raise ValueError(f"Unknown tile: {tile_id}")

        review = self.reviews[tile_id]

        # Ensure mask is 2D
        if mask.ndim == 3:
            mask = mask[0]

        # Save as GeoTIFF (preserve geospatial info if possible)
        output_path = self.reviewed_masks_dir / f"{tile_id}_mask.tif"

        # Try to copy georeferencing from original tile
        try:
            with rasterio.open(review.tile_path) as src:
                profile = src.profile.copy()
                profile.update({
                    'count': 1,
                    'dtype': 'uint8',
                    'compress': 'lzw',
                })

                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(mask.astype(np.uint8), 1)
        except Exception as e:
            # Fallback: save as simple GeoTIFF without georeferencing
            print(f"Warning: Could not preserve georeferencing: {e}")
            from rasterio.transform import Affine

            height, width = mask.shape
            transform = Affine.translation(0, 0)

            profile = {
                'driver': 'GTiff',
                'height': height,
                'width': width,
                'count': 1,
                'dtype': 'uint8',
                'compress': 'lzw',
                'transform': transform,
            }

            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(mask.astype(np.uint8), 1)

        # Update review
        review.mask_path = str(output_path)
        review.edited = True
        review.timestamp = datetime.now().isoformat()
        self.save_session()

    def next_tile(self):
        """Move to next tile."""
        tile_ids = self.get_tile_ids()
        if self.current_index < len(tile_ids) - 1:
            self.current_index += 1
            self.save_session()

    def previous_tile(self):
        """Move to previous tile."""
        if self.current_index > 0:
            self.current_index -= 1
            self.save_session()

    def goto_tile(self, index: int):
        """
        Go to specific tile by index.

        Args:
            index: Tile index (0-based)
        """
        tile_ids = self.get_tile_ids()
        if 0 <= index < len(tile_ids):
            self.current_index = index
            self.save_session()

    def get_statistics(self) -> Dict:
        """
        Get review statistics.

        Returns:
            Dictionary with counts and percentages
        """
        total = len(self.reviews)

        counts = {
            "total": total,
            "pending": 0,
            "accepted": 0,
            "rejected": 0,
            "edited": 0,
            "skipped": 0,
        }

        for review in self.reviews.values():
            counts[review.status.value] += 1
            if review.edited:
                counts["edited"] += 1

        # Calculate percentages
        reviewed = counts["accepted"] + counts["rejected"]
        progress = (reviewed / total * 100) if total > 0 else 0

        return {
            "counts": counts,
            "progress_percent": progress,
            "remaining": counts["pending"] + counts["skipped"],
        }

    def get_pending_tiles(self) -> List[TileReview]:
        """Get all pending tiles."""
        return [
            review for review in self.reviews.values()
            if review.status == ReviewStatus.PENDING
        ]

    def get_accepted_tiles(self) -> List[TileReview]:
        """Get all accepted tiles."""
        return [
            review for review in self.reviews.values()
            if review.status == ReviewStatus.ACCEPTED
        ]

    def export_results(self, output_file: Path):
        """
        Export review results to JSON.

        Args:
            output_file: Output file path
        """
        stats = self.get_statistics()

        data = {
            "session_dir": str(self.session_dir),
            "timestamp": datetime.now().isoformat(),
            "statistics": stats,
            "reviews": [review.to_dict() for review in self.reviews.values()],
        }

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Exported results to {output_file}")
