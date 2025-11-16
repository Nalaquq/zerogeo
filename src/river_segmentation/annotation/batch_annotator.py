"""
Batch annotation of tiles using zero-shot models.

Processes multiple tiles and saves results with metadata.
"""

from pathlib import Path
from typing import List, Optional, Dict
import json
from datetime import datetime
import numpy as np
import rasterio
from tqdm import tqdm
import logging

from .tiler import TileManager, TileInfo
from .zero_shot_annotator import (
    ZeroShotAnnotator,
    SegmentationResult,
    check_dependencies,
    print_dependency_status
)
from ..utils.logging_utils import setup_logger, log_exception
from ..utils.image_utils import load_image_rgb, get_image_info


class BatchAnnotator:
    """
    Batch annotation of tiles using zero-shot models.

    Processes tiles from a TileManager and saves results.
    """

    def __init__(
        self,
        annotator: ZeroShotAnnotator,
        output_dir: Path,
        save_format: str = "geotiff",
        log_dir: Optional[Path] = None
    ):
        """
        Initialize batch annotator.

        Args:
            annotator: ZeroShotAnnotator instance
            output_dir: Directory to save masks
            save_format: Output format ('geotiff', 'numpy', or 'both')
            log_dir: Directory for log files (optional)
        """
        self.annotator = annotator
        self.output_dir = Path(output_dir)
        self.save_format = save_format

        # Setup logger
        self.logger = setup_logger(
            name=__name__,
            log_dir=log_dir,
            level=logging.INFO,
            console=True
        )

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Save format: {save_format}")

        # Metadata storage
        self.metadata_file = self.output_dir / "annotation_metadata.json"
        self.metadata = {
            "created": datetime.now().isoformat(),
            "tiles": {}
        }
        self.logger.debug(f"Metadata file: {self.metadata_file}")

    def annotate_tile(
        self,
        tile_info: TileInfo,
        prompts: List[str],
        merge_masks: bool = True
    ) -> Dict:
        """
        Annotate a single tile.

        Args:
            tile_info: TileInfo with tile metadata
            prompts: Text prompts for detection
            merge_masks: Merge overlapping detections

        Returns:
            Dictionary with annotation metadata
        """
        # Load tile image
        try:
            self.logger.debug(f"Loading tile {tile_info.tile_id} from {tile_info.file_path}")

            # Get image info first for logging
            try:
                img_info = get_image_info(Path(tile_info.file_path), logger=self.logger)
                self.logger.debug(f"  Image properties: {img_info}")
            except Exception:
                pass  # Continue even if metadata extraction fails

            # Load image for annotation
            image = load_image_rgb(Path(tile_info.file_path), logger=self.logger)
            self.logger.debug(f"  Loaded as RGB: shape={image.shape}, dtype={image.dtype}")

        except Exception as e:
            self.logger.error(f"Error loading tile {tile_info.tile_id}")
            log_exception(self.logger, e, f"loading tile {tile_info.tile_id}")
            return {
                "tile_id": tile_info.tile_id,
                "success": False,
                "error": str(e)
            }

        # Run annotation
        try:
            self.logger.debug(f"Annotating tile {tile_info.tile_id} with prompts: {prompts}")
            result = self.annotator.annotate(
                image=image,
                prompts=prompts,
                merge_masks=merge_masks
            )
            self.logger.info(f"Tile {tile_info.tile_id}: Found {len(result.masks)} detections")
        except Exception as e:
            self.logger.error(f"Error annotating tile {tile_info.tile_id}")
            log_exception(self.logger, e, f"annotating tile {tile_info.tile_id}")
            return {
                "tile_id": tile_info.tile_id,
                "success": False,
                "error": str(e)
            }

        # Save masks
        try:
            self.logger.debug(f"Saving results for tile {tile_info.tile_id}")
            self._save_result(tile_info, result)
            self.logger.debug(f"Saved masks for tile {tile_info.tile_id}")
        except Exception as e:
            self.logger.error(f"Error saving results for {tile_info.tile_id}")
            log_exception(self.logger, e, f"saving tile {tile_info.tile_id}")
            return {
                "tile_id": tile_info.tile_id,
                "success": False,
                "error": str(e)
            }

        # Create metadata
        metadata = {
            "tile_id": tile_info.tile_id,
            "success": True,
            "num_detections": len(result.masks),
            "confidence_scores": result.scores.tolist() if len(result.scores) > 0 else [],
            "labels": result.labels,
            "mean_confidence": float(np.mean(result.scores)) if len(result.scores) > 0 else 0.0,
            "max_confidence": float(np.max(result.scores)) if len(result.scores) > 0 else 0.0,
            "mask_coverage": self._calculate_coverage(result.masks),
            "timestamp": datetime.now().isoformat(),
        }

        return metadata

    def annotate_tiles(
        self,
        tile_manager: TileManager,
        prompts: List[str],
        merge_masks: bool = True,
        max_tiles: Optional[int] = None
    ):
        """
        Annotate multiple tiles.

        Args:
            tile_manager: TileManager with tiles to process
            prompts: Text prompts for detection
            merge_masks: Merge overlapping detections
            max_tiles: Maximum number of tiles to process (None = all)
        """
        tiles = tile_manager.tiles[:max_tiles] if max_tiles else tile_manager.tiles

        self.logger.info("="* 60)
        self.logger.info(f"Annotating {len(tiles)} tiles")
        self.logger.info(f"Prompts: {', '.join(prompts)}")
        self.logger.info(f"Merge masks: {merge_masks}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info("=" * 60)

        # Process tiles
        start_time = datetime.now()
        for tile_info in tqdm(tiles, desc="Annotating tiles"):
            metadata = self.annotate_tile(tile_info, prompts, merge_masks)
            self.metadata["tiles"][tile_info.tile_id] = metadata

        elapsed = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"Annotation completed in {elapsed:.2f} seconds ({elapsed/len(tiles):.2f}s per tile)")

        # Save metadata
        self._save_metadata()

        # Print summary
        self._print_summary()

    def _save_result(self, tile_info: TileInfo, result: SegmentationResult):
        """Save annotation result."""
        tile_id = tile_info.tile_id

        # For GeoTIFF: save merged mask for compatibility
        if self.save_format in ["geotiff", "both"]:
            if len(result.masks) == 0:
                # No detections - save empty mask
                merged_mask = np.zeros((512, 512), dtype=np.uint8)
            else:
                # Merge all masks
                merged_mask = np.any(result.masks, axis=0).astype(np.uint8)

            output_path = self.output_dir / f"{tile_id}_mask.tif"
            self._save_geotiff(tile_info, merged_mask, output_path)

        # For NPZ: save individual masks with labels (IMPORTANT for multi-class support!)
        if self.save_format in ["numpy", "both"]:
            output_path = self.output_dir / f"{tile_id}_mask.npz"

            if len(result.masks) == 0:
                # Save empty arrays
                np.savez_compressed(
                    output_path,
                    masks=np.zeros((0, 512, 512), dtype=np.uint8),
                    boxes=np.array([]),
                    scores=np.array([]),
                    labels=np.array([], dtype=object)
                )
            else:
                # Save individual masks with their labels
                # This preserves class information for the reviewer!
                np.savez_compressed(
                    output_path,
                    masks=result.masks.astype(np.uint8),  # (N, H, W) - one mask per detection
                    boxes=result.boxes,
                    scores=result.scores,
                    labels=np.array(result.labels, dtype=object)  # Preserve class labels
                )

    def _save_geotiff(self, tile_info: TileInfo, mask: np.ndarray, output_path: Path):
        """Save mask as GeoTIFF with georeferencing."""
        # Ensure mask is 2D
        if mask.ndim > 2:
            mask = mask.squeeze()
        if mask.ndim == 1:
            # Handle edge case of 1D mask
            size = int(np.sqrt(len(mask)))
            mask = mask.reshape(size, size)

        try:
            # Try to copy georeferencing from original tile
            with rasterio.open(tile_info.file_path) as src:
                profile = src.profile.copy()
                profile.update({
                    'count': 1,
                    'dtype': 'uint8',
                    'compress': 'lzw',
                    'transform': tile_info.transform,
                })

                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(mask, 1)
        except Exception as e:
            # Fallback: save without georeferencing
            print(f"Warning: Could not preserve georeferencing: {e}")

            if mask.ndim != 2:
                print(f"Error: Mask has unexpected dimensions: {mask.shape}")
                return

            height, width = mask.shape
            profile = {
                'driver': 'GTiff',
                'height': height,
                'width': width,
                'count': 1,
                'dtype': 'uint8',
                'compress': 'lzw',
            }

            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(mask, 1)

    @staticmethod
    def _calculate_coverage(masks: np.ndarray) -> float:
        """Calculate percentage of image covered by masks."""
        if len(masks) == 0:
            return 0.0

        merged = np.any(masks, axis=0)
        coverage = np.sum(merged) / merged.size
        return float(coverage)

    def _save_metadata(self):
        """Save annotation metadata to JSON."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def _print_summary(self):
        """Print annotation summary."""
        total = len(self.metadata["tiles"])
        success = sum(1 for m in self.metadata["tiles"].values() if m.get("success", False))
        failed = total - success

        total_detections = sum(
            m.get("num_detections", 0)
            for m in self.metadata["tiles"].values()
            if m.get("success", False)
        )

        avg_confidence = np.mean([
            m.get("mean_confidence", 0.0)
            for m in self.metadata["tiles"].values()
            if m.get("success", False) and m.get("num_detections", 0) > 0
        ]) if success > 0 else 0.0

        print()
        print("=" * 60)
        print("ANNOTATION SUMMARY")
        print("=" * 60)
        print(f"Total tiles: {total}")
        print(f"Success: {success}")
        print(f"Failed: {failed}")
        print(f"Total detections: {total_detections}")
        print(f"Average confidence: {avg_confidence:.3f}")
        print(f"\nResults saved to: {self.output_dir}")
        print(f"Metadata: {self.metadata_file}")
        print("=" * 60)


def annotate_from_config(config_path: Path):
    """
    Run annotation pipeline from config file.

    Args:
        config_path: Path to configuration YAML
    """
    from ..config import load_config
    from ..utils.logging_utils import setup_pipeline_logger, log_system_info, log_config

    # Load config
    print("Loading configuration...")
    config = load_config(config_path)
    print(config.get_summary())

    # Setup logger
    logger = setup_pipeline_logger(
        log_dir=config.logs_dir,
        pipeline_name="annotation",
        level=logging.INFO
    )

    logger.info("Starting annotation pipeline")
    logger.info(f"Config file: {config_path}")

    # Log system info
    log_system_info(logger)

    # Check dependencies
    logger.info("Checking dependencies...")
    deps = check_dependencies()
    if not all(deps.values()):
        logger.error("Missing required dependencies")
        print_dependency_status()
        raise RuntimeError("Missing required dependencies")
    logger.info("All dependencies satisfied")

    # Create annotator
    logger.info("Initializing zero-shot annotator...")
    logger.info(f"  Grounding DINO config: {config.grounding_dino.config_file}")
    logger.info(f"  Grounding DINO checkpoint: {config.grounding_dino.model_checkpoint}")
    logger.info(f"  SAM checkpoint: {config.sam.checkpoint}")
    logger.info(f"  SAM model type: {config.sam.model_type}")
    logger.info(f"  Device: {config.grounding_dino.device}")

    annotator = ZeroShotAnnotator(
        grounding_dino_config=config.grounding_dino.config_file,
        grounding_dino_checkpoint=config.grounding_dino.model_checkpoint,
        sam_checkpoint=config.sam.checkpoint,
        sam_model_type=config.sam.model_type,
        box_threshold=config.grounding_dino.box_threshold,
        text_threshold=config.grounding_dino.text_threshold,
        device=config.grounding_dino.device,
        log_dir=config.logs_dir
    )

    # Load tiles
    logger.info("Loading tiles...")
    metadata_path = config.tiles_dir / f"{config.tiling.prefix}_metadata.json"
    if not metadata_path.exists():
        logger.error(f"Tile metadata not found: {metadata_path}")
        raise FileNotFoundError(
            f"Tile metadata not found: {metadata_path}\n"
            f"Run tiling first: python scripts/tile_image.py ..."
        )

    tile_manager = TileManager.load(metadata_path)
    logger.info(f"Loaded {len(tile_manager)} tiles from {metadata_path}")

    # Create batch annotator
    # IMPORTANT: Save as 'both' to preserve class information in .npz format
    # The .npz files are needed for multi-class support in the reviewer!
    logger.info("Creating batch annotator...")
    logger.info(f"  Output directory: {config.raw_masks_dir}")
    logger.info(f"  Save format: both (GeoTIFF + NPZ)")

    batch_annotator = BatchAnnotator(
        annotator=annotator,
        output_dir=config.raw_masks_dir,
        save_format="both",  # Save both GeoTIFF (merged) and NPZ (multi-class)
        log_dir=config.logs_dir
    )

    # Run annotation
    # Set merge_masks=False to preserve individual class masks
    # This enables the reviewer to show/manage each class separately
    logger.info("Starting batch annotation...")
    logger.info(f"  Prompts: {config.prompts}")
    logger.info(f"  Merge masks: False (preserving multi-class)")

    batch_annotator.annotate_tiles(
        tile_manager=tile_manager,
        prompts=config.prompts,
        merge_masks=False  # Keep individual masks for multi-class support
    )

    logger.info("Annotation pipeline complete!")
    logger.info(f"Logs saved to: {config.logs_dir}")
    logger.info(f"Next step: Review annotations with:")
    logger.info(f"  python scripts/launch_reviewer.py {config.tiles_dir} {config.raw_masks_dir}")

    print("\nAnnotation complete!")
    print(f"Next step: Review annotations with:")
    print(f"  python scripts/launch_reviewer.py {config.tiles_dir} {config.raw_masks_dir}")
