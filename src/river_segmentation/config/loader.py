"""
YAML configuration loader with validation.

Provides functions to load, save, and validate configuration files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
import copy

from .schema import (
    AnnotationConfig,
    ProjectConfig,
    TilingConfig,
    GroundingDINOConfig,
    SAMConfig,
    ReviewConfig,
    ExportConfig,
)


class ConfigLoader:
    """Configuration loader with validation and defaults."""

    @staticmethod
    def load_yaml(filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Load YAML file.

        Args:
            filepath: Path to YAML file

        Returns:
            Dictionary with configuration data

        Raises:
            FileNotFoundError: If file doesn't exist
            yaml.YAMLError: If YAML is invalid
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")

        with open(filepath, 'r') as f:
            try:
                data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML in {filepath}: {e}")

        if data is None:
            raise ValueError(f"Empty config file: {filepath}")

        return data

    @staticmethod
    def save_yaml(data: Dict[str, Any], filepath: Union[str, Path]):
        """
        Save configuration to YAML file.

        Args:
            data: Configuration dictionary
            filepath: Output file path
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, indent=2)

    @staticmethod
    def _parse_project(data: Dict[str, Any]) -> ProjectConfig:
        """Parse project configuration."""
        if "project" not in data:
            raise ValueError("Missing required 'project' section in config")

        project_data = data["project"]

        # Required fields
        required = ["name", "input_image", "output_dir"]
        missing = [field for field in required if field not in project_data]
        if missing:
            raise ValueError(f"Missing required project fields: {missing}")

        return ProjectConfig(**project_data)

    @staticmethod
    def _parse_tiling(data: Dict[str, Any]) -> TilingConfig:
        """Parse tiling configuration."""
        tiling_data = data.get("tiling", {})
        return TilingConfig(**tiling_data)

    @staticmethod
    def _parse_grounding_dino(data: Dict[str, Any]) -> GroundingDINOConfig:
        """Parse Grounding DINO configuration."""
        dino_data = data.get("grounding_dino", {})
        return GroundingDINOConfig(**dino_data)

    @staticmethod
    def _parse_sam(data: Dict[str, Any]) -> SAMConfig:
        """Parse SAM configuration."""
        sam_data = data.get("sam", {})
        return SAMConfig(**sam_data)

    @staticmethod
    def _parse_prompts(data: Dict[str, Any]) -> list:
        """Parse text prompts."""
        prompts = data.get("prompts", [])

        if not prompts:
            raise ValueError("At least one prompt is required")

        if not isinstance(prompts, list):
            raise ValueError("prompts must be a list")

        # Convert all to strings
        prompts = [str(p).strip() for p in prompts]

        # Remove empty prompts
        prompts = [p for p in prompts if p]

        if not prompts:
            raise ValueError("At least one non-empty prompt is required")

        return prompts

    @staticmethod
    def _parse_review(data: Dict[str, Any]) -> ReviewConfig:
        """Parse review configuration."""
        review_data = data.get("review", {})
        return ReviewConfig(**review_data)

    @staticmethod
    def _parse_export(data: Dict[str, Any]) -> ExportConfig:
        """Parse export configuration."""
        export_data = data.get("export", {})
        return ExportConfig(**export_data)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> AnnotationConfig:
        """
        Load and parse configuration file.

        Args:
            filepath: Path to YAML config file

        Returns:
            AnnotationConfig object

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If config is invalid
        """
        data = cls.load_yaml(filepath)

        # Parse each section
        try:
            project = cls._parse_project(data)
            tiling = cls._parse_tiling(data)
            grounding_dino = cls._parse_grounding_dino(data)
            sam = cls._parse_sam(data)
            prompts = cls._parse_prompts(data)
            review = cls._parse_review(data)
            export = cls._parse_export(data)

            # Create config
            config = AnnotationConfig(
                project=project,
                tiling=tiling,
                grounding_dino=grounding_dino,
                sam=sam,
                prompts=prompts,
                review=review,
                export=export,
            )

            return config

        except TypeError as e:
            raise ValueError(f"Invalid configuration format: {e}")

    @staticmethod
    def to_dict(config: AnnotationConfig) -> Dict[str, Any]:
        """
        Convert AnnotationConfig to dictionary.

        Args:
            config: AnnotationConfig object

        Returns:
            Dictionary representation
        """
        return {
            "project": {
                "name": config.project.name,
                "input_image": config.project.input_image,
                "output_dir": config.project.output_dir,
                "description": config.project.description,
            },
            "tiling": {
                "tile_size": config.tiling.tile_size,
                "overlap": config.tiling.overlap,
                "min_tile_size": config.tiling.min_tile_size,
                "prefix": config.tiling.prefix,
            },
            "grounding_dino": {
                "model_checkpoint": config.grounding_dino.model_checkpoint,
                "config_file": config.grounding_dino.config_file,
                "box_threshold": config.grounding_dino.box_threshold,
                "text_threshold": config.grounding_dino.text_threshold,
                "device": config.grounding_dino.device,
            },
            "sam": {
                "model_type": config.sam.model_type,
                "checkpoint": config.sam.checkpoint,
                "device": config.sam.device,
                "points_per_side": config.sam.points_per_side,
                "pred_iou_thresh": config.sam.pred_iou_thresh,
                "stability_score_thresh": config.sam.stability_score_thresh,
            },
            "prompts": config.prompts,
            "review": {
                "auto_accept_threshold": config.review.auto_accept_threshold,
                "show_tiles_per_page": config.review.show_tiles_per_page,
                "default_view": config.review.default_view,
                "enable_editing": config.review.enable_editing,
                "brush_size": config.review.brush_size,
                "confidence_colors": {
                    k: list(v) for k, v in config.review.confidence_colors.items()
                },
            },
            "export": {
                "format": config.export.format,
                "split_ratio": config.export.split_ratio,
                "seed": config.export.seed,
                "create_patches": config.export.create_patches,
                "patch_size": config.export.patch_size,
                "min_mask_coverage": config.export.min_mask_coverage,
            },
        }

    @classmethod
    def save(cls, config: AnnotationConfig, filepath: Union[str, Path]):
        """
        Save configuration to YAML file.

        Args:
            config: AnnotationConfig object
            filepath: Output file path
        """
        data = cls.to_dict(config)
        cls.save_yaml(data, filepath)

    @classmethod
    def merge_configs(
        cls,
        base_config: AnnotationConfig,
        overrides: Dict[str, Any]
    ) -> AnnotationConfig:
        """
        Merge base config with overrides.

        Args:
            base_config: Base configuration
            overrides: Dictionary with override values

        Returns:
            New AnnotationConfig with merged values
        """
        # Convert base config to dict
        base_dict = cls.to_dict(base_config)

        # Deep merge
        merged = cls._deep_merge(base_dict, overrides)

        # Parse merged config
        # Create temp file for validation
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(merged, f)
            temp_path = f.name

        try:
            merged_config = cls.load(temp_path)
        finally:
            Path(temp_path).unlink()

        return merged_config

    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> Dict:
        """
        Deep merge two dictionaries.

        Args:
            base: Base dictionary
            override: Override dictionary

        Returns:
            Merged dictionary
        """
        result = copy.deepcopy(base)

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigLoader._deep_merge(result[key], value)
            else:
                result[key] = value

        return result


def load_config(filepath: Union[str, Path]) -> AnnotationConfig:
    """
    Load configuration from YAML file.

    Convenience function for ConfigLoader.load().

    Args:
        filepath: Path to YAML config file

    Returns:
        AnnotationConfig object
    """
    return ConfigLoader.load(filepath)


def save_config(config: AnnotationConfig, filepath: Union[str, Path]):
    """
    Save configuration to YAML file.

    Convenience function for ConfigLoader.save().

    Args:
        config: AnnotationConfig object
        filepath: Output file path
    """
    ConfigLoader.save(config, filepath)


def validate_config(filepath: Union[str, Path]) -> tuple[bool, Optional[str]]:
    """
    Validate a configuration file.

    Args:
        filepath: Path to YAML config file

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        config = load_config(filepath)
        return True, None
    except Exception as e:
        return False, str(e)
