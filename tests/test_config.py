"""
Tests for configuration system.
"""

import pytest
import tempfile
from pathlib import Path
import yaml

from river_segmentation.config import (
    AnnotationConfig,
    ProjectConfig,
    TilingConfig,
    GroundingDINOConfig,
    SAMConfig,
    ReviewConfig,
    ExportConfig,
    ConfigLoader,
    load_config,
    save_config,
    validate_config,
)


@pytest.fixture
def temp_image_file(tmp_path):
    """Create a temporary dummy image file."""
    image_file = tmp_path / "test_image.tif"
    image_file.touch()
    return image_file


@pytest.fixture
def minimal_config_dict(temp_image_file):
    """Minimal valid configuration dictionary."""
    return {
        "project": {
            "name": "test_project",
            "input_image": str(temp_image_file),
            "output_dir": str(temp_image_file.parent / "output"),
        },
        "prompts": ["river", "stream"],
    }


@pytest.fixture
def full_config_dict(temp_image_file):
    """Full configuration dictionary with all options."""
    return {
        "project": {
            "name": "test_project",
            "input_image": str(temp_image_file),
            "output_dir": str(temp_image_file.parent / "output"),
            "description": "Test project",
        },
        "tiling": {
            "tile_size": 512,
            "overlap": 64,
            "min_tile_size": 256,
            "prefix": "tile",
        },
        "grounding_dino": {
            "model_checkpoint": "groundingdino.pth",
            "config_file": "config.py",
            "box_threshold": 0.35,
            "text_threshold": 0.25,
            "device": "cuda",
        },
        "sam": {
            "model_type": "vit_h",
            "checkpoint": "sam.pth",
            "device": "cuda",
            "pred_iou_thresh": 0.88,
            "stability_score_thresh": 0.95,
        },
        "prompts": ["river", "stream", "water"],
        "review": {
            "auto_accept_threshold": 0.9,
            "show_tiles_per_page": 16,
            "default_view": "grid",
            "enable_editing": True,
            "brush_size": 10,
        },
        "export": {
            "format": "geotiff",
            "split_ratio": {"train": 0.7, "val": 0.2, "test": 0.1},
            "seed": 42,
            "create_patches": True,
            "patch_size": 256,
            "min_mask_coverage": 0.01,
        },
    }


def test_project_config(temp_image_file):
    """Test ProjectConfig validation."""
    # Valid config
    config = ProjectConfig(
        name="test",
        input_image=str(temp_image_file),
        output_dir="/tmp/output"
    )
    assert config.name == "test"

    # Missing input file
    with pytest.raises(ValueError, match="Input image not found"):
        ProjectConfig(
            name="test",
            input_image="/nonexistent/file.tif",
            output_dir="/tmp/output"
        )


def test_tiling_config():
    """Test TilingConfig validation."""
    # Valid config
    config = TilingConfig(tile_size=512, overlap=64)
    assert config.stride == 448

    # Overlap >= tile_size
    with pytest.raises(ValueError, match="overlap.*must be less than tile_size"):
        TilingConfig(tile_size=512, overlap=512)

    # Negative overlap
    with pytest.raises(ValueError, match="overlap must be non-negative"):
        TilingConfig(tile_size=512, overlap=-10)

    # Auto min_tile_size
    config = TilingConfig(tile_size=512, overlap=64)
    assert config.min_tile_size == 256


def test_grounding_dino_config():
    """Test GroundingDINOConfig validation."""
    # Valid config
    config = GroundingDINOConfig()
    assert 0 <= config.box_threshold <= 1

    # Invalid box_threshold
    with pytest.raises(ValueError, match="box_threshold must be in"):
        GroundingDINOConfig(box_threshold=1.5)

    # Invalid text_threshold
    with pytest.raises(ValueError, match="text_threshold must be in"):
        GroundingDINOConfig(text_threshold=-0.1)

    # Invalid device
    with pytest.raises(ValueError, match="device must be"):
        GroundingDINOConfig(device="invalid")


def test_sam_config():
    """Test SAMConfig validation."""
    # Valid config
    config = SAMConfig(model_type="vit_h")
    assert config.model_type == "vit_h"

    # Invalid model_type
    with pytest.raises(ValueError, match="model_type must be one of"):
        SAMConfig(model_type="invalid")

    # Invalid thresholds
    with pytest.raises(ValueError, match="pred_iou_thresh"):
        SAMConfig(pred_iou_thresh=1.5)

    with pytest.raises(ValueError, match="stability_score_thresh"):
        SAMConfig(stability_score_thresh=-0.1)


def test_review_config():
    """Test ReviewConfig validation."""
    # Valid config
    config = ReviewConfig()
    assert config.auto_accept_threshold == 0.9

    # Invalid threshold
    with pytest.raises(ValueError, match="auto_accept_threshold"):
        ReviewConfig(auto_accept_threshold=1.5)

    # Invalid view
    with pytest.raises(ValueError, match="default_view"):
        ReviewConfig(default_view="invalid")

    # Invalid tiles per page
    with pytest.raises(ValueError, match="show_tiles_per_page"):
        ReviewConfig(show_tiles_per_page=0)


def test_export_config():
    """Test ExportConfig validation."""
    # Valid config
    config = ExportConfig()
    assert config.format == "geotiff"

    # Invalid format
    with pytest.raises(ValueError, match="format must be one of"):
        ExportConfig(format="invalid")

    # Invalid split ratio (doesn't sum to 1)
    with pytest.raises(ValueError, match="split_ratio values must sum"):
        ExportConfig(split_ratio={"train": 0.5, "val": 0.3, "test": 0.3})

    # Missing split keys
    with pytest.raises(ValueError, match="split_ratio must contain keys"):
        ExportConfig(split_ratio={"train": 1.0})


def test_annotation_config_missing_prompts(temp_image_file):
    """Test that AnnotationConfig requires prompts."""
    project = ProjectConfig(
        name="test",
        input_image=str(temp_image_file),
        output_dir="/tmp/output"
    )

    with pytest.raises(ValueError, match="At least one prompt is required"):
        AnnotationConfig(project=project, prompts=[])


def test_load_minimal_config(tmp_path, minimal_config_dict):
    """Test loading minimal valid config."""
    config_file = tmp_path / "config.yaml"

    with open(config_file, 'w') as f:
        yaml.dump(minimal_config_dict, f)

    config = load_config(config_file)

    assert config.project.name == "test_project"
    assert config.prompts == ["river", "stream"]
    assert config.tiling.tile_size == 512  # default


def test_load_full_config(tmp_path, full_config_dict):
    """Test loading full config with all options."""
    config_file = tmp_path / "config.yaml"

    with open(config_file, 'w') as f:
        yaml.dump(full_config_dict, f)

    config = load_config(config_file)

    assert config.project.name == "test_project"
    assert config.tiling.tile_size == 512
    assert config.grounding_dino.box_threshold == 0.35
    assert config.sam.model_type == "vit_h"
    assert len(config.prompts) == 3
    assert config.review.show_tiles_per_page == 16
    assert config.export.format == "geotiff"


def test_load_missing_file():
    """Test loading non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/config.yaml")


def test_load_invalid_yaml(tmp_path):
    """Test loading invalid YAML."""
    config_file = tmp_path / "invalid.yaml"

    with open(config_file, 'w') as f:
        f.write("invalid: yaml: content: :")

    with pytest.raises(ValueError, match="Invalid YAML"):
        load_config(config_file)


def test_load_empty_config(tmp_path):
    """Test loading empty config file."""
    config_file = tmp_path / "empty.yaml"
    config_file.touch()

    with pytest.raises(ValueError, match="Empty config file"):
        load_config(config_file)


def test_load_missing_project(tmp_path):
    """Test config without project section."""
    config_file = tmp_path / "config.yaml"

    with open(config_file, 'w') as f:
        yaml.dump({"prompts": ["river"]}, f)

    with pytest.raises(ValueError, match="Missing required 'project' section"):
        load_config(config_file)


def test_load_missing_project_fields(tmp_path):
    """Test project section missing required fields."""
    config_file = tmp_path / "config.yaml"

    config_dict = {
        "project": {"name": "test"},  # missing input_image and output_dir
        "prompts": ["river"]
    }

    with open(config_file, 'w') as f:
        yaml.dump(config_dict, f)

    with pytest.raises(ValueError, match="Missing required project fields"):
        load_config(config_file)


def test_save_config(tmp_path, temp_image_file):
    """Test saving config to file."""
    # Create config
    project = ProjectConfig(
        name="test",
        input_image=str(temp_image_file),
        output_dir=str(tmp_path / "output")
    )
    config = AnnotationConfig(project=project, prompts=["river"])

    # Save
    config_file = tmp_path / "saved_config.yaml"
    save_config(config, config_file)

    assert config_file.exists()

    # Load and verify
    loaded = load_config(config_file)
    assert loaded.project.name == "test"
    assert loaded.prompts == ["river"]


def test_config_to_dict(temp_image_file):
    """Test converting config to dictionary."""
    project = ProjectConfig(
        name="test",
        input_image=str(temp_image_file),
        output_dir="/tmp/output"
    )
    config = AnnotationConfig(project=project, prompts=["river"])

    config_dict = ConfigLoader.to_dict(config)

    assert config_dict["project"]["name"] == "test"
    assert config_dict["prompts"] == ["river"]
    assert config_dict["tiling"]["tile_size"] == 512  # default


def test_validate_config_valid(tmp_path, minimal_config_dict):
    """Test validating a valid config."""
    config_file = tmp_path / "config.yaml"

    with open(config_file, 'w') as f:
        yaml.dump(minimal_config_dict, f)

    is_valid, error = validate_config(config_file)
    assert is_valid
    assert error is None


def test_validate_config_invalid(tmp_path):
    """Test validating an invalid config."""
    config_file = tmp_path / "config.yaml"

    # Invalid config (missing prompts)
    config_dict = {
        "project": {
            "name": "test",
            "input_image": "/nonexistent.tif",
            "output_dir": "/tmp/output",
        }
    }

    with open(config_file, 'w') as f:
        yaml.dump(config_dict, f)

    is_valid, error = validate_config(config_file)
    assert not is_valid
    assert error is not None


def test_config_summary(temp_image_file):
    """Test config summary generation."""
    project = ProjectConfig(
        name="test",
        input_image=str(temp_image_file),
        output_dir="/tmp/output"
    )
    config = AnnotationConfig(
        project=project,
        prompts=["river", "stream"]
    )

    summary = config.get_summary()

    assert "test" in summary
    assert "river" in summary
    assert "stream" in summary
    assert "512" in summary  # tile size


def test_config_create_directories(tmp_path, temp_image_file):
    """Test directory creation."""
    output_dir = tmp_path / "output"

    project = ProjectConfig(
        name="test",
        input_image=str(temp_image_file),
        output_dir=str(output_dir)
    )
    config = AnnotationConfig(project=project, prompts=["river"])

    # Create directories
    config.create_directories()

    # Check that directories exist
    assert config.tiles_dir.exists()
    assert config.raw_masks_dir.exists()
    assert config.reviewed_masks_dir.exists()
    assert config.training_data_dir.exists()
    assert config.logs_dir.exists()


def test_merge_configs(tmp_path, minimal_config_dict):
    """Test merging configs."""
    config_file = tmp_path / "config.yaml"

    with open(config_file, 'w') as f:
        yaml.dump(minimal_config_dict, f)

    base_config = load_config(config_file)

    # Override some values
    overrides = {
        "tiling": {
            "tile_size": 1024,
            "overlap": 128,
        },
        "prompts": ["river", "stream", "creek"],
    }

    merged = ConfigLoader.merge_configs(base_config, overrides)

    # Check merged values
    assert merged.tiling.tile_size == 1024
    assert merged.tiling.overlap == 128
    assert len(merged.prompts) == 3

    # Check that other values remain
    assert merged.project.name == "test_project"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
