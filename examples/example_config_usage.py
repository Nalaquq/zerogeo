"""
Example demonstrating configuration system usage.

This script shows how to:
1. Load configuration from YAML
2. Validate configuration
3. Access configuration values
4. Create output directories
5. Save modified configuration
6. Merge configurations
"""

from pathlib import Path
from river_segmentation.config import (
    load_config,
    save_config,
    validate_config,
    ConfigLoader,
    ProjectConfig,
    AnnotationConfig,
)


def example_load_and_validate():
    """Example: Load and validate a config file."""
    print("=" * 60)
    print("EXAMPLE 1: Load and Validate Configuration")
    print("=" * 60)
    print()

    config_path = "config/minimal_config.yaml"

    # Validate first
    is_valid, error = validate_config(config_path)

    if is_valid:
        print(f"✓ Configuration is valid: {config_path}")
    else:
        print(f"✗ Configuration is invalid: {error}")
        return

    # Load config
    config = load_config(config_path)

    print(f"\nLoaded configuration:")
    print(f"  Project: {config.project.name}")
    print(f"  Input: {config.project.input_image}")
    print(f"  Prompts: {', '.join(config.prompts)}")
    print()


def example_access_settings():
    """Example: Access configuration settings."""
    print("=" * 60)
    print("EXAMPLE 2: Access Configuration Settings")
    print("=" * 60)
    print()

    config = load_config("config/minimal_config.yaml")

    # Project settings
    print("Project Settings:")
    print(f"  Name: {config.project.name}")
    print(f"  Input: {config.project.input_image}")
    print(f"  Output: {config.project.output_dir}")
    print()

    # Tiling settings
    print("Tiling Settings:")
    print(f"  Tile size: {config.tiling.tile_size}x{config.tiling.tile_size}")
    print(f"  Overlap: {config.tiling.overlap} pixels")
    print(f"  Stride: {config.tiling.stride} pixels")
    print()

    # Model settings
    print("Model Settings:")
    print(f"  Grounding DINO threshold: {config.grounding_dino.box_threshold}")
    print(f"  SAM model: {config.sam.model_type}")
    print()

    # Prompts
    print("Detection Prompts:")
    for i, prompt in enumerate(config.prompts, 1):
        print(f"  {i}. {prompt}")
    print()


def example_create_directories():
    """Example: Create output directories."""
    print("=" * 60)
    print("EXAMPLE 3: Create Output Directories")
    print("=" * 60)
    print()

    config = load_config("config/minimal_config.yaml")

    print(f"Creating directories in: {config.project.output_dir}")
    print()

    # Create directories
    config.create_directories()

    # Show created directories
    print("Created directories:")
    print(f"  Tiles: {config.tiles_dir}")
    print(f"  Raw masks: {config.raw_masks_dir}")
    print(f"  Reviewed masks: {config.reviewed_masks_dir}")
    print(f"  Training data: {config.training_data_dir}")
    print(f"  Logs: {config.logs_dir}")
    print()

    # Verify existence
    all_exist = all([
        config.tiles_dir.exists(),
        config.raw_masks_dir.exists(),
        config.reviewed_masks_dir.exists(),
        config.training_data_dir.exists(),
        config.logs_dir.exists(),
    ])

    if all_exist:
        print("✓ All directories created successfully")
    else:
        print("✗ Some directories were not created")

    print()


def example_show_summary():
    """Example: Display configuration summary."""
    print("=" * 60)
    print("EXAMPLE 4: Display Configuration Summary")
    print("=" * 60)
    print()

    config = load_config("config/river_annotation_example.yaml")

    # Get and print summary
    summary = config.get_summary()
    print(summary)
    print()


def example_save_config():
    """Example: Save configuration to file."""
    print("=" * 60)
    print("EXAMPLE 5: Save Configuration")
    print("=" * 60)
    print()

    # Load existing config
    config = load_config("config/minimal_config.yaml")

    # Modify some values (in practice, you'd modify the dataclass directly)
    # For demo, we'll just save as-is
    output_path = "config/example_saved_config.yaml"

    print(f"Saving configuration to: {output_path}")
    save_config(config, output_path)

    if Path(output_path).exists():
        print("✓ Configuration saved successfully")

        # Load it back to verify
        loaded = load_config(output_path)
        print(f"✓ Verified: Loaded config has project name '{loaded.project.name}'")

        # Clean up
        Path(output_path).unlink()
        print(f"✓ Cleaned up example file")
    else:
        print("✗ Failed to save configuration")

    print()


def example_merge_configs():
    """Example: Merge configurations."""
    print("=" * 60)
    print("EXAMPLE 6: Merge Configurations")
    print("=" * 60)
    print()

    # Load base config
    base = load_config("config/minimal_config.yaml")

    print("Base configuration:")
    print(f"  Tile size: {base.tiling.tile_size}")
    print(f"  Overlap: {base.tiling.overlap}")
    print(f"  Prompts: {', '.join(base.prompts)}")
    print()

    # Define overrides
    overrides = {
        "tiling": {
            "tile_size": 1024,
            "overlap": 128,
        },
        "prompts": ["river", "stream", "creek", "waterway"],
    }

    print("Overrides:")
    print(f"  New tile size: {overrides['tiling']['tile_size']}")
    print(f"  New overlap: {overrides['tiling']['overlap']}")
    print(f"  New prompts: {', '.join(overrides['prompts'])}")
    print()

    # Merge
    merged = ConfigLoader.merge_configs(base, overrides)

    print("Merged configuration:")
    print(f"  Tile size: {merged.tiling.tile_size}")
    print(f"  Overlap: {merged.tiling.overlap}")
    print(f"  Stride: {merged.tiling.stride}")
    print(f"  Prompts: {', '.join(merged.prompts)}")
    print()


def example_programmatic_config():
    """Example: Create config programmatically."""
    print("=" * 60)
    print("EXAMPLE 7: Create Configuration Programmatically")
    print("=" * 60)
    print()

    # Create project config
    project = ProjectConfig(
        name="programmatic_example",
        input_image="data/Sentinel2_1_bands.tif",
        output_dir="data/annotations/programmatic",
        description="Created programmatically"
    )

    # Create full annotation config with defaults
    config = AnnotationConfig(
        project=project,
        prompts=["river", "stream"]
    )

    print("Created configuration:")
    print(f"  Project: {config.project.name}")
    print(f"  Description: {config.project.description}")
    print(f"  Tile size: {config.tiling.tile_size} (default)")
    print(f"  SAM model: {config.sam.model_type} (default)")
    print(f"  Prompts: {', '.join(config.prompts)}")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("CONFIGURATION SYSTEM EXAMPLES")
    print("=" * 60)
    print("\n")

    try:
        example_load_and_validate()
        example_access_settings()
        example_create_directories()
        example_show_summary()
        example_save_config()
        example_merge_configs()
        example_programmatic_config()

        print("=" * 60)
        print("ALL EXAMPLES COMPLETED")
        print("=" * 60)
        print()

        # Cleanup
        print("Cleaning up example directories...")
        import shutil
        output_dir = Path("data/annotations/output")
        if output_dir.exists():
            shutil.rmtree(output_dir)
            print("✓ Cleaned up")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
