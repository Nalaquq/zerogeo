# Configuration Guide

Complete guide to configuring the annotation pipeline for river segmentation.

## Overview

The annotation pipeline uses YAML configuration files to manage all settings for:
- Project metadata
- Image tiling parameters
- Grounding DINO model settings
- SAM (Segment Anything) settings
- Text prompts for detection
- Review UI preferences
- Export and training dataset options

## Quick Start

### Minimal Configuration

The simplest valid configuration requires only:
- Project name and paths
- At least one text prompt

```yaml
project:
  name: "my_project"
  input_image: "data/image.tif"
  output_dir: "data/output"

prompts:
  - "river"
  - "stream"
```

All other settings will use sensible defaults.

### Example Configurations

The `config/` directory contains several example configurations:

- **`minimal_config.yaml`** - Minimal required settings with common overrides
- **`river_annotation_example.yaml`** - Full example with all options documented
- **`cpu_config.yaml`** - Optimized for CPU-only inference (no GPU)
- **`high_resolution_config.yaml`** - High-quality settings for detailed work

## Configuration Sections

### 1. Project Configuration

**Required fields:**
- `name`: Project identifier
- `input_image`: Path to input GeoTIFF
- `output_dir`: Where to save all outputs

**Optional fields:**
- `description`: Project description (for documentation)

```yaml
project:
  name: "alaska_rivers"
  input_image: "data/Sentinel2_1_bands.tif"
  output_dir: "data/annotations/alaska_rivers"
  description: "River segmentation for Quinhagak, Alaska"
```

### 2. Tiling Configuration

Controls how large images are split into tiles.

**Parameters:**
- `tile_size` (default: 512): Tile dimensions in pixels (width = height)
- `overlap` (default: 64): Overlap between adjacent tiles
- `min_tile_size` (default: tile_size/2): Minimum size to keep edge tiles
- `prefix` (default: "tile"): Filename prefix for tiles

**Computed property:**
- `stride`: Spacing between tile origins (= tile_size - overlap)

```yaml
tiling:
  tile_size: 512              # 512x512 pixel tiles
  overlap: 64                 # 64 pixel overlap (12.5%)
  min_tile_size: 256          # discard tiles smaller than 256x256
  prefix: "tile"
```

**Guidelines:**
- Larger tiles (1024+) provide more context but use more memory
- Higher overlap (128+) improves boundary handling but creates more tiles
- Typical overlap: 10-25% of tile size

### 3. Grounding DINO Configuration

Settings for zero-shot object detection.

**Parameters:**
- `model_checkpoint`: Path to model weights
- `config_file`: Path to model config file
- `box_threshold` (default: 0.35): Confidence threshold for detections (0-1)
- `text_threshold` (default: 0.25): Text-image similarity threshold (0-1)
- `device` (default: "cuda"): Device to use ("cuda", "cpu", or "mps")

```yaml
grounding_dino:
  model_checkpoint: "weights/groundingdino_swint_ogc.pth"
  config_file: "model_configs/GroundingDINO_SwinT_OGC.py"
  box_threshold: 0.35
  text_threshold: 0.25
  device: "cuda"
```

**Tuning tips:**
- Lower `box_threshold` → more detections (may include false positives)
- Higher `box_threshold` → fewer, more confident detections

### 4. SAM Configuration

Settings for Segment Anything Model.

**Parameters:**
- `model_type`: Model variant - "vit_h" (huge), "vit_l" (large), or "vit_b" (base)
- `checkpoint`: Path to SAM weights
- `device`: Device to use ("cuda", "cpu", or "mps")
- `points_per_side` (optional): Points for automatic mask generation
- `pred_iou_thresh` (default: 0.88): IoU threshold for predictions
- `stability_score_thresh` (default: 0.95): Stability threshold for masks

```yaml
sam:
  model_type: "vit_h"         # Best quality, slowest
  checkpoint: "weights/sam_vit_h_4b8939.pth"
  device: "cuda"
  pred_iou_thresh: 0.88
  stability_score_thresh: 0.95
```

**Model comparison:**
- `vit_h`: Best quality, largest, slowest (~2.4GB VRAM)
- `vit_l`: Good quality, medium size (~1.2GB VRAM)
- `vit_b`: Fast, smaller, lower quality (~375MB VRAM)

### 5. Prompts

Text prompts for zero-shot detection.

**Requirements:**
- Must be a list
- At least one prompt required
- Empty prompts are ignored

```yaml
prompts:
  - "river"
  - "stream"
  - "water body"
  - "waterway"
  - "flowing water"
```

**Tips:**
- Use multiple variations of the same concept
- More specific prompts may give better results
- Experiment with different phrasings

### 6. Review Configuration

Settings for manual review UI.

**Parameters:**
- `auto_accept_threshold` (default: 0.9): Auto-accept masks above this confidence
- `show_tiles_per_page` (default: 16): Grid size (4x4 = 16)
- `default_view` (default: "grid"): "grid" or "single" tile view
- `enable_editing` (default: true): Allow manual mask editing
- `brush_size` (default: 10): Default brush size for editing
- `confidence_colors`: RGB colors for visualization

```yaml
review:
  auto_accept_threshold: 0.9
  show_tiles_per_page: 16     # 4x4 grid
  default_view: "grid"
  enable_editing: true
  brush_size: 10
  confidence_colors:
    high: [0, 255, 0]         # green
    medium: [255, 255, 0]     # yellow
    low: [255, 0, 0]          # red
```

### 7. Export Configuration

Settings for creating training datasets.

**Parameters:**
- `format`: Output format - "geotiff", "numpy", or "png"
- `split_ratio`: Train/val/test split (must sum to 1.0)
- `seed` (default: 42): Random seed for reproducibility
- `create_patches` (default: true): Create training patches
- `patch_size` (default: 256): Patch dimensions if creating patches
- `min_mask_coverage` (default: 0.01): Minimum mask coverage to include

```yaml
export:
  format: "geotiff"
  split_ratio:
    train: 0.7
    val: 0.2
    test: 0.1
  seed: 42
  create_patches: true
  patch_size: 256
  min_mask_coverage: 0.01     # 1% mask coverage minimum
```

## Using Configuration Files

### Command-Line Validation

```bash
# Validate config
python scripts/validate_config.py config/my_config.yaml

# Show detailed summary
python scripts/validate_config.py config/my_config.yaml --show-summary

# Create output directories
python scripts/validate_config.py config/my_config.yaml --create-dirs
```

### Python API

```python
from river_segmentation.config import load_config, save_config

# Load config
config = load_config("config/my_config.yaml")

# Access settings
print(config.project.name)
print(config.tiling.tile_size)
print(config.prompts)

# Show summary
print(config.get_summary())

# Create output directories
config.create_directories()

# Save modified config
save_config(config, "config/modified_config.yaml")
```

### Merging Configurations

Override specific settings programmatically:

```python
from river_segmentation.config import ConfigLoader, load_config

# Load base config
base = load_config("config/base_config.yaml")

# Override specific values
overrides = {
    "tiling": {"tile_size": 1024},
    "prompts": ["river", "stream", "creek"]
}

# Merge
merged = ConfigLoader.merge_configs(base, overrides)
```

## Output Directory Structure

After running with a config, the output directory will contain:

```
output_dir/
├── tiles/              # Tiled images
├── raw_masks/          # Initial zero-shot annotations
├── reviewed_masks/     # Manually reviewed/edited masks
├── training_data/      # Final training dataset
│   ├── train/
│   ├── val/
│   └── test/
└── logs/               # Pipeline logs
```

## Common Workflows

### CPU-Only Processing

For machines without GPU:

```yaml
grounding_dino:
  device: "cpu"
  box_threshold: 0.40    # Higher threshold for speed

sam:
  model_type: "vit_b"    # Smallest/fastest model
  device: "cpu"

tiling:
  tile_size: 256         # Smaller tiles process faster
```

### High-Quality Annotation

For maximum quality:

```yaml
tiling:
  tile_size: 1024
  overlap: 128

grounding_dino:
  box_threshold: 0.30    # More detections

sam:
  model_type: "vit_h"    # Best quality
  pred_iou_thresh: 0.90
  stability_score_thresh: 0.95

review:
  auto_accept_threshold: 0.95  # Manual review more masks
```

### Fast Prototyping

For quick testing:

```yaml
tiling:
  tile_size: 256
  overlap: 32

sam:
  model_type: "vit_b"    # Fastest model

review:
  auto_accept_threshold: 0.80  # Accept more automatically
```

## Validation Rules

The config system validates:

- **Required fields**: Project name, input_image, output_dir, prompts
- **File existence**: Input image must exist
- **Value ranges**: Thresholds must be in [0, 1]
- **Consistency**: overlap < tile_size
- **Split ratios**: Must sum to 1.0
- **Enum values**: model_type, device, format, etc.

Invalid configs will fail with clear error messages.

## Best Practices

1. **Start with examples**: Copy and modify example configs
2. **Validate early**: Run `validate_config.py` before processing
3. **Use version control**: Track config changes with git
4. **Document changes**: Use YAML comments to explain settings
5. **Test incrementally**: Validate on small regions first
6. **Keep backups**: Save working configs before experimentation

## Troubleshooting

**Error: "Input image not found"**
- Check that `input_image` path is correct
- Use absolute paths or paths relative to where you run the script

**Error: "overlap must be less than tile_size"**
- Reduce `overlap` or increase `tile_size`

**Error: "split_ratio values must sum to 1.0"**
- Check that train + val + test = 1.0
- Common values: 0.7, 0.2, 0.1 or 0.8, 0.1, 0.1

**Error: "At least one prompt is required"**
- Add at least one text prompt to the `prompts` list

## Next Steps

After configuring:
1. Validate: `python scripts/validate_config.py config.yaml`
2. Tile image: `python scripts/tile_image.py` (using config settings)
3. Annotate: Run zero-shot annotation pipeline
4. Review: Use manual review UI
5. Export: Generate training dataset
6. Train: Use exported data with U-Net

See the main README and QUICKSTART guides for complete workflows.
