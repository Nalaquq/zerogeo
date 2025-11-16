# Zero-Shot Annotation Guide

Complete guide to using Grounding DINO + SAM for automated river segmentation.

## Overview

The zero-shot annotation pipeline combines two state-of-the-art models:

1. **Grounding DINO** - Detects objects using text prompts
2. **SAM (Segment Anything)** - Generates precise segmentation masks

This enables automated annotation without manually labeled training data.

## Complete Workflow

```
1. Tile large image → 2. Annotate tiles → 3. Review results → 4. Export data → 5. Train model
   (scripts/tile_image.py)  (scripts/annotate_tiles.py)  (launch_reviewer.py)  (export.py)  (train.py)
```

## Prerequisites

1. **Install dependencies:**
   ```bash
   pip install -r requirements-annotation.txt
   ```

2. **Install models:**
   See [Model Setup Guide](model_setup.md) for detailed instructions.

3. **Download weights:**
   - Grounding DINO: `groundingdino_swint_ogc.pth`
   - SAM: `sam_vit_h_4b8939.pth` (or vit_l/vit_b)

4. **Verify installation:**
   ```bash
   python scripts/annotate_tiles.py --check-deps
   ```

## Quick Start

### 1. Tile Your Image

```bash
python scripts/tile_image.py \\
    data/Sentinel2_1_bands.tif \\
    data/tiles/ \\
    --tile-size 512 \\
    --overlap 64
```

### 2. Configure Annotation

Create or edit `config/my_annotation.yaml`:

```yaml
project:
  name: "my_river_annotation"
  input_image: "data/Sentinel2_1_bands.tif"
  output_dir: "data/annotations/my_project"

tiling:
  tile_size: 512
  overlap: 64

grounding_dino:
  model_checkpoint: "weights/groundingdino_swint_ogc.pth"
  config_file: "model_configs/GroundingDINO_SwinT_OGC.py"
  box_threshold: 0.35
  text_threshold: 0.25
  device: "cuda"

sam:
  model_type: "vit_h"
  checkpoint: "weights/sam_vit_h_4b8939.pth"
  device: "cuda"

prompts:
  - "river"
  - "stream"
  - "water body"
  - "waterway"
```

### 3. Run Annotation

```bash
python scripts/annotate_tiles.py --config config/my_annotation.yaml
```

### 4. Review Results

```bash
python scripts/launch_reviewer.py \\
    data/annotations/my_project/tiles \\
    data/annotations/my_project/raw_masks
```

### 5. Export Training Data

```bash
python scripts/export_training_data.py \\
    data/annotations/my_project/session \\
    --output data/training
```

## Detailed Usage

### Annotation from Config (Recommended)

**Advantages:**
- Reproducible
- Version controllable
- All settings in one place

**Usage:**
```bash
python scripts/annotate_tiles.py --config config/river_annotation_example.yaml
```

### Direct Annotation (No Config)

**Usage:**
```bash
python scripts/annotate_tiles.py \\
    --tiles data/tiles/ \\
    --prompts "river" "stream" "waterway" \\
    --output data/masks/ \\
    --dino-config model_configs/GroundingDINO_SwinT_OGC.py \\
    --dino-checkpoint weights/groundingdino_swint_ogc.pth \\
    --sam-checkpoint weights/sam_vit_h_4b8939.pth \\
    --sam-model vit_h \\
    --device cuda
```

### Python API

```python
from river_segmentation.annotation import (
    ZeroShotAnnotator,
    BatchAnnotator,
    TileManager,
    load_image_rgb
)

# Initialize annotator
annotator = ZeroShotAnnotator(
    grounding_dino_config="model_configs/GroundingDINO_SwinT_OGC.py",
    grounding_dino_checkpoint="weights/groundingdino_swint_ogc.pth",
    sam_checkpoint="weights/sam_vit_h_4b8939.pth",
    sam_model_type="vit_h",
    box_threshold=0.35,
    text_threshold=0.25,
    device="cuda"
)

# Load tiles
tile_manager = TileManager.load("data/tiles/tile_metadata.json")

# Create batch annotator
batch_annotator = BatchAnnotator(
    annotator=annotator,
    output_dir="data/masks",
    save_format="geotiff"
)

# Annotate
batch_annotator.annotate_tiles(
    tile_manager=tile_manager,
    prompts=["river", "stream"],
    merge_masks=True
)
```

## Understanding Text Prompts

### Prompt Engineering

Text prompts guide Grounding DINO's detections:

**Good prompts:**
- Specific: "river", "stream", "creek"
- Clear: "flowing water", "water body"
- Multiple variations: ["river", "stream", "waterway"]

**Avoid:**
- Too generic: "water" (may detect lakes, oceans)
- Too specific: "small mountain stream" (may miss others)
- Ambiguous: "blue area"

### Example Prompts by Use Case

**Rivers:**
```yaml
prompts:
  - "river"
  - "stream"
  - "waterway"
  - "flowing water"
```

**Buildings:**
```yaml
prompts:
  - "building"
  - "house"
  - "structure"
```

**Roads:**
```yaml
prompts:
  - "road"
  - "highway"
  - "street"
  - "path"
```

**Vegetation:**
```yaml
prompts:
  - "forest"
  - "trees"
  - "vegetation"
```

### Testing Prompts

Start with a single tile to test:

```python
# Test on one tile
result = annotator.annotate(
    image=load_image_rgb("data/tiles/tile_0_0.tif"),
    prompts=["river", "stream"],
    merge_masks=True
)

print(f"Detections: {len(result.masks)}")
print(f"Scores: {result.scores}")
print(f"Labels: {result.labels}")
```

## Tuning Parameters

### Detection Thresholds

**box_threshold** (default: 0.35)
- Lower (0.2-0.3): More detections, more false positives
- Higher (0.4-0.5): Fewer, more confident detections
- Adjust based on your data

**text_threshold** (default: 0.25)
- Controls text-image similarity
- Lower: More lenient matching
- Higher: Stricter matching

**Example:**
```yaml
grounding_dino:
  box_threshold: 0.30  # More detections
  text_threshold: 0.20  # More lenient
```

### SAM Parameters

**pred_iou_thresh** (default: 0.88)
- Controls mask quality
- Higher: Better masks, may miss some
- Lower: More masks, lower quality

**stability_score_thresh** (default: 0.95)
- Stability of mask predictions
- Higher: More stable masks

### Model Selection

**SAM Model Trade-offs:**

| Model | Speed | Quality | VRAM | Use Case |
|-------|-------|---------|------|----------|
| vit_h | Slow | Best | 8GB | Final production |
| vit_l | Medium | Good | 4GB | Balance |
| vit_b | Fast | Decent | 2GB | Testing, CPU |

**Recommendation:**
- Start with `vit_b` for testing
- Use `vit_h` for final annotation

## Processing Large Datasets

### Batch Processing

Process tiles in batches:

```python
# Process first 100 tiles
batch_annotator.annotate_tiles(
    tile_manager=tile_manager,
    prompts=["river"],
    max_tiles=100
)
```

### Parallel Processing

For very large datasets, process in parallel:

```bash
# Split tiles into groups
# Process each group on different GPU/machine

# Group 1
python scripts/annotate_tiles.py --tiles data/tiles_1/ --output data/masks_1/

# Group 2
python scripts/annotate_tiles.py --tiles data/tiles_2/ --output data/masks_2/
```

### Resume After Interruption

The annotator skips already processed tiles:

```python
# Check for existing masks before processing
if (output_dir / f"{tile_id}_mask.tif").exists():
    print(f"Skipping {tile_id} - already processed")
    continue
```

## Output Format

### Saved Files

After annotation:

```
data/annotations/my_project/
├── tiles/                          # Original tiles
├── raw_masks/                      # Annotated masks
│   ├── tile_0_0_mask.tif
│   ├── tile_0_1_mask.tif
│   └── annotation_metadata.json   # Metadata
└── review_session/                 # Review session (after review)
```

### Metadata Format

`annotation_metadata.json`:

```json
{
  "created": "2025-01-12T10:00:00",
  "tiles": {
    "tile_0_0": {
      "tile_id": "tile_0_0",
      "success": true,
      "num_detections": 3,
      "confidence_scores": [0.92, 0.85, 0.78],
      "labels": ["river", "river", "stream"],
      "mean_confidence": 0.85,
      "max_confidence": 0.92,
      "mask_coverage": 0.15,
      "timestamp": "2025-01-12T10:05:00"
    }
  }
}
```

### Mask Format

Masks are saved as:
- **GeoTIFF**: Preserves CRS and georeferencing
- **Binary**: 0 = background, 1 = detected object
- **Single band**: uint8

## Quality Control

### Reviewing Annotations

Always review annotations before training:

```bash
python scripts/launch_reviewer.py data/tiles/ data/raw_masks/
```

**Review workflow:**
1. Accept good masks (A key)
2. Reject bad masks (R key)
3. Edit masks that need fixing (E key)
4. Export reviewed masks

See [Reviewer Guide](reviewer_guide.md) for details.

### Common Issues

**Too many false positives:**
- Increase `box_threshold`
- Refine prompts to be more specific
- Review and reject bad detections

**Missing detections:**
- Decrease `box_threshold`
- Add more prompt variations
- Check image quality

**Poor mask quality:**
- Use larger SAM model (`vit_h`)
- Increase `pred_iou_thresh`
- Manually edit in reviewer

## Performance Optimization

### GPU Optimization

```yaml
grounding_dino:
  device: "cuda"
sam:
  device: "cuda"
  model_type: "vit_h"  # Use fastest model that meets quality needs
```

### CPU Optimization

```yaml
grounding_dino:
  device: "cpu"
  box_threshold: 0.40  # Higher threshold = fewer detections = faster

sam:
  device: "cpu"
  model_type: "vit_b"  # Fastest model

tiling:
  tile_size: 256  # Smaller tiles for CPU
```

### Memory Management

For large datasets:

```python
import torch

# Clear GPU cache between batches
torch.cuda.empty_cache()

# Process in smaller batches
for batch in batches:
    process_batch(batch)
    torch.cuda.empty_cache()
```

## Troubleshooting

### No Detections

**Symptoms**: All masks are empty

**Solutions:**
1. Lower `box_threshold` to 0.25-0.30
2. Try different prompts
3. Check image quality/format
4. Verify models loaded correctly

### Poor Quality Masks

**Symptoms**: Masks don't follow object boundaries

**Solutions:**
1. Use larger SAM model (`vit_h`)
2. Increase `stability_score_thresh`
3. Manually edit in reviewer
4. Check if detected boxes are accurate

### Slow Performance

**Symptoms**: Taking too long to process

**Solutions:**
1. Use smaller SAM model (`vit_b`)
2. Increase `box_threshold` (fewer detections)
3. Use GPU instead of CPU
4. Process smaller batches
5. Use smaller tiles

### CUDA Out of Memory

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions:**
1. Use smaller SAM model
2. Reduce tile size
3. Process fewer tiles at once
4. Clear GPU cache
5. Use CPU as fallback

## Best Practices

1. **Start small**: Test on a few tiles first
2. **Tune parameters**: Adjust thresholds for your data
3. **Review everything**: Always review before training
4. **Document settings**: Save config files
5. **Version control**: Track changes to configs
6. **Backup results**: Save annotation results
7. **Iterate**: Refine prompts and parameters

## Example Workflows

### Workflow 1: High-Quality Annotation

```bash
# 1. Tile with high overlap
python scripts/tile_image.py input.tif data/tiles/ --tile-size 1024 --overlap 128

# 2. Annotate with best models
python scripts/annotate_tiles.py --config config/high_quality.yaml

# 3. Review carefully
python scripts/launch_reviewer.py data/tiles/ data/masks/

# 4. Export
python scripts/export_training_data.py data/review_session/
```

### Workflow 2: Fast Prototyping

```bash
# 1. Small tiles
python scripts/tile_image.py input.tif data/tiles/ --tile-size 256 --overlap 32

# 2. Fast models
python scripts/annotate_tiles.py --config config/cpu_config.yaml

# 3. Quick review
python scripts/launch_reviewer.py data/tiles/ data/masks/
```

## Integration with Training

After annotation and review:

```bash
# Export training data
python scripts/export_training_data.py \\
    data/review_session/ \\
    --output data/training \\
    --format geotiff

# Train U-Net
python scripts/train.py \\
    --images data/training/images \\
    --masks data/training/masks \\
    --epochs 50 \\
    --batch-size 8
```

## Next Steps

1. Set up models (see [Model Setup](model_setup.md))
2. Configure pipeline (see [Configuration Guide](configuration_guide.md))
3. Run annotation (this guide)
4. Review results (see [Reviewer Guide](reviewer_guide.md))
5. Train model (see main README)

## References

- [Grounding DINO Paper](https://arxiv.org/abs/2303.05499)
- [SAM Paper](https://arxiv.org/abs/2304.02643)
- [Main README](../README.md)
- [Configuration Guide](configuration_guide.md)
- [Model Setup](model_setup.md)
- [Reviewer Guide](reviewer_guide.md)
