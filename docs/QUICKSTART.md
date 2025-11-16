# CIVIC Quick Start Guide

Get started with CIVIC's streamlined annotation pipeline in 3 simple steps!

## Prerequisites

1. **Install CIVIC** (one command):
   ```bash
   python scripts/smart_install.py
   ```

2. **Download models** (one command):
   ```bash
   civic-download-models
   ```

That's it for setup!

## Annotate Your First Image

### Step 1: Prepare Your Data

Put your GeoTIFF/TIFF image in the `input/` directory:
```bash
# Example: input/my_satellite_image.tif
cp /path/to/your/image.tif input/
ls input/*.tif
```

### Step 2: Create Configuration

Copy and edit the example config:
```bash
cp config/river_annotation_example.yaml config/my_project.yaml
```

Edit `config/my_project.yaml` - change these key fields:
```yaml
project:
  name: "my_project"                    # Your project name
  input_image: "input/my_image.tif"     # Your input image
  output_base: "output"                 # Run dir created as output/my_project_TIMESTAMP/

prompts:
  - "river"  # Change to whatever you want to detect
  # - "road"
  # - "building"
```

### Step 3: Run the Pipeline

**One command does everything:**
```bash
python scripts/civic.py run config/my_project.yaml
```

The pipeline will:
1. ✅ Create timestamped run directory automatically (output/my_project_TIMESTAMP/)
2. ✅ Tile your image (512x512 with 64px overlap)
3. ✅ Run zero-shot annotation using Grounding DINO + SAM
4. ✅ Launch interactive reviewer for quality control
5. ✅ Reassemble masks into complete GeoTIFF for GIS viewing

**That's it!** Your annotations will be in the run directory:
- Tiles: `output/my_project_TIMESTAMP/tiles/`
- Raw masks: `output/my_project_TIMESTAMP/raw_masks/`
- Reviewed masks: `output/my_project_TIMESTAMP/reviewed_masks/`
- Final raster: `output/my_project_TIMESTAMP/final_mask_complete.tif` (GIS-ready!)
- Logs: `output/my_project_TIMESTAMP/logs/`

### View Complete Results in GIS

The pipeline automatically creates a complete GeoTIFF: `final_mask_complete.tif`

Open it in your favorite GIS software:
```bash
# QGIS
qgis output/my_project_TIMESTAMP/final_mask_complete.tif

# Or any GIS application (ArcGIS, GRASS, etc.)
```

## Tips

### Preview Before Running
```bash
# See what will happen without executing
python scripts/civic.py run config/my_project.yaml --dry-run
```

### Resume Interrupted Pipeline
```bash
# Skip steps that are already complete
python scripts/civic.py run config/my_project.yaml --skip-existing
```

### Auto Mode (Skip Interactive Review)
```bash
# Accept all masks automatically without manual review
python scripts/civic.py run config/my_project.yaml --auto
```

### Run Individual Steps
```bash
# Just tile
python scripts/civic.py tile config/my_project.yaml

# Just annotate (requires tiles)
python scripts/civic.py annotate config/my_project.yaml

# Just review (requires annotations)
python scripts/civic.py review config/my_project.yaml

# Just reassemble (requires reviewed masks)
python scripts/civic.py reassemble config/my_project.yaml
```

## Configuration Tips

### GPU vs CPU

The smart installer auto-detects your hardware, but you can override:

**For GPU (NVIDIA CUDA):**
```yaml
grounding_dino:
  device: "cuda"
sam:
  device: "cuda"
  model_type: "vit_h"  # Best quality
```

**For CPU:**
```yaml
grounding_dino:
  device: "cpu"
sam:
  device: "cpu"
  model_type: "vit_b"  # Fastest on CPU
```

### Tile Size

Larger tiles = fewer tiles but slower processing:
```yaml
tiling:
  tile_size: 512   # Standard - good balance
  # tile_size: 256  # Fast - many small tiles
  # tile_size: 1024 # Slow - fewer large tiles
  overlap: 64      # Smooth boundaries
```

### Detection Sensitivity

Adjust thresholds for more/fewer detections:
```yaml
grounding_dino:
  box_threshold: 0.35  # Higher = fewer, more confident detections
  # 0.25 = more detections (may include false positives)
  # 0.45 = fewer detections (may miss some objects)

review:
  auto_accept_threshold: 0.90  # Auto-accept masks with confidence > 90%
  # 0.95 = stricter (review more)
  # 0.85 = looser (review less)
```

### Multiple Prompts

Detect multiple classes:
```yaml
prompts:
  - "river"
  - "stream"
  - "creek"
  - "waterway"
```

## Next Steps

### Export Training Data

After reviewing, export for model training:
```bash
python scripts/export_training_data.py \
    output/my_project_TIMESTAMP/reviewed_masks/ \
    --output training/my_project/
```

This creates:
- `training/my_project/train/` (70%)
- `training/my_project/val/` (20%)
- `training/my_project/test/` (10%)

### Train a Model

Train a U-Net on your annotated data:
```bash
python scripts/train.py \
    --image-dir training/my_project/train/images \
    --mask-dir training/my_project/train/masks \
    --val-image-dir training/my_project/val/images \
    --val-mask-dir training/my_project/val/masks \
    --output-dir models/my_model \
    --epochs 50 \
    --batch-size 8
```

## Troubleshooting

### "Tile metadata not found"
Make sure you're using the streamlined workflow:
```bash
python scripts/civic.py run config/my_project.yaml
```
This automatically manages paths. If running steps manually, ensure tile output directory matches config's `{output_dir}/tiles/`.

### "CUDA out of memory"
Use a smaller SAM model or reduce tile size:
```yaml
sam:
  model_type: "vit_l"  # or "vit_b"
tiling:
  tile_size: 384  # or 256
```

### "No detections found"
Try lowering thresholds:
```yaml
grounding_dino:
  box_threshold: 0.25  # Lower = more detections
  text_threshold: 0.20
```

Or adjust your prompts to be more specific/general.

### Check Hardware
```bash
python scripts/check_hardware.py
```

## Example Projects

### River Segmentation
```yaml
project:
  name: "rivers"
  input_image: "input/sentinel2_scene.tif"
  output_base: "output"

prompts:
  - "river"
  - "stream"
```

### Road Network
```yaml
project:
  name: "roads"
  input_image: "input/aerial_imagery.tif"
  output_base: "output"

prompts:
  - "road"
  - "highway"
  - "street"
```

### Building Footprints
```yaml
project:
  name: "buildings"
  input_image: "input/city_ortho.tif"
  output_base: "output"

prompts:
  - "building"
  - "house"
  - "structure"
```

## Getting Help

- **Full documentation**: See [README.md](README.md)
- **Installation issues**: See [INSTALL.md](INSTALL.md)
- **Configuration guide**: See [docs/configuration_guide.md](docs/configuration_guide.md)
- **Report bugs**: [GitHub Issues](https://github.com/yourusername/civic/issues)

## Summary

```bash
# Complete workflow in 3 commands:

# 1. Install (once)
python scripts/smart_install.py && civic-download-models

# 2. Configure
cp config/river_annotation_example.yaml config/my_project.yaml
# Edit config/my_project.yaml

# 3. Run!
python scripts/civic.py run config/my_project.yaml
```

That's it! Happy annotating! 🎉
