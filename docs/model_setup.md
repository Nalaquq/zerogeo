# Model Setup Guide

Complete guide for setting up Grounding DINO and SAM models for zero-shot annotation.

## Overview

The annotation pipeline uses two state-of-the-art models:
1. **Grounding DINO** - Text-based object detection
2. **SAM (Segment Anything)** - High-quality segmentation

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- ~10GB disk space for models
- ~8GB GPU VRAM for vit_h models (less for smaller variants)

## Installation

### Step 1: Install All Dependencies

```bash
# Install everything (includes base + annotation + models)
pip install -r requirements.txt
```

This installs all 18 packages including:
- Core dependencies (numpy, torch, rasterio, etc.)
- Annotation dependencies (opencv, transformers, supervision, etc.)
- Model packages (groundingdino-py, segment-anything)

### Step 2: Install Models

**Recommended: Automatic Installation**

Grounding DINO and SAM are now included automatically:

```bash
# Installs everything including Grounding DINO and SAM
pip install -r requirements.txt

# Or using setup.py
pip install -e ".[annotation]"
```

**Manual Installation (if needed)**

If you need to install them separately:

```bash
pip install groundingdino-py segment-anything
```

### Step 3: Verify Installation

```bash
python scripts/annotate_tiles.py --check-deps
```

You should see:
```
Annotation Pipeline Dependencies:
  OpenCV: ✓
  PyTorch: ✓
  Grounding DINO: ✓
  SAM: ✓
```

## Download Model Weights

### Option 1: Automatic Download (Recommended) ⭐

**Download all models with one command:**

```bash
civic-download-models
```

**That's it!** This automatically downloads:
- ✅ Grounding DINO checkpoint (~670MB) → `weights/groundingdino_swint_ogc.pth`
- ✅ SAM vit_b checkpoint (~375MB) → `weights/sam_vit_b_01ec64.pth`
- ✅ Grounding DINO config → `model_model_configs/GroundingDINO_SwinT_OGC.py`

**Choose different SAM model:**

```bash
# Fast, CPU-friendly (default)
civic-download-models --sam-model vit_b  # ~375MB

# Balanced quality
civic-download-models --sam-model vit_l  # ~1.2GB

# Best quality (requires GPU)
civic-download-models --sam-model vit_h  # ~2.4GB
```

**SAM Model Comparison:**

| Model | Size | VRAM | Speed | Quality | Best For |
|-------|------|------|-------|---------|----------|
| vit_b | 375MB | ~2GB | Fast | Decent | Testing, CPU-only |
| vit_l | 1.2GB | ~4GB | Medium | Good | Balanced workflow |
| vit_h | 2.4GB | ~8GB | Slow | Best | Production, GPU |

**Advanced options:**

```bash
# List available models
civic-download-models --list

# Download to custom directory
civic-download-models --weights-dir ./my_models --configs-dir ./my_configs

# Force re-download
civic-download-models --force

# Help
civic-download-models --help
```

### Verify Downloads

```bash
ls -lh weights/
ls -lh model_model_configs/
```

Expected output:
```
weights/
├── groundingdino_swint_ogc.pth  (~670MB)
└── sam_vit_b_01ec64.pth         (~375MB, or vit_l/vit_h)

model_configs/
└── GroundingDINO_SwinT_OGC.py
```

---

### Option 2: Manual Download (Fallback)

<details>
<summary>Click to expand manual download instructions</summary>

Only use manual download if the automatic method fails.

**Create directories:**

```bash
mkdir -p weights configs
```

**Download Grounding DINO:**

```bash
cd weights
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ../configs
wget https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py
cd ..
```

**Download SAM:**

```bash
cd weights

# Choose one:
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth  # Fast
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth  # Balanced
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth  # Best

cd ..
```

</details>

## Configuration

Update your config file with model paths:

```yaml
# config/river_annotation_example.yaml

grounding_dino:
  model_checkpoint: "weights/groundingdino_swint_ogc.pth"
  config_file: "model_configs/GroundingDINO_SwinT_OGC.py"
  box_threshold: 0.35
  text_threshold: 0.25
  device: "cuda"  # or "cpu"

sam:
  model_type: "vit_h"  # or "vit_l" or "vit_b"
  checkpoint: "weights/sam_vit_h_4b8939.pth"
  device: "cuda"  # or "cpu"
  pred_iou_thresh: 0.88
  stability_score_thresh: 0.95
```

## Quick Test

Test annotation with a single tile:

```python
from river_segmentation.annotation.zero_shot_annotator import (
    ZeroShotAnnotator, load_image_rgb
)

# Initialize annotator
annotator = ZeroShotAnnotator(
    grounding_dino_config="model_configs/GroundingDINO_SwinT_OGC.py",
    grounding_dino_checkpoint="weights/groundingdino_swint_ogc.pth",
    sam_checkpoint="weights/sam_vit_h_4b8939.pth",
    sam_model_type="vit_h",
    device="cuda"
)

# Load test image
image = load_image_rgb("data/tiles/tile_0_0.tif")

# Annotate
result = annotator.annotate(
    image=image,
    prompts=["river", "stream"],
    merge_masks=True
)

print(f"Found {len(result.masks)} detections")
print(f"Confidence scores: {result.scores}")
```

## Hardware Requirements

### GPU (Recommended)

**Minimum:**
- NVIDIA GPU with CUDA support
- 8GB VRAM (for vit_h)
- 4GB VRAM (for vit_l)
- 2GB VRAM (for vit_b)

**Recommended:**
- NVIDIA RTX 3090 / A5000 or better
- 24GB VRAM

### CPU-Only

CPU inference is supported but ~10-50x slower:

```yaml
grounding_dino:
  device: "cpu"

sam:
  model_type: "vit_b"  # Use smallest model for CPU
  device: "cpu"
```

**Tips for CPU:**
- Use smaller tiles (256x256)
- Use vit_b SAM model
- Increase box_threshold for fewer detections
- Process in smaller batches

## Troubleshooting

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'groundingdino'`

**Solution:**
```bash
pip install groundingdino-py
# or from source
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO && pip install -e .
```

**Error:** `ModuleNotFoundError: No module named 'segment_anything'`

**Solution:**
```bash
pip install segment-anything
```

### CUDA Errors

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
1. Use smaller SAM model: `vit_l` or `vit_b` instead of `vit_h`
2. Reduce tile size: 256x256 instead of 512x512
3. Process fewer tiles at once
4. Use CPU: `device: "cpu"`

**Error:** `CUDA not available`

**Solution:**
```bash
# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Model Loading Errors

**Error:** `FileNotFoundError: groundingdino_swint_ogc.pth not found`

**Solution:**
- Check paths in config file
- Download model weights (see above)
- Use absolute paths if needed

**Error:** `RuntimeError: Error(s) in loading state_dict`

**Solution:**
- Ensure checkpoint matches model type
- Download correct checkpoint version
- Check file integrity (re-download if corrupted)

### Performance Issues

**Slow inference:**
- Use GPU instead of CPU
- Use smaller SAM model (vit_b)
- Increase box_threshold to reduce detections
- Process smaller batches

**High memory usage:**
- Use smaller tiles
- Use vit_b SAM model
- Clear GPU cache: `torch.cuda.empty_cache()`

## Model Updates

To update models to newer versions:

```bash
# Remove old weights
rm weights/groundingdino_swint_ogc.pth
rm weights/sam_vit_h_4b8939.pth

# Download new versions
cd weights
wget <new_model_url>
cd ..

# Update config paths if needed
```

## Alternative Models

### Using Different SAM Checkpoints

```yaml
sam:
  model_type: "vit_l"
  checkpoint: "weights/sam_vit_l_0b3195.pth"
```

### Using Mobile SAM (Faster)

For faster inference on mobile/edge devices:

```bash
pip install mobile-sam
```

Update annotator to use Mobile SAM (requires code modification).

## Security Notes

- Download models only from official sources
- Verify checksums if provided
- Keep model weights secure
- Don't commit weights to git (add to .gitignore)

## Resources

- [Grounding DINO GitHub](https://github.com/IDEA-Research/GroundingDINO)
- [Grounding DINO Paper](https://arxiv.org/abs/2303.05499)
- [SAM GitHub](https://github.com/facebookresearch/segment-anything)
- [SAM Paper](https://arxiv.org/abs/2304.02643)
- [SAM Demo](https://segment-anything.com/)

## Next Steps

After setup:
1. Configure annotation pipeline (see `docs/configuration_guide.md`)
2. Run annotation: `python scripts/annotate_tiles.py --config config.yaml`
3. Review results: `python scripts/launch_reviewer.py`
4. Export training data
5. Train U-Net model

See `docs/annotation_guide.md` for complete workflow.
