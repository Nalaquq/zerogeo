# Quick Start: Generate Training Dataset from Binary .tiff Raster

**Complete guide for generating training datasets from a single large .tiff raster using zero-shot annotation.**

## Your Data

This pipeline works with any binary .tiff raster:

- **Example**: `data/Sentinel2_1_bands.tif`
- **Size**: Any size (e.g., 1771 × 10045 pixels)
- **Bands**: Single or multi-band (will be converted to RGB for annotation)
- **Format**: .tiff, GeoTIFF with or without CRS
- **Use case**: Rivers, roads, buildings, vegetation, water bodies, etc.

## Primary Workflow: Zero-Shot Annotation ⚡

**This is the recommended approach** for generating training datasets:

**Why Zero-Shot?**
- ✅ No manual labeling required
- ✅ Automated with text prompts
- ✅ Quality control with review UI
- ✅ Generates datasets in minutes to hours (not days)
- ✅ Scales to large imagery

**Time:** ~30-60 minutes (including review)

👉 **[Zero-Shot Workflow Below](#zero-shot-annotation-workflow)**

## Alternative: Manual Annotation (Fallback) ✏️

**Use only if zero-shot doesn't meet your needs**:

- ⚠️ Time-consuming (hours to days)
- ⚠️ Requires manual labeling tools (QGIS, etc.)
- ⚠️ Difficult to scale

👉 **[Manual Workflow](#manual-annotation-workflow)** (scroll down)

---

---

## Zero-Shot Annotation Workflow

**This is the primary workflow for generating training datasets from binary .tiff rasters.**

### Prerequisites

```bash
# Install everything (ONE command!)
pip install -r requirements.txt

# Download model weights
civic-download-models
```

**That's it!** Two commands and you're ready:
1. `pip install -r requirements.txt` - Installs ALL 18 packages (including Grounding DINO & SAM)
2. `civic-download-models` - Downloads model weights (~1GB)

**What gets installed:**
- ✅ Core dependencies (numpy, torch, rasterio, PyQt5, etc.)
- ✅ Annotation dependencies (opencv, transformers, supervision, etc.)
- ✅ Grounding DINO (text-based detection)
- ✅ SAM (segmentation model)

**What gets downloaded:**
- Grounding DINO weights (~670MB)
- SAM vit_b weights (~375MB, default)
- Config files

**Optional**: Download a different SAM model:
```bash
civic-download-models --sam-model vit_h  # Best quality (~2.4GB, GPU recommended)
civic-download-models --sam-model vit_l  # Good quality (~1.2GB)
```

See [docs/model_setup.md](docs/model_setup.md) for advanced options.

### Step 1: Tile Your Image

```bash
python scripts/tile_image.py \
    data/Sentinel2_1_bands.tif \
    data/tiles/ \
    --tile-size 512 \
    --overlap 64
```

**Output:** 66 tiles in `data/tiles/` with metadata

### Step 2: Configure Annotation

```bash
# Copy example config
cp config/river_annotation_example.yaml config/my_river.yaml

# Edit with your settings
nano config/my_river.yaml
```

Update paths in config:
```yaml
project:
  name: "alaska_rivers"
  input_image: "data/Sentinel2_1_bands.tif"
  output_dir: "data/annotations/alaska_rivers"

prompts:
  - "river"
  - "stream"
  - "waterway"

# Adjust thresholds if needed
grounding_dino:
  box_threshold: 0.35  # Lower for more detections

sam:
  model_type: "vit_h"  # or "vit_b" for faster CPU inference
```

### Step 3: Run Zero-Shot Annotation

```bash
python scripts/annotate_tiles.py --config config/my_river.yaml
```

**What happens:**
1. Grounding DINO detects objects using text prompts
2. SAM generates precise segmentation masks
3. Masks saved to `data/annotations/alaska_rivers/raw_masks/`

**Expected time:** ~2-5 minutes (GPU) or ~30-60 minutes (CPU)

### Step 4: Review and Edit Annotations

```bash
python scripts/launch_reviewer.py \
    data/annotations/alaska_rivers/tiles \
    data/annotations/alaska_rivers/raw_masks
```

**Review UI Controls:**
- Press `A` to accept good masks
- Press `R` to reject bad masks
- Press `E` to enable editing mode
- Draw/erase with mouse
- Press `S` to save edits

**Tips:**
- Review all tiles for quality
- Edit masks that need minor fixes
- Reject completely wrong masks
- Use keyboard shortcuts for speed

### Step 5: Export Training Data

```bash
python scripts/export_training_data.py \
    data/annotations/alaska_rivers/review_session \
    --output data/training
```

**Output:** Training-ready dataset with train/val/test splits

### Step 6: (Optional) Train Model

**Your training dataset is now ready!** You can:
- Use it to train your own model
- Share it with collaborators
- Use it with any deep learning framework

```bash
# Optional: Train U-Net with generated dataset
python scripts/train.py \
    --image-dir data/training/images \
    --mask-dir data/training/masks \
    --output-dir outputs/alaska_rivers \
    --epochs 50 \
    --batch-size 8 \
    --n-channels 1
```

### Step 7: (Optional) Make Predictions

```bash
# Optional: Use trained model for inference
python scripts/predict.py \
    --input data/new_scene.tif \
    --output predictions/new_scene_rivers.tif \
    --checkpoint outputs/alaska_rivers/checkpoints/best_model.pth
```

**Congratulations!** You've generated a training dataset using zero-shot annotation.

---

## Manual Annotation Workflow

**⚠️ Use only if zero-shot annotation doesn't meet your needs**

This is the traditional, time-consuming approach:

### Step 1: Install Dependencies

```bash
source venv/bin/activate
pip install -e .
pip install scikit-image matplotlib  # For annotation tools
```

### Step 2: Create Initial Mask Template

Generate a starting point using automatic thresholding:

```bash
python scripts/prepare_for_annotation.py \
    --image data/Sentinel2_1_bands.tif \
    --export-for-qgis \
    --qgis-dir data/annotation
```

**Output:**
```
data/annotation/
├── image_to_annotate.tif      # Copy of your image
├── mask_template.tif           # Auto-generated mask (starting point)
└── QGIS_INSTRUCTIONS.txt      # Detailed QGIS instructions
```

### Step 3: Verify Initial Mask

Check how the auto-generated mask looks:

```bash
python scripts/visualize_annotation.py \
    --image data/annotation/image_to_annotate.tif \
    --mask data/annotation/mask_template.tif \
    --output data/annotation/initial_check.png
```

Open `data/annotation/initial_check.png` to see:
- Original image
- Auto-generated mask
- Overlay with water highlighted
- Statistics

### Step 4: Manual Annotation (Choose Your Tool)

#### Option A: QGIS (Recommended)

1. **Install QGIS**: https://qgis.org/download/

2. **Open in QGIS**:
   - Create new project
   - Add raster layer: `data/annotation/image_to_annotate.tif`
   - Add raster layer: `data/annotation/mask_template.tif`

3. **Refine the mask**:
   - Install "Serval" plugin for raster editing
   - Use Pixel Draw tool to correct:
     - Remove lakes/ponds (if you only want rivers)
     - Fix misclassified areas
     - Add missing river sections
   - Remember: 0 = water, 1 = land

4. **Save**:
   - Export/save as: `data/annotation/mask_final.tif`
   - Keep GeoTIFF format
   - Maintain same dimensions and CRS

#### Option B: Simple Threshold (Quick Start)

If the auto-generated mask looks good, use it directly:

```bash
cp data/annotation/mask_template.tif data/annotation/mask_final.tif
```

You can always refine it later if training results aren't good.

#### Option C: Python Script

Edit mask programmatically:

```python
import rasterio
import numpy as np

# Load mask
with rasterio.open('data/annotation/mask_template.tif') as src:
    mask = src.read(1)
    profile = src.profile

# Edit mask (example: remove small features)
from scipy.ndimage import binary_opening
mask = binary_opening(mask, structure=np.ones((5, 5)))

# Save
with rasterio.open('data/annotation/mask_final.tif', 'w', **profile) as dst:
    dst.write(mask, 1)
```

### Step 5: Create Training Patches

Split your large image and mask into smaller patches for training:

```bash
python scripts/create_patches.py \
    --image data/Sentinel2_1_bands.tif \
    --mask data/annotation/mask_final.tif \
    --output-dir data/patches \
    --patch-size 256 \
    --stride 128 \
    --min-water-coverage 0.05
```

**Parameters explained:**
- `--patch-size 256`: Each patch is 256×256 pixels
- `--stride 128`: 50% overlap between patches
- `--min-water-coverage 0.05`: Only keep patches with ≥5% water

**Output:**
```
data/patches/
├── images/
│   ├── patch_000.tif
│   ├── patch_001.tif
│   └── ...
└── masks/
    ├── patch_000.tif
    ├── patch_001.tif
    └── ...
```

**Expected result:** ~200-500 patches depending on image size and coverage

### Step 6: Verify Patches

Quick visual check:

```bash
python scripts/visualize_annotation.py \
    --image data/patches/images/patch_000.tif \
    --mask data/patches/masks/patch_000.tif \
    --output data/patches/check_patch_000.png
```

### Step 7: Train Model

```bash
python scripts/train.py \
    --image-dir data/patches/images \
    --mask-dir data/patches/masks \
    --output-dir outputs/manual_annotation \
    --epochs 50 \
    --batch-size 8 \
    --n-channels 1 \
    --val-split 0.2 \
    --learning-rate 0.001
```

**Training will:**
- Split data into train/validation
- Train for 50 epochs
- Save best model based on validation Dice score
- Create training history plots

**Expected time:** 10-30 minutes depending on patch count and GPU

**Outputs:**
```
outputs/manual_annotation/
├── checkpoints/
│   ├── best_model.pth          # Best model
│   └── checkpoint_epoch_*.pth  # Regular checkpoints
├── training_history.png        # Loss and metrics plots
└── config.json                 # Training configuration
```

### Step 8: Evaluate Results

```bash
# View training curves
open outputs/manual_annotation/training_history.png

# Test on new data
python scripts/predict.py \
    --input data/new_scene.tif \
    --output predictions/new_scene_rivers.tif \
    --checkpoint outputs/manual_annotation/checkpoints/best_model.pth
```

---

## Comparison: Zero-Shot vs Manual

| Aspect | Zero-Shot (Recommended) | Manual (Fallback) |
|--------|-----------|--------|
| **Time to Annotate** | 30-60 minutes ⚡ | Several hours ⏳ |
| **Accuracy** | Good (85-95%) ✅ | Excellent (95-100%) ✅ |
| **Manual Effort** | Minimal (review only) ✅ | High (all labeling) ❌ |
| **Prerequisites** | Models + GPU recommended | Just QGIS or similar |
| **Scalability** | Easy to scale ✅ | Hard to scale ❌ |
| **Best For** | **Most use cases** | Critical accuracy needs only |
| **Cost** | GPU time or cloud | Your time |
| **Training Dataset Output** | Automated ✅ | Manual ❌ |

**Recommendation**: Start with zero-shot. Only fall back to manual if results are insufficient.

## Tips for Better Results

### General Tips
- **Start small**: Test on a small region first
- **Iterate**: Train → Evaluate → Refine → Retrain
- **Augmentation**: The pipeline includes data augmentation automatically
- **Validation**: Always keep a validation set separate

### Zero-Shot Specific
- **Tune prompts**: Try different variations ("river", "stream", "waterway")
- **Adjust thresholds**: Lower `box_threshold` for more detections
- **Review carefully**: Always review auto-generated masks
- **Edit when close**: If mask is 80% correct, edit instead of rejecting

### Manual Annotation Specific
- **Use good initial mask**: Spend time on auto-threshold tuning
- **Be consistent**: Use same criteria throughout
- **Mark boundaries**: Focus on clear river boundaries
- **Include variety**: Ensure wide and narrow sections represented

## Troubleshooting

### Zero-Shot Issues

**No detections found:**
```yaml
# Lower thresholds in config
grounding_dino:
  box_threshold: 0.25  # was 0.35
```

**Poor mask quality:**
```yaml
# Use larger SAM model
sam:
  model_type: "vit_h"  # instead of vit_b
```

**Slow annotation:**
```yaml
# Use smaller/faster models
sam:
  model_type: "vit_b"
  device: "cpu"  # if no GPU
```

### Manual Annotation Issues

**Mask values are wrong:**
- Ensure 0 = water, 1 = land (not reversed)
- Check mask template generation worked correctly

**Too many/few patches:**
- Adjust `--min-water-coverage` (lower = more patches)
- Adjust `--stride` (smaller = more patches, more overlap)

**Training not improving:**
- Check validation metrics vs training metrics
- May need more diverse patches
- Try different learning rate: `--learning-rate 0.0001`

## Next Steps

After training:

1. **Evaluate on test data**
   ```bash
   python scripts/predict.py --input test_image.tif --output test_prediction.tif --checkpoint outputs/*/checkpoints/best_model.pth
   ```

2. **Visualize results**
   ```bash
   python scripts/visualize_annotation.py --image test_image.tif --mask test_prediction.tif
   ```

3. **Fine-tune if needed**
   - Add more training data from problem areas
   - Adjust model hyperparameters
   - Try different loss functions

4. **Deploy for production**
   - Use best checkpoint for inference
   - Batch process multiple scenes
   - Integrate into GIS workflow

## Additional Resources

- **[docs/annotation_guide.md](docs/annotation_guide.md)** - Complete annotation guide
- **[docs/model_setup.md](docs/model_setup.md)** - Model installation details
- **[docs/reviewer_guide.md](docs/reviewer_guide.md)** - Review UI reference
- **[docs/configuration_guide.md](docs/configuration_guide.md)** - Config system
- **[PIPELINE_COMPLETE.md](PIPELINE_COMPLETE.md)** - Full pipeline overview

## Getting Help

- Check [SETUP.md](SETUP.md) for installation issues
- See [docs/](docs/) for detailed guides
- Open an issue on GitHub
- Review example configs in `config/` directory
