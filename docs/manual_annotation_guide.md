# Manual Annotation Guide for River Segmentation

This guide explains how to manually annotate rivers in satellite imagery for training the U-Net model.

## Overview

Manual annotation creates training labels (binary masks) where:
- **0 (black)** = Water/River
- **1 (white)** = Land

## Workflow

### Step 1: Prepare Image for Annotation

Create an initial mask template using automatic thresholding:

```bash
python scripts/prepare_for_annotation.py \
  --image data/Sentinel2_1_bands.tif \
  --output-mask data/annotation/mask_template.tif
```

Or export everything ready for QGIS:

```bash
python scripts/prepare_for_annotation.py \
  --image data/Sentinel2_1_bands.tif \
  --export-for-qgis \
  --qgis-dir data/annotation
```

This creates:
- `image_to_annotate.tif` - Copy of your image
- `mask_template.tif` - Auto-generated mask as starting point
- `QGIS_INSTRUCTIONS.txt` - Step-by-step QGIS instructions

### Step 2: Manual Annotation Options

#### Option A: QGIS (Recommended for GIS users)

**Setup:**
1. Install QGIS: https://qgis.org/download/
2. Install the "Serval" plugin for raster editing:
   - Plugins → Manage and Install Plugins → Search "Serval"

**Annotation workflow:**
1. Open QGIS and create new project
2. Add layers:
   - Layer → Add Raster Layer → `image_to_annotate.tif`
   - Layer → Add Raster Layer → `mask_template.tif`
3. Set mask layer opacity to ~50% to see image underneath
4. Use Serval plugin to edit mask:
   - Plugins → Serval → Pixel Draw
   - Set brush to draw 0 (water) or 1 (land)
   - Paint corrections directly on mask
5. Save edited mask as `mask_final.tif`

**Tips for QGIS:**
- Use layer transparency to see image through mask
- Zoom in to see individual pixels
- Use different colors for visualization (blue=water, green=land)
- Save frequently!

#### Option B: Python with labelme

**Setup:**
```bash
pip install labelme
```

**Workflow:**
1. Convert GeoTIFF to PNG for annotation:
```python
python scripts/convert_for_labelme.py \
  --image data/Sentinel2_1_bands.tif \
  --output data/annotation/image.png
```

2. Annotate with labelme:
```bash
labelme data/annotation/image.png
```

3. Convert annotations back to GeoTIFF:
```python
python scripts/labelme_to_geotiff.py \
  --json data/annotation/image.json \
  --reference data/Sentinel2_1_bands.tif \
  --output data/annotation/mask_final.tif
```

#### Option C: ArcGIS Pro

1. Open ArcGIS Pro
2. Add image and mask_template to map
3. Use "Reclassify" or "Raster Calculator" tools
4. Manually digitize corrections using "Editor" toolbar
5. Export as GeoTIFF

### Step 3: Verify Annotations

Visualize your annotations to check quality:

```bash
python scripts/visualize_annotation.py \
  --image data/Sentinel2_1_bands.tif \
  --mask data/annotation/mask_final.tif \
  --output data/annotation/verification.png
```

This creates:
- Overlay visualization showing water highlighted in blue
- Histogram comparing water vs land pixel values
- Statistics on coverage

**Check for:**
- Rivers are marked as 0 (water)
- Lakes/ponds removed if not relevant
- No missing river sections
- Clean boundaries (no mixed pixels if possible)
- Reasonable water coverage (typically 5-30% for river scenes)

### Step 4: Create Training Patches

Split your large annotated image into training patches:

```bash
python scripts/create_patches.py \
  --image data/Sentinel2_1_bands.tif \
  --mask data/annotation/mask_final.tif \
  --output-dir data/patches \
  --patch-size 256 \
  --overlap 32 \
  --min-water-percent 0.01
```

**Parameters:**
- `--patch-size`: Size of patches (256x256 is standard for U-Net)
- `--overlap`: Overlap between patches (32 pixels helps continuity)
- `--min-water-percent`: Skip patches with <1% water (reduces imbalance)

This creates:
```
data/patches/
├── images/
│   ├── patch_0000_0000.tif
│   ├── patch_0000_0001.tif
│   └── ...
└── masks/
    ├── patch_0000_0000.tif
    ├── patch_0000_0001.tif
    └── ...
```

### Step 5: Train the Model

Now you can train with your patches:

```bash
python scripts/train.py \
  --image-dir data/patches/images \
  --mask-dir data/patches/masks \
  --output-dir outputs/experiment_1 \
  --n-channels 1 \
  --epochs 50 \
  --batch-size 16 \
  --val-split 0.2
```

**Note:** Use `--n-channels 1` since your image is single-band.

## Annotation Best Practices

### What to Annotate

**Include:**
- Main river channels
- Tributaries and streams
- River banks and edges
- Seasonal water flows

**Exclude (unless relevant):**
- Lakes and ponds (circular water bodies)
- Ocean/sea
- Small puddles or temporary water
- Shadows that look like water

### Quality Tips

1. **Consistency**: Use the same criteria throughout
2. **Zoom in**: Check boundaries at pixel level
3. **Use context**: Rivers are linear, connected features
4. **Check histogram**: Water should have consistently lower NIR values
5. **Save versions**: Keep backups as you work
6. **Validate often**: Run visualization script to check progress

### Common Issues

**Issue**: Shadows misclassified as water
- **Fix**: Check NIR values - shadows are darker but not as dark as water
- Use topographic data if available

**Issue**: Narrow rivers missed
- **Fix**: At 10m resolution, rivers <10m wide may be partial pixels
- Mark center line at minimum

**Issue**: Seasonal changes
- **Fix**: Annotate actual water extent at time of image
- Consider multiple seasonal images

**Issue**: Mixed pixels at boundaries
- **Fix**: Use best judgment - mark as water if >50% water
- U-Net will learn to handle boundary pixels

## Annotation Shortcuts

### Quick threshold-based annotation:

```python
# Create mask using simple threshold
python scripts/prepare_for_annotation.py \
  --image data/Sentinel2_1_bands.tif \
  --output-mask data/annotation/mask_auto.tif \
  --threshold 0.20
```

Then manually refine only problem areas.

### Region-based annotation:

For large images, annotate regions separately:

1. Extract region of interest:
```bash
# Extract specific region (row_start, row_end, col_start, col_end)
gdal_translate -srcwin 0 0 2000 2000 \
  data/Sentinel2_1_bands.tif \
  data/annotation/region_1.tif
```

2. Annotate smaller region
3. Merge back into full mask

## Next Steps

After creating patches:
1. Review patch quality with visualization script
2. Check train/val split is representative
3. Start training with small number of epochs first (10-20)
4. Evaluate results and refine annotations if needed
5. Scale up training once quality is good

## Getting Help

- Check annotation quality with: `scripts/visualize_annotation.py`
- Preview patches before training
- Start with small test set (50-100 patches) to validate workflow
- Iterate on annotation quality based on training results
