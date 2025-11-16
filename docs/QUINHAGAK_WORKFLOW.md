# Quinhagak Orthomosaic Processing Guide

## Overview

Your Quinhagak orthomosaic is **massive**:
- **Dimensions:** 86,000 x 86,000 pixels
- **Bands:** 4 (RGBA)
- **Size:** ~28 GB
- **CRS:** WGS 84 / UTM zone 4N + EGM96 height
- **Will create:** ~36,000+ tiles for annotation

## ⚠️ Important: Start with a Test Subset!

Processing the full 28GB image will take **many hours** and **100+ GB of disk space**.

**Recommended approach:** Test on a small subset first to:
1. Verify prompts work for your use case
2. Tune detection thresholds
3. Check output quality
4. Estimate full processing time

---

## Quick Start: Test Subset (Recommended)

### Step 1: Create a Test Subset

Extract a 4096x4096 pixel area from the center:

```bash
python scripts/create_subset.py data/Quinhagak-Orthomosaic.tiff \
  --output data/quinhagak_subset_4096x4096.tif \
  --size 4096 \
  --center
```

This creates a ~48 MB test file (vs. 28 GB full image).

### Step 2: Run Test Pipeline

```bash
# Process the subset (takes ~5-10 minutes)
python scripts/civic.py run config/quinhagak_test_subset.yaml
```

This will:
- Create ~64 tiles (512x512 each)
- Detect roads and buildings
- Generate multi-class masks
- Launch reviewer for manual QC

### Step 3: Review Results

The reviewer will launch automatically, or run:

```bash
python scripts/launch_reviewer.py \
  data/annotations/quinhagak_test/tiles \
  data/annotations/quinhagak_test/raw_masks
```

Check if:
- Roads are being detected correctly
- Buildings/houses are segmented well
- False positives are manageable
- Classes are labeled correctly

### Step 4: Tune if Needed

Edit `config/quinhagak_test_subset.yaml` to adjust:

```yaml
# Lower for more detections (more false positives)
# Higher for fewer detections (more false negatives)
grounding_dino:
  box_threshold: 0.30    # Try 0.25, 0.35, 0.40

# Add/remove/modify prompts
prompts:
  - "road"
  - "street"
  - "building"
  - "house"
  - "roof"           # Good for aerial imagery
  - "pathway"        # For walking paths
  - "structure"      # Other built structures
```

Re-run until you're happy with results.

---

## Full Processing: Complete Orthomosaic

**⚠️ Only proceed after test subset looks good!**

### Storage Requirements

Before starting, ensure you have:
- **Input:** 28 GB (original image)
- **Tiles:** 50-100 GB (depends on compression)
- **Masks:** 10-20 GB
- **Total needed:** ~100-150 GB free disk space

### Processing Time Estimates

With NVIDIA RTX A4000:
- **Tiling:** ~30-60 minutes
- **Annotation:** ~8-15 hours (36,000+ tiles × ~1-2 sec each)
- **Review:** Manual (can take days for 36k tiles!)

### Run Full Pipeline

```bash
# Full pipeline
python scripts/civic.py run config/quinhagak_roads_houses.yaml
```

Or step-by-step:

```bash
# Step 1: Tile (creates ~36,000 tiles)
python scripts/civic.py tile config/quinhagak_roads_houses.yaml

# Step 2: Annotate (takes many hours!)
python scripts/civic.py annotate config/quinhagak_roads_houses.yaml

# Step 3: Review (launch when annotation complete)
python scripts/civic.py review config/quinhagak_roads_houses.yaml
```

### Monitor Progress

Watch the logs:

```bash
# Follow annotation progress
tail -f data/annotations/quinhagak/logs/annotation_*.log

# Check stats
grep "Tile.*Found.*detections" data/annotations/quinhagak/logs/annotation_*.log | wc -l
```

---

## Configuration Details

### Detection Prompts

The config uses these prompts optimized for infrastructure:

```yaml
prompts:
  - "road"           # Paved and unpaved roads
  - "street"         # Streets and pathways
  - "pathway"        # Walking paths
  - "building"       # General buildings
  - "house"          # Residential structures
  - "structure"      # Other built structures
  - "roof"           # Building roofs (good for aerial view!)
```

**Tips:**
- `road` and `street` may detect similar features
- `roof` often works better than `building` for aerial imagery
- Try adding `gravel road` or `dirt road` for unpaved surfaces
- Add `boardwalk` if relevant for Quinhagak

### Detection Thresholds

```yaml
grounding_dino:
  box_threshold: 0.30   # Detection confidence
  text_threshold: 0.25  # Text-image similarity
```

**Adjustment guide:**
- **More detections:** Lower `box_threshold` (0.25)
- **Fewer false positives:** Raise `box_threshold` (0.35-0.40)
- **More semantic flexibility:** Lower `text_threshold` (0.20)

### Tiling Settings

```yaml
tiling:
  tile_size: 512      # Model optimal size
  overlap: 64         # Overlap to avoid edge artifacts
```

**Don't change these unless you have a specific reason!**

---

## Multi-Class Review Features

The reviewer now supports:

✅ **View each class in different colors**
- road = blue
- building = orange
- house = yellow
- etc.

✅ **Toggle classes on/off**
- Show only roads
- Show only buildings
- Show all together

✅ **Delete false positives**
- Click ✗ button to remove

✅ **Reassign classes**
- Click ↻ button to change (e.g., street → road)

✅ **Edit mask boundaries**
- Enable editing mode
- Draw/erase to refine
- Save changes

---

## Workflow Recommendations

### For Quick Results (Roads Only)

```yaml
prompts:
  - "road"
  - "street"
```

Process → Export roads only → Use for analysis

### For Complete Infrastructure Map

```yaml
prompts:
  - "road"
  - "street"
  - "pathway"
  - "building"
  - "house"
  - "structure"
  - "roof"
```

Process → Review all classes → Export multi-class GeoTIFF

### For Training Data Generation

1. Process with all prompts
2. Manually review and clean in reviewer
3. Accept good tiles, reject bad ones
4. Export cleaned masks for training

---

## Troubleshooting

### Out of Memory

If annotation crashes:

```yaml
processing:
  batch_size: 5      # Reduce from 10
  max_workers: 2     # Reduce from 4
```

Or process in chunks (tile first, then annotate subsets).

### Too Many False Positives

Increase thresholds:

```yaml
grounding_dino:
  box_threshold: 0.40    # Higher = stricter
  text_threshold: 0.30
```

### Missing Detections

Lower thresholds or add more prompts:

```yaml
grounding_dino:
  box_threshold: 0.25    # Lower = more permissive

prompts:
  - "road"
  - "dirt road"           # More specific
  - "gravel road"
  - "building"
  - "residential building"
  - "commercial building"
```

### Disk Space Issues

Monitor usage:

```bash
du -sh data/annotations/quinhagak/*
```

Clean up if needed:

```bash
# Remove tiles if annotations are done
rm -rf data/annotations/quinhagak/tiles

# Keep masks and metadata only
```

---

## Output Files

After processing, you'll have:

```
data/annotations/quinhagak/
├── tiles/                          # Original tiles (can delete after annotation)
│   ├── quinhagak_0_0.tif
│   └── quinhagak_metadata.json
├── raw_masks/                      # Multi-class masks
│   ├── quinhagak_0_0_mask.npz     # Multi-class (preserves labels)
│   ├── quinhagak_0_0_mask.tif     # Merged binary mask
│   └── annotation_metadata.json
├── reviewed_masks/                 # After manual review
│   ├── quinhagak_0_0_mask.npz
│   └── review_metadata.json
└── logs/                           # Detailed logs
    ├── civic_pipeline_*.log
    ├── annotation_*.log
    └── tiler_*.log
```

---

## Next Steps After Annotation

### Export as Single GeoTIFF

Merge all tiles back into one image (TODO: feature coming soon)

### Use for Analysis

Load `.npz` files in Python:

```python
import numpy as np

# Load multi-class mask
data = np.load('quinhagak_0_0_mask.npz', allow_pickle=True)
masks = data['masks']        # Individual masks (N, H, W)
labels = data['labels']      # Class names
scores = data['scores']      # Confidence scores

# Filter by class
road_mask = masks[labels == 'road']
building_mask = masks[labels == 'building']
```

### Create Training Dataset

Reviewed masks can be used to train custom models!

---

## Questions?

Check the logs:
```bash
cat data/annotations/quinhagak/logs/annotation_*.log
```

Or review the configuration:
```bash
cat config/quinhagak_roads_houses.yaml
```
