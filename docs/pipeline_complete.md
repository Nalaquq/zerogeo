# Zero-Shot Training Dataset Generation Pipeline

**Complete guide to generating training datasets for binary .tiff raster segmentation using Grounding DINO and SAM.**

## Overview

This pipeline enables **automated training dataset creation** from binary raster imagery (.tiff files) without manual annotation. It combines state-of-the-art zero-shot models with human-in-the-loop review to generate high-quality segmentation datasets for any binary classification task (rivers, roads, buildings, vegetation, etc.).

**Primary Goal**: Transform large, unlabeled .tiff rasters into ready-to-use training datasets through text prompt-driven annotation.

## Complete Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│   ZERO-SHOT TRAINING DATASET GENERATION PIPELINE               │
│   (For Binary .tiff Raster Segmentation)                       │
└─────────────────────────────────────────────────────────────────┘

INPUT: Large Binary .tiff Raster (e.g., Sentinel-2, aerial imagery)
  │
  │  Examples: 10045×1771 river scene, 5000×5000 urban area, etc.
  ↓

1. TILE LARGE IMAGE
   ├─ Tool: scripts/tile_image.py
   ├─ Process: Split into 512×512 tiles with overlap
   ├─ Preserves: CRS, georeference, spatial metadata
   ├─ Output: Tiles + tile_metadata.json
   └─ Time: ~1 second per 10k×2k image

           ↓

2. CONFIGURE PIPELINE
   ├─ File: config/my_annotation.yaml
   ├─ Settings: Text prompts, model paths, thresholds
   ├─ Validation: scripts/validate_config.py
   └─ Examples: Rivers, roads, buildings, vegetation

           ↓

3. ZERO-SHOT ANNOTATION ⭐ CORE FEATURE
   ├─ Models: Grounding DINO (detection) + SAM (segmentation)
   ├─ Tool: scripts/annotate_tiles.py
   ├─ Input: Text prompts (e.g., "river", "stream", "waterway")
   ├─ Process:
   │   ├─ Grounding DINO detects objects from text prompts
   │   └─ SAM generates precise binary masks
   ├─ Output: Binary masks (0/1) with confidence scores
   ├─ Metadata: annotation_metadata.json (scores, coverage, etc.)
   └─ Time: ~2-5 sec/tile (GPU) or ~30-60 sec/tile (CPU)

           ↓

4. MANUAL REVIEW & QUALITY CONTROL
   ├─ Tool: scripts/launch_reviewer.py
   ├─ UI: PyQt5-based interactive reviewer
   ├─ Actions:
   │   ├─ Accept (A): Good masks
   │   ├─ Reject (R): Bad masks
   │   ├─ Edit (E): Refine with brush tools
   │   └─ Skip (S): Review later
   ├─ Features: Progress tracking, session persistence
   └─ Output: Reviewed masks + review session data

           ↓

5. EXPORT TRAINING DATASET ⭐ KEY OUTPUT
   ├─ Tool: scripts/export_training_data.py
   ├─ Split: Train/Val/Test (configurable ratios)
   ├─ Formats: GeoTIFF, PNG, NumPy arrays
   ├─ Filtering: Min coverage thresholds, quality filters
   ├─ Output Structure:
   │   ├─ train/images/*.tiff
   │   ├─ train/masks/*.tiff
   │   ├─ val/images/*.tiff
   │   ├─ val/masks/*.tiff
   │   ├─ test/images/*.tiff
   │   └─ test/masks/*.tiff
   └─ Result: Ready-to-use training dataset

           ↓

6. (OPTIONAL) TRAIN MODEL
   ├─ Tool: scripts/train.py
   ├─ Architecture: U-Net or custom model
   ├─ Loss: Dice + BCE
   ├─ Input: Generated training dataset from step 5
   └─ Output: Trained model weights

           ↓

7. (OPTIONAL) DEPLOY & PREDICT
   ├─ Tool: scripts/predict.py
   ├─ Input: New .tiff rasters
   └─ Output: Binary segmentation masks
```

## Components Implemented

### 1. Tiling Module ✅

**Files:**
- `src/river_segmentation/annotation/tiler.py`
- `src/river_segmentation/annotation/__init__.py`

**Features:**
- Grid-based tiling with configurable overlap
- Geospatial metadata preservation
- Neighbor tracking for context
- Weighted blending for reconstruction
- JSON metadata storage

**Usage:**
```bash
python scripts/tile_image.py data/input.tif data/tiles/ --tile-size 512 --overlap 64
```

### 2. Configuration System ✅

**Files:**
- `src/river_segmentation/config/schema.py`
- `src/river_segmentation/config/loader.py`

**Features:**
- Type-safe dataclasses with validation
- YAML-based configuration
- Default values and overrides
- Config merging and validation
- Human-readable summaries

**Usage:**
```bash
python scripts/validate_config.py config/my_config.yaml --show-summary
```

### 3. Zero-Shot Annotation ✅

**Files:**
- `src/river_segmentation/annotation/zero_shot_annotator.py`
- `src/river_segmentation/annotation/batch_annotator.py`

**Models:**
- **Grounding DINO**: Text-based object detection
- **SAM**: High-quality segmentation

**Features:**
- Text prompt-based detection
- Automatic mask generation
- Batch processing of tiles
- Confidence scoring
- Metadata tracking

**Usage:**
```bash
python scripts/annotate_tiles.py --config config/river_annotation.yaml
```

### 4. Manual Review UI ✅

**Files:**
- `src/river_segmentation/annotation/review_session.py`
- `src/river_segmentation/annotation/reviewer_ui.py`

**Features:**
- PyQt5-based GUI
- Accept/Reject/Skip actions
- Mask editing (draw/erase)
- Progress tracking
- Session persistence
- Keyboard shortcuts (A/R/S/E)
- Dark theme

**Usage:**
```bash
python scripts/launch_reviewer.py data/tiles/ data/masks/
```

### 5. Documentation ✅

**Files:**
- `docs/configuration_guide.md` - Config system guide
- `docs/model_setup.md` - Model installation and setup
- `docs/annotation_guide.md` - Complete annotation workflow
- `docs/reviewer_guide.md` - Review UI guide

### 6. Example Configurations ✅

**Files:**
- `config/minimal_config.yaml` - Minimal setup
- `config/river_annotation_example.yaml` - Full example
- `config/cpu_config.yaml` - CPU-only config
- `config/high_resolution_config.yaml` - High-quality config

## Quick Start

### 1. Install Dependencies

```bash
# Base dependencies
pip install -r requirements.txt

# Annotation dependencies (optional)
pip install -r requirements-annotation.txt
```

### 2. Download Models

```bash
# Create directories
mkdir -p weights configs

# Download Grounding DINO
cd weights
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..

# Download SAM
cd weights
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ..

# Download config
cd configs
wget https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py
cd ..
```

### 3. Run Complete Pipeline

```bash
# 1. Tile image
python scripts/tile_image.py data/Sentinel2_1_bands.tif data/tiles/

# 2. Configure (edit config file)
cp config/river_annotation_example.yaml config/my_config.yaml
# Edit config/my_config.yaml with your settings

# 3. Validate config
python scripts/validate_config.py config/my_config.yaml --show-summary

# 4. Run annotation
python scripts/annotate_tiles.py --config config/my_config.yaml

# 5. Review results
python scripts/launch_reviewer.py data/tiles/ data/masks/

# 6. Export training data
python scripts/export_training_data.py data/review_session/ --output data/training/

# 7. Train model
python scripts/train.py --images data/training/images --masks data/training/masks
```

## API Usage

### Python API

```python
from river_segmentation.annotation import (
    ImageTiler,
    TileManager,
    ZeroShotAnnotator,
    BatchAnnotator,
    ReviewSession,
    launch_reviewer,
)
from river_segmentation.config import load_config

# 1. Tile image
tiler = ImageTiler(tile_size=512, overlap=64)
tile_manager = tiler.tile_image("input.tif", "data/tiles/")

# 2. Load config
config = load_config("config/my_config.yaml")

# 3. Initialize annotator
annotator = ZeroShotAnnotator(
    grounding_dino_config=config.grounding_dino.config_file,
    grounding_dino_checkpoint=config.grounding_dino.model_checkpoint,
    sam_checkpoint=config.sam.checkpoint,
    sam_model_type=config.sam.model_type,
    device="cuda"
)

# 4. Batch annotate
batch = BatchAnnotator(annotator, "data/masks/")
batch.annotate_tiles(tile_manager, prompts=["river", "stream"])

# 5. Review
session = ReviewSession("data/review/", "data/tiles/", "data/masks/")
launch_reviewer(session)
```

## Key Features

### Tiling
- ✅ Configurable tile size and overlap
- ✅ Automatic edge handling
- ✅ Geospatial metadata preservation
- ✅ Reconstruction with blending

### Configuration
- ✅ YAML-based configs
- ✅ Type-safe validation
- ✅ Default values
- ✅ Config merging

### Annotation
- ✅ Text prompt-based detection
- ✅ High-quality segmentation
- ✅ Batch processing
- ✅ Confidence scoring
- ✅ GPU/CPU support

### Review
- ✅ Visual inspection UI
- ✅ Accept/Reject/Edit
- ✅ Keyboard shortcuts
- ✅ Progress tracking
- ✅ Session persistence

### Documentation
- ✅ Comprehensive guides
- ✅ Example configs
- ✅ API reference
- ✅ Troubleshooting

## File Structure

```
civic/
├── src/river_segmentation/
│   ├── annotation/
│   │   ├── tiler.py                    # Tiling
│   │   ├── review_session.py           # Review state management
│   │   ├── reviewer_ui.py              # PyQt5 GUI
│   │   ├── zero_shot_annotator.py      # DINO + SAM wrappers
│   │   └── batch_annotator.py          # Batch processing
│   ├── config/
│   │   ├── schema.py                   # Config dataclasses
│   │   └── loader.py                   # YAML loader
│   ├── data/                           # Data loading
│   ├── models/                         # U-Net architecture
│   ├── training/                       # Training loops
│   └── utils/                          # Utilities
├── scripts/
│   ├── tile_image.py                   # Tiling tool
│   ├── validate_config.py              # Config validator
│   ├── annotate_tiles.py               # Annotation tool
│   ├── launch_reviewer.py              # Review UI launcher
│   └── train.py                        # Training script
├── config/
│   ├── minimal_config.yaml
│   ├── river_annotation_example.yaml
│   ├── cpu_config.yaml
│   └── high_resolution_config.yaml
├── docs/
│   ├── configuration_guide.md
│   ├── model_setup.md
│   ├── annotation_guide.md
│   └── reviewer_guide.md
├── examples/
│   ├── example_tiling.py
│   ├── example_config_usage.py
│   └── demo_reviewer.py
├── requirements.txt
├── requirements-annotation.txt
└── README.md
```

## Dependencies

### Core (Always Required)
- numpy, torch, torchvision
- rasterio, geopandas
- matplotlib, pillow, tqdm
- pyyaml

### GUI (For Reviewer)
- PyQt5

### Annotation (For Zero-Shot)
- opencv-python
- transformers
- supervision
- groundingdino-py (manual install)
- segment-anything (manual install)

## Model Requirements

### Grounding DINO
- **Checkpoint**: ~670MB
- **Config**: Python file
- **VRAM**: ~2GB

### SAM
- **vit_h**: ~2.4GB, 8GB VRAM (best quality)
- **vit_l**: ~1.2GB, 4GB VRAM (good quality)
- **vit_b**: ~375MB, 2GB VRAM (fast)

## Performance

### Tiling
- **Speed**: ~1 second for 10045×1771 → 66 tiles
- **Memory**: Minimal (~100MB)

### Annotation (GPU)
- **Speed**: ~2-5 seconds per tile (512×512, vit_h)
- **Memory**: ~8GB VRAM (vit_h)

### Annotation (CPU)
- **Speed**: ~30-60 seconds per tile (vit_b)
- **Memory**: ~4GB RAM

### Review
- **Speed**: 5-10 seconds per tile (quick review)
- **Speed**: 30-60 seconds per tile (with editing)

## Why Use This Pipeline?

**Traditional Annotation**:
- ❌ Manual labeling (hours to weeks)
- ❌ Requires domain expertise
- ❌ Expensive annotation services
- ❌ Difficult to scale

**Zero-Shot Pipeline**:
- ✅ Automated annotation (minutes to hours)
- ✅ Text prompts instead of labels
- ✅ Free and open-source
- ✅ Scales to thousands of images
- ✅ Human-in-the-loop for quality control

## Use Cases

This pipeline generates training datasets for:
- **River/stream segmentation** from satellite imagery
- **Road network extraction** from aerial photos
- **Building footprint detection** from high-res imagery
- **Vegetation mapping** from multispectral data
- **Water body delineation** from optical/radar data
- **Any binary segmentation** on .tiff rasters

## Next Steps

1. **Set up models**: See `docs/model_setup.md` (download DINO + SAM)
2. **Configure pipeline**: See `docs/configuration_guide.md` (create YAML config)
3. **Run zero-shot annotation**: See `docs/annotation_guide.md` ⭐ **START HERE**
4. **Review results**: See `docs/reviewer_guide.md` (quality control)
5. **Export dataset**: Generate training data for your model
6. **(Optional) Train model**: See main `README.md`

## Support

- **Documentation**: See `docs/` directory
- **Examples**: See `examples/` directory
- **Issues**: GitHub issues
- **Guides**: See individual guide files

## License

MIT License - See LICENSE file

## References

- [Grounding DINO Paper](https://arxiv.org/abs/2303.05499)
- [SAM Paper](https://arxiv.org/abs/2304.02643)
- [DINOcut Repository](https://github.com/Nalaquq/DINOcut)
