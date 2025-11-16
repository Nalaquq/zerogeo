# CIVIC: Zero-Shot Annotation Pipeline for Binary Raster Segmentation

**Automated training dataset generation from geospatial imagery using Grounding DINO and SAM.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

CIVIC is a zero-shot annotation pipeline that automatically generates training datasets for binary segmentation tasks from GeoTIFF imagery—no manual labeling required. Using text prompts and state-of-the-art vision models, you can annotate rivers, roads, buildings, and other features at scale.

**The entire pipeline runs with a single command:** `python scripts/civic.py run config/my_project.yaml`

**Key capabilities:**
- 🤖 **Zero-shot annotation** using Grounding DINO + SAM (Segment Anything)
- 💬 **Text prompt-driven** detection and segmentation
- 🗺️ **Full GeoTIFF support** with CRS preservation
- 🎨 **Interactive review UI** for quality control
- 📊 **Training dataset export** for downstream model development
- 🧠 **Optional U-Net training** for production deployment
- ⚡ **One-command pipeline** that manages everything automatically

## Key Features

### 🚀 Zero-Shot Annotation (Core Feature)
- **Grounding DINO**: Text prompt-based object detection
- **SAM**: State-of-the-art segmentation masks
- **Binary raster output**: Compatible with training pipelines
- **Batch processing**: Handle large multi-gigabyte .tiff files
- **No manual annotation required**: Text prompts drive everything

### 🗺️ Binary Raster & Geospatial Support
- **Full .tiff/GeoTIFF support**: Input and output
- **CRS preservation**: Maintains coordinate reference systems
- **Grid-based tiling**: Process large images efficiently with overlap
- **Georeferencing**: All spatial metadata preserved
- **Binary masks**: Clean 0/1 segmentation outputs

### 📊 Training Dataset Generation
- **Automated dataset creation**: From zero-shot annotations
- **Train/val/test splits**: Configurable ratios
- **Quality filtering**: Minimum coverage thresholds
- **Patch extraction**: Generate training patches from large rasters
- **Export formats**: GeoTIFF, PNG, NumPy arrays

### 🎨 Review & Quality Control
- **Web-based UI**: Flask-powered browser interface (works everywhere, including WSL)
- **Accept/Reject/Skip workflow**: Refine auto-generated masks
- **Visual overlay**: Toggle and adjust mask transparency
- **Session persistence**: Save and resume review sessions
- **Keyboard shortcuts**: Efficient batch review (A/R/S, arrow keys)

### 🗺️ Raster Reassembly for GIS Viewing
- **Automatic reassembly**: Combine tile masks back into complete rasters
- **Weighted blending**: Seamless overlap handling for clean results
- **GeoTIFF output**: Full CRS and geospatial metadata preservation
- **GIS-ready**: Open directly in QGIS, ArcGIS, or any GIS software
- **One command**: `reassemble_masks.py --run-dir output/project_TIMESTAMP/`

### ⚙️ Configuration-Driven Pipeline
- **YAML configs**: Reproducible annotation workflows
- **Type-safe validation**: Catch errors before processing
- **Reusable templates**: Share configs across projects
- **Easy tuning**: Adjust thresholds and prompts

### 🧠 Optional Model Training
- **U-Net architecture**: Train models on generated datasets
- **Multiple loss functions**: Dice, BCE, Focal, Tversky
- **Data augmentation**: Built-in augmentation pipeline
- **Checkpoint management**: Save and resume training

## Quick Start

### Installation

**Automated installation (recommended):**

```bash
git clone https://github.com/yourusername/civic.git
cd civic
python scripts/smart_install.py  # Auto-detects GPU/CPU and installs dependencies
```

The installer automatically:
- Detects your hardware (GPU/CPU)
- Installs PyTorch with correct CUDA/CPU support
- Downloads required models (Grounding DINO + SAM)
- Creates optimized configuration

**Manual installation:**
```bash
pip install -r requirements.txt
civic-download-models  # Download model weights
```

See [INSTALL.md](docs/INSTALL.md) for detailed instructions and troubleshooting.

### Basic Usage

#### Unified Pipeline (Recommended) ✅

**The preferred way is the unified pipeline - one command does everything:**

```bash
# 1. Create your config file
cp config/river_annotation_example.yaml config/my_project.yaml
# Edit config/my_project.yaml with your input image path and text prompts

# 2. Run the complete pipeline with one command!
python scripts/civic.py run config/my_project.yaml

# That's it! The script automatically:
# ✓ Creates correct directory structure
# ✓ Tiles your large GeoTIFF into manageable pieces
# ✓ Runs zero-shot annotation (Grounding DINO + SAM)
# ✓ Launches interactive review UI for quality control
# ✓ Reassembles reviewed masks into complete GeoTIFF
# ✓ Manages all paths and directories for you
```

**Run individual pipeline steps:**
```bash
python scripts/civic.py tile config/my_project.yaml       # Tiling only
python scripts/civic.py annotate config/my_project.yaml   # Annotation only
python scripts/civic.py review config/my_project.yaml     # Review only
python scripts/civic.py reassemble config/my_project.yaml # Reassembly only
```

**Advanced options:**
```bash
python scripts/civic.py run config/my_project.yaml --dry-run        # Preview steps without executing
python scripts/civic.py run config/my_project.yaml --skip-existing  # Resume from checkpoint
python scripts/civic.py run config/my_project.yaml --auto           # Skip interactive review (auto-accept all)
python scripts/civic.py run config/my_project.yaml --force          # Force re-run even if complete
```

**Benefits of the unified pipeline:**
- 🎯 **No path confusion** - Automatic directory management
- 🔄 **Resume capability** - Stop and resume anytime with `--skip-existing`
- 👀 **Safe preview** - Use `--dry-run` to see what will happen
- 📦 **Reproducible** - One config + one command = complete workflow

#### Alternative: Standalone Scripts

For advanced users who need fine-grained control, individual scripts are available:

```bash
# 1. Tile the image
python scripts/tile_image.py input/image.tif output/tiles/ --tile-size 512 --overlap 64

# 2. Annotate tiles
python scripts/annotate_tiles.py \
    --tiles output/tiles/ \
    --prompts "river" "stream" \
    --output output/masks/ \
    --dino-config model_configs/GroundingDINO_SwinT_OGC.py \
    --dino-checkpoint weights/groundingdino_swint_ogc.pth \
    --sam-checkpoint weights/sam_vit_h_4b8939.pth

# 3. Review masks
python scripts/launch_reviewer.py output/tiles/ output/masks/

# 4. Reassemble masks
python scripts/reassemble_masks.py --run-dir output/project_TIMESTAMP/
```

**Note:** The unified pipeline is recommended as it handles path management automatically and reduces errors.

#### Optional: Install as System Command

```bash
pip install -e .   # Install CIVIC package

# Then use 'civic' command anywhere:
civic run config/my_project.yaml
civic tile config/my_project.yaml
# etc.
```

See [QUICKSTART.md](docs/QUICKSTART.md) for a complete walkthrough.

## Pipeline Overview

```
Input: GeoTIFF imagery (Sentinel-2, aerial photos, etc.)
   │
   ├─→ 1. TILE: Split into 512×512 tiles with overlap
   │
   ├─→ 2. ANNOTATE: Zero-shot segmentation (Grounding DINO + SAM)
   │
   ├─→ 3. REVIEW: Interactive quality control (accept/reject masks)
   │
   ├─→ 4. REASSEMBLE: Combine tiles into complete GeoTIFF
   │
   └─→ 5. (Optional) TRAIN: Use dataset for model training
```

### Automatic Directory Structure

When using the unified pipeline, CIVIC automatically creates and manages this directory structure:

```
output/your_project_TIMESTAMP/
├── tiles/              # Tiled images with overlap
│   └── tile_metadata.json
├── raw_masks/          # Auto-generated annotations
│   └── annotation_metadata.json
├── reviewed_masks/     # Quality-controlled masks
├── reassembled/        # Complete GeoTIFF outputs
└── logs/              # Pipeline logs
```

No manual path management needed - just run `civic.py` and everything is organized automatically!

## Project Structure

```
civic/
├── src/
│   ├── civic/                    # CLI interface
│   │   └── cli.py               # Main command-line interface
│   └── river_segmentation/      # Core library
│       ├── annotation/          # Annotation pipeline
│       ├── config/              # Configuration system
│       ├── data/                # Data loaders & transforms
│       ├── models/              # U-Net architecture
│       ├── training/            # Training & metrics
│       └── utils/               # Utilities
├── scripts/                     # Standalone scripts
│   ├── civic.py                # Main pipeline script
│   ├── tile_image.py
│   ├── annotate_tiles.py
│   ├── launch_reviewer.py
│   └── reassemble_masks.py
├── config/                      # Example configurations
├── docs/                        # Documentation
├── tests/                       # Unit & integration tests
├── input/                       # Place input images here
├── output/                      # Generated outputs
└── weights/                     # Model weights
```

## Documentation

**Getting Started:**
- [INSTALL.md](docs/INSTALL.md) - Installation guide (start here!)
- [QUICKSTART.md](docs/QUICKSTART.md) - Your first annotation project
- [Configuration Guide](docs/configuration_guide.md) - Configuration system

**User Guides:**
- [Annotation Guide](docs/annotation_guide.md) - Complete annotation workflow
- [Reviewer Guide](docs/reviewer_guide.md) - Interactive review UI
- [Model Setup](docs/model_setup.md) - Model weights & setup

**See [docs/](docs/) for complete documentation.**

## Configuration Example

```yaml
project:
  name: "my_project"
  input_image: "input/image.tif"

prompts:
  - "river"
  - "stream"

# Optional: customize tiling, models, thresholds
tiling:
  tile_size: 512
  overlap: 64

grounding_dino:
  box_threshold: 0.35
  device: "cuda"  # or "cpu"

sam:
  model_type: "vit_b"  # vit_b, vit_l, or vit_h
  device: "cuda"
```

See [config/](config/) for complete examples and [docs/configuration_guide.md](docs/configuration_guide.md) for all options.

## Requirements

**Software:**
- Python 3.8+
- PyTorch 2.0+
- GDAL/Rasterio (for GeoTIFF support)
- See [requirements.txt](requirements.txt) for complete list

**Hardware (recommended):**
- GPU: NVIDIA with 8GB+ VRAM (CUDA support)
- RAM: 16GB+
- Storage: 20GB+ (models + data)

**CPU-only mode supported** (slower, use `sam.model_type: "vit_b"` for better performance)

## Common Issues

**CUDA out of memory:**
- Use smaller SAM model: `sam.model_type: "vit_b"`
- Switch to CPU: `sam.device: "cpu"`

**No detections found:**
- Lower threshold: `grounding_dino.box_threshold: 0.25`
- Try different/more specific prompts

**Missing dependencies:**
```bash
python scripts/check_dependencies.py
```

See [INSTALL.md](docs/INSTALL.md) for more troubleshooting.

## Contributing

Contributions welcome! See development guidelines in [docs/development/](docs/development/).

## Use Cases

- River and stream mapping from satellite imagery
- Road network extraction from aerial photos
- Building footprint detection
- Vegetation/land cover mapping
- Water body delineation
- Any binary segmentation task on GeoTIFF rasters

## License

MIT License - See [LICENSE](LICENSE) for details.

## References

- [Grounding DINO](https://arxiv.org/abs/2303.05499) - Zero-shot object detection
- [Segment Anything (SAM)](https://arxiv.org/abs/2304.02643) - Universal segmentation model
- Inspired by [DINOcut](https://github.com/Nalaquq/DINOcut)

## Citation

```bibtex
@software{ZeroGeo,
  title = {ZeroGeo: Zero-Shot Annotation Pipeline for Binary Raster Segmentation},
  author = {Sean Gleason},
  year = {2025},
  url = {https://github.com/yourusername/nalaquq}
}
```
