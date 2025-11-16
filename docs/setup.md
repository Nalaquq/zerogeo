# Setup Guide

Complete setup instructions for the **Civic zero-shot training dataset pipeline** for binary raster segmentation.

## Prerequisites

- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows (WSL recommended)
- **GPU** (strongly recommended): NVIDIA GPU with 8GB+ VRAM for zero-shot annotation
- **Disk Space**: ~20GB (for models, data, and dependencies)
- **RAM**: 16GB+ recommended for processing large .tiff files

## Installation Paths

### Recommended: Full Installation

**This installs EVERYTHING you need for zero-shot annotation:**

```bash
# Clone repository
git clone https://github.com/yourusername/civic.git
cd civic

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install everything (ONE COMMAND!)
pip install -r requirements.txt

# Verify installation
python scripts/check_dependencies.py
```

**What this installs (18 packages total):**
- ✅ All core dependencies (numpy, torch, rasterio, PyQt5, etc.)
- ✅ Zero-shot annotation dependencies (opencv, transformers, supervision, etc.)
- ✅ **Grounding DINO** (text-based object detection)
- ✅ **SAM** (universal segmentation model)
- ✅ Everything needed for zero-shot annotation!

**Alternative (same result):**
```bash
pip install -e ".[annotation]"
```

**Ready to use immediately:**
- ✅ Zero-shot annotation with text prompts (primary feature)
- ✅ Process binary .tiff rasters
- ✅ Generate training datasets automatically
- ✅ Review and refine annotations with GUI
- ✅ Export training data in multiple formats
- ✅ (Optional) Train U-Net models

**No separate model installation needed!** Grounding DINO and SAM are now included.

### Minimal Installation (Not Recommended)

If you only want core dependencies without zero-shot capabilities:

```bash
# Install minimal dependencies only
pip install -e .
```

**What you get:**
- ✅ Tile large .tiff images
- ✅ Train U-Net models (if you already have labeled data)
- ❌ Zero-shot annotation (main feature unavailable)

**Note**: This is NOT recommended. The whole point of Civic is zero-shot annotation. Use `pip install -r requirements.txt` instead.

### Development Installation

For contributors and developers:

```bash
# Install with development tools
pip install -e ".[dev]"
pip install -r requirements-annotation.txt

# Install pre-commit hooks (optional)
pre-commit install
```

## Verify Installation

### 1. Check Base Installation

```bash
# Test import
python -c "import river_segmentation; print('✓ Base package installed')"

# Check PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2. Check All Dependencies

```bash
# Comprehensive dependency check
python scripts/check_dependencies.py
```

You should see all required dependencies marked with ✓:
```
Core Dependencies:
  ✓ numpy
  ✓ torch
  ✓ torchvision
  ✓ pillow
  ✓ tqdm
  ✓ pyyaml

Geospatial Dependencies:
  ✓ rasterio
  ✓ geopandas

...

Model Dependencies:
  ✓ groundingdino-py
  ✓ segment-anything
```

Alternatively, check annotation-specific dependencies:
```bash
python scripts/annotate_tiles.py --check-deps
```

### 3. Run Basic Tests

```bash
# Run unit tests
pytest tests/test_basic.py -v

# Run tiling tests (doesn't require models)
pytest tests/test_tiler.py -v

# Run config tests
pytest tests/test_config.py -v
```

## Download Models (Automatic - One Command!)

**Models are essential** for the zero-shot annotation pipeline. Use the automatic downloader:

### Quick Download (Recommended)

```bash
# Download all models with one command
civic-download-models
```

**That's it!** The command automatically downloads:
- ✅ Grounding DINO checkpoint (~670MB)
- ✅ SAM vit_b checkpoint (~375MB, default)
- ✅ Grounding DINO config file

### Choose SAM Model Type

```bash
# Fast, CPU-friendly (default)
civic-download-models --sam-model vit_b  # ~375MB

# Good quality, balanced
civic-download-models --sam-model vit_l  # ~1.2GB

# Best quality (requires GPU)
civic-download-models --sam-model vit_h  # ~2.4GB
```

**Recommendation:**
- **Testing/CPU-only**: `vit_b` (default)
- **Production/GPU**: `vit_h` (best quality)
- **Balanced**: `vit_l`

### Advanced Options

```bash
# List available models
civic-download-models --list

# Download to custom directory
civic-download-models --weights-dir ./my_models --configs-dir ./my_configs

# Force re-download existing models
civic-download-models --force

# See all options
civic-download-models --help
```

### Verify Downloads

```bash
ls -lh weights/
ls -lh model_configs/
```

Expected output:
```
weights/:
-rw-r--r--  groundingdino_swint_ogc.pth  (~670MB)
-rw-r--r--  sam_vit_b_01ec64.pth         (~375MB, or vit_h/vit_l)

model_configs/:
-rw-r--r--  GroundingDINO_SwinT_OGC.py
```

### Manual Download (Not Recommended)

<details>
<summary>Click to expand manual download instructions (only if automatic download fails)</summary>

If the automatic download doesn't work, you can manually download:

**Grounding DINO:**
```bash
mkdir -p weights configs
cd weights
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ../configs
wget https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py
cd ..
```

**SAM:**
```bash
cd weights
# Choose one:
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth  # Fast
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth  # Balanced
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth  # Best
cd ..
```

</details>

## GPU Setup

### Check GPU Availability

```bash
# Check NVIDIA GPU
nvidia-smi
```

### Install CUDA PyTorch

If PyTorch doesn't detect CUDA:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Verify CUDA

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device count: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")
```

## Configuration

### Create Your First Config

```bash
# Copy example config
cp config/minimal_config.yaml config/my_project.yaml

# Edit with your settings
nano config/my_project.yaml  # or use your preferred editor
```

Update paths:
```yaml
project:
  name: "my_river_project"
  input_image: "data/your_image.tif"  # Update this
  output_dir: "data/annotations/my_project"

prompts:
  - "river"
  - "stream"
```

### Validate Config

```bash
python scripts/validate_config.py config/my_project.yaml --show-summary
```

## Quick Test

### Test Tiling (No models required)

```bash
# Tile the sample image
python scripts/tile_image.py \
    data/Sentinel2_1_bands.tif \
    data/test_tiles/ \
    --tile-size 256 \
    --overlap 32

# Check output
ls -lh data/test_tiles/
```

### Test GUI Reviewer (Requires PyQt5)

```bash
# Demo reviewer with sample data
python examples/demo_reviewer.py
```

### Test Full Pipeline (Requires models)

```bash
# Small test with limited tiles
python scripts/annotate_tiles.py \
    --tiles data/test_tiles/ \
    --prompts "river" "water" \
    --output data/test_masks/ \
    --max-tiles 5
```

## Troubleshooting

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'river_segmentation'`

**Solution:**
```bash
# Reinstall in editable mode
pip install -e .
```

### Grounding DINO Installation

**Error:** `No module named 'groundingdino'`

**Solution:**
```bash
# Try PyPI first
pip install groundingdino-py

# If that fails, install from source
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
cd ..
```

### SAM Installation

**Error:** `No module named 'segment_anything'`

**Solution:**
```bash
pip install segment-anything
# or from source
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### CUDA Errors

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
1. Use smaller SAM model:
   ```yaml
   sam:
     model_type: "vit_b"  # instead of vit_h
   ```
2. Reduce tile size:
   ```bash
   --tile-size 256  # instead of 512
   ```
3. Use CPU:
   ```yaml
   sam:
     device: "cpu"
   ```

**Error:** `CUDA not available`

**Solution:**
```bash
# Check PyTorch installation
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Rasterio Installation Issues

**Ubuntu/Debian:**
```bash
sudo apt-get install gdal-bin libgdal-dev
pip install rasterio
```

**macOS:**
```bash
brew install gdal
pip install rasterio
```

**Windows:**
```bash
# Use conda for easier installation
conda install -c conda-forge rasterio
```

### PyQt5 Issues

**Error:** `No module named 'PyQt5'`

**Solution:**
```bash
pip install PyQt5
```

**Linux display issues:**
```bash
# Install Qt dependencies
sudo apt-get install python3-pyqt5
```

### Memory Issues

**Training:**
- Reduce batch size: `--batch-size 4` or `--batch-size 2`
- Use smaller patches
- Use CPU: `--device cpu`

**Annotation:**
- Use smaller tiles: `--tile-size 256`
- Use smaller SAM model: `vit_b` instead of `vit_h`
- Process fewer tiles at once: `--max-tiles 10`

## Platform-Specific Notes

### Linux (Ubuntu/Debian)

```bash
# System dependencies
sudo apt-get update
sudo apt-get install python3-pip python3-venv gdal-bin libgdal-dev

# Python packages
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

### macOS

```bash
# Install Homebrew if needed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# System dependencies
brew install gdal

# Python packages
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

### Windows (WSL Recommended)

**Option 1: WSL (Recommended)**
```bash
# Install WSL2 with Ubuntu
wsl --install

# Follow Linux instructions above
```

**Option 2: Native Windows**
```bash
# Use Anaconda for easier setup
conda create -n civic python=3.10
conda activate civic
conda install -c conda-forge rasterio gdal
pip install -e .
```

## Next Steps

After setup is complete:

1. **Read Documentation** (Start here!)
   - [docs/annotation_guide.md](docs/annotation_guide.md) - **Zero-shot annotation guide (PRIMARY)**
   - [docs/configuration_guide.md](docs/configuration_guide.md) - Config system
   - [QUICKSTART_SINGLE_IMAGE.md](QUICKSTART_SINGLE_IMAGE.md) - Manual workflow (fallback)

2. **Try Examples**
   ```bash
   # Tiling example
   python examples/example_tiling.py

   # Config example
   python examples/example_config_usage.py

   # Reviewer demo
   python examples/demo_reviewer.py
   ```

3. **Run Your First Zero-Shot Annotation**
   ```bash
   # Create config file
   cp config/river_annotation_example.yaml config/my_project.yaml
   # Edit config/my_project.yaml with your settings

   # Run zero-shot annotation (primary workflow)
   python scripts/annotate_tiles.py --config config/my_project.yaml

   # Review generated annotations
   python scripts/launch_reviewer.py data/tiles/ data/masks/
   ```

4. **Check Advanced Docs**
   - [Model Setup](docs/model_setup.md) - Detailed model installation
   - [Reviewer Guide](docs/reviewer_guide.md) - UI usage
   - [Pipeline Complete](PIPELINE_COMPLETE.md) - Full workflow

## Getting Help

- **Documentation**: See `docs/` directory
- **Examples**: See `examples/` directory
- **Issues**: https://github.com/yourusername/civic/issues
- **Dependencies Check**: `python scripts/annotate_tiles.py --check-deps`

## Development Setup

For contributors:

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run tests
pytest tests/ -v

# Check code style
black --check src/ tests/
isort --check src/ tests/
flake8 src/ tests/

# Type checking
mypy src/
```

## Summary Checklist

**For Zero-Shot Pipeline (Recommended)**:
- [ ] Python 3.8+ installed
- [ ] Virtual environment created
- [ ] Base dependencies installed (`pip install -e .`)
- [ ] **Annotation dependencies installed** (`pip install -r requirements-annotation.txt`)
- [ ] **Grounding DINO & SAM installed** (`pip install groundingdino-py segment-anything`)
- [ ] **PyTorch with CUDA working** (verify with `torch.cuda.is_available()`)
- [ ] **Models downloaded** (`civic-download-models` - that's it!)
- [ ] Config YAML created (`cp config/river_annotation_example.yaml config/my_project.yaml`)
- [ ] Test tiling on sample .tiff (optional)
- [ ] **Ready for zero-shot annotation!** 🚀

**What you just did:**
1. Installed Python packages (3 pip commands)
2. Downloaded models (1 command: `civic-download-models`)
3. Created config file (1 copy command)

**Total setup time: ~5-10 minutes** (mostly waiting for downloads)

---

**For Base Installation Only** (not recommended - missing main features):
- [ ] Python 3.8+ installed
- [ ] Base dependencies installed
- [ ] Manual annotation tools ready (QGIS, etc.)

