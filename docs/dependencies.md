# Complete Dependency List

This document lists all dependencies required for the Civic zero-shot training dataset pipeline.

## Quick Check

Run the dependency checker to verify your installation:

```bash
python scripts/check_dependencies.py
```

## Core Dependencies (Required)

These are installed automatically with `pip install -e .`:

| Package | Version | Purpose |
|---------|---------|---------|
| **numpy** | >=1.21.0 | Array operations and numerical computing |
| **torch** | >=2.0.0 | Deep learning framework |
| **torchvision** | >=0.15.0 | Computer vision utilities for PyTorch |
| **rasterio** | >=1.3.0 | Reading/writing geospatial raster data (.tiff) |
| **geopandas** | >=0.12.0 | Geospatial data processing |
| **scikit-learn** | >=1.0.0 | Machine learning utilities |
| **matplotlib** | >=3.5.0 | Visualization and plotting |
| **pillow** | >=9.0.0 | Image processing |
| **tqdm** | >=4.60.0 | Progress bars |
| **pyyaml** | >=6.0 | YAML configuration file parsing |
| **PyQt5** | >=5.15.0 | GUI for annotation reviewer |

## Zero-Shot Annotation Dependencies (Included in requirements.txt)

These enable the primary zero-shot annotation workflow. **Automatically installed with:**
- `pip install -r requirements.txt` (recommended)
- OR `pip install -e ".[annotation]"`

| Package | Version | Purpose |
|---------|---------|---------|
| **opencv-python** | >=4.5.0 | Computer vision operations |
| **transformers** | >=4.30.0 | Transformer models and utilities |
| **supervision** | >=0.16.0 | Detection and tracking utilities |
| **huggingface-hub** | >=0.16.0 | Model hub access |
| **scipy** | >=1.9.0 | Scientific computing utilities |
| **groundingdino-py** | >=0.1.0 | Zero-shot object detection (auto-installed!) |
| **segment-anything** | >=1.0 | Universal segmentation (auto-installed!) |

## Model Dependencies (Now Included Automatically!)

**These are now installed automatically with annotation dependencies:**

### Grounding DINO

**Installed automatically with:**
```bash
pip install -e ".[annotation]"
# OR
pip install -r requirements-annotation.txt
```

**Purpose:** Text-based zero-shot object detection

**Package:** `groundingdino-py>=0.1.0`

### Segment Anything (SAM)

**Installed automatically with:**
```bash
pip install -e ".[annotation]"
# OR
pip install -r requirements-annotation.txt
```

**Purpose:** Universal image segmentation

**Package:** `segment-anything>=1.0`

**Note:** You no longer need to install these separately! They're included in the annotation dependencies.

## Development Dependencies (Optional)

For contributors and developers. Install with `pip install -e ".[dev]"`:

| Package | Version | Purpose |
|---------|---------|---------|
| **pytest** | >=7.0.0 | Testing framework |
| **pytest-cov** | >=3.0.0 | Code coverage |
| **black** | >=22.0.0 | Code formatting |
| **flake8** | >=4.0.0 | Linting |
| **mypy** | >=0.950 | Type checking |
| **isort** | >=5.10.0 | Import sorting |

## Documentation Dependencies (Optional)

For building documentation. Install with `pip install -e ".[docs]"`:

| Package | Version | Purpose |
|---------|---------|---------|
| **sphinx** | >=4.5.0 | Documentation generator |
| **sphinx-rtd-theme** | >=1.0.0 | Read the Docs theme |
| **myst-parser** | >=0.18.0 | Markdown support for Sphinx |

## Installation Options

### Recommended: Complete Installation
```bash
pip install -r requirements.txt
```
**Installs EVERYTHING you need** (18 packages total). Includes all core dependencies, annotation dependencies, Grounding DINO, and SAM.

### Alternative: Using setup.py
```bash
pip install -e ".[annotation]"
```
Same as above, just using pyproject.toml instead of requirements.txt.

### Minimal (Not Recommended)
```bash
pip install -e .
```
Installs only 11 core packages. **Does not include zero-shot annotation** (the main feature). Only use if you already have labeled data.

## System Dependencies

### For rasterio (geospatial)

**Ubuntu/Debian:**
```bash
sudo apt-get install gdal-bin libgdal-dev
```

**macOS:**
```bash
brew install gdal
```

**Windows:**
Use conda for easier installation:
```bash
conda install -c conda-forge rasterio
```

### For PyQt5 (GUI)

**Linux:**
```bash
sudo apt-get install python3-pyqt5
```

## GPU Support

For GPU acceleration (highly recommended for zero-shot annotation):

### CUDA 11.8
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### CUDA 12.1
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Verify GPU
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
```

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError`, reinstall:
```bash
pip install -e .
```

### GDAL/Rasterio Issues

Install system GDAL libraries first:
```bash
# Ubuntu/Debian
sudo apt-get install gdal-bin libgdal-dev

# macOS
brew install gdal

# Then reinstall rasterio
pip install rasterio --force-reinstall
```

### PyQt5 Display Issues

On Linux:
```bash
sudo apt-get install python3-pyqt5 python3-pyqt5.qtsvg
```

### Grounding DINO Installation Fails

Try from source:
```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
```

## Dependency Graph

```
civic (base)
├── numpy
├── torch ─┐
├── torchvision ─┤
├── rasterio
├── geopandas ─── requires: pandas, shapely, fiona
├── scikit-learn
├── matplotlib
├── pillow
├── tqdm
├── pyyaml
└── PyQt5

civic[annotation]
├── opencv-python
├── transformers ─── requires: tokenizers, huggingface-hub
├── supervision
├── scipy
└── huggingface-hub

External Models
├── groundingdino-py ─── requires: torch, torchvision, transformers
└── segment-anything ─── requires: torch, torchvision
```

## Minimum Versions

Python version: **3.8 or higher** (3.9+ recommended)

## Recommended Versions (Tested)

- Python: 3.10 or 3.11
- PyTorch: 2.0+ with CUDA 11.8 or 12.1
- CUDA: 11.8 or 12.1 (for GPU support)

## Disk Space Requirements

- Base install: ~5GB
- With annotation dependencies: ~8GB
- Model weights (Grounding DINO + SAM vit_h): ~3GB
- Total recommended: **15-20GB**

## RAM Requirements

- Minimum: 8GB
- Recommended: 16GB+
- For large imagery: 32GB+

## GPU Requirements

- Minimum: NVIDIA GPU with 4GB VRAM (for vit_b)
- Recommended: 8GB+ VRAM (for vit_h)
- Best: 24GB+ VRAM (for large-scale processing)

## Check Your Installation

After installation, verify everything:

```bash
# Check all dependencies
python scripts/check_dependencies.py

# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Check annotation dependencies
python scripts/annotate_tiles.py --check-deps

# List installed packages
pip list | grep -E "numpy|torch|rasterio|opencv|groundingdino|segment"
```

## See Also

- [SETUP.md](SETUP.md) - Complete setup guide
- [POST_INSTALL.md](POST_INSTALL.md) - Post-installation steps
- [README.md](README.md) - Main documentation
