# Post-Installation Setup

**Quick setup guide after installing Civic.**

## Installation Complete! Now What?

After running `pip install -r requirements.txt` (which includes EVERYTHING - Grounding DINO, SAM, all dependencies), follow these steps:

### Step 1: Download Models (One Command!)

```bash
civic-download-models
```

This command automatically downloads:
- Grounding DINO checkpoint (~670MB)
- SAM checkpoint (~375MB, vit_b by default)
- Grounding DINO config file

**Total download:** ~1GB (vit_b) or ~3GB (vit_h)

### Step 2: Create Your First Config

```bash
# Copy example config
cp config/river_annotation_example.yaml config/my_project.yaml

# Edit with your settings
nano config/my_project.yaml  # or use your favorite editor
```

Update these fields:
```yaml
project:
  name: "my_project_name"
  input_image: "data/your_image.tif"
  output_dir: "data/annotations/my_project"

prompts:
  - "river"  # Replace with your target class
  - "stream"
```

### Step 3: Run Your First Annotation

```bash
# Tile your image
python scripts/tile_image.py data/your_image.tif data/tiles/

# Run zero-shot annotation
python scripts/annotate_tiles.py --config config/my_project.yaml

# Review results
python scripts/launch_reviewer.py data/tiles/ data/masks/
```

## Alternative: SAM Model Options

If you need a different SAM model:

```bash
# Best quality (requires GPU with 8GB+ VRAM)
civic-download-models --sam-model vit_h

# Balanced (4GB+ VRAM)
civic-download-models --sam-model vit_l

# Fast, CPU-friendly (default)
civic-download-models --sam-model vit_b
```

## Python API Usage

You can also use the downloader in Python:

```python
from river_segmentation.utils import download_all_models

# Download all models
paths = download_all_models(sam_model_type="vit_b")

print(f"Grounding DINO: {paths['grounding_dino_checkpoint']}")
print(f"SAM: {paths['sam_checkpoint']}")
```

## Verify Installation

```bash
# Check if models exist
ls -lh weights/
ls -lh model_configs/

# Expected output:
# weights/groundingdino_swint_ogc.pth  (~670MB)
# weights/sam_vit_b_01ec64.pth         (~375MB)
# model_configs/GroundingDINO_SwinT_OGC.py
```

## Next Steps

1. **Read the guides:**
   - [docs/annotation_guide.md](docs/annotation_guide.md) - Complete annotation workflow
   - [docs/configuration_guide.md](docs/configuration_guide.md) - Config system
   - [QUICKSTART_SINGLE_IMAGE.md](QUICKSTART_SINGLE_IMAGE.md) - Quick start guide

2. **Try examples:**
   ```bash
   python examples/example_tiling.py
   python examples/example_config_usage.py
   ```

3. **Run your first annotation:**
   - Follow Step 3 above

## Troubleshooting

### Command not found: civic-download-models

If you get "command not found", try reinstalling:

```bash
pip install -e .
```

Or run directly with Python:

```bash
python -m river_segmentation.utils.model_downloader
```

### Download fails

If automatic download fails, see manual instructions in [docs/model_setup.md](docs/model_setup.md).

### GPU not detected

Check CUDA availability:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

If False, you can still use CPU mode (slower):

```yaml
grounding_dino:
  device: "cpu"
sam:
  device: "cpu"
  model_type: "vit_b"  # Use smallest model for CPU
```

## Getting Help

- **Documentation**: See `docs/` directory
- **Issues**: https://github.com/yourusername/civic/issues
- **Configuration guide**: [docs/configuration_guide.md](docs/configuration_guide.md)

## Ready to Go! 🚀

You're all set! Run your first zero-shot annotation:

```bash
civic-download-models
cp config/river_annotation_example.yaml config/my_project.yaml
# Edit config/my_project.yaml
python scripts/annotate_tiles.py --config config/my_project.yaml
```
