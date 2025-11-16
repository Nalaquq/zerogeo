# Installation Guide

Civic provides a **smart installer** that automatically detects your hardware (GPU/CPU) and installs the correct dependencies. This is the easiest and recommended way to install Civic!

## Quick Install (Recommended)

### One-Command Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/civic.git
cd civic

# Run the smart installer - it handles everything!
python scripts/smart_install.py
```

That's it! The installer will:
1. ✅ Detect your GPU/CPU automatically
2. ✅ Install the correct PyTorch version (CUDA or CPU)
3. ✅ Install all Civic dependencies
4. ✅ Download model weights optimized for your hardware
5. ✅ Create a configuration file optimized for your system

### What the Installer Does

The smart installer detects:
- **GPU**: NVIDIA GPU model and VRAM
- **CUDA**: Available CUDA version
- **PyTorch**: Whether it's installed and configured correctly

Based on your hardware, it:
- Installs **CUDA PyTorch** if you have an NVIDIA GPU
- Installs **CPU PyTorch** if no GPU is detected
- Downloads the **optimal SAM model** for your VRAM:
  - `vit_h` (2.4GB) for 8GB+ VRAM - best quality
  - `vit_l` (1.2GB) for 4-8GB VRAM - good balance
  - `vit_b` (375MB) for <4GB VRAM or CPU - fastest
- Suggests optimal **tile size** and **configuration**

## Check Your Hardware

Before or after installation, you can check your hardware:

```bash
python scripts/check_hardware.py
```

This shows:
- GPU information (model, VRAM, CUDA version)
- PyTorch status (installed, version, CUDA support)
- Recommended configuration settings
- Expected inference speed

Example output:
```
======================================================================
HARDWARE DETECTION REPORT
======================================================================
Platform: Linux 6.6.87.2-microsoft-standard-WSL2
Python: 3.10.12

GPU Detection:
  NVIDIA GPU Found: ✓ YES
  GPU Model: NVIDIA RTX A4000 Laptop GPU
  GPU Memory: 8192 MB (8.0 GB)
  CUDA Version: 573.57

PyTorch Status:
  Installed: ✓ YES
  Version: 2.9.1+cpu
  CUDA Support: ✗ NO (CPU-only)

RECOMMENDED CONFIGURATION
======================================================================
SAM Model: vit_h
Tile Size: 512
Device: cuda
💪 Your GPU is excellent for this task!
```

## Installation Options

### Dry Run (Preview)

See what would be installed without making changes:

```bash
python scripts/smart_install.py --dry-run
```

### Force CPU Installation

Install CPU-only version even if GPU is available:

```bash
python scripts/smart_install.py --force-cpu
```

### Force GPU Installation

Install GPU version even if no GPU detected (for cluster deployment):

```bash
python scripts/smart_install.py --force-gpu
```

### Skip Steps

Skip specific installation steps:

```bash
# Skip PyTorch (if already installed correctly)
python scripts/smart_install.py --skip-pytorch

# Skip model download (download later)
python scripts/smart_install.py --skip-models

# Skip config generation
python scripts/smart_install.py --skip-config
```

## Manual Installation

If you prefer manual control over the installation:

### Step 1: Install PyTorch

**For GPU (CUDA 12.1):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**For CPU only:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Step 2: Install Civic Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download Models

```bash
# After installation, civic-download-models command will be available
pip install -e .

# Download models (choose SAM model based on your hardware)
civic-download-models --sam-model vit_h  # Best quality (8GB+ VRAM)
civic-download-models --sam-model vit_l  # Good balance (4-8GB VRAM)
civic-download-models --sam-model vit_b  # Fast, CPU-friendly (<4GB VRAM)
```

## Verify Installation

After installation, verify everything is working:

```bash
# Check hardware and PyTorch status
python scripts/check_hardware.py

# Check annotation dependencies
python scripts/check_dependencies.py

# Verify model weights are downloaded
ls -lh weights/
```

Expected output in `weights/`:
- `groundingdino_swint_ogc.pth` (~670MB)
- `sam_vit_b_01ec64.pth` (~375MB) OR
- `sam_vit_l_0b3195.pth` (~1.2GB) OR
- `sam_vit_h_4b8939.pth` (~2.4GB)

## Upgrading from CPU to GPU

If you initially installed CPU version but now have a GPU:

```bash
# Uninstall CPU PyTorch
pip uninstall torch torchvision -y

# Install GPU PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA is now available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Troubleshooting

### PyTorch Not Using GPU

**Problem:** You have a GPU but PyTorch is in CPU mode.

**Solution:**
```bash
# Check current status
python scripts/check_hardware.py

# Reinstall with smart installer
python scripts/smart_install.py
```

### CUDA Out of Memory

**Problem:** GPU runs out of memory during annotation.

**Solutions:**
1. Use smaller SAM model: `vit_b` instead of `vit_h`
2. Reduce tile size in config: `tile_size: 256`
3. Use CPU mode temporarily

### Models Not Found

**Problem:** Error says model weights not found.

**Solution:**
```bash
# Download models manually
civic-download-models --sam-model vit_h

# Or specify a different location in your config
```

### Import Errors

**Problem:** Cannot import groundingdino or segment_anything.

**Solution:**
```bash
# Reinstall annotation dependencies
pip install -r requirements.txt

# Or install just the missing packages
pip install groundingdino-py segment-anything
```

## Hardware Requirements

### Minimum (CPU-only)
- **CPU**: Any modern processor
- **RAM**: 8GB
- **Storage**: 5GB for dependencies + models
- **Expected speed**: ~30-60 seconds per tile

### Recommended (GPU)
- **GPU**: NVIDIA with 4GB+ VRAM
- **RAM**: 16GB
- **Storage**: 10GB
- **Expected speed**: ~2-15 seconds per tile

### Optimal (GPU)
- **GPU**: NVIDIA with 8GB+ VRAM (RTX 3060, RTX 4060, A4000, etc.)
- **RAM**: 16GB+
- **Storage**: 20GB+
- **Expected speed**: ~2-5 seconds per tile

## Next Steps

After installation:

1. **Verify setup:**
   ```bash
   python scripts/check_hardware.py
   ```

2. **Check the auto-generated config:**
   ```bash
   cat config/auto_generated_config.yaml
   ```

3. **Run your first annotation:**
   ```bash
   python scripts/annotate_tiles.py --config config/auto_generated_config.yaml
   ```

4. **Explore example configs:**
   - `config/river_annotation_example.yaml` - Full featured
   - `config/cpu_config.yaml` - CPU-optimized
   - `config/high_resolution_config.yaml` - High quality
   - `config/minimal_config.yaml` - Minimal setup

## Additional Resources

- [README.md](README.md) - Project overview
- [QUICKSTART_SINGLE_IMAGE.md](QUICKSTART_SINGLE_IMAGE.md) - Quick start guide
- [docs/configuration_guide.md](docs/configuration_guide.md) - Configuration details
- [docs/model_setup.md](docs/model_setup.md) - Advanced model setup

## Getting Help

If you encounter issues:

1. Check hardware status: `python scripts/check_hardware.py`
2. Review error messages carefully
3. Check [GitHub Issues](https://github.com/yourusername/civic/issues)
4. Open a new issue with:
   - Output of `python scripts/check_hardware.py`
   - Full error message
   - Steps to reproduce

---

**Made installation as easy as possible!** 🚀

The smart installer handles all the complexity of GPU/CPU detection and PyTorch installation automatically.
