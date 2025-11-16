#!/usr/bin/env python3
"""
Smart installation script for Civic.

Automatically detects hardware (GPU/CPU) and installs the correct dependencies.
This makes installation as easy as possible for end users!

Usage:
    python scripts/smart_install.py
    python scripts/smart_install.py --dry-run  # Show what would be installed
    python scripts/smart_install.py --force-cpu  # Force CPU installation
    python scripts/smart_install.py --force-gpu  # Force GPU installation
"""

import sys
import subprocess
import argparse
from pathlib import Path

# Add src to path so we can import our hardware detection
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from river_segmentation.utils.hardware_detect import detect_hardware, HardwareInfo


def run_command(cmd: str, description: str, dry_run: bool = False) -> bool:
    """
    Run a shell command with nice output.

    Args:
        cmd: Command to run
        description: Human-readable description
        dry_run: If True, only print command without running

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'[DRY RUN] ' if dry_run else ''}{'='*70}")
    print(f"📦 {description}")
    print(f"{'='*70}")
    print(f"Command: {cmd}\n")

    if dry_run:
        print("[DRY RUN MODE - Not actually running this command]")
        return True

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            text=True
        )
        print(f"\n✓ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with error code {e.returncode}")
        return False


def install_pytorch(hw_info: HardwareInfo, force_mode: str = None, dry_run: bool = False) -> bool:
    """
    Install PyTorch with correct GPU/CPU support.

    Args:
        hw_info: Hardware information
        force_mode: Force 'cpu' or 'gpu' installation
        dry_run: If True, only show what would be installed

    Returns:
        True if successful
    """
    # Determine installation mode
    if force_mode:
        use_gpu = (force_mode == "gpu")
    else:
        use_gpu = hw_info.has_nvidia_gpu

    # Check if already installed correctly
    if hw_info.pytorch_installed and not dry_run:
        if use_gpu and hw_info.pytorch_has_cuda:
            print("\n✓ PyTorch with CUDA is already installed correctly!")
            return True
        elif not use_gpu and not hw_info.pytorch_has_cuda:
            print("\n✓ PyTorch (CPU) is already installed correctly!")
            return True
        else:
            print("\n⚠️  PyTorch is installed but in wrong mode. Reinstalling...")
            # Uninstall first
            if not run_command(
                "pip uninstall torch torchvision torchaudio -y",
                "Uninstalling old PyTorch",
                dry_run
            ):
                return False

    # Install PyTorch
    install_cmd = hw_info.get_pytorch_install_command()

    if use_gpu:
        description = f"Installing PyTorch with CUDA support for {hw_info.gpu_name}"
    else:
        description = "Installing PyTorch (CPU-only)"

    return run_command(install_cmd, description, dry_run)


def install_civic_deps(dry_run: bool = False) -> bool:
    """
    Install Civic dependencies.

    Args:
        dry_run: If True, only show what would be installed

    Returns:
        True if successful
    """
    # Get path to requirements.txt
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    requirements_file = repo_root / "requirements.txt"

    if not requirements_file.exists():
        print(f"\n✗ Requirements file not found: {requirements_file}")
        return False

    cmd = f"pip install -r {requirements_file}"
    return run_command(cmd, "Installing Civic dependencies", dry_run)


def download_models(hw_info: HardwareInfo, dry_run: bool = False) -> bool:
    """
    Download model weights.

    Args:
        hw_info: Hardware information
        dry_run: If True, only show what would be downloaded

    Returns:
        True if successful
    """
    sam_model = hw_info.get_sam_model_recommendation()
    cmd = f"civic-download-models --sam-model {sam_model}"

    description = f"Downloading model weights (SAM: {sam_model})"
    return run_command(cmd, description, dry_run)


def create_test_config(hw_info: HardwareInfo, dry_run: bool = False) -> bool:
    """
    Create an optimized test configuration file.

    Args:
        hw_info: Hardware information
        dry_run: If True, only show what would be created

    Returns:
        True if successful
    """
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    config_dir = repo_root / "config"
    config_file = config_dir / "auto_generated_config.yaml"

    sam_model = hw_info.get_sam_model_recommendation()
    tile_size = hw_info.get_tile_size_recommendation()
    device = "cuda" if (hw_info.has_nvidia_gpu and hw_info.pytorch_has_cuda) else "cpu"

    # Determine SAM checkpoint filename
    sam_checkpoints = {
        "vit_h": "sam_vit_h_4b8939.pth",
        "vit_l": "sam_vit_l_0b3195.pth",
        "vit_b": "sam_vit_b_01ec64.pth"
    }
    sam_checkpoint = sam_checkpoints[sam_model]

    config_content = f"""# AUTO-GENERATED CONFIGURATION
# Created by smart_install.py based on your hardware
# Hardware: {hw_info.gpu_name or 'CPU'}
# Generated for: {hw_info.platform}

project:
  name: "test_annotation"
  input_image: "data/Sentinel2_1_bands.tif"
  output_dir: "data/annotations/test_output"
  description: "Auto-configured annotation project"

tiling:
  tile_size: {tile_size}              # Optimized for your hardware
  overlap: {tile_size // 8}                 # 12.5% overlap
  min_tile_size: {tile_size // 2}
  prefix: "tile"

grounding_dino:
  model_checkpoint: "weights/groundingdino_swint_ogc.pth"
  config_file: "model_configs/GroundingDINO_SwinT_OGC.py"
  box_threshold: 0.35
  text_threshold: 0.25
  device: "{device}"

sam:
  model_type: "{sam_model}"         # Optimized for your hardware
  checkpoint: "weights/{sam_checkpoint}"
  device: "{device}"
  pred_iou_thresh: 0.88
  stability_score_thresh: 0.95

prompts:
  - "river"
  - "stream"
  - "waterway"

review:
  auto_accept_threshold: 0.90
  show_tiles_per_page: 16
  default_view: "grid"
  enable_editing: true
  brush_size: 10

export:
  format: "geotiff"
  split_ratio:
    train: 0.7
    val: 0.2
    test: 0.1
  seed: 42
  create_patches: true
  patch_size: 256
  min_mask_coverage: 0.01

# PERFORMANCE NOTES:
# Device: {device.upper()}
# SAM Model: {sam_model} ({sam_checkpoints[sam_model]})
# Tile Size: {tile_size}x{tile_size}
"""

    if hw_info.has_nvidia_gpu:
        vram_gb = hw_info.gpu_memory_mb / 1024 if hw_info.gpu_memory_mb else 0
        config_content += f"# GPU: {hw_info.gpu_name} ({vram_gb:.1f} GB VRAM)\n"

        if device == "cuda":
            config_content += "# Expected speed: ~2-5 seconds per tile (GPU accelerated)\n"
        else:
            config_content += "# Note: Install CUDA PyTorch for GPU acceleration!\n"
    else:
        config_content += "# Expected speed: ~30-60 seconds per tile (CPU)\n"

    print(f"\n{'[DRY RUN] ' if dry_run else ''}{'='*70}")
    print("📝 Creating optimized configuration file")
    print(f"{'='*70}")
    print(f"File: {config_file}")
    print(f"Device: {device.upper()}")
    print(f"SAM Model: {sam_model}")
    print(f"Tile Size: {tile_size}x{tile_size}\n")

    if dry_run:
        print("[DRY RUN MODE - Not actually creating file]")
        print("\nFile contents preview:")
        print("-" * 70)
        print(config_content[:500] + "..." if len(config_content) > 500 else config_content)
        return True

    try:
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file.write_text(config_content)
        print(f"\n✓ Created configuration file: {config_file}")
        return True
    except Exception as e:
        print(f"\n✗ Failed to create config file: {e}")
        return False


def print_summary(hw_info: HardwareInfo, success: bool):
    """Print installation summary."""
    print("\n" + "=" * 70)
    print("INSTALLATION SUMMARY")
    print("=" * 70)

    if success:
        print("\n🎉 Installation completed successfully!")
        print("\nYour system is ready to use Civic!")

        print("\n" + "-" * 70)
        print("QUICK START:")
        print("-" * 70)
        print("\n1. Verify installation:")
        print("   python scripts/check_hardware.py")

        print("\n2. Use the auto-generated config:")
        print("   python scripts/annotate_tiles.py --config config/auto_generated_config.yaml")

        print("\n3. Or create your own config based on the examples in config/")

        recommendation = hw_info.get_recommendation()
        if recommendation.startswith("cuda"):
            print("\n💪 GPU-accelerated inference enabled!")
            print("   Your annotations will be FAST! (~2-5 seconds per tile)")
        else:
            print("\n⏱️  CPU inference enabled")
            print("   Annotations will take ~30-60 seconds per tile")
            if hw_info.has_nvidia_gpu:
                print("   💡 Tip: Install CUDA PyTorch for 10-20x speedup!")
    else:
        print("\n⚠️  Installation encountered errors.")
        print("Please check the output above for details.")
        print("\nFor help, visit: https://github.com/yourusername/civic/issues")

    print("=" * 70)


def main():
    """Main installation script."""
    parser = argparse.ArgumentParser(
        description="Smart installation script for Civic - auto-detects GPU/CPU",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be installed without actually installing"
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU-only installation even if GPU is available"
    )
    parser.add_argument(
        "--force-gpu",
        action="store_true",
        help="Force GPU installation even if no GPU is detected"
    )
    parser.add_argument(
        "--skip-pytorch",
        action="store_true",
        help="Skip PyTorch installation (use if already installed correctly)"
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip model download"
    )
    parser.add_argument(
        "--skip-config",
        action="store_true",
        help="Skip config file generation"
    )

    args = parser.parse_args()

    # Detect hardware
    print("=" * 70)
    print("CIVIC SMART INSTALLER")
    print("Automatic GPU/CPU detection and dependency installation")
    print("=" * 70)

    print("\n🔍 Detecting hardware...")
    hw_info = detect_hardware()
    print(hw_info)

    # Determine force mode
    force_mode = None
    if args.force_cpu:
        force_mode = "cpu"
        print("\n⚠️  Forcing CPU-only installation")
    elif args.force_gpu:
        force_mode = "gpu"
        print("\n⚠️  Forcing GPU installation")

    if args.dry_run:
        print("\n" + "=" * 70)
        print("DRY RUN MODE - No changes will be made")
        print("=" * 70)

    # Installation steps
    success = True

    # Step 1: Install PyTorch
    if not args.skip_pytorch:
        if not install_pytorch(hw_info, force_mode, args.dry_run):
            success = False
    else:
        print("\n⏭️  Skipping PyTorch installation (--skip-pytorch)")

    # Step 2: Install Civic dependencies
    if success:
        if not install_civic_deps(args.dry_run):
            success = False

    # Step 3: Download models
    if success and not args.skip_models:
        if not download_models(hw_info, args.dry_run):
            print("\n⚠️  Model download failed, but you can download them later with:")
            print(f"    civic-download-models --sam-model {hw_info.get_sam_model_recommendation()}")
            # Don't fail the whole installation if model download fails
    elif args.skip_models:
        print("\n⏭️  Skipping model download (--skip-models)")

    # Step 4: Create test config
    if success and not args.skip_config:
        create_test_config(hw_info, args.dry_run)
    elif args.skip_config:
        print("\n⏭️  Skipping config generation (--skip-config)")

    # Print summary
    print_summary(hw_info, success)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
