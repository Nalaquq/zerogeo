"""Automatic model downloader for Grounding DINO and SAM models."""

import os
import urllib.request
from pathlib import Path
from typing import Optional
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for file downloads."""

    def update_to(self, b=1, bsize=1, tsize=None):
        """Update progress bar.

        Args:
            b: Number of blocks transferred
            bsize: Size of each block (bytes)
            tsize: Total size (bytes)
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path, description: str = "Downloading"):
    """Download a file with progress bar.

    Args:
        url: URL to download from
        output_path: Path to save file
        description: Description for progress bar
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=description) as t:
        urllib.request.urlretrieve(url, output_path, reporthook=t.update_to)


# Model URLs
GROUNDING_DINO_CHECKPOINT_URL = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
GROUNDING_DINO_CONFIG_URL = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"

SAM_CHECKPOINT_URLS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}

SAM_CHECKPOINT_SIZES = {
    "vit_h": "~2.4GB (best quality)",
    "vit_l": "~1.2GB (good quality)",
    "vit_b": "~375MB (fastest, good for CPU)",
}


def get_default_model_dir() -> Path:
    """Get default directory for model weights.

    Returns:
        Path to models directory (civic/weights)
    """
    # Get the project root (assuming this file is in src/river_segmentation/utils/)
    package_dir = Path(__file__).parent.parent.parent.parent
    return package_dir / "weights"


def get_default_config_dir() -> Path:
    """Get default directory for model configs.

    Returns:
        Path to model_configs directory (civic/model_configs)
    """
    package_dir = Path(__file__).parent.parent.parent.parent
    return package_dir / "model_configs"


def download_grounding_dino(
    checkpoint_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
    force: bool = False
) -> tuple[Path, Path]:
    """Download Grounding DINO model and config.

    Args:
        checkpoint_path: Path to save checkpoint (default: weights/groundingdino_swint_ogc.pth)
        config_path: Path to save config (default: model_configs/GroundingDINO_SwinT_OGC.py)
        force: Force re-download even if files exist

    Returns:
        Tuple of (checkpoint_path, config_path)
    """
    if checkpoint_path is None:
        checkpoint_path = get_default_model_dir() / "groundingdino_swint_ogc.pth"
    if config_path is None:
        config_path = get_default_config_dir() / "GroundingDINO_SwinT_OGC.py"

    # Download checkpoint
    if not checkpoint_path.exists() or force:
        print(f"\nDownloading Grounding DINO checkpoint (~670MB)...")
        print(f"URL: {GROUNDING_DINO_CHECKPOINT_URL}")
        print(f"Saving to: {checkpoint_path}")
        download_file(GROUNDING_DINO_CHECKPOINT_URL, checkpoint_path, "Grounding DINO")
        print(f"✓ Downloaded: {checkpoint_path}")
    else:
        print(f"✓ Grounding DINO checkpoint already exists: {checkpoint_path}")

    # Download config
    if not config_path.exists() or force:
        print(f"\nDownloading Grounding DINO config...")
        print(f"URL: {GROUNDING_DINO_CONFIG_URL}")
        print(f"Saving to: {config_path}")
        download_file(GROUNDING_DINO_CONFIG_URL, config_path, "DINO Config")
        print(f"✓ Downloaded: {config_path}")
    else:
        print(f"✓ Grounding DINO config already exists: {config_path}")

    return checkpoint_path, config_path


def download_sam(
    model_type: str = "vit_b",
    checkpoint_path: Optional[Path] = None,
    force: bool = False
) -> Path:
    """Download SAM model checkpoint.

    Args:
        model_type: SAM model type ('vit_h', 'vit_l', or 'vit_b')
        checkpoint_path: Path to save checkpoint (default: weights/sam_{model_type}_*.pth)
        force: Force re-download even if file exists

    Returns:
        Path to downloaded checkpoint
    """
    if model_type not in SAM_CHECKPOINT_URLS:
        raise ValueError(f"Invalid model_type: {model_type}. Choose from: {list(SAM_CHECKPOINT_URLS.keys())}")

    url = SAM_CHECKPOINT_URLS[model_type]
    filename = url.split("/")[-1]

    if checkpoint_path is None:
        checkpoint_path = get_default_model_dir() / filename

    if not checkpoint_path.exists() or force:
        print(f"\nDownloading SAM {model_type} checkpoint ({SAM_CHECKPOINT_SIZES[model_type]})...")
        print(f"URL: {url}")
        print(f"Saving to: {checkpoint_path}")
        download_file(url, checkpoint_path, f"SAM {model_type}")
        print(f"✓ Downloaded: {checkpoint_path}")
    else:
        print(f"✓ SAM {model_type} checkpoint already exists: {checkpoint_path}")

    return checkpoint_path


def download_all_models(
    sam_model_type: str = "vit_b",
    weights_dir: Optional[Path] = None,
    configs_dir: Optional[Path] = None,
    force: bool = False
) -> dict:
    """Download all required models for zero-shot annotation.

    Args:
        sam_model_type: SAM model type to download ('vit_h', 'vit_l', or 'vit_b')
        weights_dir: Directory for model weights (default: civic/weights)
        configs_dir: Directory for configs (default: civic/configs)
        force: Force re-download even if files exist

    Returns:
        Dictionary with paths to downloaded files
    """
    print("=" * 70)
    print("CIVIC MODEL DOWNLOADER")
    print("Zero-Shot Annotation Models for Binary Raster Segmentation")
    print("=" * 70)

    # Set default directories
    if weights_dir is None:
        weights_dir = get_default_model_dir()
    if configs_dir is None:
        configs_dir = get_default_config_dir()

    # Create directories
    weights_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nModel directories:")
    print(f"  Weights: {weights_dir}")
    print(f"  Configs: {configs_dir}")

    # Download Grounding DINO
    print("\n" + "-" * 70)
    print("1/2: Grounding DINO (Text-based Object Detection)")
    print("-" * 70)
    dino_checkpoint = weights_dir / "groundingdino_swint_ogc.pth"
    dino_config = configs_dir / "GroundingDINO_SwinT_OGC.py"
    dino_checkpoint, dino_config = download_grounding_dino(
        checkpoint_path=dino_checkpoint,
        config_path=dino_config,
        force=force
    )

    # Download SAM
    print("\n" + "-" * 70)
    print(f"2/2: SAM (Segment Anything Model) - {sam_model_type}")
    print("-" * 70)
    print(f"\nAvailable SAM models:")
    for model, size in SAM_CHECKPOINT_SIZES.items():
        marker = "→" if model == sam_model_type else " "
        print(f"  {marker} {model}: {size}")

    sam_checkpoint = download_sam(
        model_type=sam_model_type,
        checkpoint_path=None,
        force=force
    )

    # Summary
    print("\n" + "=" * 70)
    print("✓ MODEL DOWNLOAD COMPLETE")
    print("=" * 70)
    print("\nDownloaded files:")
    print(f"  1. Grounding DINO checkpoint: {dino_checkpoint}")
    print(f"  2. Grounding DINO config:     {dino_config}")
    print(f"  3. SAM {sam_model_type} checkpoint:      {sam_checkpoint}")

    total_size = sum(f.stat().st_size for f in [dino_checkpoint, dino_config, sam_checkpoint] if f.exists())
    print(f"\nTotal size: {total_size / (1024**3):.2f} GB")

    print("\n" + "=" * 70)
    print("READY FOR ZERO-SHOT ANNOTATION!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Create a config file: cp config/river_annotation_example.yaml config/my_project.yaml")
    print("  2. Edit your config with prompts and paths")
    print("  3. Run annotation: python scripts/annotate_tiles.py --config config/my_project.yaml")
    print("\nSee docs/annotation_guide.md for complete workflow.")

    return {
        "grounding_dino_checkpoint": str(dino_checkpoint),
        "grounding_dino_config": str(dino_config),
        "sam_checkpoint": str(sam_checkpoint),
        "sam_model_type": sam_model_type,
    }


def main():
    """CLI entry point for model downloader."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download Grounding DINO and SAM models for zero-shot annotation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download default models (vit_b - fastest, CPU-friendly)
  civic-download-models

  # Download best quality SAM model (vit_h - requires GPU)
  civic-download-models --sam-model vit_h

  # Download to custom directories
  civic-download-models --weights-dir ./my_models --configs-dir ./my_configs

  # Force re-download existing models
  civic-download-models --force

SAM Model Options:
  vit_h: ~2.4GB - Best quality, requires 8GB+ VRAM (GPU recommended)
  vit_l: ~1.2GB - Good quality, requires 4GB+ VRAM
  vit_b: ~375MB - Fast, works on CPU, recommended for testing
        """
    )

    parser.add_argument(
        "--sam-model",
        type=str,
        default="vit_b",
        choices=["vit_h", "vit_l", "vit_b"],
        help="SAM model type to download (default: vit_b)",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=None,
        help="Directory to save model weights (default: civic/weights)",
    )
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=None,
        help="Directory to save config files (default: civic/configs)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit",
    )

    args = parser.parse_args()

    if args.list:
        print("\nAvailable SAM models:")
        for model, size in SAM_CHECKPOINT_SIZES.items():
            print(f"  {model}: {size}")
        print("\nGrounding DINO: ~670MB")
        return

    try:
        download_all_models(
            sam_model_type=args.sam_model,
            weights_dir=args.weights_dir,
            configs_dir=args.configs_dir,
            force=args.force,
        )
    except Exception as e:
        print(f"\n❌ Error downloading models: {e}")
        raise


if __name__ == "__main__":
    main()
