"""
Hardware detection utilities for automatic GPU/CPU configuration.

This module detects available hardware and determines the best
installation options for PyTorch and other dependencies.
"""

import subprocess
import sys
import platform
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class HardwareInfo:
    """Information about detected hardware."""

    has_nvidia_gpu: bool
    gpu_name: Optional[str]
    gpu_memory_mb: Optional[int]
    cuda_version: Optional[str]
    pytorch_installed: bool
    pytorch_version: Optional[str]
    pytorch_has_cuda: bool
    platform: str
    python_version: str

    def __str__(self) -> str:
        """Human-readable hardware summary."""
        lines = [
            "=" * 70,
            "HARDWARE DETECTION REPORT",
            "=" * 70,
            f"Platform: {self.platform}",
            f"Python: {self.python_version}",
            "",
            "GPU Detection:",
            f"  NVIDIA GPU Found: {'✓ YES' if self.has_nvidia_gpu else '✗ NO'}",
        ]

        if self.has_nvidia_gpu:
            lines.extend([
                f"  GPU Model: {self.gpu_name}",
                f"  GPU Memory: {self.gpu_memory_mb} MB ({self.gpu_memory_mb / 1024:.1f} GB)",
                f"  CUDA Version: {self.cuda_version or 'Not detected'}",
            ])

        lines.extend([
            "",
            "PyTorch Status:",
            f"  Installed: {'✓ YES' if self.pytorch_installed else '✗ NO'}",
        ])

        if self.pytorch_installed:
            lines.extend([
                f"  Version: {self.pytorch_version}",
                f"  CUDA Support: {'✓ YES' if self.pytorch_has_cuda else '✗ NO (CPU-only)'}",
            ])

        lines.append("=" * 70)
        return "\n".join(lines)

    def get_recommendation(self) -> str:
        """Get installation recommendation based on hardware."""
        if not self.has_nvidia_gpu:
            return "cpu"

        if self.gpu_memory_mb and self.gpu_memory_mb >= 8192:
            return "cuda-high"  # Can use vit_h model
        elif self.gpu_memory_mb and self.gpu_memory_mb >= 4096:
            return "cuda-medium"  # Use vit_l or vit_b model
        elif self.gpu_memory_mb and self.gpu_memory_mb >= 2048:
            return "cuda-low"  # Use vit_b model only
        else:
            return "cpu"  # GPU too small, use CPU

    def get_pytorch_install_command(self) -> str:
        """Get the appropriate PyTorch installation command."""
        if not self.has_nvidia_gpu:
            return "pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"

        # For CUDA GPUs, install CUDA 12.1 version (widely compatible)
        return "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"

    def get_sam_model_recommendation(self) -> str:
        """Recommend best SAM model for this hardware."""
        rec = self.get_recommendation()

        if rec == "cuda-high":
            return "vit_h"  # Best quality
        elif rec == "cuda-medium":
            return "vit_l"  # Good balance
        elif rec == "cuda-low":
            return "vit_b"  # Fastest
        else:
            return "vit_b"  # CPU-friendly

    def get_tile_size_recommendation(self) -> int:
        """Recommend tile size based on hardware."""
        rec = self.get_recommendation()

        if rec in ["cuda-high", "cuda-medium"]:
            return 512  # Standard size
        elif rec == "cuda-low":
            return 256  # Smaller for limited VRAM
        else:
            return 256  # Smaller for CPU


def check_nvidia_gpu() -> Tuple[bool, Optional[str], Optional[int], Optional[str]]:
    """
    Check for NVIDIA GPU using nvidia-smi.

    Returns:
        Tuple of (has_gpu, gpu_name, memory_mb, cuda_version)
    """
    try:
        # Try to run nvidia-smi
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0 and result.stdout.strip():
            # Parse output: "GPU Name, Memory"
            output = result.stdout.strip()
            parts = output.split(',')

            if len(parts) >= 2:
                gpu_name = parts[0].strip()
                memory_str = parts[1].strip().split()[0]  # Extract number before "MiB"
                memory_mb = int(memory_str)

                # Try to get CUDA version
                cuda_result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                cuda_version = cuda_result.stdout.strip() if cuda_result.returncode == 0 else None

                return True, gpu_name, memory_mb, cuda_version

        return False, None, None, None

    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        return False, None, None, None


def check_pytorch() -> Tuple[bool, Optional[str], bool]:
    """
    Check if PyTorch is installed and has CUDA support.

    Returns:
        Tuple of (installed, version, has_cuda)
    """
    try:
        import torch
        version = torch.__version__
        has_cuda = torch.cuda.is_available()
        return True, version, has_cuda
    except ImportError:
        return False, None, False


def detect_hardware() -> HardwareInfo:
    """
    Detect all hardware and return comprehensive information.

    Returns:
        HardwareInfo object with detected hardware details
    """
    # Detect GPU
    has_nvidia, gpu_name, gpu_memory, cuda_version = check_nvidia_gpu()

    # Check PyTorch
    pytorch_installed, pytorch_version, pytorch_cuda = check_pytorch()

    # System info
    system_platform = f"{platform.system()} {platform.release()}"
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    return HardwareInfo(
        has_nvidia_gpu=has_nvidia,
        gpu_name=gpu_name,
        gpu_memory_mb=gpu_memory,
        cuda_version=cuda_version,
        pytorch_installed=pytorch_installed,
        pytorch_version=pytorch_version,
        pytorch_has_cuda=pytorch_cuda,
        platform=system_platform,
        python_version=python_version
    )


def print_installation_guide(hw_info: HardwareInfo):
    """Print installation guide based on detected hardware."""
    print("\n" + "=" * 70)
    print("INSTALLATION GUIDE")
    print("=" * 70)

    recommendation = hw_info.get_recommendation()

    if not hw_info.pytorch_installed:
        print("\n⚠️  PyTorch is NOT installed.")
        print("\n📦 STEP 1: Install PyTorch")
        print("-" * 70)
        print(f"Run this command:\n")
        print(f"  {hw_info.get_pytorch_install_command()}")
        print()
    elif hw_info.pytorch_installed and not hw_info.pytorch_has_cuda and hw_info.has_nvidia_gpu:
        print("\n⚠️  PyTorch is installed but in CPU-only mode.")
        print("   You have an NVIDIA GPU but PyTorch can't use it!")
        print("\n📦 STEP 1: Reinstall PyTorch with CUDA support")
        print("-" * 70)
        print("Run these commands:\n")
        print("  pip uninstall torch torchvision -y")
        print(f"  {hw_info.get_pytorch_install_command()}")
        print()
    else:
        print("\n✓ PyTorch is properly configured!")
        print()

    print("📦 STEP 2: Install Civic dependencies")
    print("-" * 70)
    print("Run this command:\n")
    print("  pip install -r requirements.txt")
    print()

    print("📦 STEP 3: Download model weights")
    print("-" * 70)
    sam_model = hw_info.get_sam_model_recommendation()
    print(f"Run this command:\n")
    print(f"  civic-download-models --sam-model {sam_model}")
    print()

    print("=" * 70)
    print("RECOMMENDED CONFIGURATION")
    print("=" * 70)
    print(f"SAM Model: {sam_model}")
    print(f"Tile Size: {hw_info.get_tile_size_recommendation()}")
    print(f"Device: {'cuda' if hw_info.has_nvidia_gpu and hw_info.pytorch_has_cuda else 'cpu'}")

    if recommendation == "cuda-high":
        print("\n💪 Your GPU is excellent for this task!")
        print("   Expected speed: ~2-5 seconds per tile (very fast!)")
    elif recommendation in ["cuda-medium", "cuda-low"]:
        print("\n👍 Your GPU should work well for this task!")
        print("   Expected speed: ~5-15 seconds per tile")
    else:
        print("\n⏱️  CPU-only mode (slower but works)")
        print("   Expected speed: ~30-60 seconds per tile")

    print("=" * 70)


def main():
    """Command-line interface for hardware detection."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect hardware and provide installation guidance"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check hardware, don't print installation guide"
    )

    args = parser.parse_args()

    # Detect hardware
    hw_info = detect_hardware()

    if args.json:
        import json
        data = {
            "has_nvidia_gpu": hw_info.has_nvidia_gpu,
            "gpu_name": hw_info.gpu_name,
            "gpu_memory_mb": hw_info.gpu_memory_mb,
            "cuda_version": hw_info.cuda_version,
            "pytorch_installed": hw_info.pytorch_installed,
            "pytorch_version": hw_info.pytorch_version,
            "pytorch_has_cuda": hw_info.pytorch_has_cuda,
            "platform": hw_info.platform,
            "python_version": hw_info.python_version,
            "recommendation": hw_info.get_recommendation(),
            "sam_model": hw_info.get_sam_model_recommendation(),
            "tile_size": hw_info.get_tile_size_recommendation(),
        }
        print(json.dumps(data, indent=2))
    else:
        print(hw_info)

        if not args.check_only:
            print_installation_guide(hw_info)


if __name__ == "__main__":
    main()
