#!/usr/bin/env python3
"""
Check if all required dependencies are installed.
Run this script to verify your installation.
"""

import sys
from importlib import import_module

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def check_import(module_name, package_name=None, optional=False):
    """Check if a module can be imported.

    Args:
        module_name: Name of module to import
        package_name: Display name (if different from module_name)
        optional: Whether this is an optional dependency

    Returns:
        True if import successful, False otherwise
    """
    if package_name is None:
        package_name = module_name

    try:
        mod = import_module(module_name)
        version = getattr(mod, '__version__', 'unknown')
        status = f"{GREEN}✓{RESET}"
        label = "OPTIONAL" if optional else "REQUIRED"
        print(f"  {status} {package_name:30s} {version:15s} [{label}]")
        return True
    except ImportError as e:
        status = f"{YELLOW}⚠{RESET}" if optional else f"{RED}✗{RESET}"
        label = "OPTIONAL" if optional else "REQUIRED"
        print(f"  {status} {package_name:30s} {'NOT INSTALLED':15s} [{label}]")
        if not optional:
            print(f"       {RED}Error: {e}{RESET}")
        return False

def main():
    print("=" * 80)
    print(f"{BLUE}CIVIC DEPENDENCY CHECK{RESET}")
    print("Zero-Shot Training Dataset Pipeline for Binary Raster Segmentation")
    print("=" * 80)

    all_ok = True

    # Core dependencies
    print(f"\n{BLUE}Core Dependencies:{RESET}")
    all_ok &= check_import("numpy")
    all_ok &= check_import("torch")
    all_ok &= check_import("torchvision")
    all_ok &= check_import("PIL", "pillow")
    all_ok &= check_import("tqdm")
    all_ok &= check_import("yaml", "pyyaml")

    # Geospatial dependencies
    print(f"\n{BLUE}Geospatial Dependencies:{RESET}")
    all_ok &= check_import("rasterio")
    all_ok &= check_import("geopandas")

    # ML utilities
    print(f"\n{BLUE}Machine Learning Utilities:{RESET}")
    all_ok &= check_import("sklearn", "scikit-learn")
    all_ok &= check_import("matplotlib")

    # GUI dependencies
    print(f"\n{BLUE}GUI Dependencies:{RESET}")
    all_ok &= check_import("PyQt5", optional=False)

    # Annotation dependencies (optional but needed for zero-shot)
    print(f"\n{BLUE}Zero-Shot Annotation Dependencies:{RESET}")
    has_cv2 = check_import("cv2", "opencv-python", optional=True)
    has_transformers = check_import("transformers", optional=True)
    has_supervision = check_import("supervision", optional=True)
    has_scipy = check_import("scipy", optional=True)
    has_hf_hub = check_import("huggingface_hub", "huggingface-hub", optional=True)

    # Model-specific dependencies (now part of annotation group)
    print(f"\n{BLUE}Model Dependencies (Included with annotation):{RESET}")
    has_dino = check_import("groundingdino", "groundingdino-py", optional=True)
    has_sam = check_import("segment_anything", "segment-anything", optional=True)

    # Summary
    print("\n" + "=" * 80)
    if all_ok:
        print(f"{GREEN}✓ All required dependencies are installed!{RESET}")
    else:
        print(f"{RED}✗ Some required dependencies are missing.{RESET}")
        print(f"\n{YELLOW}Install missing dependencies:{RESET}")
        print(f"  pip install -e .")

    # Zero-shot annotation check
    annotation_deps = [has_cv2, has_transformers, has_supervision, has_scipy, has_hf_hub, has_dino, has_sam]
    if not all(annotation_deps):
        print(f"\n{YELLOW}⚠ Zero-shot annotation dependencies incomplete.{RESET}")
        print(f"{YELLOW}  Install all dependencies (including Grounding DINO & SAM):{RESET}")
        print(f'  pip install -e ".[annotation]"')
        print(f"  {YELLOW}or{RESET}")
        print(f"  pip install -r requirements-annotation.txt")
    elif has_dino and has_sam:
        print(f"\n{GREEN}✓ All annotation dependencies installed!{RESET}")
        print(f"\n{BLUE}Download model weights (if not already done):{RESET}")
        print(f"  civic-download-models")

    print("=" * 80)

    # Final recommendations
    if all_ok and all(annotation_deps) and has_dino and has_sam:
        print(f"\n{GREEN}🚀 You're all set! Ready for zero-shot annotation.{RESET}\n")
        print(f"{BLUE}Next steps:{RESET}")
        print(f"  1. Download models: civic-download-models")
        print(f"  2. Create config: cp config/river_annotation_example.yaml config/my_project.yaml")
        print(f"  3. Run annotation: python scripts/annotate_tiles.py --config config/my_project.yaml")
    else:
        print(f"\n{YELLOW}Complete installation to use zero-shot annotation.{RESET}\n")

    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
