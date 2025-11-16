================================================================================
CIVIC - ZERO-SHOT TRAINING DATASET PIPELINE
Installation Summary
================================================================================

AUTOMATIC MODEL DOWNLOADS NOW AVAILABLE!

After installing Civic, you can now download all required models with one command:

    civic-download-models

This automatically downloads:
  ✅ Grounding DINO checkpoint (~670MB)
  ✅ SAM checkpoint (~375MB for vit_b, default)
  ✅ Grounding DINO config file

NO MANUAL DOWNLOADS NEEDED!

================================================================================
QUICK START
================================================================================

1. Install dependencies:
   pip install -e .
   pip install -r requirements-annotation.txt
   pip install groundingdino-py segment-anything

2. Download models:
   civic-download-models

3. Create config:
   cp config/river_annotation_example.yaml config/my_project.yaml

4. Run annotation:
   python scripts/annotate_tiles.py --config config/my_project.yaml

================================================================================
WHAT'S NEW
================================================================================

✨ AUTOMATIC MODEL DOWNLOADER
   - New command: civic-download-models
   - Progress bars for downloads
   - Automatic directory creation
   - Skip already downloaded files
   - Choose SAM model type (vit_b/vit_l/vit_h)
   - Custom directories supported

📦 NEW FILE: src/river_segmentation/utils/model_downloader.py
   - Handles automatic downloads
   - Can be imported: from river_segmentation.utils import download_all_models

🔧 UPDATED: pyproject.toml
   - Added console script entry point
   - Updated description to emphasize zero-shot pipeline

📖 UPDATED DOCUMENTATION:
   - README.md: Shows automatic download
   - SETUP.md: Automatic download as primary method
   - QUICKSTART_SINGLE_IMAGE.md: One-command setup
   - docs/model_setup.md: Automatic download with fallback
   - POST_INSTALL.md: New quick reference guide

================================================================================
COMMAND OPTIONS
================================================================================

Basic usage:
  civic-download-models

Choose SAM model:
  civic-download-models --sam-model vit_b  # Fast, CPU-friendly (default)
  civic-download-models --sam-model vit_l  # Balanced
  civic-download-models --sam-model vit_h  # Best quality

Advanced:
  civic-download-models --list                      # List models
  civic-download-models --weights-dir ./my_weights  # Custom directory
  civic-download-models --force                     # Re-download
  civic-download-models --help                      # Full help

================================================================================
BENEFITS
================================================================================

Before:
  ❌ Manual wget commands for each file
  ❌ Multiple steps to download models
  ❌ Easy to forget config file
  ❌ No verification of downloads

After:
  ✅ One command downloads everything
  ✅ Automatic directory creation
  ✅ Progress bars and status messages
  ✅ Skip already downloaded files
  ✅ Verify all files present

================================================================================
NEXT STEPS
================================================================================

See POST_INSTALL.md for complete post-installation guide.
See SETUP.md for detailed installation instructions.
See docs/annotation_guide.md for zero-shot annotation workflow.

Ready to generate training datasets! 🚀

================================================================================
