# CIVIC Documentation

Complete documentation for the CIVIC zero-shot annotation pipeline.

## Getting Started

**New users start here:**

1. [../INSTALL.md](../INSTALL.md) - Installation guide
2. [../QUICKSTART.md](../QUICKSTART.md) - Your first annotation project
3. [streamlined_pipeline.md](streamlined_pipeline.md) - Pipeline overview

## Core Guides

**Essential documentation:**

- [configuration_guide.md](configuration_guide.md) - Configuration system
- [annotation_guide.md](annotation_guide.md) - Zero-shot annotation workflow
- [reviewer_guide.md](reviewer_guide.md) - Interactive review interface
- [model_setup.md](model_setup.md) - Model weights and setup
- [reassemble_guide.md](reassemble_guide.md) - Mask reassembly for GIS

## Installation & Setup

- [../INSTALL.md](../INSTALL.md) - Main installation guide
- [setup.md](setup.md) - Detailed setup (legacy)
- [dependencies.md](dependencies.md) - Dependency information
- [post_install.md](post_install.md) - Post-installation configuration

## Pipeline Documentation

- [pipeline_complete.md](pipeline_complete.md) - Complete pipeline overview
- [streamlined_pipeline.md](streamlined_pipeline.md) - Streamlined workflow
- [quickstart_single_image.md](quickstart_single_image.md) - Single image workflow

## Alternative Workflows

- [manual_annotation_guide.md](manual_annotation_guide.md) - Manual annotation

## Development

Developer documentation and summaries:

- [development/](development/) - Development summaries and changelogs
  - CLEANUP_SUMMARY.md
  - GROUNDING_DINO_FIX_SUMMARY.md
  - RESTRUCTURING_SUMMARY.md
  - UNIFIED_PIPELINE_SUMMARY.md
  - UNIT_TESTS_SUMMARY.md
  - END_TO_END_TEST_RESULTS.md

## Quick Reference

**Common commands:**
```bash
# Download model weights (first time setup)
civic-download-models

# Run the complete pipeline (RECOMMENDED)
python scripts/civic.py run config/my_project.yaml

# Or individual steps
python scripts/civic.py tile config/my_project.yaml
python scripts/civic.py annotate config/my_project.yaml
python scripts/civic.py review config/my_project.yaml
python scripts/civic.py reassemble config/my_project.yaml

# Utility commands
python scripts/check_hardware.py   # Check GPU/CPU capabilities
python scripts/civic.py --help     # Show all available commands
```

**Project structure:**
```
civic/
├── src/                 # Core library
│   ├── civic/          # CLI interface
│   └── river_segmentation/  # Core modules
├── scripts/            # Standalone scripts
├── config/             # Example configurations
├── docs/               # Documentation (you are here)
├── tests/              # Unit & integration tests
├── input/              # Input images
└── output/             # Generated outputs
```

## Need Help?

- **Overview**: [../README.md](../README.md)
- **Installation issues**: [../INSTALL.md](../INSTALL.md)
- **Workflow questions**: [annotation_guide.md](annotation_guide.md)
- **Configuration**: [configuration_guide.md](configuration_guide.md)
