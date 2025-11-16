#!/usr/bin/env python3
"""
Wrapper script for CIVIC CLI.

This script provides backward compatibility for running civic from scripts directory.
When CIVIC is installed as a package, use the 'civic' command instead.

Usage:
    python scripts/civic.py run config/my_project.yaml

Or after installation:
    civic run config/my_project.yaml
"""

import sys
from pathlib import Path

# Add src to path for development use
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Import and run the main CLI
from civic.cli import main

if __name__ == "__main__":
    sys.exit(main())
