#!/usr/bin/env python3
"""
Quick hardware check script for Civic.

Run this to verify your installation and get recommendations.

Usage:
    python scripts/check_hardware.py
    python scripts/check_hardware.py --json  # Output as JSON
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from river_segmentation.utils.hardware_detect import main

if __name__ == "__main__":
    main()
