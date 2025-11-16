"""
Path management utilities for CIVIC annotation system.

Handles the creation and organization of input/output directories,
with timestamped run directories for each annotation session.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


class RunDirectoryManager:
    """Manages output directory structure for annotation runs."""

    def __init__(self, output_base: str = "output"):
        """
        Initialize the run directory manager.

        Args:
            output_base: Base output directory (default: "output")
        """
        self.output_base = Path(output_base)
        self.output_base.mkdir(parents=True, exist_ok=True)

    def create_run_directory(
        self,
        project_name: str,
        timestamp: Optional[str] = None,
        input_image: Optional[str] = None,
        metadata_extra: Optional[Dict] = None
    ) -> Path:
        """
        Create a new run directory with project name and timestamp.

        Args:
            project_name: Name of the project/annotation session
            timestamp: Optional custom timestamp (default: current time in YYYYMMDD_HHMMSS)
            input_image: Optional path to input image (for reassembly)
            metadata_extra: Optional extra metadata to include

        Returns:
            Path to the created run directory
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        run_name = f"{project_name}_{timestamp}"
        run_dir = self.output_base / run_name

        # Create subdirectories
        (run_dir / "tiles").mkdir(parents=True, exist_ok=True)
        (run_dir / "raw_masks").mkdir(parents=True, exist_ok=True)
        (run_dir / "reviewed_masks").mkdir(parents=True, exist_ok=True)
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)

        # Create run metadata file
        metadata = {
            "project_name": project_name,
            "timestamp": timestamp,
            "created_at": datetime.now().isoformat(),
            "run_directory": str(run_dir)
        }

        if input_image:
            metadata["input_image"] = str(input_image)

        if metadata_extra:
            metadata.update(metadata_extra)

        with open(run_dir / "run_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return run_dir

    def get_latest_run(self, project_name: Optional[str] = None) -> Optional[Path]:
        """
        Get the most recent run directory, optionally filtered by project name.

        Args:
            project_name: Optional project name filter

        Returns:
            Path to latest run directory, or None if no runs exist
        """
        if not self.output_base.exists():
            return None

        # Get all run directories
        run_dirs = [d for d in self.output_base.iterdir() if d.is_dir()]

        # Filter by project name if specified
        if project_name:
            run_dirs = [d for d in run_dirs if d.name.startswith(f"{project_name}_")]

        if not run_dirs:
            return None

        # Sort by timestamp (embedded in directory name)
        run_dirs.sort(reverse=True)
        return run_dirs[0]

    def list_runs(self, project_name: Optional[str] = None) -> list[Path]:
        """
        List all run directories, optionally filtered by project name.

        Args:
            project_name: Optional project name filter

        Returns:
            List of run directory paths, sorted by timestamp (newest first)
        """
        if not self.output_base.exists():
            return []

        run_dirs = [d for d in self.output_base.iterdir() if d.is_dir()]

        if project_name:
            run_dirs = [d for d in run_dirs if d.name.startswith(f"{project_name}_")]

        run_dirs.sort(reverse=True)
        return run_dirs


def get_run_paths(run_dir: Path) -> Dict[str, Path]:
    """
    Get all standard paths within a run directory.

    Args:
        run_dir: The run directory

    Returns:
        Dictionary mapping path keys to Path objects
    """
    return {
        "run_dir": run_dir,
        "tiles": run_dir / "tiles",
        "raw_masks": run_dir / "raw_masks",
        "reviewed_masks": run_dir / "reviewed_masks",
        "logs": run_dir / "logs",
        "tile_metadata": run_dir / "tiles" / "tile_metadata.json",
        "annotation_metadata": run_dir / "raw_masks" / "annotation_metadata.json",
        "session": run_dir / "reviewed_masks" / "session.json",
        "run_metadata": run_dir / "run_metadata.json"
    }


def ensure_input_directory(input_base: str = "input") -> Path:
    """
    Ensure the input directory exists.

    Args:
        input_base: Base input directory (default: "input")

    Returns:
        Path to the input directory
    """
    input_path = Path(input_base)
    input_path.mkdir(parents=True, exist_ok=True)
    return input_path
