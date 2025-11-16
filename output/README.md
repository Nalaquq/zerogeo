# Output Directory

This directory contains timestamped run directories for each annotation session.

## Structure

Each run directory follows this naming convention:
```
output/{project_name}_{YYYYMMDD_HHMMSS}/
```

For example:
```
output/quinhagak_20251113_143022/
├── tiles/              # Generated image tiles
├── raw_masks/          # Auto-generated masks from annotation
├── reviewed_masks/     # Manually reviewed and approved masks
├── logs/               # Processing logs
└── run_metadata.json   # Run information and parameters
```

## Benefits of This Structure

1. **Reproducibility**: Each run is self-contained with all outputs
2. **No Conflicts**: Timestamps prevent overwriting previous runs
3. **Easy Comparison**: Compare results from different runs/parameters
4. **Clean Organization**: All outputs for a run in one place

## Working with Run Directories

### Using the latest run
```bash
# The CLI scripts automatically use the latest run directory
python scripts/annotate_tiles.py --run-dir output/myproject_20251113_143022/
python scripts/launch_reviewer.py --run-dir output/myproject_20251113_143022/
```

### Finding specific runs
```bash
# List all runs for a project
ls -lt output/myproject_*

# Find the latest run
ls -t output/myproject_* | head -1
```

## Notes

- This directory is not tracked by git (contains generated data)
- Each run preserves all intermediate outputs for debugging
- Logs are automatically saved to `{run_dir}/logs/`
