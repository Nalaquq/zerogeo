# Streamlined Pipeline Update

## What Changed

The CIVIC annotation workflow has been completely streamlined to match the patterns from your other repos (blender-synth, ocr_nunalleq). The old multi-script workflow with manual path management has been replaced with a **single unified pipeline** that handles everything automatically.

### Before (Old Way) ❌
```bash
# Multiple commands, manual path management, easy to mess up!
python scripts/tile_image.py data/input.tif data/tiles/ --tile-size 512 --overlap 64
# Wait... should tiles go in data/tiles/ or data/annotations/project/tiles/?

python scripts/annotate_tiles.py --config config/my_project.yaml
# Error: Tile metadata not found! 😫

python scripts/launch_reviewer.py data/tiles/ data/masks/
# Which directories again? 🤔
```

### After (New Way) ✅
```bash
# One command does everything!
python scripts/civic.py run config/my_project.yaml

# The pipeline automatically:
# ✓ Creates correct directory structure
# ✓ Runs tiling with config settings
# ✓ Runs annotation
# ✓ Launches reviewer
# ✓ Manages all paths correctly
```

## New Features

### 1. Automatic Directory Management
No more path confusion! The pipeline creates this structure automatically:
```
data/annotations/your_project/
├── tiles/              # Tiled images
│   └── tile_metadata.json
├── raw_masks/          # Auto-generated annotations
│   └── annotation_metadata.json
├── reviewed_masks/     # Quality-controlled masks
├── training_data/      # Exported training data
└── logs/              # Pipeline logs
```

### 2. Unified Entry Point
One script (`scripts/civic.py`) handles everything:
- **Run full pipeline**: `python scripts/civic.py run config.yaml`
- **Run individual steps**: `tile`, `annotate`, or `review`
- **Preview mode**: `--dry-run` flag
- **Resume capability**: `--skip-existing` flag

### 3. Safe Preview (Dry Run)
See what will happen before executing:
```bash
python scripts/civic.py run config/my_project.yaml --dry-run
```

Shows:
- What directories will be created
- What files will be processed
- What commands will run
- **No changes made to disk**

### 4. Smart Resume
Stop and resume anytime:
```bash
# First run - completes tiling, starts annotation
python scripts/civic.py run config.yaml

# Interrupt with Ctrl+C

# Resume - skips completed tiling, continues from annotation
python scripts/civic.py run config.yaml --skip-existing
```

### 5. Flexible Execution
Run just what you need:
```bash
# Just create tiles
python scripts/civic.py tile config.yaml

# Just run annotation (requires tiles)
python scripts/civic.py annotate config.yaml

# Just launch reviewer (requires annotations)
python scripts/civic.py review config.yaml

# Force re-run a step
python scripts/civic.py annotate config.yaml --force
```

### 6. Optional System Command
Install as a global command:
```bash
sudo ln -s $(pwd)/civic-annotate /usr/local/bin/civic-annotate

# Use anywhere
cd ~/myproject/
civic-annotate run config/rivers.yaml
```

## How to Use

### Quick Start
```bash
# 1. Create config
cp config/river_annotation_example.yaml config/my_project.yaml

# 2. Edit config (set input_image, prompts, etc.)
vim config/my_project.yaml

# 3. Run!
python scripts/civic.py run config/my_project.yaml
```

### Advanced Options
```bash
# Preview without executing
python scripts/civic.py run config.yaml --dry-run

# Skip completed steps
python scripts/civic.py run config.yaml --skip-existing

# Run individual steps
python scripts/civic.py tile config.yaml
python scripts/civic.py annotate config.yaml
python scripts/civic.py review config.yaml

# Force re-run even if complete
python scripts/civic.py annotate config.yaml --force
```

## Configuration

Your existing config files work without changes! The pipeline reads the config and:
- Uses `output_dir` as the base directory
- Creates `{output_dir}/tiles/`, `{output_dir}/raw_masks/`, etc.
- Automatically manages all paths

Example config:
```yaml
project:
  name: "my_project"
  input_image: "data/satellite_image.tif"
  output_dir: "data/annotations/my_project"  # Everything goes here

tiling:
  tile_size: 512
  overlap: 64

prompts:
  - "river"

# ... rest of config ...
```

## Migration Guide

If you have existing work-in-progress projects:

### Option 1: Move Files (Recommended)
Move your tiles to the expected location:
```bash
# If tiles are in data/tiles/
mv data/tiles/ data/annotations/my_project/

# Now run the pipeline
python scripts/civic.py run config/my_project.yaml --skip-existing
```

### Option 2: Keep Using Old Scripts
The old individual scripts still work:
```bash
python scripts/tile_image.py ...
python scripts/annotate_tiles.py ...
python scripts/launch_reviewer.py ...
```

Just make sure paths match your config's `{output_dir}/tiles/` structure.

## Benefits

### 1. No More Path Confusion
The pipeline manages all paths automatically. No more:
- "Tile metadata not found" errors
- Wondering which directory to use
- Manual directory creation

### 2. Reproducible Workflows
One config + one command = complete pipeline:
```bash
python scripts/civic.py run config/project_v1.yaml
python scripts/civic.py run config/project_v2.yaml
python scripts/civic.py run config/experiment3.yaml
```

Each project is self-contained and reproducible.

### 3. Safe Experimentation
Dry-run mode lets you preview before executing:
```bash
# Try different configs safely
python scripts/civic.py run config/aggressive_threshold.yaml --dry-run
python scripts/civic.py run config/conservative_threshold.yaml --dry-run

# Run the one you like
python scripts/civic.py run config/aggressive_threshold.yaml
```

### 4. Efficient Iterations
Skip completed steps when iterating:
```bash
# First attempt
python scripts/civic.py run config.yaml

# Adjust annotation settings in config

# Re-run just annotation and review
python scripts/civic.py annotate config.yaml --force
python scripts/civic.py review config.yaml
```

### 5. Batch Processing
Run multiple projects in sequence:
```bash
# Process multiple regions
for region in north south east west; do
    python scripts/civic.py run config/${region}_rivers.yaml
done
```

## Architecture

The new pipeline follows patterns from your other repos:

### Like blender-synth:
- Single entry point (`blender-synth generate` → `civic.py run`)
- Configuration-driven
- Automatic output management
- Modular components

### Like ocr_nunalleq:
- Safe-first (dry-run mode like `preview`)
- Flexible execution (CLI + programmatic)
- Batch processing support
- Explicit output directories

### Plus CIVIC-specific:
- Multi-step pipeline orchestration
- Step completion tracking
- Resume capability
- Geospatial path handling

## Files Created

New files:
- `scripts/civic.py` - Main pipeline orchestrator
- `civic-annotate` - CLI entry point wrapper
- `QUICKSTART.md` - Beginner-friendly guide
- `STREAMLINED_PIPELINE.md` - This document

Modified files:
- `README.md` - Added streamlined workflow section

Unchanged:
- All existing scripts still work
- All configs compatible
- No breaking changes to core functionality

## Future Enhancements

Potential additions:
- Web UI (like ocr_nunalleq)
- Progress bars for long operations
- Email notifications on completion
- Multi-config batch processing
- Checkpoint resumption mid-step
- Pipeline timing/profiling

## Support

- **Quick start**: See [QUICKSTART.md](QUICKSTART.md)
- **Full docs**: See [README.md](README.md)
- **Config guide**: See [docs/configuration_guide.md](docs/configuration_guide.md)

## Summary

**Old way**: Multiple scripts, manual paths, confusion
**New way**: One command, automatic everything, clarity

```bash
# That's it!
python scripts/civic.py run config/my_project.yaml
```

Welcome to the streamlined CIVIC pipeline! 🎉
