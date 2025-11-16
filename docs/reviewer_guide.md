## Annotation Reviewer UI Guide

Complete guide to using the manual annotation reviewer for quality control and mask editing.

## Overview

The Annotation Reviewer is a PyQt5-based GUI application for reviewing and editing zero-shot annotations. It provides:

- **Visual review** of tiles with mask overlays
- **Accept/Reject/Skip** actions for quality control
- **Manual editing** tools for fixing masks
- **Progress tracking** and session persistence
- **Keyboard shortcuts** for efficient workflow
- **Dark theme** for reduced eye strain

## Installation

The reviewer requires PyQt5:

```bash
pip install PyQt5>=5.15.0
```

This is included in `requirements.txt`.

## Quick Start

### Launch from Command Line

```bash
# Basic usage
python scripts/launch_reviewer.py data/tiles/

# With masks
python scripts/launch_reviewer.py data/tiles/ data/masks/

# Custom session directory
python scripts/launch_reviewer.py data/tiles/ data/masks/ --session-dir data/my_review/
```

### Launch from Python

```python
from river_segmentation.annotation import ReviewSession, launch_reviewer

# Create session
session = ReviewSession(
    session_dir="data/review_session",
    tiles_dir="data/tiles",
    masks_dir="data/masks",  # optional
    auto_accept_threshold=0.9
)

# Launch UI
launch_reviewer(session)
```

### Run Demo

Try the demo with sample data:

```bash
python examples/demo_reviewer.py
```

## UI Components

### Main Window Layout

```
┌─────────────────────────────────────────────────┐
│ Annotation Reviewer - tile_0_0                 │
├──────────────────────────────┬──────────────────┤
│                              │  Review Actions  │
│                              │  ✓ Accept (A)    │
│      Tile Viewer             │  ✗ Reject (R)    │
│   (with mask overlay)        │  ⏭ Skip (S)      │
│                              │                  │
│                              │  Editing Tools   │
│                              │  [Enable Edit]   │
│                              │  Draw | Erase    │
│                              │  Brush: ▬▬▬      │
│                              │  [Save Edits]    │
│  [Overlay: ▬▬▬▬] [Hide Mask] │                  │
│                              │  Notes           │
│  [Previous] Tile 1/66 [Next] │  [text box]      │
│                              │                  │
│                              │  Progress        │
│                              │  ████░░░ 60%     │
│                              │  Reviewed: 40/66 │
│                              │                  │
│                              │  [Export Results]│
└──────────────────────────────┴──────────────────┘
```

### 1. Tile Viewer (Left Panel)

**Purpose**: Display tile image with mask overlay

**Features**:
- Shows tile with colored mask overlay (green by default)
- Adjustable overlay transparency
- Toggle mask visibility
- Zoom/pan (via mouse editing mode)

**Controls**:
- **Overlay slider**: Adjust mask transparency (0-100%)
- **Hide/Show Mask button**: Toggle mask visibility
- **Navigation**: Previous/Next buttons and tile counter

### 2. Review Actions (Right Panel)

**Purpose**: Accept, reject, or skip tiles

**Actions**:
- **Accept (A)**: Mark tile as good - mask is correct
- **Reject (R)**: Mark tile as bad - mask is incorrect/unusable
- **Skip (S)**: Skip for now - review later

**Behavior**:
- Automatically moves to next tile after action
- Saves notes with each review
- Tracks timestamp of review

### 3. Editing Tools

**Purpose**: Manually edit masks

**Workflow**:
1. Click "Enable Editing"
2. Select Draw or Erase mode
3. Adjust brush size
4. Click and drag to edit mask
5. Click "Save Edits" to save changes

**Draw Mode**:
- Adds pixels to mask (value = 1)
- Use for expanding masks or adding missed areas
- Circular brush follows mouse

**Erase Mode**:
- Removes pixels from mask (value = 0)
- Use for removing false positives
- Same brush size as draw mode

**Brush Size**:
- Range: 1-50 pixels
- Adjust with slider
- Shows current size below slider

### 4. Notes

**Purpose**: Add comments about tiles

**Use cases**:
- Note why tile was rejected
- Mark tiles for special attention
- Document manual edits
- Track issues or patterns

**Tips**:
- Keep notes brief but descriptive
- Use consistent terminology
- Notes are saved with each review action

### 5. Progress Panel

**Purpose**: Track review progress

**Displays**:
- Progress bar (% completed)
- Total reviewed count
- Accepted count
- Rejected count
- Edited count

**Updates**: Real-time after each review action

## Keyboard Shortcuts

Keyboard shortcuts enable fast review workflow:

| Key | Action |
|-----|--------|
| `A` | Accept current tile |
| `R` | Reject current tile |
| `S` | Skip current tile |
| `E` | Toggle editing mode |
| `Left Arrow` | Previous tile |
| `Right Arrow` | Next tile |

**Pro tip**: Use keyboard-only workflow for maximum speed!

## Review Workflow

### Basic Review (No Editing)

1. **Load session** - Tiles and masks are loaded
2. **Review tile** - Look at tile with mask overlay
3. **Make decision**:
   - Press `A` if mask is good
   - Press `R` if mask is bad
   - Press `S` if unsure (come back later)
4. **Repeat** - Automatically moves to next tile
5. **Export results** when done

**Speed**: ~5-10 seconds per tile

### Review with Editing

1. **Load session**
2. **Review tile** - Identify issues with mask
3. **Enable editing** (press `E`)
4. **Fix mask**:
   - Select Draw to add areas
   - Select Erase to remove areas
   - Adjust brush size as needed
5. **Save edits** - Click "Save Edits" button
6. **Accept tile** (press `A`)
7. **Repeat**

**Speed**: ~30-60 seconds per edited tile

### Efficient Strategy

1. **First pass** - Quick review, accept good ones, reject obviously bad ones
2. **Second pass** - Review skipped tiles more carefully
3. **Edit pass** - Go back and edit tiles that need fixing

## Session Management

### Session Persistence

The review session is automatically saved:
- After each review action
- When editing masks
- On application exit

**Session directory contains**:
```
review_session/
├── session.json          # Session state and review data
└── reviewed_masks/       # Edited masks
    ├── tile_0_0_mask.tif
    ├── tile_0_1_mask.tif
    └── ...
```

### Resume Session

Sessions are automatically resumed:

```bash
# First run - creates new session
python scripts/launch_reviewer.py data/tiles/

# Later - resumes from where you left off
python scripts/launch_reviewer.py data/tiles/
```

The session remembers:
- Current tile position
- All review decisions
- Edited masks
- Notes

### Export Results

Click "Export Results" or use menu to save:

```json
{
  "session_dir": "data/review_session",
  "timestamp": "2025-01-12T10:30:00",
  "statistics": {
    "counts": {
      "total": 66,
      "accepted": 45,
      "rejected": 12,
      "pending": 9,
      "edited": 8
    },
    "progress_percent": 86.4
  },
  "reviews": [
    {
      "tile_id": "tile_0_0",
      "status": "accepted",
      "confidence": 0.95,
      "edited": false,
      "notes": "",
      "timestamp": "2025-01-12T10:25:00"
    },
    ...
  ]
}
```

## Advanced Features

### Auto-Accept Threshold

Set confidence threshold for auto-accepting masks:

```python
session = ReviewSession(
    session_dir="data/review",
    tiles_dir="data/tiles",
    masks_dir="data/masks",
    auto_accept_threshold=0.9  # Auto-accept if confidence >= 0.9
)
```

Tiles with confidence above threshold are marked as "accepted" but can still be reviewed/modified.

### Custom Mask Colors

Currently masks are green. To customize (future feature):

```python
# In config
review:
  confidence_colors:
    high: [0, 255, 0]      # green
    medium: [255, 255, 0]  # yellow
    low: [255, 0, 0]       # red
```

### Batch Operations

Review multiple tile sets:

```bash
# Review set 1
python scripts/launch_reviewer.py data/tiles_1/ data/masks_1/ --session-dir data/review_1/

# Review set 2
python scripts/launch_reviewer.py data/tiles_2/ data/masks_2/ --session-dir data/review_2/
```

## Tips & Best Practices

### Efficient Review

1. **Use keyboard shortcuts** - Much faster than clicking
2. **Adjust overlay** - Find transparency that works for your eyes
3. **Work in sessions** - Take breaks every 30-60 minutes
4. **Focus on quality** - Better to review carefully than quickly

### When to Edit vs Reject

**Edit if**:
- Mask is mostly correct but needs minor fixes
- Easy to add/remove small areas
- Will save time vs re-annotating

**Reject if**:
- Mask is completely wrong
- More than 50% needs editing
- Easier to re-annotate from scratch

### Quality Control

1. **Consistency** - Use same criteria for all tiles
2. **Documentation** - Add notes for rejected tiles
3. **Double-check edits** - Review after making changes
4. **Export regularly** - Save results every 30-60 minutes

### Performance

For large datasets (1000+ tiles):

1. **Review in batches** - 100-200 tiles per session
2. **Use SSD** - Faster loading of tiles/masks
3. **Close other apps** - More memory for image loading
4. **Reduce overlay quality** - If display is slow

## Troubleshooting

### UI doesn't launch

**Error**: `ModuleNotFoundError: No module named 'PyQt5'`

**Solution**:
```bash
pip install PyQt5>=5.15.0
```

### Tiles/masks not loading

**Check**:
- Files are `.tif` format
- Paths are correct
- Files are readable

**Debug**:
```python
from pathlib import Path
print(list(Path("data/tiles").glob("*.tif")))
```

### Edits not saving

**Check**:
- Editing mode is enabled
- Clicked "Save Edits" button
- Have write permissions in session directory

**Solution**: Check session log in `session.json`

### Slow performance

**Causes**:
- Large tile sizes (>2048x2048)
- Many bands in GeoTIFF
- Slow disk I/O

**Solutions**:
- Use smaller tiles (512x512 or 256x256)
- Reduce overlay transparency updates
- Use SSD for data storage

## Integration with Pipeline

### Full Workflow

1. **Tile** large image:
   ```bash
   python scripts/tile_image.py input.tif data/tiles/
   ```

2. **Annotate** with zero-shot (future):
   ```bash
   python scripts/annotate_tiles.py data/tiles/ data/masks/ --config config.yaml
   ```

3. **Review** annotations:
   ```bash
   python scripts/launch_reviewer.py data/tiles/ data/masks/
   ```

4. **Export** reviewed data:
   ```bash
   python scripts/export_training_data.py data/review_session/
   ```

5. **Train** U-Net:
   ```bash
   python scripts/train.py --data data/training/ --epochs 50
   ```

## API Reference

### ReviewSession

```python
class ReviewSession:
    """Manages annotation review session."""

    def __init__(
        self,
        session_dir: Path,
        tiles_dir: Path,
        masks_dir: Optional[Path] = None,
        auto_accept_threshold: float = 0.9
    ):
        """Initialize session."""

    def get_statistics(self) -> Dict:
        """Get review statistics."""

    def accept_current(self, notes: str = ""):
        """Accept current tile."""

    def reject_current(self, notes: str = ""):
        """Reject current tile."""

    def skip_current(self):
        """Skip current tile."""

    def save_edited_mask(self, tile_id: str, mask: np.ndarray):
        """Save edited mask."""

    def export_results(self, output_file: Path):
        """Export review results to JSON."""
```

### launch_reviewer

```python
def launch_reviewer(session: ReviewSession):
    """
    Launch reviewer UI.

    Args:
        session: ReviewSession to review
    """
```

## Future Enhancements

Planned features:
- Grid view (multiple tiles at once)
- Batch accept/reject
- Undo/redo for edits
- Custom color schemes
- Zoom/pan in viewer
- Side-by-side comparison
- Filter by status/confidence
- Statistics dashboard

## Contributing

To extend the reviewer:

1. **Add features** to `reviewer_ui.py`
2. **Update session** management in `review_session.py`
3. **Add tests** in `tests/test_reviewer.py`
4. **Update docs** in this file

See `src/river_segmentation/annotation/` for source code.

## References

- [PyQt5 Documentation](https://doc.qt.io/qt-5/)
- [Main README](../README.md)
- [Configuration Guide](configuration_guide.md)
- [Tiling Guide](../QUICKSTART_SINGLE_IMAGE.md)
