# Reassemble Masks Guide

After completing annotation and review, you'll want to view the complete results as a single raster in GIS software. The `reassemble_masks.py` script combines your tile masks back into a full-size GeoTIFF.

## Quick Start

```bash
# Basic usage - reassemble reviewed masks
python scripts/reassemble_masks.py --run-dir output/myproject_20251113_143022/

# Output: output/myproject_20251113_143022/final_mask_reviewed.tif
```

## How It Works

The reassemble script:

1. **Reads tile metadata** - Finds tile positions and overlap info
2. **Loads individual masks** - Reads each tile mask
3. **Weighted blending** - Seamlessly merges overlapping regions
4. **Preserves geospatial data** - Maintains CRS, transform, and all metadata
5. **Outputs GeoTIFF** - Creates a complete, GIS-ready raster

### Weighted Blending

Tiles overlap by default (typically 64 pixels). The reassemble script uses weighted blending in overlap regions to avoid visible seams:

- **Center of tiles**: Full weight (1.0)
- **Overlap edges**: Linear ramp from 0.0 to 1.0
- **Result**: Seamless transitions, no visible tile boundaries

## Usage Examples

### 1. Basic Usage (Reviewed Masks)

```bash
# Reassemble reviewed masks from run directory
python scripts/reassemble_masks.py --run-dir output/rivers_20251113_143022/

# Creates: output/rivers_20251113_143022/final_mask_reviewed.tif
```

### 2. Use Raw Masks Instead

```bash
# Reassemble raw (auto-generated) masks without review
python scripts/reassemble_masks.py \
    --run-dir output/rivers_20251113_143022/ \
    --use-raw

# Creates: output/rivers_20251113_143022/final_mask_raw.tif
```

### 3. Custom Output Location

```bash
# Specify custom output path
python scripts/reassemble_masks.py \
    --run-dir output/rivers_20251113_143022/ \
    --output final_results/river_segmentation_complete.tif
```

### 4. Adjust Overlap Blending

```bash
# If you used custom overlap during tiling
python scripts/reassemble_masks.py \
    --run-dir output/rivers_20251113_143022/ \
    --overlap 128

# Disable blending (simple averaging)
python scripts/reassemble_masks.py \
    --run-dir output/rivers_20251113_143022/ \
    --no-blend
```

### 5. Keep Float Values

```bash
# Keep confidence scores instead of binary output
python scripts/reassemble_masks.py \
    --run-dir output/rivers_20251113_143022/ \
    --binary-threshold -1

# This creates a float32 raster with values 0.0-1.0
```

### 6. Custom Binary Threshold

```bash
# Use stricter threshold for binary output
python scripts/reassemble_masks.py \
    --run-dir output/rivers_20251113_143022/ \
    --binary-threshold 0.7

# Only pixels with confidence >= 0.7 will be marked as positive
```

### 7. Legacy Workflow (Manual Paths)

```bash
# If you're not using run directories
python scripts/reassemble_masks.py \
    --masks-dir path/to/masks/ \
    --tiles-dir path/to/tiles/ \
    --original-image input/original.tif \
    --output reassembled.tif
```

## Command-Line Options

### Required (Choose One)

- `--run-dir PATH` - Run directory containing tiles/ and masks/ (recommended)
- `--masks-dir PATH` + `--original-image PATH` - Manual paths (legacy)

### Optional

- `--use-raw` - Use raw_masks/ instead of reviewed_masks/ (only with --run-dir)
- `--output PATH` - Custom output path (default: {run_dir}/final_mask_{type}.tif)
- `--overlap N` - Overlap used during tiling in pixels (default: 64)
- `--no-blend` - Disable weighted blending in overlap regions
- `--binary-threshold T` - Threshold for binary output, 0-1 (default: 0.5, use -1 for float)
- `--tiles-dir PATH` - Directory with tile metadata (legacy, auto-detected from run-dir)

## Output Files

### Binary Output (Default)

```
final_mask_reviewed.tif
- Data type: uint8
- Values: 0 (background), 1 (detected class)
- Compression: LZW
- Size: ~1-10% of original image (depends on coverage)
```

### Float Output (--binary-threshold -1)

```
final_mask_reviewed.tif
- Data type: float32
- Values: 0.0-1.0 (confidence scores)
- Compression: LZW
- Size: ~4x larger than binary
```

## Viewing in GIS Software

### QGIS

1. Open QGIS
2. **Layer → Add Layer → Add Raster Layer**
3. Select `final_mask_reviewed.tif`
4. The raster will load with correct georeferencing

**Styling:**
- **Single band grayscale** for binary masks
- **Singleband pseudocolor** for confidence visualization
- Adjust transparency to overlay on original imagery

### ArcGIS

1. Open ArcGIS Pro
2. **Map → Add Data → Data**
3. Browse to `final_mask_reviewed.tif`
4. The raster loads with CRS preserved

**Symbology:**
- Use **Unique Values** for binary (0, 1)
- Use **Classified** or **Stretched** for confidence scores

### Google Earth Engine

Export to asset or use locally:

```python
import ee
import rasterio

# Read with rasterio, convert to ee.Image
# ... (see GEE documentation)
```

## Workflow Integration

### Complete Pipeline

```bash
# 1. Tile image
python scripts/tile_image.py input/image.tif --project rivers

# 2. Annotate tiles
python scripts/annotate_tiles.py \
    --run-dir output/rivers_20251113_143022/ \
    --prompts "river" "stream"

# 3. Review annotations
python scripts/launch_reviewer.py --run-dir output/rivers_20251113_143022/

# 4. Reassemble for viewing
python scripts/reassemble_masks.py --run-dir output/rivers_20251113_143022/

# 5. Open in QGIS
qgis output/rivers_20251113_143022/final_mask_reviewed.tif
```

### Comparing Raw vs Reviewed

```bash
# Generate both for comparison
python scripts/reassemble_masks.py \
    --run-dir output/rivers_20251113_143022/ \
    --use-raw \
    --output output/rivers_20251113_143022/raw_complete.tif

python scripts/reassemble_masks.py \
    --run-dir output/rivers_20251113_143022/ \
    --output output/rivers_20251113_143022/reviewed_complete.tif

# Load both in QGIS and compare side-by-side
```

## Technical Details

### Coordinate System Preservation

The script preserves all geospatial metadata:

- **CRS**: Copied from original image
- **Transform**: Reconstructed from tile positions
- **Bounds**: Calculated from original dimensions
- **NoData value**: Set to 0

### Memory Efficiency

For large images:

- **Tiles processed sequentially** (not all in memory)
- **Accumulation arrays**: Only full image size needed
- **Example**: 10,000 x 10,000 image ≈ 200 MB RAM (float32)

### Overlap Handling

```
Tile 1:  [####======]
Tile 2:       [======####]
           ^overlap^

Weights in overlap:
Tile 1:  [1.0 → 0.0]
Tile 2:  [0.0 → 1.0]
Result:  [smooth blend]
```

## Troubleshooting

### "Tile metadata not found"

**Problem**: Can't find tile metadata JSON.

**Solution**:
```bash
# Ensure you're using the correct run directory
ls output/myproject_TIMESTAMP/tiles/*_metadata.json

# Or specify tiles directory manually
python scripts/reassemble_masks.py \
    --masks-dir output/.../reviewed_masks/ \
    --tiles-dir output/.../tiles/ \
    --original-image input/image.tif
```

### "Original image required"

**Problem**: Script can't find original image path.

**Solution**:
```bash
# Provide original image explicitly
python scripts/reassemble_masks.py \
    --run-dir output/myproject_TIMESTAMP/ \
    --original-image input/original_image.tif
```

### "No mask files found"

**Problem**: Mask directory is empty or wrong path.

**Solution**:
```bash
# Check if masks exist
ls output/myproject_TIMESTAMP/reviewed_masks/*_mask.*

# If empty, try raw masks
python scripts/reassemble_masks.py --run-dir ... --use-raw

# Or check if review was completed
python scripts/launch_reviewer.py --run-dir ...
```

### Visible Tile Boundaries

**Problem**: Can see lines between tiles in output.

**Solution**:
```bash
# Ensure correct overlap value
python scripts/reassemble_masks.py \
    --run-dir output/myproject_TIMESTAMP/ \
    --overlap 64  # Use same value as during tiling

# Try enabling blending if disabled
python scripts/reassemble_masks.py \
    --run-dir output/myproject_TIMESTAMP/
    # (blending enabled by default)
```

### File Size Too Large

**Problem**: Output file is very large.

**Solution**:
```bash
# Use binary output instead of float
python scripts/reassemble_masks.py \
    --run-dir output/myproject_TIMESTAMP/ \
    --binary-threshold 0.5  # Default, much smaller than float

# Binary is typically 1/4 the size of float32
```

## Performance

### Processing Time

- **Small images** (5,000 x 5,000): ~5-10 seconds
- **Medium images** (20,000 x 20,000): ~30-60 seconds
- **Large images** (86,000 x 86,000): ~5-10 minutes

Time scales with number of tiles and image size.

### Optimization Tips

1. **Use binary output** - Much faster and smaller than float
2. **Disable blending** - Faster processing if seams aren't an issue
3. **Filter tiles first** - Only process accepted tiles if needed

## Advanced Usage

### Scripting Integration

```python
from pathlib import Path
from scripts.reassemble_masks import reassemble_masks

# Call directly from Python
reassemble_masks(
    masks_dir=Path("output/project/reviewed_masks"),
    tiles_dir=Path("output/project/tiles"),
    original_image=Path("input/image.tif"),
    output_path=Path("output/final.tif"),
    overlap=64,
    blend_overlap=True,
    binary_threshold=0.5
)
```

### Batch Processing

```bash
# Reassemble all projects
for project in output/*/; do
    echo "Processing $project"
    python scripts/reassemble_masks.py --run-dir "$project"
done
```

## Best Practices

1. **Always use reviewed masks** - Better quality than raw
2. **Keep original overlap value** - Ensures proper blending
3. **Enable weighted blending** - Prevents visible seams
4. **Use binary output** - Smaller files, GIS-compatible
5. **Verify in QGIS** - Check quality before using

## See Also

- [Annotation Guide](annotation_guide.md) - Complete annotation workflow
- [Reviewer Guide](reviewer_guide.md) - Review process details
- [QUICKSTART.md](../QUICKSTART.md) - Quick start guide
- [README.md](../README.md) - Main documentation
