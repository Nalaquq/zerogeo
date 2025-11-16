# Input Directory

Place your source GeoTIFF/TIFF images here for annotation.

## Usage

```bash
# Copy your images to this directory
cp /path/to/your/image.tif input/

# Then reference them in your config files
```

## Example Files

- `Quinhagak-Orthomosaic.tiff` - Large orthomosaic image
- `Sentinel2_1_bands.tif` - Sentinel-2 satellite imagery
- `quinhagak_subset_4096x4096.tif` - Test subset

## Notes

- This directory contains large binary files that are not tracked by git
- Keep source images here for organization
- Images are read-only; processed outputs go to `output/`
