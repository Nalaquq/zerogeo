# Quick Start Guide

## Installation

1. **Clone the repository** (if not already done):
```bash
cd civic
```

2. **Activate the virtual environment**:
```bash
source venv/bin/activate
```

3. **Install the package**:
```bash
pip install -e .
```

Or for development:
```bash
pip install -e ".[dev]"
```

## Data Preparation

### Using Google Earth Engine

1. Open the GEE script: `scripts/sentinel2_water_land_classification.js`
2. Copy and paste into the Google Earth Engine Code Editor
3. Draw your area of interest (polygon or rectangle)
4. Select date range
5. Adjust NIR threshold (default: 0.15)
6. Click "Run Analysis"
7. Export images from the Tasks tab

### Expected Data Structure

```
data/
├── images/          # Sentinel-2 multi-band GeoTIFFs
│   ├── image_001.tif
│   ├── image_002.tif
│   └── ...
└── masks/           # Binary water masks from GEE
    ├── image_001.tif  # 0 = water, 1 = land
    ├── image_002.tif
    └── ...
```

**Important**: Image and mask filenames must match!

## Training a Model

### Basic Training

```bash
python scripts/train.py \
  --image-dir data/images \
  --mask-dir data/masks \
  --output-dir outputs/experiment_1 \
  --n-channels 4 \
  --epochs 50 \
  --batch-size 8
```

### Training Parameters

- `--image-dir`: Directory with input satellite images
- `--mask-dir`: Directory with binary masks
- `--output-dir`: Where to save outputs (checkpoints, plots, etc.)
- `--n-channels`: Number of input bands (4 for RGB+NIR, 6 for RGB+NIR+SWIR)
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size (reduce if GPU memory issues)
- `--learning-rate`: Learning rate (default: 1e-4)
- `--val-split`: Validation split ratio (default: 0.2)
- `--resume`: Path to checkpoint to resume from

### Training Outputs

The training script creates:
- `checkpoints/best_model.pth` - Best model based on validation Dice score
- `checkpoints/checkpoint_epoch_*.pth` - Periodic checkpoints
- `training_history.json` - Metrics for each epoch
- `training_history.png` - Training curves plot
- `prediction_sample_*.png` - Sample predictions on validation set

## Making Predictions

### Single Image Prediction

```bash
python scripts/predict.py \
  --input data/images/new_image.tif \
  --output outputs/predictions/new_image_pred.tif \
  --checkpoint outputs/experiment_1/checkpoints/best_model.pth \
  --n-channels 4 \
  --binary
```

### Prediction Parameters

- `--input`: Input satellite image (GeoTIFF)
- `--output`: Output path for prediction
- `--checkpoint`: Trained model checkpoint
- `--n-channels`: Number of input channels (must match training)
- `--binary`: Save as binary (0/1) instead of probabilities
- `--device`: Device to use (cuda or cpu)

## Monitoring Training

The training script displays:
- Progress bars for each epoch
- Real-time metrics (loss, Dice, IoU)
- Learning rate updates
- Best model saves

Example output:
```
Epoch 25/50
[Train]: 100%|████████| Loss: 0.1234, Dice: 0.8567, IoU: 0.7890
[Val]:   100%|████████| Loss: 0.1456, Dice: 0.8234, IoU: 0.7456

Train - Loss: 0.1234, Dice: 0.8567, IoU: 0.7890
Val   - Loss: 0.1456, Dice: 0.8234, IoU: 0.7456
Best model saved! Dice: 0.8234
```

## Evaluating Results

### Metrics Explained

- **Dice Coefficient**: Measures overlap between prediction and ground truth (0-1, higher is better)
- **IoU (Intersection over Union)**: Similar to Dice, measures segmentation accuracy
- **Pixel Accuracy**: Percentage of correctly classified pixels
- **Loss**: Combined BCE + Dice loss (lower is better)

### Interpreting Predictions

In the output masks:
- **0 (black)** = Water / River
- **1 (white)** = Land

### Visualizations

The training script generates:
1. **Training curves**: Shows model learning over epochs
2. **Sample predictions**: Visual comparison of input, ground truth, and prediction

## Tips for Better Results

1. **More training data**: Aim for 100+ image-mask pairs
2. **Data diversity**: Include different seasons, river types, lighting conditions
3. **Proper validation split**: Keep 20% for validation
4. **Monitor overfitting**: Watch for diverging train/val metrics
5. **Adjust threshold**: If too much/little water detected, adjust GEE threshold
6. **Data augmentation**: Enabled by default (flips, rotations, brightness)
7. **Experiment with bands**: Try different band combinations (RGB+NIR, add SWIR, etc.)

## Troubleshooting

### GPU Out of Memory
- Reduce `--batch-size` (try 4, 2, or 1)
- Use smaller images or patch-based training

### Poor Performance
- Check data quality and alignment between images and masks
- Increase training epochs
- Collect more diverse training data
- Adjust NIR threshold in GEE script

### File Not Found Errors
- Ensure image and mask filenames match exactly
- Check file extensions (.tif vs .tiff)
- Verify paths are correct

## Next Steps

- Experiment with different model architectures
- Add temporal analysis for river change detection
- Implement post-processing to filter small water bodies
- Export to vector format (shapefile) for GIS integration
