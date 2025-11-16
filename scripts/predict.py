"""
Inference script for river segmentation.

This script performs inference on new Sentinel-2 imagery using a trained model.
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.transform import from_bounds

from river_segmentation.models import UNet
from river_segmentation.utils import load_checkpoint


def load_image(image_path: Path, n_channels: int = 4) -> tuple:
    """
    Load a GeoTIFF image for inference.

    Args:
        image_path: Path to GeoTIFF file
        n_channels: Number of channels to load

    Returns:
        Tuple of (image_array, profile) where image_array is (C, H, W)
    """
    with rasterio.open(image_path) as src:
        # Read all bands or specific number of channels
        if n_channels is not None:
            image = src.read(list(range(1, n_channels + 1)))
        else:
            image = src.read()

        profile = src.profile

        # Normalize
        image = image.astype(np.float32)
        if image.max() > 100:
            image = image / 10000.0
        image = np.clip(image, 0, 1)

    return image, profile


def save_prediction(
    prediction: np.ndarray,
    output_path: Path,
    profile: dict,
    binary: bool = True,
):
    """
    Save prediction as GeoTIFF.

    Args:
        prediction: Prediction array (H, W)
        output_path: Path to save output
        profile: Rasterio profile from input image
        binary: Whether to save as binary (0/1) or probability (0.0-1.0)
    """
    # Update profile for single-band output
    profile.update(
        count=1,
        dtype=rasterio.uint8 if binary else rasterio.float32
    )

    # Convert to binary if needed
    if binary:
        prediction = (prediction > 0.5).astype(np.uint8)
    else:
        prediction = prediction.astype(np.float32)

    # Save
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(prediction, 1)

    print(f"Prediction saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Perform inference with trained river segmentation model"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input satellite image (GeoTIFF)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save prediction (GeoTIFF)"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--n-channels",
        type=int,
        default=4,
        help="Number of input channels"
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Save as binary mask (0/1) instead of probabilities"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on"
    )
    parser.add_argument(
        "--batch-inference",
        action="store_true",
        help="Process image in patches (for large images)"
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=512,
        help="Patch size for batch inference"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("River Segmentation Inference")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Load model
    model = UNet(n_channels=args.n_channels, n_classes=1)
    load_checkpoint(str(args.checkpoint), model, device=args.device)
    model = model.to(args.device)
    model.eval()

    # Load image
    print("Loading image...")
    image, profile = load_image(args.input, n_channels=args.n_channels)
    print(f"Image shape: {image.shape}")

    # Perform inference
    print("Running inference...")
    with torch.no_grad():
        # Convert to tensor
        image_tensor = torch.from_numpy(image).float()
        image_tensor = image_tensor.unsqueeze(0).to(args.device)  # Add batch dimension

        # Predict
        output = model(image_tensor)
        prediction = torch.sigmoid(output).squeeze(0).squeeze(0)  # Remove batch and channel dims

        # Move to CPU and convert to numpy
        prediction = prediction.cpu().numpy()

    print(f"Prediction shape: {prediction.shape}")
    print(f"Prediction range: [{prediction.min():.4f}, {prediction.max():.4f}]")

    # Calculate water percentage
    water_pixels = (prediction > 0.5).sum()
    total_pixels = prediction.size
    water_percentage = (water_pixels / total_pixels) * 100
    print(f"Water coverage: {water_percentage:.2f}%")

    # Save prediction
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_prediction(prediction, args.output, profile, binary=args.binary)

    print("=" * 60)
    print("Inference completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
