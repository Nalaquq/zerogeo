"""Visualization utilities."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Optional, Dict, List


def visualize_predictions(
    image: torch.Tensor,
    mask: torch.Tensor,
    prediction: torch.Tensor,
    save_path: Optional[str] = None,
    band_indices: Optional[List[int]] = None,
):
    """
    Visualize input image, ground truth mask, and prediction.

    Args:
        image: Input image tensor (C, H, W)
        mask: Ground truth mask (1, H, W)
        prediction: Model prediction (1, H, W)
        save_path: Optional path to save the figure
        band_indices: Indices of bands to use for RGB visualization (default: [0,1,2])
    """
    if band_indices is None:
        band_indices = [0, 1, 2]  # Use first 3 bands for RGB

    # Convert to numpy and move to CPU
    image = image.cpu().numpy()
    mask = mask.cpu().numpy().squeeze()
    prediction = prediction.cpu().numpy().squeeze()

    # Apply sigmoid if needed
    if prediction.max() > 1 or prediction.min() < 0:
        prediction = 1 / (1 + np.exp(-prediction))

    # Threshold prediction
    pred_binary = (prediction > 0.5).astype(np.float32)

    # Create RGB image for visualization
    if image.shape[0] >= 3:
        # Use specified bands for RGB
        rgb = image[band_indices, :, :]
        # Normalize to [0, 1] for display
        rgb = np.transpose(rgb, (1, 2, 0))
        rgb = np.clip(rgb, 0, 1)
    else:
        # Grayscale
        rgb = image[0, :, :]
        rgb = np.clip(rgb, 0, 1)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Input image (RGB or first 3 bands)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title('Input Image (RGB)')
    axes[0, 0].axis('off')

    # Ground truth mask
    axes[0, 1].imshow(mask, cmap='RdYlGn', vmin=0, vmax=1)
    axes[0, 1].set_title('Ground Truth (0=Water, 1=Land)')
    axes[0, 1].axis('off')

    # Prediction (continuous)
    axes[1, 0].imshow(prediction, cmap='RdYlGn', vmin=0, vmax=1)
    axes[1, 0].set_title('Prediction Probability')
    axes[1, 0].axis('off')

    # Prediction (binary)
    axes[1, 1].imshow(pred_binary, cmap='RdYlGn', vmin=0, vmax=1)
    axes[1, 1].set_title('Binary Prediction (Threshold=0.5)')
    axes[1, 1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.close()


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
):
    """
    Plot training and validation metrics over epochs.

    Args:
        history: Dictionary with keys like 'train_loss', 'val_loss', 'train_dice', etc.
        save_path: Optional path to save the figure
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Dice Coefficient
    axes[0, 1].plot(epochs, history['train_dice'], 'b-', label='Train')
    axes[0, 1].plot(epochs, history['val_dice'], 'r-', label='Validation')
    axes[0, 1].set_title('Dice Coefficient')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # IoU Score
    axes[1, 0].plot(epochs, history['train_iou'], 'b-', label='Train')
    axes[1, 0].plot(epochs, history['val_iou'], 'r-', label='Validation')
    axes[1, 0].set_title('IoU Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('IoU')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Accuracy
    axes[1, 1].plot(epochs, history['train_accuracy'], 'b-', label='Train')
    axes[1, 1].plot(epochs, history['val_accuracy'], 'r-', label='Validation')
    axes[1, 1].set_title('Pixel Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history to {save_path}")

    plt.close()
