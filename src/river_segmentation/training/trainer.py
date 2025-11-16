"""Training and validation loops."""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict

from .metrics import dice_coefficient, iou_score, pixel_accuracy


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: The model to train
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number

    Returns:
        Dictionary with training metrics
    """
    model.train()

    total_loss = 0
    total_dice = 0
    total_iou = 0
    total_accuracy = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")

    for batch_idx, (images, masks) in enumerate(progress_bar):
        # Move to device
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Calculate metrics
        with torch.no_grad():
            dice = dice_coefficient(outputs, masks)
            iou = iou_score(outputs, masks)
            accuracy = pixel_accuracy(outputs, masks)

        # Update running totals
        total_loss += loss.item()
        total_dice += dice
        total_iou += iou
        total_accuracy += accuracy

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice:.4f}',
            'iou': f'{iou:.4f}',
        })

    # Calculate average metrics
    num_batches = len(dataloader)
    metrics = {
        'loss': total_loss / num_batches,
        'dice': total_dice / num_batches,
        'iou': total_iou / num_batches,
        'accuracy': total_accuracy / num_batches,
    }

    return metrics


def validate_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """
    Validate for one epoch.

    Args:
        model: The model to validate
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number

    Returns:
        Dictionary with validation metrics
    """
    model.eval()

    total_loss = 0
    total_dice = 0
    total_iou = 0
    total_accuracy = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")

    with torch.no_grad():
        for images, masks in progress_bar:
            # Move to device
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, masks)

            # Calculate metrics
            dice = dice_coefficient(outputs, masks)
            iou = iou_score(outputs, masks)
            accuracy = pixel_accuracy(outputs, masks)

            # Update running totals
            total_loss += loss.item()
            total_dice += dice
            total_iou += iou
            total_accuracy += accuracy

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{dice:.4f}',
                'iou': f'{iou:.4f}',
            })

    # Calculate average metrics
    num_batches = len(dataloader)
    metrics = {
        'loss': total_loss / num_batches,
        'dice': total_dice / num_batches,
        'iou': total_iou / num_batches,
        'accuracy': total_accuracy / num_batches,
    }

    return metrics
