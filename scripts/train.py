"""
Training script for river segmentation model.

This script trains a U-Net model on Sentinel-2 imagery with binary water masks
exported from Google Earth Engine.
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from river_segmentation.data import (
    RiverSegmentationDataset,
    get_train_transforms,
    get_val_transforms,
)
from river_segmentation.models import UNet
from river_segmentation.training import (
    CombinedLoss,
    train_epoch,
    validate_epoch,
)
from river_segmentation.utils import (
    save_checkpoint,
    plot_training_history,
    visualize_predictions,
)


def main():
    parser = argparse.ArgumentParser(
        description="Train a river segmentation model"
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Path to directory containing input satellite images (GeoTIFF)"
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        required=True,
        help="Path to directory containing binary water masks (GeoTIFF)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="./outputs",
        help="Path to save model outputs and checkpoints"
    )
    parser.add_argument(
        "--n-channels",
        type=int,
        default=4,
        help="Number of input channels (e.g., 4 for RGBN, 6 for RGBN+SWIR)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio"
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on (cuda or cpu)"
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("River Segmentation Training")
    print("=" * 60)
    print(f"Image directory: {args.image_dir}")
    print(f"Mask directory: {args.mask_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Input channels: {args.n_channels}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Validation split: {args.val_split}")
    print("=" * 60)

    # Create full dataset
    full_dataset = RiverSegmentationDataset(
        image_dir=str(args.image_dir),
        mask_dir=str(args.mask_dir),
    )

    # Split into train and validation
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Apply transforms
    train_dataset.dataset.transform = get_train_transforms()
    val_dataset.dataset.transform = get_val_transforms()

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if args.device == "cuda" else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if args.device == "cuda" else False
    )

    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    print("=" * 60)

    # Create model
    model = UNet(n_channels=args.n_channels, n_classes=1)
    model = model.to(args.device)

    # Loss function and optimizer
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, verbose=True
    )

    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        from river_segmentation.utils import load_checkpoint
        checkpoint = load_checkpoint(
            str(args.resume), model, optimizer, device=args.device
        )
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    # Training history
    history = defaultdict(list)
    best_val_dice = 0.0

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, args.device, epoch
        )

        # Validate
        val_metrics = validate_epoch(
            model, val_loader, criterion, args.device, epoch
        )

        # Update learning rate
        scheduler.step(val_metrics['loss'])

        # Log metrics
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
              f"Dice: {train_metrics['dice']:.4f}, "
              f"IoU: {train_metrics['iou']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Dice: {val_metrics['dice']:.4f}, "
              f"IoU: {val_metrics['iou']:.4f}")

        # Save history
        for key in train_metrics:
            history[f'train_{key}'].append(train_metrics[key])
        for key in val_metrics:
            history[f'val_{key}'].append(val_metrics[key])

        # Save checkpoint
        if val_metrics['dice'] > best_val_dice:
            best_val_dice = val_metrics['dice']
            checkpoint_path = checkpoint_dir / "best_model.pth"
            save_checkpoint(
                model, optimizer, epoch, val_metrics, str(checkpoint_path)
            )
            print(f"Best model saved! Dice: {best_val_dice:.4f}")

        # Save latest checkpoint
        if epoch % 5 == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            save_checkpoint(
                model, optimizer, epoch, val_metrics, str(checkpoint_path)
            )

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best validation Dice: {best_val_dice:.4f}")
    print("=" * 60)

    # Save training history
    history_path = args.output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(dict(history), f, indent=2)
    print(f"Training history saved to {history_path}")

    # Plot training history
    plot_path = args.output_dir / "training_history.png"
    plot_training_history(dict(history), save_path=str(plot_path))

    # Visualize some predictions
    print("\nGenerating sample predictions...")
    model.eval()
    with torch.no_grad():
        for i in range(min(3, len(val_dataset))):
            image, mask = val_dataset[i]
            image_input = image.unsqueeze(0).to(args.device)
            prediction = model(image_input).squeeze(0)

            viz_path = args.output_dir / f"prediction_sample_{i+1}.png"
            visualize_predictions(image, mask, prediction, save_path=str(viz_path))

    print(f"\nAll outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
