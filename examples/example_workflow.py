"""
Example workflow for river segmentation.

This script demonstrates the complete workflow:
1. Load data
2. Create data loaders
3. Initialize model
4. Train model
5. Make predictions
"""

import torch
from torch.utils.data import DataLoader, random_split

from river_segmentation.data import (
    RiverSegmentationDataset,
    get_train_transforms,
    get_val_transforms,
)
from river_segmentation.models import UNet
from river_segmentation.training import CombinedLoss, train_epoch, validate_epoch
from river_segmentation.utils import save_checkpoint, visualize_predictions


def example_workflow():
    """Example training workflow."""

    # Configuration
    IMAGE_DIR = "data/images"
    MASK_DIR = "data/masks"
    N_CHANNELS = 4  # RGB + NIR
    BATCH_SIZE = 8
    EPOCHS = 10
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {DEVICE}")

    # 1. Create dataset
    print("\n1. Loading dataset...")
    dataset = RiverSegmentationDataset(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
    )

    # Split into train/val
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Apply transforms
    train_dataset.dataset.transform = get_train_transforms()
    val_dataset.dataset.transform = get_val_transforms()

    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")

    # 2. Create data loaders
    print("\n2. Creating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

    # 3. Initialize model
    print("\n3. Initializing U-Net model...")
    model = UNet(n_channels=N_CHANNELS, n_classes=1)
    model = model.to(DEVICE)

    # 4. Setup training
    print("\n4. Setting up training...")
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 5. Training loop
    print("\n5. Starting training...")
    best_dice = 0.0

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE, epoch
        )

        # Validate
        val_metrics = validate_epoch(
            model, val_loader, criterion, DEVICE, epoch
        )

        print(f"Train - Dice: {train_metrics['dice']:.4f}, IoU: {train_metrics['iou']:.4f}")
        print(f"Val   - Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}")

        # Save best model
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                "outputs/best_model.pth"
            )
            print(f"Best model saved! Dice: {best_dice:.4f}")

    # 6. Make predictions
    print("\n6. Making predictions on validation set...")
    model.eval()
    with torch.no_grad():
        for i in range(min(3, len(val_dataset))):
            image, mask = val_dataset[i]
            image_input = image.unsqueeze(0).to(DEVICE)
            prediction = model(image_input).squeeze(0)

            visualize_predictions(
                image, mask, prediction,
                save_path=f"outputs/prediction_{i+1}.png"
            )

    print("\n✓ Workflow completed!")
    print(f"Best validation Dice: {best_dice:.4f}")
    print("Check outputs/ directory for results")


if __name__ == "__main__":
    example_workflow()
