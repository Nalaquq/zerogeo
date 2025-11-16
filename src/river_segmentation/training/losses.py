"""Loss functions for river segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.

    Dice coefficient = 2 * |X ∩ Y| / (|X| + |Y|)
    Dice loss = 1 - Dice coefficient
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Dice loss.

        Args:
            pred: Predictions (logits or probabilities) of shape (B, 1, H, W)
            target: Ground truth of shape (B, 1, H, W)

        Returns:
            Dice loss
        """
        # Apply sigmoid if pred contains logits
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)

        # Flatten tensors
        pred = pred.view(-1)
        target = target.view(-1)

        # Calculate intersection and union
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

        return 1 - dice


class CombinedLoss(nn.Module):
    """
    Combination of Binary Cross Entropy and Dice Loss.

    This is often effective for segmentation tasks.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined loss.

        Args:
            pred: Predictions (logits) of shape (B, 1, H, W)
            target: Ground truth of shape (B, 1, H, W)

        Returns:
            Combined loss
        """
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Useful when water/river pixels are much less frequent than land pixels.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate focal loss.

        Args:
            pred: Predictions (logits) of shape (B, 1, H, W)
            target: Ground truth of shape (B, 1, H, W)

        Returns:
            Focal loss
        """
        # Binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        # Calculate pt
        pt = torch.exp(-bce_loss)

        # Apply focal term
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        return focal_loss.mean()
