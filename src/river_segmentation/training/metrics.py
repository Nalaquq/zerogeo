"""Evaluation metrics for segmentation."""

import torch


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    """
    Calculate Dice coefficient (F1 score) for binary segmentation.

    Args:
        pred: Predictions of shape (B, 1, H, W)
        target: Ground truth of shape (B, 1, H, W)
        smooth: Smoothing factor to avoid division by zero

    Returns:
        Dice coefficient in range [0, 1]
    """
    # Apply sigmoid if needed
    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)

    # Flatten
    pred = pred.view(-1)
    target = target.view(-1)

    # Calculate
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return dice.item()


def iou_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5,
              smooth: float = 1e-6) -> float:
    """
    Calculate Intersection over Union (IoU) / Jaccard Index.

    Args:
        pred: Predictions of shape (B, 1, H, W)
        target: Ground truth of shape (B, 1, H, W)
        threshold: Threshold to binarize predictions
        smooth: Smoothing factor

    Returns:
        IoU score in range [0, 1]
    """
    # Apply sigmoid and threshold
    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)

    pred = (pred > threshold).float()

    # Flatten
    pred = pred.view(-1)
    target = target.view(-1)

    # Calculate
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)

    return iou.item()


def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Calculate pixel-wise accuracy.

    Args:
        pred: Predictions of shape (B, 1, H, W)
        target: Ground truth of shape (B, 1, H, W)
        threshold: Threshold to binarize predictions

    Returns:
        Accuracy in range [0, 1]
    """
    # Apply sigmoid and threshold
    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)

    pred = (pred > threshold).float()

    # Calculate accuracy
    correct = (pred == target).sum()
    total = target.numel()

    accuracy = (correct / total).item()

    return accuracy


def precision_recall(pred: torch.Tensor, target: torch.Tensor,
                     threshold: float = 0.5, smooth: float = 1e-6):
    """
    Calculate precision and recall.

    Args:
        pred: Predictions of shape (B, 1, H, W)
        target: Ground truth of shape (B, 1, H, W)
        threshold: Threshold to binarize predictions
        smooth: Smoothing factor

    Returns:
        Tuple of (precision, recall)
    """
    # Apply sigmoid and threshold
    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)

    pred = (pred > threshold).float()

    # Flatten
    pred = pred.view(-1)
    target = target.view(-1)

    # True positives, false positives, false negatives
    tp = (pred * target).sum()
    fp = pred.sum() - tp
    fn = target.sum() - tp

    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)

    return precision.item(), recall.item()
