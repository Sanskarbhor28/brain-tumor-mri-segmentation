import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        predictions = predictions.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        intersection = (predictions * targets).sum()

        dice = (
            (2.0 * intersection + self.smooth)
            /
            (
                predictions.sum()
                + targets.sum()
                + self.smooth
            )
        )

        return 1.0 - dice


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):

        # Binary Cross Entropy
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none"
        )

        # Convert logits to probabilities
        probabilities = torch.sigmoid(logits)

        # Probability of the correct class
        pt = torch.where(
            targets == 1,
            probabilities,
            1.0 - probabilities
        )

        # Alpha weighting
        alpha_t = torch.where(
            targets == 1,
            self.alpha,
            1.0 - self.alpha
        )

        # Focal Loss
        focal_loss = (
            alpha_t
            * (1.0 - pt) ** self.gamma
            * bce
        )

        return focal_loss.mean()


class FocalDiceLoss(nn.Module):
    def __init__(
        self,
        alpha=0.75,
        gamma=2.0,
        dice_weight=1.0,
        focal_weight=1.0
    ):
        super().__init__()

        self.focal = FocalLoss(
            alpha=alpha,
            gamma=gamma
        )

        self.dice = DiceLoss()

        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, logits, targets):

        # Focal Loss
        focal_loss = self.focal(
            logits,
            targets
        )

        # Convert logits to probabilities for Dice
        probabilities = torch.sigmoid(logits)

        # Dice Loss
        dice_loss = self.dice(
            probabilities,
            targets
        )

        # Combined loss
        total_loss = (
            self.focal_weight * focal_loss
            +
            self.dice_weight * dice_loss
        )

        return total_loss