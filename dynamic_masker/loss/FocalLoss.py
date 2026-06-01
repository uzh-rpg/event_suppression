import torch
import torch.nn.functional as F
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, **kwargs):
        super(FocalLoss, self).__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)
        if not hasattr(self, "alpha") or not hasattr(self, "gamma"):
            raise ValueError("FocalLoss must have 'alpha' and 'gamma' attributes.")

    @staticmethod
    def focal_loss(pred, target, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Compute the focal loss for binary classification.

        Args:
            pred: Raw logits from the model (before sigmoid), shape (N, H, W)
            target: Binary ground truth (0 or 1), same shape as pred
            alpha: Balancing factor for class imbalance (0 < alpha < 1) higher alpha reduces FP, lower alpha reduces FN
            gamma: Focusing parameter (higher => more focus on hard examples)
            reduction: 'mean' (default), 'sum', or 'none'

        Returns:
            Focal loss value (scalar or tensor)
        """
        # Convert logits to probabilities
        pred_probs = torch.sigmoid(pred)

        # Compute standard binary cross-entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        # Compute focal weight
        p_t = pred_probs * target + (1 - pred_probs) * (1 - target)  # p_t = p if y=1, 1-p if y=0
        focal_weight = (1 - p_t) ** gamma

        # Apply alpha balancing
        alpha_t = alpha * target + (1 - alpha) * (1 - target)

        # Compute focal loss
        loss = alpha_t * focal_weight * bce_loss

        # Apply reduction
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss  # Return per-element loss

    def __call__(self, pred, target):
        return self.focal_loss(pred, target, self.alpha, self.gamma)

if __name__ == "__main__":
    pred = torch.randn(1, 256, 256, requires_grad=True)  # Raw logits (before sigmoid)
    target = torch.zeros_like(pred) # predict all zeros

    # Compute focal loss
    focal_loss = FocalLoss(alpha=0, gamma=2.0)
    loss = focal_loss(pred, target)
    print(f"Focal Loss: {loss.item()}")