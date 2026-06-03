import torch
import torch.nn as nn

# Binary Cross-Entropy Loss
bce_loss = nn.BCEWithLogitsLoss()  # Logits version is more stable for binary tasks

# Dice Loss
def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)  # Convert logits to probabilities
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    return 1 - dice  # We subtract from 1 because we minimize the loss

# Combined BCE + Dice Loss
class BCEDiceLoss(nn.Module):
    def __init__(self, **kwargs):
        super(BCEDiceLoss, self).__init__()
        self.class_weights = 1  # Default class weights
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        if not hasattr(self, "bce_weight") or not hasattr(self, "dice_weight"):
            raise ValueError("BCEDiceLoss must have 'bce_weight' and 'dice_weight' attributes.")
        
        class_weights = torch.tensor(self.class_weights) # > 1 increases the contribution of positive samples
        self.bce = nn.BCEWithLogitsLoss(pos_weight=class_weights)

    def forward(self, pred, target):
        bce = self.bce(pred, target)
        dice = dice_loss(pred, target)
        return self.bce_weight * bce + self.dice_weight * dice
    

if __name__ == "__main__":
    # Example usage

    # pred: output from the U-Net (logits, shape [B, 1, H, W])
    pred = torch.randn(4, 1, 128, 128)  # Example logits
    
    # target: ground truth mask (binary, shape [B, 1, H, W])
    target = torch.randint(0, 2, (4, 1, 128, 128)).float()  # Example binary mask

    loss_fn = BCEDiceLoss(bce_weight=0.7, dice_weight=0.3)
    loss = loss_fn(pred, target)

    print(f"Loss: {loss.item()}")