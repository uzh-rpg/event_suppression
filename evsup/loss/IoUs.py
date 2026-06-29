import torch

class IoUs():
    def __init__(self):
        pass

    @staticmethod
    def iou(pred_mask, target_mask):
        intersection_1 = (pred_mask & target_mask).sum().float()
        union_1 = (pred_mask | target_mask).sum().float()

        iou_1 = intersection_1 / union_1 if union_1 > 0 else torch.tensor(float('nan'))
        return iou_1

    @staticmethod
    def piou(pred_mask, target_mask, event_mask):
        """computes the point wise IoU taking into account only the part
            of the mask which is related to an event
        """
        pred_mask = pred_mask & event_mask
        target_mask = target_mask & event_mask

        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()

        piou = intersection / union if union > 0 else torch.tensor(float('nan'))
        return piou

    def __call__(self, pred, target, event_voxel, apply_sigmoid=True):
        """
        Compute the Intersection over Union (IoU) for binary segmentation.
        Args:
            pred: torch.Tensor, shape: (N, H, W), predicted mask
            target: torch.Tensor, shape: (N, H, W), target mask
            event_mask: torch.Tensor, shape: (1, H, W), TRUE when there is an event, FALSE otherwise
        Returns:
            
        """
        # apply sigmoid to the prediction
        if apply_sigmoid:
            pred_probs = torch.sigmoid(pred)
        else:
            pred_probs = pred

        # IoU for positive class
        pred_mask_1 = (pred_probs > 0.5)
        target_mask_1 = (target > 0.5)
        iou_1 = self.iou(pred_mask_1, target_mask_1)

        # IoU for negative class
        pred_mask_0 = (pred_probs <= 0.5)
        target_mask_0 = (target <= 0.5)
        iou_0 = self.iou(pred_mask_0, target_mask_0)

        # Mean IoU
        miou = torch.nanmean(torch.tensor([iou_0, iou_1])).item()

        # event-wise IoU only for positive class
        event_mask = torch.any(event_voxel, dim=0)
        piou = self.piou(pred_mask_1, target_mask_1, event_mask).item()

        return {
            "IoU": [iou_1.item(), iou_0.item()],  # [dynamic object IoU, background IoU]
            "mIoU": miou,
            "pIoU": piou,
            "IoU%": [iou_1.item() * 100, iou_0.item() * 100], # [dynamic object IoU, background IoU]
            "mIoU%": miou * 100,
            "pIoU%": piou * 100
        }
    
if __name__ == "__main__":
    pred_mask = torch.tensor([[0, 1, 1], [0, 1, 1], [1, 1, 0]])  # Example prediction
    gt_mask = torch.tensor([[0, 1, 1], [1, 1, 1], [1, 0, 0]])    # Example ground truth
    event_mask = torch.tensor([[1, 1, 1], [1, 0, 0], [1, 1, 1]])  # Example event mask

    results = IoUs()(pred_mask, gt_mask, event_mask)
    print(results)