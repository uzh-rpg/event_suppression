import torch

def epe_loss(predicted, gt):
    """
    Computes the End-Point Error (EPE) between predicted and ground truth optical flow.
    """
    if len(predicted.shape) == 4:
        norm = torch.norm(predicted - gt, dim=1).mean()  # Mean over all pixels
        return norm.mean()  # Mean over all batches
    else:
        return torch.norm(predicted - gt, dim=0).mean()

def npe_loss(predicted, gt, threshold=1):
    """
    Computes the percentage of pixels where the End-Point Error (EPE) is greater than `threshold`.
    - threshold = 1 for 1PE
    - threshold = 2 for 2PE
    - threshold = 3 for 3PE
    """
    epe = epe_loss(predicted, gt)  # Compute per-pixel EPE
    pixels_exceeding_threshold = (epe > threshold).float().mean()
    return pixels_exceeding_threshold *100 # Percentage of pixels exceeding threshold

def mean_angular_error(predicted, gt):
    """
    Computes the Angular Error (AE) between predicted and ground truth flow vectors.
    - Uses dot product to compute the angle difference.
    """
    assert predicted.shape == gt.shape, "Predicted and ground truth shapes must match"
    if len(predicted.shape) == 4:
        dim = 1
    else:
        dim = 0
    dot_product = torch.sum(predicted * gt, dim=dim)  # Compute dot product u1*u2 + v1*v2
    norm_pred = torch.norm(predicted, dim=dim)  # ||F_pred||
    norm_gt = torch.norm(gt, dim=dim)  # ||F_gt||

    # Clamp to prevent invalid values due to numerical precision errors
    cos_theta = torch.clamp(dot_product / (norm_pred * norm_gt + 1e-8), -1, 1)
    ae = torch.acos(cos_theta)
    return ae.mean()  # Mean over all pixels

def charbonnier_loss(predicted, gt, epsilon=1e-3):
    return torch.mean(torch.sqrt((predicted - gt) ** 2 + epsilon ** 2))