import torch
import wandb
import numpy as np
import torch.nn.functional as F

from dynamic_masker.loss.BCEDiceLoss import BCEDiceLoss
from dynamic_masker.loss.ClassificationLoss import ClassificationLoss
from dynamic_masker.loss.FocalLoss import FocalLoss
from dynamic_masker.loss.IoUs import IoUs

from dynamic_masker.loss.flow_classic_loss import epe_loss, npe_loss, mean_angular_error, charbonnier_loss


def binary_segmentation_losses(pred, target, event_voxel, config):
    classification_losses = ClassificationLoss()
    classification_losses(pred, target)
    fp = classification_losses.fp
    fn = classification_losses.fn
    tp = classification_losses.tp
    tn = classification_losses.tn
    accuracy = classification_losses.accuracy
    precision = classification_losses.precision
    recall = classification_losses.recall
    F1_score = classification_losses.F1_score
    focal = FocalLoss(**config["loss"])(pred, target)
    bce_loss = BCEDiceLoss(bce_weight=1, dice_weight=0)(pred, target)
    dice = BCEDiceLoss(bce_weight=0, dice_weight=1)(pred, target)
    ious = IoUs()(pred, target, event_voxel)
    return {
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "tn": tn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "F1_score": F1_score,
        "focal": focal,
        "bce_loss": bce_loss,
        "dice": dice,
        "ious": ious
    }

def plot_segmentation_masks(pred, smooth_target, event_voxel):
    # Debugging images
    if event_voxel.shape[0] > 1:
        event_voxel = event_voxel[0].unsqueeze(0)
        smooth_target = smooth_target[0].unsqueeze(0)
        pred = pred[0].unsqueeze(0)

    # TODO debug this ugly padding
    event_rgb_image = F.pad(event_voxel.squeeze(), (0, 0, 0, 0, 0, 1), value=0)
    event_rgb_image = event_rgb_image.permute(1, 2, 0)
    ground_truth = smooth_target.squeeze().detach().cpu() > 0.5
    predictions = torch.sigmoid(pred.squeeze()) > 0.5

    events_rgb = wandb.Image(event_rgb_image.numpy(), masks={
            "predictions": {"mask_data": predictions.numpy()},
            "ground_truth": {"mask_data": ground_truth.numpy()}
            },
        )
    return events_rgb

def train_log(run, pred, target, event_voxel, smooth_target, loss, config, total_seen_samples, pos_pixels, neg_pixels):
    """
    Function to log on wandb the training process of the model
    Args:
        pred: Predictions of the model
        target: Ground truth of the model
        loss: Loss of the model
    Returns:
        None
"""
    # Debugging losses
    losses = binary_segmentation_losses(pred, target, event_voxel, config)

    events_rgb = plot_segmentation_masks(pred, smooth_target, event_voxel)

    run.log(
            {   
                "train/sample_loss": loss.item(),
                "train/dice_loss": losses["dice"],
                "train/focal_loss": losses["focal"],
                "train/true_positives": losses["tp"],
                "train/true_negatives": losses["tn"],
                "train/false_positives": losses["fp"],
                "train/false_negatives": losses["fn"],
                "train/IoU_positive_class": losses["ious"]["IoU"][0],
                "train/IoU_negative_class": losses["ious"]["IoU"][1],
                "train/mIoU": losses["ious"]["mIoU"],
                "train/pIoU": losses["ious"]["pIoU"],
                "train/accuracy": losses["accuracy"],
                "train/precision": losses["precision"],
                "train/recall": losses["recall"],
                "train/F1_score": losses["F1_score"],
                "train/bce_loss": losses["bce_loss"],
                "images/event_for_loss_compute": events_rgb,
                "dataset_info/positive_pixels_percentage": pos_pixels,
                "dataset_info/negative_pixels_percentage": neg_pixels
            },
            step=total_seen_samples
        )
    
def log_hydra_masks(mask_pred_t0, mask_target_t0, mask_pred_t1, mask_target_t1, event_voxel, config):
    # Debugging losses
    mask_pred_t0 = mask_pred_t0.detach()
    mask_target_t0 = mask_target_t0.detach()
    mask_pred_t1 = mask_pred_t1.detach()
    mask_target_t1 = mask_target_t1.detach()
    event_voxel = event_voxel.detach()

    losses_t0 = binary_segmentation_losses(mask_pred_t0, mask_target_t0, event_voxel, config)
    losses_t1 = binary_segmentation_losses(mask_pred_t1, mask_target_t1, event_voxel, config)
    
    logging_dict = {
        "train/mask/t0/true_positives": losses_t0["tp"],
        "train/mask/t0/true_negatives": losses_t0["tn"],
        "train/mask/t0/false_positives": losses_t0["fp"],
        "train/mask/t0/false_negatives": losses_t0["fn"],
        "train/mask/t0/IoU_positive_class": losses_t0["ious"]["IoU"][0],
        "train/mask/t0/IoU_negative_class": losses_t0["ious"]["IoU"][1],
        "train/mask/t0/mIoU": losses_t0["ious"]["mIoU"],
        "train/mask/t0/pIoU": losses_t0["ious"]["pIoU"],

        "train/mask/t1/true_positives": losses_t1["tp"],
        "train/mask/t1/true_negatives": losses_t1["tn"],
        "train/mask/t1/false_positives": losses_t1["fp"],
        "train/mask/t1/false_negatives": losses_t1["fn"],
        "train/mask/t1/IoU_positive_class": losses_t1["ious"]["IoU"][0],
        "train/mask/t1/IoU_negative_class": losses_t1["ious"]["IoU"][1],
        "train/mask/t1/mIoU": losses_t1["ious"]["mIoU"],
        "train/mask/t1/pIoU": losses_t1["ious"]["pIoU"],
    }
    return logging_dict

def flow_losses(flow_pred_t1, flow_target_t1):
    if flow_target_t1 is None:
        return {
            "l1": np.nan,
            "epe": np.nan,
            "charbonnier": np.nan,
            "one_PE": np.nan,
            "two_PE": np.nan,
            "three_PE": np.nan,
            "AE": np.nan
        }
    flow_pred_t1 = flow_pred_t1.squeeze(0).detach()
    flow_target_t1 = flow_target_t1.squeeze(0).detach()
    
    l1_loss = torch.nn.L1Loss()(flow_pred_t1, flow_target_t1).cpu()
    epe = epe_loss(flow_pred_t1, flow_target_t1).cpu()
    charb = charbonnier_loss(flow_pred_t1, flow_target_t1).cpu()
    one_PE = npe_loss(flow_pred_t1, flow_target_t1, threshold=1).cpu()
    two_PE = npe_loss(flow_pred_t1, flow_target_t1, threshold=2).cpu()
    three_PE = npe_loss(flow_pred_t1, flow_target_t1, threshold=3).cpu()
    AE = mean_angular_error(flow_pred_t1, flow_target_t1).cpu()
    return {
        "l1": l1_loss.numpy(),
        "epe": epe.numpy(),
        "charbonnier": charb.numpy(),
        "one_PE": one_PE.numpy(),
        "two_PE": two_PE.numpy(),
        "three_PE": three_PE.numpy(),
        "AE": AE.numpy()
    }

def log_hydra_flow(flow_pred_t1, flow_target_t1):
    losses = flow_losses(flow_pred_t1, flow_target_t1)
    logging_dict = {
        "train/flow/t1/l1_loss": losses["l1"],
        "train/flow/t1/epe": losses["epe"],
        "train/flow/t1/charbonnier": losses["charbonnier"],
        "train/flow/t1/one_PE": losses["one_PE"],
        "train/flow/t1/two_PE": losses["two_PE"],
        "train/flow/t1/three_PE": losses["three_PE"],
        "train/flow/t1/AE": losses["AE"]
    }
    return logging_dict

def train_log_hydra(mask_pred_t0, mask_target_t0, mask_pred_t1, mask_target_t1, event_voxel, config, flow_pred_t1, flow_target_t1):
    logging_dict = log_hydra_masks(mask_pred_t0, mask_target_t0, mask_pred_t1, mask_target_t1, event_voxel, config)
    if flow_target_t1 is not None:
        logging_dict.update(log_hydra_flow(flow_pred_t1, flow_target_t1))
    return logging_dict

def train_log_herm(mask_pred_t0, mask_target_t0, mask_pred_t1, mask_target_t1, event_voxel, config):
    logging_dict = log_hydra_masks(mask_pred_t0, mask_target_t0, mask_pred_t1, mask_target_t1, event_voxel, config)
    return logging_dict


def log_future_hydra_masks(mask_pred_t1, mask_target_t1, event_voxel, config):
    # Debugging losses
    mask_pred_t1 = mask_pred_t1.detach()
    mask_target_t1 = mask_target_t1.detach()
    event_voxel = event_voxel.detach()

    losses_t1 = binary_segmentation_losses(mask_pred_t1, mask_target_t1, event_voxel, config)
    
    logging_dict = {
        "train/mask/t1/true_positives": losses_t1["tp"],
        "train/mask/t1/true_negatives": losses_t1["tn"],
        "train/mask/t1/false_positives": losses_t1["fp"],
        "train/mask/t1/false_negatives": losses_t1["fn"],
        "train/mask/t1/IoU_positive_class": losses_t1["ious"]["IoU"][0],
        "train/mask/t1/IoU_negative_class": losses_t1["ious"]["IoU"][1],
        "train/mask/t1/mIoU": losses_t1["ious"]["mIoU"],
        "train/mask/t1/pIoU": losses_t1["ious"]["pIoU"],
    }
    return logging_dict