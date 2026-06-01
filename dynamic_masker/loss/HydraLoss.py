import torch
import torch.nn as nn
from .BCEDiceLoss import BCEDiceLoss
from dynamic_masker.loss.flow import Iterative

# Multi-task Supervision Loss with flow smoothness
class HydraLoss(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        bce_weight = config["loss"].get('bce_weight', 0.7)
        dice_weight = config["loss"].get('dice_weight', 0.3)
        class_weights = config["loss"].get('class_weights', 1)
        lambda_future_mask = config["loss"].get('lambda_future_mask', 1.0)
        lambda_flow = config["loss"].get('lambda_flow', 0.1)
        lambda_flow_smooth = config["loss"].get('lambda_flow_smooth', 0.05)

        self.w_sup = config["loss"].get('w_sup', 1.0)
        self.w_unsup = config["loss"].get('w_unsup', 1.0)
        self.config = config

        self.mask_loss = BCEDiceLoss(
            bce_weight=bce_weight, dice_weight=dice_weight, class_weights=class_weights
            )
        self.lambda_future_mask = lambda_future_mask
        self.lambda_flow = lambda_flow
        self.lambda_flow_smooth = lambda_flow_smooth
        self.flow_loss = nn.L1Loss()
        self.unsupervised_flow_loss = Iterative(config, device=kwargs["device"])

        self.supervised_mask_loss = torch.tensor(0.0).to('cuda')

    @staticmethod
    def charbonnier_loss(predicted, gt, epsilon=1e-3):
        return torch.mean(torch.sqrt((predicted - gt) ** 2 + epsilon ** 2))
    
    def supervised_flow_loss(self, pred_f, gt_f, mask):
        loss = 0
        gt_f = gt_f[mask]
        for pred_flow_ in pred_f:
            pred_flow_ = pred_flow_[mask]
            flow_loss = self.flow_loss(pred_flow_, gt_f)
            loss += self.lambda_flow * flow_loss

            flow_smooth = HydraLoss.charbonnier_loss(pred_flow_[:, :, :-1, :], pred_flow_[:, :, 1:, :]) + \
                        HydraLoss.charbonnier_loss(pred_flow_[:, :, :, :-1], pred_flow_[:, :, :, 1:])
            loss += self.lambda_flow_smooth * flow_smooth
        return loss

    def update_(self, 
                pred_mask_t0: list, 
                gt_mask_t0: torch.Tensor,
                pred_mask_t1: list = None, 
                gt_mask_t1: torch.Tensor = None,
                pred_flow_t0: list = None, 
                gt_flow_t0: torch.Tensor = None,
                mask_invalid_flows_t0: torch.Tensor = torch.tensor([False]),
                pred_flow_t1: list = None,
                gt_flow_t1: torch.Tensor = None,
                mask_invalid_flows_t1: torch.Tensor = torch.tensor([False])
                ):
        
        
        # Always compute mask loss at t=0
        if pred_mask_t0 is not None and gt_mask_t0 is not None:
            loss = torch.tensor(0.0).to(gt_mask_t0.device)
            for pred_mask in pred_mask_t0:
                loss += self.mask_loss(pred_mask, gt_mask_t0)
        elif pred_mask_t1 is not None:
            loss = torch.tensor(0.0).to(pred_mask_t1[-1].device)
        else:
            raise ValueError("At least one of pred_mask_t0 or pred_mask_t1 must be provided.")

        # Optional: future mask loss
        if pred_mask_t1 is not None and gt_mask_t1 is not None:
            for pred_mask in pred_mask_t1:
                future_mask_loss = self.mask_loss(pred_mask, gt_mask_t1)
                loss += self.lambda_future_mask * future_mask_loss

        # compute the unsupervised loss

        # Optional: Compute loss for optical flow at t=0 + smoothness
        if mask_invalid_flows_t0.any().item():
            flow_loss = self.supervised_flow_loss(pred_flow_t0, gt_flow_t0, mask_invalid_flows_t0)
            loss += flow_loss
        
        if mask_invalid_flows_t1.any().item():
            flow_loss = self.supervised_flow_loss(pred_flow_t1, gt_flow_t1, mask_invalid_flows_t1)
            loss += flow_loss

        self.supervised_mask_loss += loss

    def update(self, 
                pred_mask_t0: list = None, 
                gt_mask_t0: torch.Tensor = None,
                pred_mask_t1: list = None, 
                gt_mask_t1: torch.Tensor = None,
                pred_flow_t0: list = None, 
                gt_flow_t0: torch.Tensor = None,
                mask_invalid_flows_t0: torch.Tensor = torch.tensor([False]),
                pred_flow_t1: list = None,
                gt_flow_t1: torch.Tensor = None,
                mask_invalid_flows_t1: torch.Tensor = torch.tensor([False]),
                event_list: list = None, 
                pol_mask: list = None, 
                d_event_list: list = None, 
                d_pol_mask: list = None
                ):
        self.update_(
            pred_mask_t0=pred_mask_t0,
            gt_mask_t0=gt_mask_t0,
            pred_mask_t1=pred_mask_t1,
            gt_mask_t1=gt_mask_t1,
            pred_flow_t0=pred_flow_t0,
            gt_flow_t0=gt_flow_t0,
            mask_invalid_flows_t0=mask_invalid_flows_t0,
            pred_flow_t1=pred_flow_t1,
            gt_flow_t1=gt_flow_t1,
            mask_invalid_flows_t1=mask_invalid_flows_t1
            )
        
        self.unsupervised_flow_loss.update(
            flow_list=pred_flow_t1, 
            event_list=event_list, 
            pol_mask=pol_mask, 
            d_event_list=d_event_list, 
            d_pol_mask=d_pol_mask
            )
    

    def forward(self, *args, **kwargs):
        return self.w_sup*self.supervised_mask_loss + self.w_unsup*self.unsupervised_flow_loss()
    
    def reset(self):
        self.supervised_mask_loss = torch.tensor(0.0).to('cuda')
        self.unsupervised_flow_loss.reset()
