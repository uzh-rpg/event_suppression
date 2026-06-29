import torch
from .HydraLoss import HydraLoss

# Multi-task Supervision Loss with flow smoothness
class HermLoss(HydraLoss):
    def __init__(self,config, **kwargs):
        super().__init__(config=config, **kwargs)
        self.supervised_mask_loss = torch.tensor(0.0, device=self.device)

    def update_(self, 
                pred_mask_t0: list, 
                gt_mask_t0: torch.Tensor,
                pred_mask_t1: list = None, 
                gt_mask_t1: torch.Tensor = None,
                ):
        
        
        # Always compute mask loss at t=0
        loss = torch.tensor(0.0).to(gt_mask_t0.device)
        for pred_mask in pred_mask_t0:
            loss += self.mask_loss(pred_mask, gt_mask_t0)

        # Optional: future mask loss
        if pred_mask_t1 is not None and gt_mask_t1 is not None:
            for pred_mask in pred_mask_t1:
                future_mask_loss = self.mask_loss(pred_mask, gt_mask_t1)
                loss += self.lambda_future_mask * future_mask_loss

        self.supervised_mask_loss += loss

    def update(self, 
                pred_mask_t0: list, 
                gt_mask_t0: torch.Tensor,
                pred_mask_t1: list = None, 
                gt_mask_t1: torch.Tensor = None,
                pred_flow_t1: list = None,
                event_list: list = None, 
                pol_mask: list = None, 
                d_event_list: list = None, 
                d_pol_mask: list = None
                ):
        self.supervised_mask_loss = torch.tensor(0.0, device=self.device)
        self.update_(
            pred_mask_t0=pred_mask_t0,
            gt_mask_t0=gt_mask_t0,
            pred_mask_t1=pred_mask_t1,
            gt_mask_t1=gt_mask_t1,
            )
        
        self.unsupervised_flow_loss.update(
            flow_list=pred_flow_t1, 
            event_list=event_list, 
            pol_mask=pol_mask, 
            d_event_list=d_event_list, 
            d_pol_mask=d_pol_mask
            )
    
    def forward(self, *args, **kwargs):
        return self.supervised_mask_loss + self.unsupervised_flow_loss()
    
    def reset(self):
        self.supervised_mask_loss = torch.tensor(0.0, device=self.device)
        self.unsupervised_flow_loss.reset()
