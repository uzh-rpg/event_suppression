import os
import sys
import torch
import numpy as np
from tqdm import tqdm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from Validate_herm import ValidateHerm
from Validate import open_config_json

from suppressor.metrics.LinearMaskExtrapolator import LinearMaskExtrapolator


class ValidateLinExtrapEVIMO(ValidateHerm):
    def __init__(self, config, model_path):
        super().__init__(config, model_path)
        self.linear_extrapolator = LinearMaskExtrapolator()

    def _evaluate_sequence(self, sequence, plot_path):
        plot_path_flow = plot_path / "flow"
        plot_path_flow.mkdir(parents=True, exist_ok=True)
        plot_path_segmentation = plot_path / "segmentation"
        plot_path_segmentation.mkdir(parents=True, exist_ok=True)

        iou_t0_sets, miou_t0_sets, piou_t0_sets = [], [], []
        iou_t1_sets, miou_t1_sets, piou_t1_sets = [], [], []
        
        self.linear_extrapolator.reset()

        for ind in tqdm(range(len(sequence)-1)):
            data_t0 = sequence[ind]
            data_t1 = sequence[ind+1]
            
            voxel = data_t0["representation"].to(self.device)
            dt = data_t1["sampled_dt"].to(self.device) # future dt

            # Run the model
            with torch.no_grad():
                output = self.model(x=voxel.unsqueeze(0), dt=dt)
                mask_t0 = output["mask"][-1].squeeze(0).squeeze(0)
                extrapolated_mask = self.linear_extrapolator(pred_logits=mask_t0)
                if ind == 0: continue
                
            gt_mask_t0 = data_t0["dynamic_mask"].to(self.device)
            gt_mask_t1 = data_t1["dynamic_mask"].to(self.device)
            
            if config["vis"]["plot"]:
                self._plot_prediction(
                    pred_t0=torch.sigmoid(mask_t0).unsqueeze(0), 
                    gt_t0=gt_mask_t0, 
                    pred_t1=extrapolated_mask.unsqueeze(0), 
                    gt_t1=gt_mask_t1, 
                    plot_path=plot_path_segmentation, 
                    ind=ind, 
                    apply_sigmoid=False
                    )
            
            ious_t0 = self.iou_metric(mask_t0, gt_mask_t0, voxel)
            # NOTE we don't apply sigmoid correctly, but we trashold TWICE, this imo should work
            ious_t1 = self.iou_metric(extrapolated_mask, gt_mask_t1, voxel, apply_sigmoid=False)

            iou_t0_sets.append(ious_t0["IoU"])
            miou_t0_sets.append(ious_t0["mIoU"])
            piou_t0_sets.append(ious_t0["pIoU"])
            iou_t1_sets.append(ious_t1["IoU"])
            miou_t1_sets.append(ious_t1["mIoU"])
            piou_t1_sets.append(ious_t1["pIoU"])

        return {
            "IoU/t0": np.nanmean(np.array(iou_t0_sets) * 100, axis=0).tolist(),
            "mIoU/t0": np.nanmean(np.array(miou_t0_sets) * 100).tolist(),
            "pIoU/t0": np.nanmean(np.array(piou_t0_sets) * 100).tolist(),
            "IoU/t1": np.nanmean(np.array(iou_t1_sets) * 100, axis=0).tolist(),
            "mIoU/t1": np.nanmean(np.array(miou_t1_sets) * 100).tolist(),
            "pIoU/t1": np.nanmean(np.array(piou_t1_sets) * 100).tolist(),
        }
    

if __name__ == "__main__":
    config = open_config_json("configs/train_herm.json")
    validator = ValidateLinExtrapEVIMO(
        config=config,
        model_path="checkpoints/Herm_2025-04-15_15-49-03/model_epoch_15.pth"
        )
    validator.validate_model(save_path="results/Herm_2025-04-15_15-49-03/linear_extrapolation_epoch_15")