import os
import sys
import torch
import numpy as np
from tqdm import tqdm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from Validate_herm import ValidateHerm
from Validate import open_config_json
from utils.train_log import binary_segmentation_losses


class ValidateCurrentAsFutureEVIMO(ValidateHerm):
    def __init__(self, config, model_path):
        super().__init__(config, model_path)

    def _evaluate_sequence(self, sequence, plot_path):
        plot_path_flow = plot_path / "flow"
        plot_path_flow.mkdir(parents=True, exist_ok=True)
        plot_path_segmentation = plot_path / "segmentation"
        plot_path_segmentation.mkdir(parents=True, exist_ok=True)

        iou_t0_sets, miou_t0_sets, piou_t0_sets = [], [], []
        iou_t1_sets, miou_t1_sets, piou_t1_sets = [], [], []

        for ind in tqdm(range(len(sequence)-1)):
            data_t0 = sequence[ind]
            data_t1 = sequence[ind+1]

            voxel = data_t0["representation"].to(self.device)
            dt = data_t1["sampled_dt"].to(self.device) # future dt
            with torch.no_grad():
                output = self.model(x=voxel.unsqueeze(0), dt=dt)
                mask_t0 = output["mask"][-1].squeeze(0)
                mask_t1 = output["future_mask"][-1].squeeze(0)

                gt_mask_t0 = data_t0["dynamic_mask"].to(self.device)
                gt_mask_t1 = data_t1["dynamic_mask"].to(self.device)

            if self.config["vis"]["plot"]:
                self._plot_prediction(mask_t0, gt_mask_t0, mask_t1, gt_mask_t1, plot_path_segmentation, ind)

            losses_mask_t0 = binary_segmentation_losses(mask_t0, gt_mask_t0, voxel, self.config)
            losses_mask_t1 = binary_segmentation_losses(mask_t0, gt_mask_t1, voxel, self.config)

            iou_t0_sets.append(losses_mask_t0["ious"]["IoU"])
            miou_t0_sets.append(losses_mask_t0["ious"]["mIoU"])
            piou_t0_sets.append(losses_mask_t0["ious"]["pIoU"])
            iou_t1_sets.append(losses_mask_t1["ious"]["IoU"])
            miou_t1_sets.append(losses_mask_t1["ious"]["mIoU"])
            piou_t1_sets.append(losses_mask_t1["ious"]["pIoU"])

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
    validator = ValidateCurrentAsFutureEVIMO(
        config=config,
        model_path="checkpoints/Herm_2025-04-15_15-49-03/model_epoch_15.pth"
        )
    validator.validate_model(save_path="results/Herm_2025-04-15_15-49-03/current_as_future")

    # binary_mask = torch.tensor([
    # [0, 0, 1, 1, 0],
    # [0, 0, 1, 1, 0],
    # [0, 0, 0, 0, 0],
    # [0, 1, 1, 0, 0],
    # [0, 1, 1, 0, 0]
    #     ], dtype=torch.uint8)

    # centroids_A, labels_A = ValidateLinearExtrap.get_centroids(binary_mask)
    # print(centroids_A)  # → [(0.5, 2.5), (3.5, 1.5)]

    # binary_mask = torch.tensor([
    # [0, 0, 0, 1, 1],
    # [0, 0, 0, 1, 1],
    # [0, 0, 0, 0, 0],
    # [0, 0, 1, 1, 0],
    # [0, 0, 1, 1, 0]
    #     ], dtype=torch.uint8)

    # centroids_B, labels_B = ValidateLinearExtrap.get_centroids(binary_mask)
    # print(centroids_B) # → [(0.5, 3.5), (3.5, 2.5)]

    # matches = ValidateLinearExtrap.optimal_match_centroids(centroids_A, centroids_B)
    # print(matches)  # → [(0, 3), (1, 4)]

    # future_mask = ValidateLinearExtrap.extrapolate_dynamic_mask(
    #     centroids_A, centroids_B, matches, labels_B
    #     )
    # print(future_mask) 
