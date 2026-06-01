import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from Validate import Validate

class ValidateCurrentAsFutureDSEC(Validate):
    def __init__(self, config_path, model_path):
        super().__init__(config_path, model_path)

    def _evaluate_sequence(self, sequence, plot_path):
        iou_sets, miou_sets, piou_sets = [], [], []
        tps, tns, fps, fns = [], [], [], []
        gt_timestamps = len(sequence.timestamps)

        for ind, data in tqdm(enumerate(sequence)):
            if ind+1 >= gt_timestamps: continue

            voxel = data["representation"]["left"].unsqueeze(0).to(self.device)

            # Run the model
            with torch.no_grad():
                output = self.model(voxel)
                pred = output["dynamic_mask"][-1].squeeze(0).squeeze(0)
                dynamic_mask_t1 = (torch.sigmoid(pred) > 0.5).cpu()

            if data["dynamic_mask_gt"] is not None:
                gt_future = sequence[ind+1]["dynamic_mask_gt"]
                gt_future = gt_future.squeeze(0).cpu()
                voxel = voxel.squeeze(0).cpu()

                ious = self.iou_metric(dynamic_mask_t1, gt_future, voxel, apply_sigmoid=False)

                loss_eval = self.loss_class()
                loss_eval(dynamic_mask_t1, gt_future)

                iou_sets.append(ious["IoU"])
                miou_sets.append(ious["mIoU"])
                piou_sets.append(ious["pIoU"])
                tps.append(loss_eval.tp.cpu().numpy())
                tns.append(loss_eval.tn.cpu().numpy())
                fps.append(loss_eval.fp.cpu().numpy())
                fns.append(loss_eval.fn.cpu().numpy())

                if self.config["vis"]["plot"]:
                    self._plot_prediction(dynamic_mask_t1, gt_future, plot_path, ind)

        return {
            "IoU": np.nanmean(np.array(iou_sets) * 100, axis=0).tolist(),
            "mIoU": np.nanmean(np.array(miou_sets) * 100).tolist(),
            "pIoU": np.nanmean(np.array(piou_sets) * 100).tolist(),
            "true_positives": np.nanmean(tps).tolist(),
            "true_negatives": np.nanmean(tns).tolist(),
            "false_positives": np.nanmean(fps).tolist(),
            "false_negatives": np.nanmean(fns).tolist(),
        }
    

if __name__ == "__main__":
    validator = ValidateCurrentAsFuture(
        config_path="configs/validate.json",
        model_path="checkpoints/2025-02-17_17-04-37/model_epoch_2.pth"
        )
    validator.validate_model(save_path="results/2025-02-17_17-04-37/model_epoch_2")

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
