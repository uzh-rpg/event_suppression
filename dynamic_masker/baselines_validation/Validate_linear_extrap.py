from matplotlib import pyplot as plt
from Validate import Validate
from tqdm import tqdm

import torch
import numpy as np
from scipy.ndimage import label, center_of_mass
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


class ValidateLinearExtrap(Validate):
    def __init__(self, config_path, model_path):
        super().__init__(config_path, model_path)

    @staticmethod
    def get_centroids(binary_mask: torch.Tensor):
        # Ensure it's a CPU numpy array
        mask_np = binary_mask.cpu().numpy().astype(np.uint8)
        
        # Step 1: Label connected components
        labeled_array, num_features = label(mask_np) # labeled_array: shape (H, W), num_features: int

        # Step 2: Compute centroids
        centroids = center_of_mass(mask_np, labeled_array, range(1, num_features + 1))
        centroids_array = np.array(centroids).T  # shape: (2, N_centroids)

        return centroids_array, labeled_array
    
    @staticmethod
    def optimal_match_centroids(centroids_A, centroids_B):
        A = centroids_A.T
        B = centroids_B.T
        distances = cdist(A, B)

        row_ind, col_ind = linear_sum_assignment(distances)
        return np.array(list(zip(row_ind, col_ind)))
    
    @staticmethod
    def extrapolated_centroids(match_, centroids_t0, centroids_t1):
        # match_[0] should take the centroid number associated with centroid number match_[1]
        coords_cluster = np.stack((centroids_t0[:, match_[0]], centroids_t1[:, match_[1]]), axis=0)
        t = [0, 1]
        coeffs_x = np.polyfit(t, coords_cluster[:, 0], deg=1)
        coeffs_y = np.polyfit(t, coords_cluster[:, 1], deg=1)
        extrapolated_x = np.polyval(coeffs_x, x=2)
        extrapolated_y = np.polyval(coeffs_y, x=2)
        
        # the future centroid position
        return np.array([extrapolated_x, extrapolated_y])

    @staticmethod
    def shifted_coords_(coords, flow, H, W):
        # coords are (row, col) | flow is (y, x) = (row, col)
        shifted_coords = coords + flow
        # Round and clip to valid indices
        shifted_coords = np.round(shifted_coords).astype(int)
        valid = (0 <= shifted_coords[:, 0]) & (shifted_coords[:, 0] < H) & \
                (0 <= shifted_coords[:, 1]) & (shifted_coords[:, 1] < W)
        return shifted_coords[valid]

    @staticmethod
    def extrapolate_dynamic_mask(centroids_t0, centroids_t1, matches, labled_mask_t1):

        future_mask = np.zeros_like(labled_mask_t1)
        for match_ in matches:
            # the future centroid position
            x_y_t0 = centroids_t0[:, match_[0]]
            x_y_t1 = centroids_t1[:, match_[1]]
            x_y_t2 = ValidateLinearExtrap.extrapolated_centroids(match_, centroids_t0, centroids_t1)
            flow = x_y_t2 - x_y_t1

            cluster_label = match_[0] + 1
            cluster_mask = (labled_mask_t1 == cluster_label)
            coords = np.argwhere(cluster_mask)

            # Shift each coordinate
            H, W = future_mask.shape
            shifted_coords = ValidateLinearExtrap.shifted_coords_(coords, flow, H, W)
            future_mask[shifted_coords[:, 0], shifted_coords[:, 1]] = 1
        
        return future_mask

    @staticmethod
    def has_an_object(mask):
        return (mask == True).any().item()

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

            if ind == 0:
                dynamic_mask_t0 = dynamic_mask_t1
                continue

            # calculate the centroids ONLY if the mask has objects
            if self.has_an_object(dynamic_mask_t0) and self.has_an_object(dynamic_mask_t1):
                centroids_t0, labeld_mask_t0 = self.get_centroids(dynamic_mask_t0)
                centroids_t1, labeld_mask_t1 = self.get_centroids(dynamic_mask_t1)
                matches = self.optimal_match_centroids(centroids_t0, centroids_t1)
                future_mask = self.extrapolate_dynamic_mask(
                    centroids_t0, centroids_t1, matches, labeld_mask_t1
                    )
                future_mask = torch.tensor(future_mask)
            else:
                future_mask = dynamic_mask_t1
            
            # Save current mask for the next iter
            dynamic_mask_t0 = dynamic_mask_t1

            if data["dynamic_mask_gt"] is not None:
                gt_future = sequence[ind+1]["dynamic_mask_gt"]
                gt_future = gt_future.squeeze(0).cpu()
                voxel = voxel.squeeze(0).cpu()

                ious = self.iou_metric(future_mask, gt_future, voxel, apply_sigmoid=False)

                loss_eval = self.loss_class()
                loss_eval(future_mask, gt_future)

                iou_sets.append(ious["IoU"])
                miou_sets.append(ious["mIoU"])
                piou_sets.append(ious["pIoU"])
                tps.append(loss_eval.tp.cpu().numpy())
                tns.append(loss_eval.tn.cpu().numpy())
                fps.append(loss_eval.fp.cpu().numpy())
                fns.append(loss_eval.fn.cpu().numpy())

                if self.config["vis"]["plot"]:
                    self._plot_prediction(future_mask, gt_future, plot_path, ind)

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
    validator = ValidateLinearExtrap(
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
