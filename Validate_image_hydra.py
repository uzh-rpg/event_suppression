import torch
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import json
from pathlib import Path
import wandb


from dynamic_masker.utils.utils import load_model
from dynamic_masker.models.model_hydra import HydraEVNet
from dynamic_masker.utils.train_log import binary_segmentation_losses, flow_losses

from Validate import open_config_json
from Validate_hydra import ValidateHydra


class ValidateImageHydra(ValidateHydra):
    def __init__(self, config, model_path):
        super().__init__(config, model_path)

    def _evaluate_sequence(self, sequence, plot_path):
        plot_path_flow = plot_path / "flow"
        plot_path_flow.mkdir(parents=True, exist_ok=True)
        plot_path_segmentation = plot_path / "segmentation"
        plot_path_segmentation.mkdir(parents=True, exist_ok=True)

        iou_t0_sets, miou_t0_sets, piou_t0_sets = [], [], []
        iou_t1_sets, miou_t1_sets, piou_t1_sets = [], [], []
        epe, one_epe, two_epe, three_epe, ae = [], [], [], [], []

        for ind in tqdm(range(len(sequence)-1)):
            data_t0 = sequence[ind]
            data_t1 = sequence[ind+1]

            with torch.no_grad():
                voxel = data_t0["representation"]["left"].to(self.device)
                frame = data_t0["frame"].to(self.device)
                input_ = torch.cat((voxel, frame), dim=0)

                output = self.model(x=input_.unsqueeze(0), dt=torch.tensor([100], device=self.device))

                flow_tot = output["flow"][-1].squeeze(0)
                mask_t0 = output["mask"][-1].squeeze(0)
                mask_t1 = output["future_mask"][-1].squeeze(0)

            gt_mask_t0 = data_t0["dynamic_mask"].to(self.device)
            gt_mask_t1 = data_t1["dynamic_mask"].to(self.device)
            gt_flow_total = self._total_flow(data_t0, data_t1)

            if self.config["vis"]["plot"]:
                self._plot_prediction(mask_t0, gt_mask_t0, mask_t1, gt_mask_t1, plot_path_segmentation, ind)
                if gt_flow_total is not None:
                    self._plot_flow(flow_tot, gt_flow_total, plot_path_flow, ind)

            losses_mask_t0 = binary_segmentation_losses(mask_t0, gt_mask_t0, voxel, self.config)
            losses_mask_t1 = binary_segmentation_losses(mask_t1, gt_mask_t1, voxel, self.config)
            losses_flow = flow_losses(flow_tot, gt_flow_total)

            iou_t0_sets.append(losses_mask_t0["ious"]["IoU"])
            miou_t0_sets.append(losses_mask_t0["ious"]["mIoU"])
            piou_t0_sets.append(losses_mask_t0["ious"]["pIoU"])
            iou_t1_sets.append(losses_mask_t1["ious"]["IoU"])
            miou_t1_sets.append(losses_mask_t1["ious"]["mIoU"])
            piou_t1_sets.append(losses_mask_t1["ious"]["pIoU"])
            epe.append(losses_flow["epe"])
            one_epe.append(losses_flow["one_PE"])
            two_epe.append(losses_flow["two_PE"])
            three_epe.append(losses_flow["three_PE"])
            ae.append(losses_flow["AE"])

        return {
            "IoU/t0": np.nanmean(np.array(iou_t0_sets) * 100, axis=0).tolist(),
            "mIoU/t0": np.nanmean(np.array(miou_t0_sets) * 100).tolist(),
            "pIoU/t0": np.nanmean(np.array(piou_t0_sets) * 100).tolist(),
            "IoU/t1": np.nanmean(np.array(iou_t1_sets) * 100, axis=0).tolist(),
            "mIoU/t1": np.nanmean(np.array(miou_t1_sets) * 100).tolist(),
            "pIoU/t1": np.nanmean(np.array(piou_t1_sets) * 100).tolist(),
            "EPE": np.nanmean(epe).tolist(),
            "1PE": np.nanmean(one_epe).tolist(),
            "2PE": np.nanmean(two_epe).tolist(),
            "3PE": np.nanmean(three_epe).tolist(),
            "AE": np.nanmean(ae).tolist()
        }

    def setup_model(self, model_path):
        model_config = self.config["model"].copy()
        final_w_scale_flow = self.config["custom"]["final_w_scale_flow"]
        current_flow_sup = self.config["custom"]["current_flow_sup"]
        event_dt_ms = self.config["loader"]["event_dt_ms"]

        model = HydraEVNet(
            kwargs=model_config, 
            num_bins=self.num_bins +3,  # +3 for image channels
            final_w_scale_flow=final_w_scale_flow,
            current_flow_sup=current_flow_sup,
            current_flow_scaling=event_dt_ms
            )
        model = model.to(self.device)
        return load_model(model, self.device, model_dir=model_path)
    
    
if __name__ == "__main__":
    config = open_config_json("dynamic_masker/configs/train_image_hydra.json")
    validator = ValidateImageHydra(
                        config=config,
                        model_path="checkpoints/ImageHydra_2025-06-05_12-17-08/model_epoch_1.pth"
    )
    # validator.validate_model(
    #     save_path="results/ImageHydra_2025-06-05_12-17-08/model_epoch_1"
    # )
    validator.validate_all_models_in_folder(
        model_folder="checkpoints/ImageHydra_2025-06-11_09-49-15",
        results_folder="results/ImageHydra_2025-06-11_09-49-15"
    )
