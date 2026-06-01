import json
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt

from dynamic_masker.utils.utils import load_model
from dynamic_masker.models.model_hydra import HydraEVNet
from dynamic_masker.utils.train_log import binary_segmentation_losses, flow_losses
from suppressor.visualization.plotting_func import (
                                                plot_prediction_and_gt, 
                                                plot_all_events, 
                                                plot_filtered_events,
                                                plot_all_events_only,
                                                plot_flow
                                                )

from Validate import Validate, open_config_json
from Validate_hydra import ValidateHydra
from evlicious import Events

class VizHydra(ValidateHydra):
    def __init__(self, config, model_path):
        super().__init__(config, model_path)

    def validate_model(self, save_path):
        """Validate the dynamic masker model on the test dataset

        Args:
            save_path (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.dataset = self._get_dataset()
        self.model = self.setup_model(self.model_path)
        self.model.eval()

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        for sequence in tqdm(self.dataset):
            sequence_id = sequence[0]["sequence_id"]
            if sequence_id != "zurich_city_14_b":
                continue
            save_sequence_path = save_path / sequence_id
            save_sequence_path.mkdir(exist_ok=True)
            plot_path = save_sequence_path / "plots_{}".format(sequence_id)
            plot_path.mkdir(exist_ok=True)

            self._evaluate_sequence(sequence, plot_path)
            self.model.reset_states()
    
    @staticmethod
    def _plot_paper(pred_t1, gt_t1, plot_path, ind, events, flow, apply_sigmoid=True):
        pred_t1 = pred_t1.cpu()
        gt_t1 = gt_t1.cpu()
        # Plot prediction and ground truth separately
        # plot_prediction_and_gt(pred_t1, gt_t1, plot_path, ind, apply_sigmoid=apply_sigmoid)

        # Convert events to standard format
        events_np = events.numpy() if hasattr(events, 'numpy') else events

        # Plot all events
        plot_all_events(events_np, plot_path, ind)
        # plot_all_events_only(events_np, plot_path, ind)

        dynamic_mask = torch.sigmoid(pred_t1) > 0.5
        dynamic_mask = dynamic_mask.squeeze(0)
        plot_filtered_events(events_np, dynamic_mask, plot_path, ind)
        # plot_flow(flow, plot_path, ind)
        

    def _evaluate_sequence(self, sequence, plot_path):
        plot_path = plot_path / "paper"
        plot_path.mkdir(parents=True, exist_ok=True)

        for ind in tqdm(range(len(sequence)-1)):
            data_t0 = sequence[ind]
            data_t1 = sequence[ind+1]

            voxel = data_t0["representation"]["left"].to(self.device)
            with torch.no_grad():
                output = self.model(x=voxel.unsqueeze(0), dt=torch.tensor([100], device=self.device))
                flow_tot = output["flow"][-1].squeeze(0)
                mask_t1 = output["future_mask"][-1].squeeze(0)
                gt_mask_t1 = data_t1["dynamic_mask"].to(self.device)

            if self.config["vis"]["plot"]:
                self._plot_paper(
                    pred_t1=mask_t1, 
                    gt_t1=gt_mask_t1, 
                    plot_path=plot_path, 
                    events=data_t0['events'],
                    flow=flow_tot,
                    ind=ind,
                    )
        
if __name__ == "__main__":
    config = open_config_json("dynamic_masker/configs/validate_hydra.json")
    validator = VizHydra(
                        config=config,
                        model_path="dynamic_masker/checkpoints/Hydra_2025-04-06_10-35-01/model_epoch_64_best.pth"
    )
    validator.validate_model(
        save_path="dynamic_masker/results/Hydra_2025-04-06_10-35-01_plot/model_epoch_64_best"
    )
