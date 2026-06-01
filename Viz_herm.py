import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

from Validate import open_config_json
from Validate_hydra import ValidateHydra

from suppressor.visualization.plotting_func import (
    plot_prediction_and_gt,
    plot_all_events,
    plot_filtered_events,
    plot_all_events_only,
    plot_flow_arrows_colored,
    plot_events_with_flow
)

class VizHerm(ValidateHydra):
    def __init__(self, config, model_path):
        super().__init__(config, model_path)

    def _get_dataset(self):
        return self.dataset_provider.get_evimo_test_dataset()

    @staticmethod
    def _plot_paper(pred_t1, gt_t1, plot_path, ind, events, flow, apply_sigmoid=True):
        pred_t1 = pred_t1.cpu()
        gt_t1 = gt_t1.cpu()

        # Convert events to numpy if needed
        events_np = events.numpy() if hasattr(events, 'numpy') else events

        # Plot all events
        plot_all_events(events_np, plot_path, ind)
        plot_all_events_only(events_np, plot_path, ind)

        # Uncomment to plot prediction/GT masks or filtered events/flow
        plot_prediction_and_gt(pred_t1, gt_t1, plot_path, ind, apply_sigmoid=apply_sigmoid)
        dynamic_mask = torch.sigmoid(pred_t1) > 0.5
        plot_filtered_events(events_np, dynamic_mask.squeeze(0), plot_path, ind)
        # plot_flow(flow, plot_path, ind)
        plot_events_with_flow(events_np, flow*100, plot_path, ind)

    def _evaluate_sequence(self, sequence, plot_path):
        plot_path = plot_path / "paper"
        plot_path.mkdir(parents=True, exist_ok=True)

        self.model = self.model.to(self.device)

        for ind in tqdm(range(len(sequence) - 1)):
            data_t0 = sequence[ind]
            data_t1 = sequence[ind + 1]

            voxel = data_t0["representation"].to(self.device)
            dt = data_t1["sampled_dt"].to(self.device)

            with torch.no_grad():
                output = self.model(x=voxel.unsqueeze(0), dt=dt)
                flow_tot = output["flow"][-1].squeeze(0)
                mask_t1 = output["future_mask"][-1].squeeze(0)
                gt_mask_t1 = data_t1["dynamic_mask"].to(self.device)

            if self.config["vis"]["plot"]:
                self._plot_paper(
                    pred_t1=mask_t1,
                    gt_t1=gt_mask_t1,
                    plot_path=plot_path,
                    events=data_t0["events"],
                    flow=flow_tot,
                    ind=ind,
                )

    def validate_model(self, save_path):
        self.dataset = self._get_dataset()
        self.model = self.setup_model(self.model_path)
        self.model.eval()

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        for sequence in tqdm(self.dataset):
            sequence_id = sequence[0]["sequence_id"]
            # If you want to visualize only a specific sequence, set this:
            if sequence_id != "floor_seq_00":
                continue
            save_sequence_path = save_path / sequence_id
            save_sequence_path.mkdir(exist_ok=True)
            plot_path = save_sequence_path / f"plots_{sequence_id}"
            plot_path.mkdir(exist_ok=True)

            self._evaluate_sequence(sequence, plot_path)
            self.model.reset_states()


if __name__ == "__main__":
    config = open_config_json("dynamic_masker/configs/validate_herm.json")
    validator = VizHerm(
        config=config,
        model_path="dynamic_masker/checkpoints/Herm_2025-04-15_15-49-03/model_epoch_49.pth"
    )
    validator.validate_model(
        save_path="dynamic_masker/results/Herm_2025-04-15_15-49-03_plot/model_epoch_49"
    )
