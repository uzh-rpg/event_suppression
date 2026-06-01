import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import json
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

from Validate import open_config_json
from Validate_hydra import ValidateHydra
from evlicious import Events
import matplotlib.pyplot as plt

from suppressor.EVIMOv1_dataloader.EVIMOTestSequence import EVIMOTestSequence

import evlicious
from evlicious.io.utils.images import Images



sequences_to_ignore = [
    "box_seq_00",
    "box_seq_01",
    "box_seq_02",
    "fast_seq_00",
    "fast_seq_01",
    "floor_seq_01",
    "table_seq_00",
    "table_seq_01",
    "tabletop_seq_00"
    "tabletop_seq_01"
    "tabletop_seq_02",
    "tabletop_seq_03",
    ]

def debug_plot(raw_events, dynamic_mask_t0):
    events = Events(
        x=raw_events['x'].astype(np.int16).astype(np.uint16),
        y=raw_events['y'].astype(np.int16).astype(np.uint16),
        t=raw_events['t'].astype(np.int64),
        p=raw_events['p'].astype(np.int8),
        width=640,
        height=480
    )
    rendered_events = events.render()

    plt.clf()
    plt.imshow(dynamic_mask_t0.squeeze().numpy())
    plt.imshow(rendered_events, alpha=0.5)
    plt.savefig('debug_plot.png')
    plt.close()

class VisualizationsHerm(ValidateHydra):
    def __init__(self, config, model_path):
        super().__init__(config, model_path)
        self.delta_t_ms = self.config["loader"]["event_dt_ms"]
        self.num_bins = self.config["data"]["voxel_bins"]

    def _get_test_sequence(self, grandchild):
        test_sequence = EVIMOTestSequence(h5_path=grandchild, window_ms=self.delta_t_ms, num_bins=self.num_bins)
        return test_sequence
    
    def viz_events(self, sequence, output_path=None):
        sequence = self._get_test_sequence(sequence)
        data = sequence[0]
        events = data["events"]
        dynamic_mask_t0 = data["dynamic_mask"]
        depth_map = data["depth_map"]

        # if images are given, load them
        # images = data["images"]
        # images = Images.from_path(images)
        # images = images[::1]

        # if feature tracks are given, load them
        # tracks_data = load_feature_tracks(args.tracks)

        mask_x = (events[:, 0] > 0) & (events[:, 0] < 346)
        mask_y = (events[:, 1] > 0) & (events[:, 1] < 260)
        mask_events = mask_x & mask_y  # Use '&' instead of 'and'

        events[:,2]*=1e4

        events_ = Events(
            x=events[:, 0][mask_events].astype(np.int16).astype(np.uint16),
            y=events[:, 1][mask_events].astype(np.int16).astype(np.uint16),
            t=events[:, 2][mask_events].astype(np.int64),
            p=events[:, 3][mask_events].astype(np.int8),
            width=346,
            height=260
        )

        evlicious.art.visualize_3d(
                                events_, 
                                time_window_us=self.delta_t_ms * 1000,
                                time_step_us=1000,
                                images=None,
                                factor=1,
                                tracks=None,
                                output_path=output_path,
                                loop=False)
    
    
    def visualize_sequence(self, sequence, plot_path):
        plot_path = Path(plot_path)
        plot_path_flow = plot_path / "flow"
        plot_path_flow.mkdir(parents=True, exist_ok=True)
        plot_path_segmentation = plot_path / "segmentation"
        plot_path_segmentation.mkdir(parents=True, exist_ok=True)

        self.model = self.setup_model(self.model_path)
        self.model.eval()
        sequence = self._get_test_sequence(sequence)

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


if __name__ == "__main__":
    config = open_config_json("dynamic_masker/configs/validate_herm.json")
    validator = VisualizationsHerm(
                        config=config,
                        model_path="checkpoints/Herm_2025-04-15_15-49-03/model_epoch_49.pth"
    )
    # validator.visualize_sequence(
    #     sequence = "/home/roberto/datasets/EVIMO1/test/box/seq_03.h5",
    #     plot_path = "results/Herm_2025-04-15_15-49-03/model_epoch_49"
    # )
    validator.viz_events(
        sequence = "/home/roberto/datasets/EVIMO1/test/box/seq_03.h5",
        output_path = "results/Herm_2025-04-15_15-49-03/model_epoch_49/events_viz.png"
    )

