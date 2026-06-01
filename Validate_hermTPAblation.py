"""
The following script is for ablating the model capability to different time prediction horizons dt.
We vary the temporal horizon dt during validation, for each dt we evaluate the segmentation performance.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only the first GPU for validation

import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

from Validate import open_config_json
from Validate_herm import ValidateHerm, sequences_to_ignore
from suppressor.metrics.piou_evimo import piou_evimo
from dynamic_masker.utils.train_log import binary_segmentation_losses


class ValidateHermTPAblation(ValidateHerm):
    def __init__(self, config, model_path):
        super().__init__(config, model_path)
        self.pred_times_ms = np.arange(0, 101, 10)  # From 0ms to 100ms in steps of 10ms
        
    def _get_dataset(self, dt_pred_time_ms):
        dataset = self.dataset_provider.get_evimo_test_dataset_by_varying_event_window(dt_pred_time_ms=dt_pred_time_ms)
        return dataset
    
    def validate_model(self, save_path_):
        """Validate the dynamic masker model on the test dataset

        Args:
            save_path_ (_type_): _description_

        Returns:
            _type_: _description_
        """
        
        for pred_time in self.pred_times_ms:
            save_path = f"{save_path_}/dt_{int(pred_time)}ms"
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            self.dataset = self._get_dataset(dt_pred_time_ms=pred_time)
            self.model = self.setup_model(self.model_path)
            self.model.eval()

            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            results = {}

            for sequence in tqdm(self.dataset):
                sequence_id = sequence[0]["sequence_id"]
                # Do not validate sequences that would be discarted by EVMO metrics
                if sequence_id in sequences_to_ignore:
                    print(f"Skipping sequence {sequence_id} as it is in the ignore list.")
                    continue
                save_sequence_path = save_path / sequence_id
                save_sequence_path.mkdir(exist_ok=True)
                plot_path = save_sequence_path / "plots_{}".format(sequence_id)
                plot_path.mkdir(exist_ok=True)

                stats = self._evaluate_sequence(sequence, plot_path)
                
                results[sequence_id] = stats
                logs = self._aggregate_logs(stats, sequence_id)

                self._write_json(results, save_path / "results.json")

                if self.config["vis"]["verbose"]:
                    self._print_stats(sequence_id, stats)
                    
                self.model.reset_states()

            aggregated = self._aggregate_results(results)
            logs = self._add_tot_to_logs(logs, aggregated)

            results["test/total"] = aggregated
            self._write_json(results, save_path / "results.json")
        return results

    
    def _evaluate_sequence(self, sequence, plot_path):
        plot_path_flow = plot_path / "flow"
        plot_path_flow.mkdir(parents=True, exist_ok=True)
        plot_path_segmentation = plot_path / "segmentation"
        plot_path_segmentation.mkdir(parents=True, exist_ok=True)

        iou_t0_sets, miou_t0_sets, piou_t0_sets = [], [], []
        iou_t1_sets, miou_t1_sets, piou_t1_sets = [], [], []
        piou_evimo_t0_sets, pious_evimo_t1_sets = [], []

        # make sure everything is on the same device
        self.model = self.model.to(self.device)

        for ind in tqdm(range(len(sequence)-1)):
            data_t0 = sequence[ind]
            data_t1 = sequence[ind+1]

            voxel = data_t0["representation"].to(self.device)
            dt = data_t1["sampled_dt"].to(self.device) # future dt
            # transform dt to be in milliseconds and integer type to save memory
            dt = (dt * 1000.0).to(torch.int8)
            
            with torch.no_grad():
                output = self.model(x=voxel.unsqueeze(0), dt=dt)
                mask_t0 = output["mask"][-1].squeeze(0)
                mask_t1 = output["future_mask"][-1].squeeze(0)

                gt_mask_t0 = data_t0["dynamic_mask"].to(self.device)
                gt_mask_t1 = data_t1["dynamic_mask"].to(self.device)

            if self.config["vis"]["plot"]:
                self._plot_prediction(mask_t0, gt_mask_t0, mask_t1, gt_mask_t1, plot_path_segmentation, ind)

            piou_evimo_t0 = piou_evimo(
                events=torch.from_numpy(data_t0["events"]).cpu(),
                pred_mask=mask_t0.squeeze(0).cpu(),
                gt_mask=gt_mask_t0.squeeze(0).cpu(),
                depth_map_start=data_t0["depth_map"],
                depth_map_end=data_t1["depth_map"]
            )
            piou_evimo_t1 = piou_evimo(
                events=torch.from_numpy(data_t1["events"]).cpu(),
                pred_mask=mask_t1.squeeze(0).cpu(),
                gt_mask=gt_mask_t1.squeeze(0).cpu(),
                depth_map_start=data_t0["depth_map"],
                depth_map_end=data_t1["depth_map"]
            )
            losses_mask_t0 = binary_segmentation_losses(mask_t0, gt_mask_t0, voxel, self.config)
            losses_mask_t1 = binary_segmentation_losses(mask_t1, gt_mask_t1, voxel, self.config)

            piou_evimo_t0_sets.append(piou_evimo_t0)
            pious_evimo_t1_sets.append(piou_evimo_t1)
            iou_t0_sets.append(losses_mask_t0["ious"]["IoU"])
            miou_t0_sets.append(losses_mask_t0["ious"]["mIoU"])
            piou_t0_sets.append(losses_mask_t0["ious"]["pIoU"])
            iou_t1_sets.append(losses_mask_t1["ious"]["IoU"])
            miou_t1_sets.append(losses_mask_t1["ious"]["mIoU"])
            piou_t1_sets.append(losses_mask_t1["ious"]["pIoU"])

        return {
            "pIoU_evimo/t0": np.nanmean(piou_evimo_t0_sets).tolist(),
            "pIoU_evimo/t1": np.nanmean(pious_evimo_t1_sets).tolist(),
            "IoU/t0": np.nanmean(np.array(iou_t0_sets) * 100, axis=0).tolist(),
            "mIoU/t0": np.nanmean(np.array(miou_t0_sets) * 100).tolist(),
            "pIoU/t0": np.nanmean(np.array(piou_t0_sets) * 100).tolist(),
            "IoU/t1": np.nanmean(np.array(iou_t1_sets) * 100, axis=0).tolist(),
            "mIoU/t1": np.nanmean(np.array(miou_t1_sets) * 100).tolist(),
            "pIoU/t1": np.nanmean(np.array(piou_t1_sets) * 100).tolist(),
        }


if __name__ == "__main__":
    config_name = "dynamic_masker/configs/train_herm_ablation_TP_ablation.json"
    model_name = "Herm_Tp_2025-10-28_14-09-42/model_epoch_12"

    config = open_config_json(config_name)
    validator = ValidateHermTPAblation(
                        config=config,
                        model_path="checkpoints/" + model_name + ".pth"
    )
    validator.validate_model(
        save_path_="dynamic_masker/results/" + model_name
    )
    
    # validator.validate_all_models_in_folder(
    #     model_folder="checkpoints/HermNoAtten_2025-06-19_17-59-22",
    #     results_folder="results/HermNoAtten_2025-06-19_17-59-22"
    # )