import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

from Validate import open_config_json
from Validate_hydra import ValidateHydra
from suppressor.metrics.piou_evimo import piou_evimo
from dynamic_masker.utils.train_log import binary_segmentation_losses
from suppressor.metrics.ratio_correctly_segmented_objects import instance_success_from_binary, object_success_proxy

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

class ValidateHerm(ValidateHydra):
    def __init__(self, config, model_path):
        super().__init__(config, model_path)

    def _get_dataset(self):
        dataset = self.dataset_provider.get_evimo_test_dataset()
        return dataset
    
    def _aggregate_results(self, results):
        total = lambda key: np.nanmean([results[seq][key] for seq in results if seq != "total"], axis=0).tolist()
        return {
            "pIoU_evimo/t0": total("pIoU_evimo/t0"),
            "pIoU_evimo/t1": total("pIoU_evimo/t1"),
            "IoU/t0": total("IoU/t0"),
            "mIoU/t0": total("mIoU/t0"),
            "pIoU/t0": total("pIoU/t0"),
            "IoU/t1": total("IoU/t1"),
            "mIoU/t1": total("mIoU/t1"),
            "pIoU/t1": total("pIoU/t1"),
        }
    
    def _evaluate_sequence(self, sequence, plot_path):
        plot_path_flow = plot_path / "flow"
        plot_path_flow.mkdir(parents=True, exist_ok=True)
        plot_path_segmentation = plot_path / "segmentation"
        plot_path_segmentation.mkdir(parents=True, exist_ok=True)

        iou_t0_sets, miou_t0_sets, piou_t0_sets = [], [], []
        iou_t1_sets, miou_t1_sets, piou_t1_sets = [], [], []
        piou_evimo_t0_sets, pious_evimo_t1_sets = [], []
        inst_succ = []
        object_succ = []

        # make sure everything is on the same device
        self.model = self.model.to(self.device)

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
            
            # Compute instance success rate
            s_inst, k_inst, r_inst = instance_success_from_binary(
                gt_mask_t1.squeeze(0).cpu().numpy(), 
                (torch.sigmoid(mask_t1) > 0.5).cpu().numpy(), 
                iou_thresh=0.5
                )
            # Compute object success rate
            s_obj, k_obj, r_obj = object_success_proxy(
                gt_mask_t1.squeeze(0).cpu().numpy(), 
                (torch.sigmoid(mask_t1) > 0.5).cpu().numpy(), 
                iou_thresh=0.5
                )

            piou_evimo_t0_sets.append(piou_evimo_t0)
            pious_evimo_t1_sets.append(piou_evimo_t1)
            iou_t0_sets.append(losses_mask_t0["ious"]["IoU"])
            miou_t0_sets.append(losses_mask_t0["ious"]["mIoU"])
            piou_t0_sets.append(losses_mask_t0["ious"]["pIoU"])
            iou_t1_sets.append(losses_mask_t1["ious"]["IoU"])
            miou_t1_sets.append(losses_mask_t1["ious"]["mIoU"])
            piou_t1_sets.append(losses_mask_t1["ious"]["pIoU"])
            inst_succ.append(r_inst)
            object_succ.append(r_obj)

        return {
            "pIoU_evimo/t0": np.nanmean(piou_evimo_t0_sets).tolist(),
            "pIoU_evimo/t1": np.nanmean(pious_evimo_t1_sets).tolist(),
            "IoU/t0": np.nanmean(np.array(iou_t0_sets) * 100, axis=0).tolist(),
            "mIoU/t0": np.nanmean(np.array(miou_t0_sets) * 100).tolist(),
            "pIoU/t0": np.nanmean(np.array(piou_t0_sets) * 100).tolist(),
            "IoU/t1": np.nanmean(np.array(iou_t1_sets) * 100, axis=0).tolist(),
            "mIoU/t1": np.nanmean(np.array(miou_t1_sets) * 100).tolist(),
            "pIoU/t1": np.nanmean(np.array(piou_t1_sets) * 100).tolist(),
            "IS@0.5/t0": np.nanmean(np.array(inst_succ) * 100).tolist(),
            "OS@0.5/t0": np.nanmean(np.array(object_succ) * 100).tolist(),
        }
    
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
        results = {}

        for sequence in tqdm(self.dataset):
            sequence_id = sequence[0]["sequence_id"]
            # Do not validate sequences that would be discarted by EVMO metrics
            # if sequence_id in sequences_to_ignore:
            #     print(f"Skipping sequence {sequence_id} as it is in the ignore list.")
            #     continue
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


def validate_models(config_path, models_dir, save_path):
    config = open_config_json(config_path)
    models_dir = Path(models_dir)
    general_save_path = Path(save_path, models_dir.stem)
    if not general_save_path.exists():
        general_save_path.mkdir(parents=True)
    
    all_results = {}
    for model_path in sorted(models_dir.iterdir()):
        print(f"Validating model: {model_path}")
        save_path = Path(general_save_path, model_path.stem)
        validator = ValidateHerm(
                        config=config,
                        model_path=model_path
        )
        model_results = validator.validate_model(save_path=str(save_path))
        all_results[model_path.stem] = model_results["test/total"]
        print(f"Model {model_path} validated and results saved at {save_path}")

    with open(Path(general_save_path, "all_results.json"), 'w') as json_file:
        json.dump(all_results, json_file, indent=4)

    
if __name__ == "__main__":
    config = open_config_json("dynamic_masker/configs/validate_herm.json")
    validator = ValidateHerm(
                        config=config,
                        model_path="checkpoints/checkpoints_prerebuttal/Herm_2025-04-15_15-49-03/model_epoch_49.pth"
    )
    validator.validate_model(
        save_path="dynamic_masker/results/Herm_2025-04-15_15-49-03/model_epoch_49"
    )
    # validate_models(
    #     config_path="dynamic_masker/configs/validate_herm.json",
    #     models_dir="dynamic_masker/checkpoints/Herm_2025-04-15_15-49-03",
    #     save_path="dynamic_masker/results/Herm_2025-04-15_15-49-03"
    # )
    # validate_models(
    #     config_path="dynamic_masker/configs/train_herm_ablation_XXS.json",
    #     models_dir="checkpoints/Herm_XXS_2025-10-29_08-50-02_",
    #     save_path="dynamic_masker/results/Herm_XXS_2025-10-29_08-50-02"
    # )