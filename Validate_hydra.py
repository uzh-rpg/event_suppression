import json
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt

from dynamic_masker.utils.utils import load_model
from dynamic_masker.models.model_hydra import HydraEVNet
from dynamic_masker.utils.train_log import binary_segmentation_losses, flow_losses
from dynamic_masker.utils.visualization import Visualization
from suppressor.visualization.plotting_func import (
                                                plot_prediction_and_gt, 
                                                plot_all_events, 
                                                plot_filtered_events,
                                                plot_all_events_only
                                                )

from Validate import Validate, open_config_json
from evlicious import Events

class ValidateHydra(Validate):
    def __init__(self, config, model_path):
        super().__init__(config, model_path)

        self.model_path = model_path
        optical_flow_dt_ms = self.config["loader"]["optical_flow_dt_ms"]
        event_dt_ms = self.config["loader"]["event_dt_ms"]
        # NOTE: we need this ratio to scale the optical flow GT
        self.ev_flow_dt_ratio = event_dt_ms / optical_flow_dt_ms

    def setup_model(self, model_path):
        model_config = self.config["model"].copy()
        final_w_scale_flow = self.config["custom"]["final_w_scale_flow"]
        current_flow_sup = self.config["custom"]["current_flow_sup"]
        event_dt_ms = self.config["loader"]["event_dt_ms"]

        model = HydraEVNet(
            kwargs=model_config, 
            num_bins=self.num_bins,
            final_w_scale_flow=final_w_scale_flow,
            current_flow_sup=current_flow_sup,
            current_flow_scaling=event_dt_ms
            )
        model = model.to(self.device)
        return load_model(model, self.device, model_dir=model_path)
    
    def _get_dataset(self):
        dataset = self.dataset_provider.get_hydra_test_dataset()
        return dataset
    
    def _total_flow(self, data_t0, data_t1):
        flow_gt_t0 = data_t0["forward_flow_gt"]
        flow_gt_t1 = data_t1["forward_flow_gt"]
        if flow_gt_t0 is None or flow_gt_t1 is None:
            return None
        flow_gt_t0 = flow_gt_t0 * self.ev_flow_dt_ratio
        total_flow_gt = flow_gt_t0 + flow_gt_t1
        return total_flow_gt.to(self.device)
    
    @staticmethod
    def _plot_prediction(pred_t0, gt_t0, pred_t1, gt_t1, plot_path, ind, apply_sigmoid=True):
        plt.clf()
        fig, ax = plt.subplots(2, 2, figsize=(24, 20))

        assert len(pred_t0.shape) == 3
        assert len(gt_t0.shape) == 3
        assert len(pred_t1.shape) == 3
        assert len(gt_t1.shape) == 3

        pred_t0 = pred_t0.squeeze(0)
        gt_t0 = gt_t0.squeeze(0)
        pred_t1 = pred_t1.squeeze(0)
        gt_t1 = gt_t1.squeeze(0)

        if apply_sigmoid:
            pred_t0 = torch.sigmoid(pred_t0)
            pred_t1 = torch.sigmoid(pred_t1)
        
        ax[0,0].imshow((pred_t0 > 0.5).cpu().numpy())
        ax[0,0].axis("off")
        ax[0,0].set_title("Predicted dynamic objects at t=0")

        ax[0,1].imshow(gt_t0.squeeze().cpu().numpy())
        ax[0,1].axis("off")
        ax[0,1].set_title("Ground Truth dynamic objects at t=0")

        ax[1,0].imshow((pred_t1 > 0.5).cpu().numpy())
        ax[1,0].axis("off")
        ax[1,0].set_title("Predicted dynamic objects at t=1")

        ax[1,1].imshow(gt_t1.squeeze().cpu().numpy())
        ax[1,1].axis("off")
        ax[1,1].set_title("Ground Truth dynamic objects at t=1")

        plt.tight_layout()
        plt.savefig(plot_path / f"{str(ind).zfill(6)}.png")
        plt.close(fig)

    def _plot_flow(self, flow, gt_flow, plot_path, ind):
        plt.clf()
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        assert len(flow.shape) == 3

        flow_to_plot = flow.permute(1,2, 0).detach().cpu().numpy()
        rendered_flow = Visualization.flow_to_image(flow_to_plot)

        ax[0].imshow(rendered_flow)
        ax[0].axis("off")
        ax[0].set_title("Predicted optical flow")

        gt_flow_to_plot = gt_flow.permute(1,2, 0).detach().cpu().numpy()
        rendered_gt_flow = Visualization.flow_to_image(gt_flow_to_plot)

        ax[1].imshow(rendered_gt_flow)
        ax[1].axis("off")
        ax[1].set_title("Ground Truth optical flow")

        plt.tight_layout()
        plt.savefig(plot_path / f"{str(ind).zfill(6)}.png")
        plt.close(fig)

    def _aggregate_results(self, results):
        total = lambda key: np.nanmean([results[seq][key] for seq in results if seq != "total"], axis=0).tolist()
        return {
            "IoU/t0": total("IoU/t0"),
            "mIoU/t0": total("mIoU/t0"),
            "pIoU/t0": total("pIoU/t0"),
            "IoU/t1": total("IoU/t1"),
            "mIoU/t1": total("mIoU/t1"),
            "pIoU/t1": total("pIoU/t1"),
            "EPE": total("EPE"),
            "1PE": total("1PE"),
            "2PE": total("2PE"),
            "3PE": total("3PE"),
            "AE": total("AE")
        }

    @staticmethod
    def _aggregate_logs(stats, sequence_id):
        logs = {}
        for key in stats:
            new_key = f"test/{sequence_id}/{key}"
            value = stats[key]
            if "/IoU/" in new_key:
                print(key)
                value = value[0]
            logs[new_key] = value
        return logs
    
    @staticmethod
    def _add_tot_to_logs(logs, aggegated):
        for key in aggegated:
            new_key = f"test/total/{key}"
            value = aggegated[key]
            if "/IoU/" in new_key:
                value = value[0]
            logs[new_key] = value
        return logs

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
        return logs
    
    @staticmethod
    def _plot_paper(pred_t1, gt_t1, plot_path, ind, events, apply_sigmoid=True):
        pred_t1 = pred_t1.cpu()
        gt_t1 = gt_t1.cpu()
        # Plot prediction and ground truth separately
        plot_prediction_and_gt(pred_t1, gt_t1, plot_path, ind, apply_sigmoid=apply_sigmoid)

        # Convert events to standard format
        events_np = events.numpy() if hasattr(events, 'numpy') else events

        # Plot all events
        plot_all_events(events_np, plot_path, ind)
        plot_all_events_only(events_np, plot_path, ind)

        dynamic_mask = torch.sigmoid(pred_t1) > 0.5
        dynamic_mask = dynamic_mask.squeeze(0)
        plot_filtered_events(events_np, dynamic_mask, plot_path, ind)

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

            voxel = data_t0["representation"]["left"].to(self.device)
            with torch.no_grad():
                output = self.model(x=voxel.unsqueeze(0), dt=torch.tensor([100], device=self.device))
                flow_tot = output["flow"][-1].squeeze(0)
                mask_t0 = output["mask"][-1].squeeze(0)
                mask_t1 = output["future_mask"][-1].squeeze(0)

            gt_mask_t0 = data_t0["dynamic_mask"].to(self.device)
            gt_mask_t1 = data_t1["dynamic_mask"].to(self.device)
            gt_flow_total = self._total_flow(data_t0, data_t1)

            if self.config["vis"]["plot"]:
                self._plot_paper(
                    pred_t1=mask_t1, 
                    gt_t1=gt_mask_t1, 
                    plot_path=plot_path_segmentation, 
                    events=data_t0['events'],
                    ind=ind,
                    )
                
                # self._plot_prediction(mask_t0, gt_mask_t0, mask_t1, gt_mask_t1, plot_path_segmentation, ind)
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
        
    def validate_all_models_in_folder(self, model_folder, results_folder):
        model_folder = Path(model_folder)
        results_folder = Path(results_folder)
        results_folder.mkdir(parents=True, exist_ok=True)

        model_files = sorted(model_folder.glob("*.pth"))

        for model_file in model_files:
            print(f"Validating model: {model_file.name}")
            self.model_path = model_file

            model_result_dir = results_folder / model_file.stem
            model_result_dir.mkdir(parents=True, exist_ok=True)
            result_json_path = results_folder / f"{model_file.stem}_metrics.json"

            if result_json_path.exists():
                print(f"Results already exist for {model_file.name}, skipping...")
                continue

            result = self.validate_model(save_path=model_result_dir)

            with open(result_json_path, "w") as f:
                json.dump(result, f, indent=4)

            print(f"Saved results to: {result_json_path}")
        return results_folder

    
if __name__ == "__main__":
    config = open_config_json("dynamic_masker/configs/validate_hydra.json")
    validator = ValidateHydra(
                        config=config,
                        model_path="dynamic_masker/checkpoints/Hydra_2025-04-06_10-35-01/model_epoch_64_best.pth"
    )
    validator.validate_model(
        save_path="dynamic_masker/results/Hydra_2025-04-06_10-35-01_plot/model_epoch_64_best"
    )
