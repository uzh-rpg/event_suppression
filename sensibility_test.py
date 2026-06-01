import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from matplotlib import pyplot as plt

from Validate_herm import ValidateHerm # Assuming this is correctly importable
from suppressor.metrics.MaskFlow import MaskFlow
from dynamic_masker.utils.train_log import binary_segmentation_losses
from suppressor.utils.utils import open_config_json


class SensibiliyTest(ValidateHerm):
    def __init__(self, config, model_path):
        super().__init__(config, model_path)

    
    def _get_dataset(self, window_ev_num, only_sequence=""):
        dataset = self.dataset_provider.get_evimo_test_dataset_by_event(window_ev_num=window_ev_num, only_sequence=only_sequence)
        return dataset

    def _evaluate_sequence(self, sequence, plot_path):
        plot_path_flow = plot_path / "flow"
        plot_path_flow.mkdir(parents=True, exist_ok=True)
        plot_path_segmentation = plot_path / "segmentation"
        plot_path_segmentation.mkdir(parents=True, exist_ok=True)

        iou_t0_sets, miou_t0_sets, piou_t0_sets = [], [], []
        iou_t1_sets, miou_t1_sets, piou_t1_sets = [], [], []
        
        num_events_list = [] # To store number of events for each frame

        maskflow = MaskFlow()
        total_vel = []
        total_br_to_object_ratio = []
        for ind in tqdm(range(len(sequence)-1), leave=False, desc="Eval Seq"): # Added leave=False, desc
            data_t0 = sequence[ind]
            data_t1 = sequence[ind+1]

            flows = maskflow.flow_calculation_with_lables(
                dynamic_mask_t1=data_t0["mask"].numpy().astype(int)
                )

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

            if flows is None:
                continue

            br_to_object_ratio = self.background_object_ratio(gt_mask_t0.cpu(), voxel.cpu())
            if br_to_object_ratio is None:
                continue
            
            # Calculate number of events from the input voxel
            # Sum of absolute values of all elements in the voxel representation
            current_num_events = torch.abs(voxel).sum().item()

            losses_mask_t0 = binary_segmentation_losses(mask_t0, gt_mask_t0, voxel, self.config)
            losses_mask_t1 = binary_segmentation_losses(mask_t1, gt_mask_t1, voxel, self.config)

            iou_t0_sets.append(losses_mask_t0["ious"]["IoU"])
            miou_t0_sets.append(losses_mask_t0["ious"]["mIoU"])
            piou_t0_sets.append(losses_mask_t0["ious"]["pIoU"])
            iou_t1_sets.append(losses_mask_t1["ious"]["IoU"])
            miou_t1_sets.append(losses_mask_t1["ious"]["mIoU"])
            piou_t1_sets.append(losses_mask_t1["ious"]["pIoU"])
            
            num_events_list.append(current_num_events) # Store number of events

            if br_to_object_ratio < 0.5:
                pass

            objects_vel = []
            for label in flows:
                current_flow = flows[label]["flow"]
                objects_vel.append(np.linalg.norm(current_flow))
            total_vel.append(np.nanmean(objects_vel))

            total_br_to_object_ratio.append(br_to_object_ratio)

        return {
            "IoU/t0": np.nanmean(np.array(iou_t0_sets) * 100, axis=0).tolist() if iou_t0_sets else [],
            "mIoU/t0": np.nanmean(np.array(miou_t0_sets) * 100).tolist() if miou_t0_sets else np.nan,
            "pIoU/t0": np.nanmean(np.array(piou_t0_sets) * 100).tolist() if piou_t0_sets else np.nan,
            "IoU/t1": np.nanmean(np.array(iou_t1_sets) * 100, axis=0).tolist() if iou_t1_sets else [],
            "mIoU/t1": np.nanmean(np.array(miou_t1_sets) * 100).tolist() if miou_t1_sets else np.nan,
            "pIoU/t1": np.nanmean(np.array(piou_t1_sets) * 100).tolist() if piou_t1_sets else np.nan,
            "velocities": total_vel,
            "mious": miou_t0_sets, # This is a list of frame-wise mIoU (not *100)
            "br_to_object_ratio": total_br_to_object_ratio,
            "num_events_frames": num_events_list, # Added list of event counts per frame
        }

    def measure_dataset_velocities(self, sequence, plot_path):
        plot_path_flow = plot_path / "flow"
        plot_path_flow.mkdir(parents=True, exist_ok=True)
        plot_path_segmentation = plot_path / "segmentation"
        plot_path_segmentation.mkdir(parents=True, exist_ok=True)

        maskflow = MaskFlow()
        total_vel = []
        for ind in tqdm(range(len(sequence)-1), leave=False, desc="Eval Seq"): # Added leave=False, desc
            data_t0 = sequence[ind]

            flows = maskflow.flow_calculation_with_lables(
                dynamic_mask_t1=data_t0["mask"].numpy().astype(int)
                )

            if flows is None:
                continue
            
            objects_vel = []
            for label in flows:
                current_flow = flows[label]["flow"]
                pixel_velocity = np.linalg.norm(current_flow)
                objects_vel.append(pixel_velocity)
            total_vel.append(np.nanmean(objects_vel))

        return {
            "velocities": total_vel,
        }
    
    @staticmethod
    def background_object_ratio(mask, events_voxel):
        # Weight mask by event locations
        events_weights = np.abs(events_voxel.numpy()).sum(axis=0)
        mask = mask.squeeze(0).numpy()
        mask = mask * events_weights

        mask = mask.flatten()
        background_pixels = np.sum(mask == 0)
        object_pixels = np.sum(mask != 0)

        if object_pixels == 0:
            return None
        else:
            return background_pixels / object_pixels
    
    @staticmethod
    def correlation_plot_general(columns, rows, column_name, rows_name, title, save_path):
        columns = np.array(columns)
        rows = np.array(rows) # Assumed to be in %

        plt.figure(figsize=(8, 6))
        plt.scatter(
            columns,
            rows,
            s=20,
            edgecolors='gray',
            facecolors='none',
            linewidth=0.8,
            label='Data points'
        )

        if len(columns) > 1 and len(rows) > 1:
            try:
                z = np.polyfit(columns, rows, 1)
                p = np.poly1d(z)
                sorted_indices = np.argsort(columns)
                plt.plot(columns[sorted_indices], p(columns[sorted_indices]), 'r--', label='Trend line')
            except (np.linalg.LinAlgError, ValueError) as e:
                print(f"Could not fit trend line for velocity vs IoU: {e}")
        
        plt.xlabel(column_name)
        plt.ylabel(rows_name)
        plt.title(title) # Simplified title
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def correlation_log_plot_general(columns, rows,  column_name, rows_name, title, save_path, measure_trend=False):
        columns = np.array(columns)
        rows = np.array(rows)  # Assumed to be in %
        
        # plt.rcParams['ps.useafm'] = True
        # plt.rcParams['pdf.use14corefonts'] = True
        # plt.rcParams['text.usetex'] = True

        plt.figure(figsize=(8, 6))
        plt.scatter(
            columns,
            rows,
            s=20,
            edgecolors='gray',
            facecolors='none',
            linewidth=0.8,
            label='Data points'
        )

        if measure_trend:
            try:
                # Fit a trend line in log-space for the x-axis
                log_num_events = np.log10(columns)
                z = np.polyfit(log_num_events, rows, 1)  # Linear fit in log-space
                p = np.poly1d(z)
                sorted_indices = np.argsort(columns)
                plt.plot(
                    columns[sorted_indices],
                    p(np.log10(columns[sorted_indices])),
                    'r--',
                    label='Trend line'
                )
            except (np.linalg.LinAlgError, ValueError) as e:
                print(f"Could not fit trend line for events vs IoU: {e}")

        plt.xlabel(column_name)
        plt.ylabel(rows_name)  # Using mIoU at t0, as a percentage
        if title:
            plt.title(title)
        plt.xscale('log')  # Apply logarithmic scale
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def correlation_plot_vel_iou(velocities, ious, save_path):
        velocities = np.array(velocities)
        ious = np.array(ious) # Assumed to be in %

        plt.figure(figsize=(8, 6))
        plt.scatter(
            velocities,
            ious,
            s=20,
            edgecolors='gray',
            facecolors='none',
            linewidth=0.8,
            label='Data points'
        )

        if len(velocities) > 1 and len(ious) > 1:
            try:
                z = np.polyfit(velocities, ious, 1)
                p = np.poly1d(z)
                sorted_indices = np.argsort(velocities)
                plt.plot(velocities[sorted_indices], p(velocities[sorted_indices]), 'r--', label='Trend line')
            except (np.linalg.LinAlgError, ValueError) as e:
                print(f"Could not fit trend line for velocity vs IoU: {e}")
        
        plt.xlabel('Velocity (px/100ms)')
        plt.ylabel('mIoU at t0 (%)') # Updated label
        plt.title('Sensitivity of Segmentation Performance to Object Velocity') # Simplified title
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    @staticmethod
    def correlation_plot_br_ratio_iou(br, ious, save_path):
        br = np.array(br)
        ious = np.array(ious) # Assumed to be in %

        plt.figure(figsize=(8, 6))
        plt.scatter(
            br,
            ious,
            s=20,
            edgecolors='gray',
            facecolors='none',
            linewidth=0.8,
            label='Data points'
        )

        if len(br) > 3 and len(ious) > 3: # polyfit degree 3 needs at least 4 points
            try:
                z = np.polyfit(br, ious, 3)
                p = np.poly1d(z)
                sorted_indices = np.argsort(br)
                plt.plot(br[sorted_indices], p(br[sorted_indices]), 'r--', label='Trend line')
            except (np.linalg.LinAlgError, ValueError) as e:
                print(f"Could not fit trend line for BR ratio vs IoU: {e}")

        plt.xlabel('Background to Objects Pixels Ratio')
        plt.ylabel('mIoU at t0 (%)') # Updated label
        plt.title('Sensitivity of Segmentation Performance to Object Size') # Simplified title
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def correlation_plot_events_iou(num_events, ious, save_path):
        num_events = np.array(num_events)
        ious = np.array(ious)  # Assumed to be in %
        
        
        plt.rcParams['ps.useafm'] = True
        plt.rcParams['pdf.use14corefonts'] = True
        plt.rcParams['text.usetex'] = True

        plt.figure(figsize=(8, 6))
        plt.scatter(
            num_events,
            ious,
            s=20,
            edgecolors='gray',
            facecolors='none',
            linewidth=0.8,
            label='Data points'
        )

        if len(num_events) > 1 and len(ious) > 1:
            try:
                # Fit a trend line in log-space for the x-axis
                log_num_events = np.log10(num_events)
                z = np.polyfit(log_num_events, ious, 1)  # Linear fit in log-space
                p = np.poly1d(z)
                sorted_indices = np.argsort(num_events)
                plt.plot(
                    num_events[sorted_indices],
                    p(np.log10(num_events[sorted_indices])),
                    'r--',
                    label='Trend line'
                )
            except (np.linalg.LinAlgError, ValueError) as e:
                print(f"Could not fit trend line for events vs IoU: {e}")

        plt.xlabel('Number of Events')
        plt.ylabel('mIoU(%)')  # Using mIoU at t0, as a percentage
        # plt.title('Sensitivity of Segmentation Performance to Number of Input Events')
        plt.xscale('log')  # Apply logarithmic scale
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def validate_model(self, save_path):
        self.dataset = self._get_dataset()
        self.model = self.setup_model(self.model_path)
        self.model.eval()

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        results = {}

        all_velocities = []
        all_mious_t0 = [] # Stores frame-wise mIoU (not *100) from stats["mious"]
        all_iou_t0 = [] # Stores frame-wise IoU at t0
        all_br_to_object_ratios = []
        all_num_events_frames = [] # Stores frame-wise event counts

        for sequence_idx, sequence in enumerate(tqdm(self.dataset, desc="Processing Sequences")):
            sequence_id = sequence[0]["sequence_id"] if sequence else f"unknown_sequence_{sequence_idx}"
            save_sequence_path = save_path / str(sequence_id)
            save_sequence_path.mkdir(exist_ok=True)
            plot_path = save_sequence_path / "plots_{}".format(sequence_id)
            # plot_path.mkdir(exist_ok=True) # _evaluate_sequence handles this

            stats = self._evaluate_sequence(sequence, plot_path)
            
            results[sequence_id] = stats
            logs = self._aggregate_logs(stats, sequence_id) # Ensure logs is initialized if stats is empty

            self._write_json(results, save_path / "results.json")

            if self.config["vis"]["verbose"]:
                self._print_stats(sequence_id, stats)

            self.model.reset_states()

            if stats["velocities"]: all_velocities.append(stats["velocities"])
            if stats["mious"]: all_mious_t0.append(stats["mious"]) # Appending list of mIoUs
            if stats["br_to_object_ratio"]: all_br_to_object_ratios.append(stats["br_to_object_ratio"])
            if stats["num_events_frames"]: all_num_events_frames.append(stats["num_events_frames"])
            if stats['IoU/t0'] and not(stats['IoU/t0'][0] == 0):
                all_iou_t0.append(stats['IoU/t0'][0])
        
        # Consolidate all frame-wise data
        # Ensure lists are not empty before concatenating
        velocities_flat = np.concatenate(all_velocities) if all_velocities else np.array([])
        mious_t0_flat = np.concatenate(all_mious_t0) if all_mious_t0 else np.array([])
        br_ratios_flat = np.concatenate(all_br_to_object_ratios) if all_br_to_object_ratios else np.array([])
        num_events_flat = np.concatenate(all_num_events_frames) if all_num_events_frames else np.array([])

        # Save raw data
        np.save(save_path /'velocities.npy', velocities_flat)
        np.save(save_path /'mious_t0_raw.npy', mious_t0_flat) # Saving raw mIoU values (0-1)
        np.save(save_path /'br_to_object_ratios.npy', br_ratios_flat)
        np.save(save_path /'num_events.npy', num_events_flat)

        # For plotting, mIoUs are typically presented as percentages
        mious_t0_percent = mious_t0_flat * 100

        # Filter and plot for velocity vs mIoU
        # Make sure mious_t0_percent has same length as filtered velocities_flat
        valid_vel_indices = velocities_flat < 60
        if velocities_flat[valid_vel_indices].size > 0 and mious_t0_percent[valid_vel_indices].size > 0 :
            self.correlation_plot_vel_iou(
                velocities_flat[valid_vel_indices], 
                mious_t0_percent[valid_vel_indices], 
                save_path / "correlation_plot_velocity_miou.png" # Renamed for clarity
            )
        
        # Plot for BR ratio vs mIoU
        if br_ratios_flat.size > 0 and mious_t0_percent.size == br_ratios_flat.size: # Ensure same length
             self.correlation_plot_br_ratio_iou(
                br_ratios_flat, mious_t0_percent, 
                save_path / "correlation_plot_br_ratio_miou.png" # Renamed for clarity
            )

        # Plot for Number of Events vs mIoU
        if num_events_flat.size > 0 and mious_t0_percent.size == num_events_flat.size: # Ensure same length
            self.correlation_plot_events_iou(
                num_events_flat,
                mious_t0_percent,
                save_path / "correlation_plot_events_miou.png"
            )

        aggregated = self._aggregate_results(results)
        if 'logs' not in locals(): # if all sequences were empty or skipped
            logs = {}
        logs = self._add_tot_to_logs(logs, aggregated)

        results["test/total"] = aggregated
        self._write_json(results, save_path / "results.json")
        return logs

    def validate_model_event_window(self, window_ev_num):
        all_mious_t0 = [] # Stores frame-wise mIoU (not *100) from stats["mious"]
        all_iou_t0 = [] # Stores frame-wise IoU at t0

        for sequence_idx, sequence in enumerate(tqdm(self.dataset, desc="Processing Sequences")):
            # if sequence_idx in [0, 1,2,3,4,5]:  # Skip the first two sequences as per original logic
                # continue  # Skip the first sequence as per original logic

            stats = self._evaluate_sequence(sequence, plot_path=Path('dynamic_masker/results/sensitivity_plots_new'))
            self.model.reset_states()

            if stats["mious"]: all_mious_t0.append(stats["mious"]) # Appending list of mIoUs
            if stats['IoU/t0'] and not(stats['IoU/t0'][0] == 0): all_iou_t0.append(stats['IoU/t0'][0])

            break
        mious_t0_flat = np.concatenate(all_mious_t0) if all_mious_t0 else np.array([])
        mious_t0_percent = mious_t0_flat * 100

        ious_t0_flat = np.array(all_iou_t0) if all_iou_t0 else np.array([])

        return mious_t0_percent.mean(), np.nanmean(ious_t0_flat)
    
    @staticmethod
    def correlation_plot_events_window_iou(num_events, ious, save_path):
        num_events = np.array(num_events)
        ious = np.array(ious) # Assumed to be in %

        plt.figure(figsize=(8, 6))
        plt.scatter(
            num_events,
            ious,
            s=20,
            edgecolors='gray',
            facecolors='none',
            linewidth=0.8,
            label='Data points'
        )
        
        if len(num_events) > 1 and len(ious) > 1:
            try:
                # Consider if data is too sparse or non-linear for a simple polyfit
                z = np.polyfit(num_events, ious, 1) # Linear trend line
                p = np.poly1d(z)
                sorted_indices = np.argsort(num_events)
                plt.plot(num_events[sorted_indices], p(num_events[sorted_indices]), 'r--', label='Trend line')
            except (np.linalg.LinAlgError, ValueError) as e:
                print(f"Could not fit trend line for events vs IoU: {e}")

        plt.xlabel('Number of Events')
        plt.ylabel('mIoU(%)') # Using mIoU at t0, as a percentage
        plt.title('Sensitivity of Segmentation Performance to Number of Input Events')
        # Optional: Use a log scale for x-axis if event counts vary widely
        # if np.min(num_events) > 0 and np.max(num_events) / np.min(num_events) > 100: # Heuristic for log scale
        plt.xscale('log')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
    def correlation_events_number_iou(self, save_path):
        self.model = self.setup_model(self.model_path)
        self.model.eval()

        # Logarithmic scale for better distribution
        event_windows = list(np.logspace(start=0, stop=6, num=100).astype(int)) 
        mious = []
        ious = []

        for window in tqdm(event_windows, desc="Processed event windows"):
            self.dataset = self._get_dataset(window_ev_num=window)
            miou, iou = self.validate_model_event_window(window)
            mious.append(miou)
            ious.append(iou)

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)  # <-- ADD THIS LINE

        ious = np.array(ious)
        np.save(save_path / 'ious_t0_raw.npy', ious)
        mious = np.array(mious)
        np.save(save_path / 'mious_t0_raw.npy', mious)

        self.correlation_plot_events_iou(
            event_windows,
            mious,
            save_path / "correlation_plot_events_window_miou.png"
        )

if __name__ == "__main__":

    config_path = "dynamic_masker/configs/train_herm.json" # Path to your config
    model_path = "checkpoints/Herm_2025-04-15_15-49-03/model_epoch_49.pth" # Path to your model
    save_path = "dynamic_masker/results/sensitivity_plots" # Output path

    try:
        config = open_config_json(config_path)
    except FileNotFoundError:
        print(f"Config file not found at {config_path}. Please ensure it exists.")
        # Create a dummy config for the script to run if needed for placeholder
        config = {"vis": {"plot": False, "verbose": True}, "dataset_params": {"name": "dummy_dataset"},
                  "model_params": {}, "dataloader_params": {}} # Adjust as per ValidateHerm needs
        print("Using a placeholder config.")


    validator = SensibiliyTest(config=config, model_path=model_path)
    print(f"Starting validation, saving results to: {save_path}")
    # validator.correlation_events_number_iou(save_path=save_path)
    # print(f"Validation finished. Plots saved in {save_path}")
    
    mious = np.load("dynamic_masker/results/sensitivity_plots_new/mious_t0_raw.npy")
    validator.correlation_plot_events_iou(
                num_events = list(np.logspace(start=0, stop=6, num=100).astype(int)),
                ious=mious,
                save_path="dynamic_masker/results/sensitivity_plots_new/correlation_plot_events_window_miou.pdf"
    )
