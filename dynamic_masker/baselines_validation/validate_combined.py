import json
import torch
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt
from pathlib import Path

from utils.utils import load_model
from loss.IoUs import IoUs
from models.model import RecEVFlowNet
from models.model_time_conditioned import TimeCondRecEVFlowNet
from configs.utils import get_device

from suppressor.DSEC_dataloader.provider import DatasetProvider
from suppressor.utils.utils import render_events
from loss.ClassificationLoss import ClassificationLoss
from tqdm import tqdm

from utils.visualization import Visualization
from old_training_scripts.train_time_conditioned import plot_events_from_data

import torch.jit
import torch.nn as nn

def get_config(config_path):
    config = json.load(open(config_path, 'r'))
    return config

def get_device_from_path(config_path):
    config = json.load(open(config_path, 'r'))
    device = get_device(gpu_num=config["loader"]["gpu"])
    return device

def get_model_dynamic_mask(config_path, model_path):
    config = json.load(open(config_path, 'r'))
    num_bins = config["data"]["voxel_bins"]
    model_config = config["model"].copy()

    model = RecEVFlowNet(model_config, num_bins, key="dynamic_mask")
    model = load_model(model, 'cpu', model_dir=model_path)
    return model

def get_dataset_dynamic_mask(config_path):
    config = json.load(open(config_path, 'r'))
    num_bins = config["data"]["voxel_bins"]
    representation = config["data"]["representation"]
    dataset_path = config["data"]["path"]

    dataset_provider = DatasetProvider(dataset_path=dataset_path, num_bins=num_bins, representation=representation)
    dataset = dataset_provider.get_test_dataset(FPS=10)
    return dataset

def get_model_optical_flow(config_path, model_path):
    config = json.load(open(config_path, 'r'))
    num_bins = config["data"]["voxel_bins"]
    model_config = config["model"].copy()

    model = TimeCondRecEVFlowNet(model_config, num_bins, key="future_OF")
    model = load_model(model, 'cpu', model_dir=model_path)
    return model

def get_dataset_optical_flow(config_path):
    config = json.load(open(config_path, 'r'))
    num_bins = config["data"]["voxel_bins"]
    representation = config["data"]["representation"]
    dataset_path = config["data"]["path"]

    dataset_provider = DatasetProvider(dataset_path=dataset_path, num_bins=num_bins, representation=representation)
    dataset = dataset_provider.get_flow_test_dataset()
    return dataset

def high_frequency_future_mask_prediction(model_dm, model_of, event_voxel, event_cnt, device):
    """_description_

    Args:
        model_dm (): Dynamica mask model
        model_of (): Optical flow model
        event_voxel (torch.Tensor): voxelised inputs for dynamic mask prediction Size: [B=1, C=2, H, W]
        event_cnt (torch.Tensor): Event count image for optical flow prediction Size: [B=1, C=2, H, W]
        device (str): Device to run the models on
    return (torch.Tensor): Predicted dynamic mask
    """

    event_voxel = event_voxel.unsqueeze(0).to(device)
    event_cnt = event_cnt.unsqueeze(0).to(device)
    dt=torch.tensor([50.0]).to(device)

    # dm = model_dm(event_voxel)
    # of = model_of(event_cnt, dt=dt)

    # Run the two models in parallel
    handle_dm = torch.jit.fork(model_dm, event_voxel)
    handle_of = torch.jit.fork(model_of, event_cnt, dt=dt)
    dm = torch.jit.wait(handle_dm)
    of = torch.jit.wait(handle_of)

    # Post process the outputs
    pred_dm = dm["dynamic_mask"][-1].squeeze(0).squeeze(0)
    dynamic_mask = (torch.sigmoid(pred_dm) > 0.5)

    pred_of = of["future_OF"][-1].squeeze(0)

    if dynamic_mask.sum() == 0:
        return dynamic_mask, dynamic_mask, pred_of

    # Generate normalized coordinate grid
    C, H, W = pred_of.shape
    y_coords, x_coords = torch.meshgrid(
                torch.linspace(-1, 1, H), 
                torch.linspace(-1, 1, W), 
                indexing="ij"
                )
    grid = torch.stack((x_coords, y_coords), dim=0).to(dynamic_mask.device)  # Shape: (2, H, W)

    flow_norm = pred_of.clone()
    flow_norm[0, :, :] /= (W / 2)  # Normalize x displacement
    flow_norm[1, :, :] /= (H / 2)  # Normalize y displacement

    future_grid = grid + flow_norm  # Shape: (2, H, W)
    future_grid = future_grid.permute(1, 2, 0).unsqueeze(0)  # Shape: (B, H, W, 2)
    dynamic_mask = dynamic_mask.unsqueeze(0).unsqueeze(0).float()

    future_mask = F.grid_sample(dynamic_mask, future_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    
    future_mask = future_mask.squeeze(0).squeeze(0)
    dynamic_mask = dynamic_mask.squeeze(0).squeeze(0)
    return future_mask, dynamic_mask, pred_of


def validate(config_path_dm, config_path_of, model_path_dm, model_path_of, save_path):
    device = get_device_from_path(config_path_dm)

    model_dm = get_model_dynamic_mask(config_path_dm, model_path_dm)
    dataset_dm = get_dataset_dynamic_mask(config_path_dm)

    model_of = get_model_optical_flow(config_path_of, model_path_of)
    dataset_of = get_dataset_optical_flow(config_path_of)

    model_dm = model_dm.to(device)
    model_of = model_of.to(device)

    config = get_config(config_path_dm)

    save_path = Path(save_path)
    if not save_path.exists():
        save_path.mkdir()
    
    results = {}
    for sequence_dm, sequence_of in tqdm(zip(dataset_dm, dataset_of)):
        results_json_path = Path(save_path, "results.json")
        with open(results_json_path, 'w') as json_file:
            json.dump(results, json_file, indent=4)

        sequence_id = sequence_dm[0]["sequence_id"]
        save_sequence_path = Path(save_path, sequence_id)
        if not save_sequence_path.exists():
            save_sequence_path.mkdir()

        iou_sets = []
        miou_sets = []
        piou_sets = []
        for ind, (data_dm, data_of) in tqdm(enumerate(zip(sequence_dm, sequence_of))):
            if ind+1 >= len(sequence_dm.timestamps):
                continue

            event_voxel = data_dm["representation"]["left"]
            event_cnt = data_of["representation"]["left"]

            with torch.no_grad():
                future_mask, dynamic_mask, pred_of = high_frequency_future_mask_prediction(
                    model_dm=model_dm, 
                    model_of=model_of, 
                    event_voxel=event_voxel, 
                    event_cnt=event_cnt, 
                    device=device
                    )
                future_mask = future_mask.cpu()
                pred_of = pred_of.cpu()


            if data_dm["dynamic_mask_gt"] is not None:
                sequence_id = data_dm["sequence_id"]
                next_data_sample = sequence_dm[ind+1]
                dynamic_mask_gt = next_data_sample["dynamic_mask_gt"]
                
                ious = IoUs()(future_mask, dynamic_mask_gt, event_voxel)
                classification_losses = ClassificationLoss()
                classification_losses(future_mask, dynamic_mask_gt)

                iou_sets.append(ious["IoU"])
                miou_sets.append(ious["mIoU"])
                piou_sets.append(ious["pIoU"])

            # Plot images
            if config["vis"]["plot"]:

                plt.clf()
                fig, ax = plt.subplots(2, 2, figsize=(24, 20))

                # Plot events and dynamic mask on top 
                rendered_events = plot_events_from_data(data_of, config)
                next_data_sample = sequence_dm[ind+1]
                dynamic_mask_gt = next_data_sample["dynamic_mask_gt"].squeeze(0)
                
                # Set colors: Green for ground truth, Red for prediction, yellow for overlap
                overlay = plot_gt_pred_dynamic_mask(dynamic_mask_gt, future_mask)

                ax[0, 0].imshow(rendered_events)
                ax[0, 0].imshow(overlay, alpha=0.5)
                ax[0, 0].axis("off")  # Hide axes
                ax[0, 0].set_title("Gt (green) Predicted (Red) Future dynamic objects")

                # Plot current dynamic mask and predicted GT
                dynamic_mask_gt = data_dm["dynamic_mask_gt"].squeeze(0)
                dynamic_mask = dynamic_mask.cpu()
                overlay_ = plot_gt_pred_dynamic_mask(dynamic_mask_gt, dynamic_mask)

                ax[0, 1].imshow(rendered_events)
                ax[0, 1].imshow(overlay_, alpha=0.5)
                ax[0, 1].axis("off")
                ax[0, 1].set_title("Gt (green) Predicted (Red) Current dynamic objects")

                flow_to_plot = pred_of.permute(1, 2, 0).detach().cpu().numpy()
                rendered_flow = Visualization.flow_to_image(flow_to_plot)

                ax[1, 0].imshow(rendered_flow)
                ax[1, 0].axis("off")  # Hide axes
                ax[1, 0].set_title("Predicted Optical Flow in the dt = 50ms")

                if "forward_flow_gt" in data_of:
                    forward_flow_gt = data_of["forward_flow_gt"]
                    rendered_flow = Visualization.flow_to_image(forward_flow_gt)

                    ax[1, 1].imshow(rendered_flow)
                    ax[1, 1].axis("off")
                    ax[1, 1].set_title("Ground Truth Optical Flow in dt = 0ms")
                
                plt.tight_layout()
                plot_file = Path(save_sequence_path, f"plots_{sequence_id}")
                if not plot_file.exists():
                    plot_file.mkdir()
                plt.savefig(plot_file / Path(f"{str(ind).zfill(6)}.png"))
                plt.close(fig)
                
        # reset the states after a sequence
        model_dm.reset_states()
        model_of.reset_states()

        # report values for each sequence
        ious_perc = np.array(iou_sets)*100
        miou_perc = np.array(miou_sets)*100
        piou_perc = np.array(piou_sets)*100

        results[sequence_id] = {
            "IoU": np.nanmean(ious_perc, axis=0).tolist(),
            "mIoU": np.nanmean(miou_perc).tolist(),
            "pIoU": np.nanmean(piou_perc).tolist(),
        }

        with open(results_json_path, 'w') as json_file:
            json.dump(results, json_file, indent=4)

        if config["vis"]["verbose"]:
            print(f"Sequence: {sequence_id}")
            print(f"Mean IoU %: {np.nanmean(ious_perc, axis=0)}")
            print(f"Mean mIoU %: {np.nanmean(miou_perc)}")
            print(f"Mean pIoU %: {np.nanmean(piou_perc)}")

    # report values for all sequences for the given model
    all_seq_ious = np.array([results[seq]["IoU"] for seq in results])
    all_seq_miou = np.array([results[seq]["mIoU"] for seq in results])
    all_seq_piou = np.array([results[seq]["pIoU"] for seq in results])
    tot_mean_iou = np.nanmean(all_seq_ious, axis=0)
    tot_mean_miou = np.nanmean(all_seq_miou)
    tot_mean_piou = np.nanmean(all_seq_piou)

    results["total"] = {
        "IoU": tot_mean_iou.tolist(),
        "mIoU": tot_mean_miou.tolist(),
        "pIoU": tot_mean_piou.tolist(),
    }
    with open(results_json_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)
    
    return results

def plot_gt_pred_dynamic_mask(dynamic_mask_gt, dynamic_mask):
    H, W = dynamic_mask_gt.shape
    overlay = np.zeros((H, W, 3), dtype=np.uint8)
    overlay[dynamic_mask_gt == 1] = [0, 255, 0]   # Green
    overlay[dynamic_mask == 1] = [255, 0, 0]  # Red
    overlay[(dynamic_mask_gt == 1) & (dynamic_mask == 1)] = [255, 255, 0]  # Yellow for overlap
    return overlay

def validate_models(config_path, models_dir, save_path):
    models_dir = Path(models_dir)
    general_save_path = Path(save_path, models_dir.stem)
    if not general_save_path.exists():
        general_save_path.mkdir(parents=True)
    
    all_results = {}
    for model_path in sorted(models_dir.iterdir()):
        print(f"Validating model: {model_path}")
        save_path = Path(general_save_path, model_path.stem)
        model_results = validate(config_path=config_path, model_path=model_path, save_path=save_path)
        all_results[model_path.stem] = model_results["total"]
        print(f"Model {model_path} validated and results saved at {save_path}")

    with open(Path(general_save_path, "all_results.json"), 'w') as json_file:
        json.dump(all_results, json_file, indent=4)

# TODO implement separate logging for validation
def write_report(results, save_path):
    pass

if __name__ == "__main__":
    # validate(
    #     config_path_dm="configs/validate.json", 
    #     config_path_of="configs/train_time_cond.json", 
    #     model_path_dm="checkpoints/2025-02-17_17-04-37/model_epoch_2.pth", 
    #     model_path_of="checkpoints/2025-03-03_09-51-29/model_epoch_3.pth",
    #     save_path="results_combined"
    #     )
    # validate_models(
    #     config_path="configs/validate.json", 
    #     models_dir="checkpoints/2025-03-03_09-51-29",
    #     save_path="results"
    #     )
    validate(
        config_path_dm="configs/validate.json", 
        config_path_of="configs/train_time_cond.json", 
        model_path_dm="checkpoints/2025-02-17_17-04-37/model_epoch_2.pth", 
        model_path_of="checkpoints/TimeCond_2025-03-18_09-22-45/model_epoch_0.pth",
        save_path="results_combined"
        )
