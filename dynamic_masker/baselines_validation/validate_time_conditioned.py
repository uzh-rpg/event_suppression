import json
import torch
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

from utils.utils import load_model
from configs.utils import get_device

from suppressor.DSEC_dataloader.provider import DatasetProvider
from suppressor.utils.utils import render_events
from utils.visualization import Visualization
from tqdm import tqdm

from loss.L1Loss import L1Loss

from models.model_time_conditioned import TimeCondRecEVFlowNet
from loss.flow_classic_loss import epe_loss, npe_loss, mean_angular_error, charbonnier_loss


def validate(config_path, model_path, save_path):
    config = json.load(open(config_path, 'r'))
    num_bins = config["data"]["voxel_bins"]
    representation = config["data"]["representation"]
    model_config = config["model"].copy()
    device = get_device(gpu_num=config["loader"]["gpu"])

    model = TimeCondRecEVFlowNet(model_config, num_bins, key="future_OF")
    model = load_model(model, device, model_dir=model_path)
    model = model.to(device)

    dataset_provider = DatasetProvider(dataset_path=config["data"]["path"], num_bins=num_bins, representation=representation)
    dataset = dataset_provider.get_flow_test_dataset()

    save_path = Path(save_path)
    if not save_path.exists():
        save_path.mkdir()
    
    results_json_path = Path(save_path, "results.json")
    results = {}
    for sequence in tqdm(dataset):
        sequence_id = sequence[0]["sequence_id"]
        print(f"Validating sequence: {sequence_id}")

        save_sequence_path = Path(save_path, sequence_id)
        if not save_sequence_path.exists():
            save_sequence_path.mkdir()

        # Initialize metrics
        l1_loss_ = []
        EPE_ = []
        charb_ = []
        one_PE_ = []
        two_PE_ = []
        three_PE_ = []
        AE_ = []

        # basically we give the old event voxel but evaluate on the new optical flow
        # for future flow pred
        dt = torch.tensor([0.0]).to(device)
        predict_future_flow = False
        if dt.item() > 0:
            print(f"Predicting future flow at timestamp {dt.item()} ms")
            predict_future_flow = True

        for ind, data in tqdm(enumerate(sequence)):
            if ind == 0:
                continue

            if predict_future_flow:
                event_cnt = sequence[ind-1]["representation"]["left"].to(device)
            else:
                event_cnt = data["representation"]["left"].to(device)

            with torch.no_grad():
                x = model(event_cnt.unsqueeze(0).to(device), dt)

            # Compute metrics
            if "forward_flow_gt" in data:
                predicted_flow = x["future_OF"][-1].squeeze(0).to(device)
                forward_flow_gt = torch.from_numpy(data["forward_flow_gt"]).to(device)
                forward_flow_gt = forward_flow_gt.permute(2, 0, 1)

                l1_loss_.append(L1Loss()(predicted_flow, forward_flow_gt).cpu())
                EPE_.append(epe_loss(predicted_flow, forward_flow_gt).cpu())
                charb_.append(charbonnier_loss(predicted_flow, forward_flow_gt).cpu())
                one_PE_.append(npe_loss(predicted_flow, forward_flow_gt, threshold=1).cpu())
                two_PE_.append(npe_loss(predicted_flow, forward_flow_gt, threshold=2).cpu())
                three_PE_.append(npe_loss(predicted_flow, forward_flow_gt, threshold=3).cpu())
                AE_.append(mean_angular_error(predicted_flow, forward_flow_gt).cpu())
            
            # Plot images
            if config["vis"]["plot"]:
                plt.clf()

                flow_to_plot = x["future_OF"][-1][0].permute(1,2, 0).detach().cpu().numpy()
                rendered_flow = Visualization.flow_to_image(flow_to_plot)
                fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                ax[0].imshow(rendered_flow)
                ax[0].axis("off")  # Hide axes
                ax[0].set_title("Predicted Optical Flow")

                forward_flow = data['forward_flow_gt']
                rendered_flow_ = Visualization.flow_to_image(forward_flow)
                ax[1].imshow(rendered_flow_)
                ax[1].axis("off")  # Hide axes
                ax[1].set_title("Ground Truth Forward Flow")

                plt.tight_layout()
                sequence_id = data["sequence_id"]
                plot_file = Path(save_sequence_path, f"plots_{sequence_id}")
                if not plot_file.exists():
                    plot_file.mkdir()
                plt.savefig(plot_file / Path(f"{str(ind).zfill(6)}.png"))
                plt.close(fig)
            
        results[sequence_id] = {
            "l1_loss": np.mean(l1_loss_).astype(float),
            "EPE": np.mean(EPE_).astype(float),
            "charb": np.mean(charb_).astype(float),
            "one_PE": np.mean(one_PE_).astype(float),
            "two_PE": np.mean(two_PE_).astype(float),
            "three_PE": np.mean(three_PE_).astype(float),
            "AE": np.mean(AE_).astype(float)
        }

        # reset the states after a sequence
        model.reset_states()
    
    results['total'] = {
        "l1_loss": np.nanmean([results[seq]["l1_loss"] for seq in results]),
        "EPE": np.nanmean([results[seq]["EPE"] for seq in results]),
        "charb": np.nanmean([results[seq]["charb"] for seq in results]),
        "one_PE": np.nanmean([results[seq]["one_PE"] for seq in results]),
        "two_PE": np.nanmean([results[seq]["two_PE"] for seq in results]),
        "three_PE": np.nanmean([results[seq]["three_PE"] for seq in results]),
        "AE": np.nanmean([results[seq]["AE"] for seq in results])
    }
    with open(results_json_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)
    
    return results


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
    #     config_path="configs/train_time_cond.json", 
    #     model_path="checkpoints/TamingCM 2025-03-14_08-53-18/model_epoch_0.pth",
    #     save_path="results"
    #     )
    validate_models(
        config_path="configs/train_time_cond.json", 
        models_dir="checkpoints/TimeCond_2025-03-18_09-22-45",
        save_path="results"
        )
