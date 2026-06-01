import json
import torch
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

from utils.utils import load_model
from loss.IoUs import IoUs
from models.model import RecEVFlowNet
from configs.utils import get_device

from suppressor.DSEC_dataloader.provider import DatasetProvider
from suppressor.utils.utils import render_events
from loss.ClassificationLoss import ClassificationLoss
from tqdm import tqdm



def validate(config_path, model_path, save_path):
    config = json.load(open(config_path, 'r'))
    num_bins = config["data"]["voxel_bins"]
    representation = config["data"]["representation"]
    model_config = config["model"].copy()
    device = get_device(gpu_num=config["loader"]["gpu"])

    model = RecEVFlowNet(model_config, num_bins, key="dynamic_mask")
    model = model.to(device)
    model = load_model(model, device, model_dir=model_path)

    dataset_provider = DatasetProvider(dataset_path=config["data"]["path"], num_bins=num_bins, representation=representation)
    dataset = dataset_provider.get_test_dataset(FPS=10)

    save_path = Path(save_path)
    if not save_path.exists():
        save_path.mkdir()
    
    results = {}
    for sequence in tqdm(dataset):
        results_json_path = Path(save_path, "results.json")
        with open(results_json_path, 'w') as json_file:
            json.dump(results, json_file, indent=4)

        save_sequence_path = Path(save_path, sequence[0]["sequence_id"])
        if not save_sequence_path.exists():
            save_sequence_path.mkdir()
        
        iou_sets = []
        miou_sets = []
        piou_sets = []
        tps = []
        tns = []
        fps = []
        fns = []
        for ind, data in tqdm(enumerate(sequence)):
            event_voxel = data["representation"]["left"].to(device)

            with torch.no_grad():
                x = model(event_voxel.unsqueeze(0).to(device))
                pred = x["dynamic_mask"][-1].squeeze(0).squeeze(0)

            if data["dynamic_mask_gt"] is not None:
                sequence_id = data["sequence_id"]
                dynamic_mask_gt = data["dynamic_mask_gt"].to(device)
                
                ious = IoUs()(pred, dynamic_mask_gt, event_voxel)
                classification_losses = ClassificationLoss()
                classification_losses(pred, dynamic_mask_gt)

                iou_sets.append(ious["IoU"])
                miou_sets.append(ious["mIoU"])
                piou_sets.append(ious["pIoU"])
                tps.append(classification_losses.tp.cpu().numpy())
                tns.append(classification_losses.tn.cpu().numpy())
                fps.append(classification_losses.fp.cpu().numpy())
                fns.append(classification_losses.fn.cpu().numpy())

            # Plot images
            if config["vis"]["plot"]:
                plt.clf()
                # TODO here is an ERROR as the visualization should be on 
                # the prediction after passing from a sigmoid function
                fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                ax[0].imshow((pred>0.5).cpu().numpy())
                ax[0].axis("off")  # Hide axes
                ax[0].set_title("Predicted dynamic objects")

                ax[1].imshow(dynamic_mask_gt.squeeze().cpu().numpy())
                ax[1].axis("off")  # Hide axes
                ax[1].set_title("Ground Truth dynamic objects")

                plt.tight_layout()
                plot_file = Path(save_sequence_path, f"plots_{sequence_id}")
                if not plot_file.exists():
                    plot_file.mkdir()
                plt.savefig(plot_file / Path(f"{str(ind).zfill(6)}.png"))
                plt.close(fig)
                
        # reset the states after a sequence
        model.reset_states()

        # report values for each sequence
        ious_perc = np.array(iou_sets)*100
        miou_perc = np.array(miou_sets)*100
        piou_perc = np.array(piou_sets)*100
        tps = np.array(tps)
        tns = np.array(tns)
        fps = np.array(fps)
        fns = np.array(fns)

        results[sequence_id] = {
            "IoU": np.nanmean(ious_perc, axis=0).tolist(),
            "mIoU": np.nanmean(miou_perc).tolist(),
            "pIoU": np.nanmean(piou_perc).tolist(),
            "true_positives": np.nanmean(tps).tolist(),
            "true_negatives": np.nanmean(tns).tolist(),
            "false_positives": np.nanmean(fps).tolist(),
            "false_negatives": np.nanmean(fns).tolist(),
        }

        with open(results_json_path, 'w') as json_file:
            json.dump(results, json_file, indent=4)

        if config["vis"]["verbose"]:
            print(f"Sequence: {sequence_id}")
            print(f"Mean IoU %: {np.nanmean(ious_perc, axis=0)}")
            print(f"Mean mIoU %: {np.nanmean(miou_perc)}")
            print(f"Mean pIoU %: {np.nanmean(piou_perc)}")
            print(f"Mean True Positives px: {np.nanmean(tps)}")
            print(f"Mean True Negatives px: {np.nanmean(tns)}")
            print(f"Mean False Positives px: {np.nanmean(fps)}")
            print(f"Mean False Negatives px: {np.nanmean(fns)}")

    # report values for all sequences for the given model
    all_seq_ious = np.array([results[seq]["IoU"] for seq in results])
    all_seq_miou = np.array([results[seq]["mIoU"] for seq in results])
    all_seq_piou = np.array([results[seq]["pIoU"] for seq in results])
    tot_mean_iou = np.nanmean(all_seq_ious, axis=0)
    tot_mean_miou = np.nanmean(all_seq_miou)
    tot_mean_piou = np.nanmean(all_seq_piou)

    # Classification scores
    all_seq_tps = np.array([results[seq]["true_positives"] for seq in results])
    all_seq_tns = np.array([results[seq]["true_negatives"] for seq in results])
    all_seq_fps = np.array([results[seq]["false_positives"] for seq in results])
    all_seq_fns = np.array([results[seq]["false_negatives"] for seq in results])
    tot_mean_tps = np.nanmean(all_seq_tps)
    tot_mean_tns = np.nanmean(all_seq_tns)
    tot_mean_fps = np.nanmean(all_seq_fps)
    tot_mean_fns = np.nanmean(all_seq_fns)

    results["total"] = {
        "IoU": tot_mean_iou.tolist(),
        "mIoU": tot_mean_miou.tolist(),
        "pIoU": tot_mean_piou.tolist(),
        "true_positives": tot_mean_tps.tolist(),
        "true_negatives": tot_mean_tns.tolist(),
        "false_positives": tot_mean_fps.tolist(),
        "false_negatives": tot_mean_fns.tolist(),
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
    validate(
        config_path="configs/validate.json", 
        model_path="checkpoints/2025-02-17_17-04-37/model_epoch_2.pth",
        save_path="results/2025-02-17_17-04-37/model_epoch_2"
        )
    # validate_models(
    #     config_path="configs/validate.json", 
    #     models_dir="checkpoints/2025-03-03_09-51-29",
    #     save_path="results"
    #     )
