import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from loss.IoUs import IoUs
from loss.ClassificationLoss import ClassificationLoss
from utils.utils import load_model
from utils.visualization import Visualization

from models.model_hydra import HydraEVNet
from configs.utils import get_device

from suppressor.utils.representations import VoxelGrid
from suppressor.DSEC_dataloader.sequence import Sequence
from suppressor.Evimo2Loader import Evimo2Loader 


class EvalEvimo2:
    def __init__(self, config, model_path, loss_class=ClassificationLoss):
        self.config = config
        self.device = get_device(gpu_num=self.config["loader"]["gpu"])
        self.num_bins = self.config["data"]["voxel_bins"]
        self.loss_class = loss_class
        self.model_path = model_path
        self.dataset_path = self.config["data"]["path"]
        resolution = self.config["loader"]["resolution"]
        self.height = resolution[0]
        self.width = resolution[1]

        self.voxel_grid = VoxelGrid(self.num_bins, self.height, self.width, normalize=True)

        self.iou_metric = IoUs()

    def setup_model(self):
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
        return load_model(model, self.device, model_dir=self.model_path)
    
    def _events_to_voxel_grid(self, x, y, p, t):
        return Sequence.events_to_voxel_grid(self.voxel_grid, x, y, p, t)

    def evaluate_sequence(self, sequence_loader: Evimo2Loader, save_dir: Path):
        self.model = self.setup_model()
        save_dir.mkdir(parents=True, exist_ok=True)
        plot_path = save_dir / sequence_loader.sequence_name
        plot_path.mkdir(exist_ok=True)

        ious, mious, pious = [], [], []
        tps, tns, fps, fns = [], [], [], []

        for i in tqdm(range(1, len(sequence_loader))):
            data_t0 = sequence_loader[i]
            data_t1 = sequence_loader[i]

            x = data_t0["voxel"].to(self.device)
            flow = data_t1["flow"].to(self.device)
            mask = data_t1["mask"].to(self.device)
            dt = torch.tensor([100.0]).to(self.device)

            with torch.no_grad():
                output = self.model(x, dt)
                pred = output["mask"][-1].squeeze()

            loss_eval = self.loss_class()
            iou_result = self.iou_metric(pred, pred, x)

            loss_eval(pred, pred)

            ious.append(iou_result["IoU"])
            mious.append(iou_result["mIoU"])
            pious.append(iou_result["pIoU"])
            tps.append(loss_eval.tp.cpu().numpy())
            tns.append(loss_eval.tn.cpu().numpy())
            fps.append(loss_eval.fp.cpu().numpy())
            fns.append(loss_eval.fn.cpu().numpy())

            if self.config["vis"]["plot"]:
                self._plot_prediction(pred, mask, plot_path, i)

        return {
            "IoU": np.nanmean(np.array(ious) * 100, axis=0).tolist(),
            "mIoU": np.nanmean(np.array(mious) * 100).tolist(),
            "pIoU": np.nanmean(np.array(pious) * 100).tolist(),
            "true_positives": np.nanmean(tps).tolist(),
            "true_negatives": np.nanmean(tns).tolist(),
            "false_positives": np.nanmean(fps).tolist(),
            "false_negatives": np.nanmean(fns).tolist(),
        }

    def _plot_prediction(self, pred, gt, plot_path, idx):
        plt.clf()
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].imshow((pred > 0.5).cpu().numpy())
        ax[0].axis("off")
        ax[0].set_title("Predicted dynamic objects")

        ax[1].imshow(gt.squeeze().cpu().numpy())
        ax[1].axis("off")
        ax[1].set_title("Ground Truth dynamic objects")

        plt.tight_layout()
        plt.savefig(plot_path / f"{str(idx).zfill(6)}.png")
        plt.close(fig)

    def _write_json(self, data, path):
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
    
    def evaluate(self, sequence_name, save_dir):
        sequence_loader = Evimo2Loader(self.dataset_path, sequence_name, events_source="samsung", num_bins=self.num_bins)
        stats = self.evaluate_sequence(sequence_loader, Path(save_dir))
        print(f"Results for sequence {sequence_name}:")
        for k, v in stats.items():
            print(f"{k}: {v}")
        self._write_json(stats, Path(save_dir) / "results.json")
        return stats
    
    def __call__(self):
        pass



if __name__ == "__main__":
    with open("configs/validate_evimo2.json", "r") as f:
        config = json.load(f)

    evaluator = EvalEvimo2(
        config=config,
        model_path="checkpoints/Hydra_2025-03-27_12-33-07/model_epoch_6.pth"
    )
    evaluator.evaluate(
        sequence_name="scene13_dyn_test_00_000000", 
        save_dir="/home/rpg/Desktop/dynamic_masker/results/Hydra_2025-03-27_12-33-07/model_epoch_6"
        )
