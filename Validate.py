import json
import torch
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from tqdm import tqdm

from dynamic_masker.utils.utils import load_model
from dynamic_masker.loss.IoUs import IoUs

from dynamic_masker.models.model import RecEVFlowNet

from suppressor.utils.utils import get_device
from suppressor.DSEC_dataloader.provider import DatasetProvider
from dynamic_masker.loss.ClassificationLoss import ClassificationLoss

def open_config_json(config_path):
    config = json.load(open(config_path, 'r'))
    return config

class Validate:
    def __init__(self, config, model_path, model_class=RecEVFlowNet, loss_class=ClassificationLoss):
        self.config = config
        self.device = get_device(gpu_num=self.config["loader"]["gpu"])
        self.num_bins = self.config["data"]["voxel_bins"]
        self.model_class = model_class
        self.loss_class = loss_class
        self.model_path = model_path

        self.dataset_provider = DatasetProvider(
            dataset_path=self.config["data"]["path"],
            num_bins=self.num_bins,
            delta_t_ms=self.config["loader"]["event_dt_ms"],
            representation=self.config["data"]["representation"]
        )
        self.iou_metric = IoUs()

    def setup_model(self, model_path):
        model = self.model_class(self.config["model"], self.num_bins, key="dynamic_mask")
        model = model.to(self.device)
        return load_model(model, self.device, model_dir=model_path)

    def _evaluate_sequence(self, sequence, plot_path):
        iou_sets, miou_sets, piou_sets = [], [], []
        tps, tns, fps, fns = [], [], [], []

        for ind, data in tqdm(enumerate(sequence)):
            voxel = data["representation"]["left"].to(self.device)
            with torch.no_grad():
                output = self.model(voxel.unsqueeze(0))
                pred = output["dynamic_mask"][-1].squeeze()

            if data["dynamic_mask_gt"] is not None:
                gt = data["dynamic_mask_gt"].to(self.device)
                ious = self.iou_metric(pred, gt, voxel)
                loss_eval = self.loss_class()
                loss_eval(pred, gt)

                iou_sets.append(ious["IoU"])
                miou_sets.append(ious["mIoU"])
                piou_sets.append(ious["pIoU"])
                tps.append(loss_eval.tp.cpu().numpy())
                tns.append(loss_eval.tn.cpu().numpy())
                fps.append(loss_eval.fp.cpu().numpy())
                fns.append(loss_eval.fn.cpu().numpy())

                if self.config["vis"]["plot"]:
                    self._plot_prediction(pred, gt, plot_path, ind)

        return {
            "IoU": np.nanmean(np.array(iou_sets) * 100, axis=0).tolist(),
            "mIoU": np.nanmean(np.array(miou_sets) * 100).tolist(),
            "pIoU": np.nanmean(np.array(piou_sets) * 100).tolist(),
            "true_positives": np.nanmean(tps).tolist(),
            "true_negatives": np.nanmean(tns).tolist(),
            "false_positives": np.nanmean(fps).tolist(),
            "false_negatives": np.nanmean(fns).tolist(),
        }

    def _plot_prediction(self, pred, gt, plot_path, ind):
        plt.clf()
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].imshow((pred > 0.5).cpu().numpy())
        ax[0].axis("off")
        ax[0].set_title("Predicted dynamic objects")

        ax[1].imshow(gt.squeeze().cpu().numpy())
        ax[1].axis("off")
        ax[1].set_title("Ground Truth dynamic objects")

        plt.tight_layout()
        plt.savefig(plot_path / f"{str(ind).zfill(6)}.png")
        plt.close(fig)

    def _aggregate_results(self, results):
        total = lambda key: np.nanmean([results[seq][key] for seq in results if seq != "total"], axis=0).tolist()
        return {
            "IoU/t0": total("IoU"),
            "mIoU": total("mIoU"),
            "pIoU": total("pIoU"),
            "true_positives": total("true_positives"),
            "true_negatives": total("true_negatives"),
            "false_positives": total("false_positives"),
            "false_negatives": total("false_negatives"),
        }

    def _print_stats(self, sequence_id, stats):
        print(f"Sequence: {sequence_id}")
        for k, v in stats.items():
            print(f"{k}: {v}")

    def _write_json(self, data, path):
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)

    def validate_multiple_models(self, models_dir, save_root):
        models_dir = Path(models_dir)
        save_root = Path(save_root) / models_dir.stem
        save_root.mkdir(parents=True, exist_ok=True)

        all_results = {}
        for model_path in tqdm(sorted(models_dir.iterdir())):
            print(f"Validating model: {model_path.name}")
            save_path = save_root / model_path.stem
            model_results = self.validate_model(model_path, save_path)
            all_results[model_path.stem] = model_results["total"]

        self._write_json(all_results, save_root / "all_results.json")

    def _get_dataset(self):
        return self.dataset_provider.get_test_dataset(FPS=10)

    def validate_model(self, save_path):
        """Validate the dynamic masker model on the test dataset

        Args:
            save_path (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.dataset = self._get_dataset()
        self.model = self.setup_model(self.model_path)

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

            self._write_json(results, save_path / "results.json")

            if self.config["vis"]["verbose"]:
                self._print_stats(sequence_id, stats)

            self.model.reset_states()

        results["total"] = self._aggregate_results(results)
        self._write_json(results, save_path / "results.json")
        return results

    
if __name__ == "__main__":
    config = open_config_json("dynamic_masker/configs/validate.json")
    validator = Validate(
        config=config, 
        model_path="dynamic_masker/checkpoints/2025-02-17_17-04-37/model_epoch_2.pth",
        )
    validator.validate_model(
        save_path="dynamic_masker/results/2025-02-17_17-04-37/model_epoch_2"
        )
    # validate_models(
    #     config_path="configs/validate.json", 
    #     models_dir="checkpoints/2025-03-03_09-51-29",
    #     save_path="results"
    #     )
