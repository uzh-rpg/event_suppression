import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import json
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from dynamic_masker.utils.utils import load_model
from dynamic_masker.models.model_hydra import HydraEVNet
from Validate import Validate, open_config_json
from Validate_hydra import ValidateHydra
from dynamic_masker.utils.train_log import binary_segmentation_losses


def interpolate_mask(gt_mask_t0, gt_mask_t1, dt, delta_t_ms=100):
    alpha = dt / delta_t_ms
    alpha = min(max(alpha, 0.0), 1.0)  # Clamp to [0, 1]
    # Interpolate ground truth masks
    interpolated_gt_mask = (1 - alpha) * gt_mask_t0 + alpha * gt_mask_t1
    return interpolated_gt_mask

class ValidateHydraMultiStep(ValidateHydra):
    def __init__(self, config, model_path):
        super().__init__(config, model_path)
        # self.dt_range = list(range(1, 100+1, 10))  # 1 to 100ms prediction
        self.dt_range = list(range(10, 100+1, 10))  # 1 to 100ms prediction

    def _get_dataset(self):
        return self.dataset_provider.get_hydra_test_dataset()

    def validate_multistep_iou(self, save_path):
        self.dataset = self._get_dataset()
        self.model = self.setup_model(self.model_path)
        self.model.eval()

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        all_dt_ious = {dt: [] for dt in self.dt_range}

        for dt in tqdm(self.dt_range, desc='dt'):
            for sequence in tqdm(self.dataset, desc="Sequences"):
                for ind in tqdm(range(len(sequence) - 1), desc="samples"):
                    data_t0 = sequence[ind]
                    data_t1 = sequence[ind+1]
                    voxel = data_t0["representation"]["left"].to(self.device)
                    gt_mask_t0 = data_t0["dynamic_mask"].to(self.device)
                    gt_mask_t1 = data_t1["dynamic_mask"].to(self.device)

                    gt_mask_tx = interpolate_mask(gt_mask_t0=gt_mask_t0, gt_mask_t1=gt_mask_t1, dt=dt)
                    with torch.no_grad():
                        output = self.model(x=voxel.unsqueeze(0), dt=torch.tensor([dt], device=self.device))
                        pred_mask_tx = output["future_mask"][-1].squeeze(0)

                    losses = binary_segmentation_losses(pred_mask_tx, gt_mask_tx, voxel, self.config)
                    iou = losses["ious"]["IoU"]
                    all_dt_ious[dt].append(iou)

                    self.model.reset_states()
                break

        # Save per-dt results in separate folders
        for dt in self.dt_range:
            dt_folder = save_path / f"dt_{dt}ms"
            dt_folder.mkdir(parents=True, exist_ok=True)
            mean_iou = float(np.nanmean(all_dt_ious[dt]) * 100)
            with open(dt_folder / "iou.json", "w") as f:
                json.dump({"IoU": mean_iou}, f, indent=4)

        print(f"Validation complete. Per-dt IoU results saved under: {save_path}")
        return {dt: float(np.nanmean(all_dt_ious[dt]) * 100) for dt in self.dt_range}



if __name__ == "__main__":
    config = open_config_json("dynamic_masker/configs/validate_HighFreqHydra.json")
    validator = ValidateHydraMultiStep(
        config=config,
        model_path="checkpoints/HighFreq_2025-06-24_14-59-34/model_epoch_1.pth"
    )
    validator.validate_multistep_iou(
        save_path=Path("results/HighFreq_MultiStepEval")
    )
