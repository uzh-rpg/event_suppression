import argparse
import math
import os
from datetime import datetime

import torch
from torch.utils.data import Subset

from TrainBaseHydra import TrainBaseHydra
from Validate_eed import ValidateEED
from dynamic_masker.loss.HermLoss import HermLoss
from dynamic_masker.models.model_util import label_smoothing
from suppressor.DSEC_dataloader.provider import DatasetProvider


class TrainHermEED(TrainBaseHydra):
    """
    Fine-tune a Herm/Hydra checkpoint on the EED dataset.

    This trainer keeps the standard HydraEVNet model and Herm loss, but swaps the
    dataloader and validation pipeline to the EED dataset, which provides dynamic
    masks and event lists without supervised flow ground truth.
    """

    def init_model_name(self):
        if self.checkpoint_path:
            return os.path.basename(os.path.dirname(self.checkpoint_path))
        return "HermEED_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def _align_temporal_loss_config(self):
        available_passes = int(self.config["data"]["sequence_len"])
        requested_passes = int(self.config["data"]["passes_loss"])

        if available_passes < 1:
            raise ValueError("EED training requires data.sequence_len >= 1.")

        if requested_passes > available_passes:
            print(
                f"Adjusting data.passes_loss from {requested_passes} to {available_passes} "
                "to match the number of available sequence transitions."
            )
            self.config["data"]["passes_loss"] = available_passes
            requested_passes = available_passes

        requested_scales = int(self.config["data"]["scales_loss"])
        max_scales = max(1, int(math.floor(math.log2(requested_passes))) + 1)
        if requested_scales > max_scales:
            print(
                f"Adjusting data.scales_loss from {requested_scales} to {max_scales} "
                f"because passes_loss={requested_passes} only supports {max_scales} temporal scales."
            )
            self.config["data"]["scales_loss"] = max_scales

    def build_dataloader(self):
        print("Building dataloader for EED dataset...")
        provider = DatasetProvider(
            dataset_path=self.config["data"]["path"],
            representation=self.config["data"]["representation"],
            num_bins=self.config["data"]["voxel_bins"],
            delta_t_ms=self.config["loader"]["event_dt_ms"],
        )

        eed_config = self.config.get("eed", {})
        train_dataset = provider.get_eed_dataset(
            sequence_len=self.config["data"]["sequence_len"],
            batch_size=self.config["loader"]["batch_size"],
            augment=self.config["loader"].get("augment", []),
            augment_prob=self.config["loader"].get("augment_prob", []),
            max_num_grad_events=self.config["loader"].get("max_num_grad_events", 20000),
            max_num_detach_events=self.config["loader"].get("max_num_detach_events", 20000),
            event_filter=eed_config.get("event_filter"),
            filter_events_to_dynamic_mask=eed_config.get("filter_events_to_dynamic_mask", False),
        )

        required_steps = int(self.config["data"]["sequence_len"]) + 1
        valid_indices = [idx for idx in range(len(train_dataset)) if len(train_dataset[idx]) == required_steps]
        if len(valid_indices) != len(train_dataset):
            print(
                f"Keeping {len(valid_indices)}/{len(train_dataset)} EED samples with "
                f"the full temporal window of {required_steps} steps."
            )
        train_dataset = Subset(train_dataset, valid_indices)

        n_workers = int(self.config["loader"]["n_workers"])
        loader_kwargs = dict(
            dataset=train_dataset,
            drop_last=True,
            batch_size=self.config["loader"]["batch_size"],
            shuffle=self.config["loader"]["shuffle"],
            num_workers=n_workers,
            worker_init_fn=self.seed_worker,
            pin_memory=True,
        )
        if n_workers > 0:
            loader_kwargs["prefetch_factor"] = self.config["loader"]["prefetch_factor"]

        return torch.utils.data.DataLoader(**loader_kwargs)

    def build_loss_function(self):
        self._align_temporal_loss_config()
        self.loss_function = HermLoss(self.config, device=self.device)

    def train_batch(self, batch_data):
        loss_sequence = torch.tensor(0.0, device=self.device)
        last_outputs = None
        last_event_voxel = None
        last_mask_t0 = None
        last_mask_t1 = None

        for data_ind in range(len(batch_data) - 1):
            data_t0 = batch_data[data_ind]
            data_t1 = batch_data[data_ind + 1]

            event_voxel = data_t0["representation"].to(self.device)
            dt = data_t1["sampled_dt"].float().to(self.device)

            outputs = self.model(event_voxel, dt)

            dynamic_mask_t0_gt = label_smoothing(data_t0["dynamic_mask"].to(self.device))
            dynamic_mask_t1_gt = label_smoothing(data_t1["dynamic_mask"].to(self.device))

            self.loss_function.update(
                pred_mask_t0=outputs["mask"],
                gt_mask_t0=dynamic_mask_t0_gt,
                pred_mask_t1=outputs["future_mask"],
                gt_mask_t1=dynamic_mask_t1_gt,
                pred_flow_t1=outputs["flow"],
                event_list=data_t1["event_list"].float().to(self.device).detach(),
                pol_mask=data_t1["polarity_mask"].float().to(self.device).detach(),
                d_event_list=data_t1["d_event_list"].float().to(self.device).detach(),
                d_pol_mask=data_t1["d_polarity_mask"].float().to(self.device).detach(),
            )

            last_outputs = outputs
            last_event_voxel = event_voxel
            last_mask_t0 = dynamic_mask_t0_gt
            last_mask_t1 = dynamic_mask_t1_gt

        loss_sequence = self.loss_function()
        loss_sequence.backward()

        if self.config["loss"]["clip_grad"] is not None:
            torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), self.config["loss"]["clip_grad"])

        if self.config["vis"]["log_wandb"] and last_outputs is not None:
            self.log_wandb(
                mask=last_outputs["mask"],
                dynamic_mask_t0_gt=last_mask_t0,
                future_mask=last_outputs["future_mask"],
                dynamic_mask_t1_gt=last_mask_t1,
                event_voxel=last_event_voxel,
                flow_t1=last_outputs["flow"],
                flow_gt_t1=None,
            )

        return loss_sequence

    def run_validation(self, model_dir, epoch):
        validation_frequency = max(int(self.config["loader"].get("validation_frequency", 1)), 1)
        if epoch % validation_frequency != 0:
            return

        validator = ValidateEED(config=self.config, model_path=model_dir)
        results_root = self.config["loader"].get("results_path", "dynamic_masker/results/EED_finetune")
        save_path = f"{results_root}/{self.model_name}/model_epoch_{epoch}"
        results = validator.validate_model(save_path=save_path)

        if self.config["vis"]["log_wandb"] and self.run:
            self.run.log(results, step=self.total_seen_samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="dynamic_masker/configs/train_herm_eed.json",
        help="Training configuration for EED fine-tuning.",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/checkpoints_prerebuttal/Herm_2025-04-15_15-49-03/model_epoch_11.pth",
        help="Checkpoint to resume or fine-tune from.",
    )
    args = parser.parse_args()

    trainer = TrainHermEED(config_path=args.config, checkpoint_path=args.checkpoint)
    trainer.train()
    trainer.finish()


if __name__ == "__main__":
    main()
