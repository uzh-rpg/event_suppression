from __future__ import annotations

import argparse
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

from dynamic_masker.config import get_device, load_config, save_config
from dynamic_masker.data import build_train_dataset
from dynamic_masker.loss.HermLoss import HermLoss
from dynamic_masker.loss.HydraLoss import HydraLoss
from dynamic_masker.models.model_hydra import HydraEVNet
from dynamic_masker.models.model_util import initialize_weights, label_smoothing
from dynamic_masker.utils.utils import custom_collate, load_model, load_optimizer_epoch_seen_samples, save_model


LOSS_REGISTRY = {
    "HydraLoss": HydraLoss,
    "HermLoss": HermLoss,
}


def seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


class EventSuppressorTrainer:
    def __init__(self, config: dict, checkpoint_path: str = ""):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.device = get_device(config.get("loader", {}).get("gpu"))
        self.model_name = self._model_name()
        self.model = None
        self.optimizer = None
        self.loss_function = None
        self.starting_epoch = -1
        self.total_seen_samples = 0

    def _model_name(self) -> str:
        if self.checkpoint_path:
            return Path(self.checkpoint_path).parent.name
        dataset = self.config["data"]["dataset"].upper()
        return f"EventSuppressor_{dataset}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    def build_model(self) -> HydraEVNet:
        model_config = self.config["model"].copy()
        model = HydraEVNet(
            kwargs=model_config,
            num_bins=self.config["data"].get("voxel_bins", 2),
            final_w_scale_flow=self.config["custom"].get("final_w_scale_flow", 0.01),
            current_flow_sup=self.config["custom"].get("current_flow_sup", False),
            current_flow_scaling=self.config["loader"].get("event_dt_ms", 50),
        ).to(self.device)

        if self.checkpoint_path:
            model = load_model(model, self.device, self.checkpoint_path)
        else:
            model.apply(initialize_weights)

        self.model = model.train()
        return self.model

    def build_optimizer(self) -> None:
        self.optimizer = AdamW(self.model.parameters(), lr=self.config["optimizer"]["lr"])
        self.optimizer, self.starting_epoch, self.total_seen_samples = load_optimizer_epoch_seen_samples(
            self.optimizer, self.device, self.checkpoint_path
        )

    def build_loss(self) -> None:
        loss_name = self.config["loss"].get("name", "HermLoss")
        self.loss_function = LOSS_REGISTRY[loss_name](self.config, device=self.device)

    def build_loader(self) -> torch.utils.data.DataLoader:
        dataset = build_train_dataset(self.config)
        loader = self.config["loader"]
        kwargs = {
            "dataset": dataset,
            "batch_size": loader.get("batch_size", 1),
            "shuffle": loader.get("shuffle", True),
            "drop_last": True,
            "num_workers": loader.get("n_workers", 0),
            "worker_init_fn": seed_worker,
            "pin_memory": torch.cuda.is_available(),
            "collate_fn": custom_collate,
        }
        if kwargs["num_workers"] > 0:
            kwargs["prefetch_factor"] = loader.get("prefetch_factor", 2)
        return torch.utils.data.DataLoader(**kwargs)

    def _dt(self, data_t1: dict) -> torch.Tensor:
        dt = data_t1["sampled_dt"]
        if not isinstance(dt, torch.Tensor):
            dt = torch.as_tensor(dt)
        return dt.float().to(self.device).view(-1, 1)

    def _events_to_device(self, value) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value
        else:
            tensor = torch.stack(value)
        return tensor.float().to(self.device).detach()

    def train_batch(self, batch_data: list[dict]) -> torch.Tensor:
        for data_ind in range(len(batch_data) - 1):
            data_t0 = batch_data[data_ind]
            data_t1 = batch_data[data_ind + 1]

            event_voxel = data_t0["representation"].float().to(self.device)
            dt = self._dt(data_t1)
            outputs = self.model(event_voxel, dt)

            gt_t0 = label_smoothing(data_t0["dynamic_mask"].float().to(self.device))
            gt_t1 = label_smoothing(data_t1["dynamic_mask"].float().to(self.device))

            update_kwargs = {
                "pred_mask_t0": outputs["mask"],
                "gt_mask_t0": gt_t0,
                "pred_mask_t1": outputs["future_mask"],
                "gt_mask_t1": gt_t1,
                "pred_flow_t1": outputs["flow"],
                "event_list": self._events_to_device(data_t1["event_list"]),
                "pol_mask": self._events_to_device(data_t1["polarity_mask"]),
                "d_event_list": self._events_to_device(data_t1["d_event_list"]),
                "d_pol_mask": self._events_to_device(data_t1["d_polarity_mask"]),
            }

            if "forward_flow_gt" in data_t0 and "has_flow" in data_t0:
                has_flow_t0 = data_t0["has_flow"].to(self.device)
                has_flow_t1 = data_t1.get("has_flow", has_flow_t0).to(self.device)
                flow_scale = self.config["loader"].get("event_dt_ms", 50) / self.config["loader"].get("optical_flow_dt_ms", 100)
                update_kwargs.update(
                    pred_flow_t0=outputs["flow_t0"],
                    gt_flow_t0=data_t0["forward_flow_gt"].float().to(self.device) * flow_scale,
                    mask_invalid_flows_t0=has_flow_t0,
                    gt_flow_t1=data_t1.get("forward_flow_gt", data_t0["forward_flow_gt"]).float().to(self.device),
                    mask_invalid_flows_t1=has_flow_t1,
                )

            self.loss_function.update(**update_kwargs)

        loss = self.loss_function()
        loss.backward()
        clip_grad = self.config["loss"].get("clip_grad")
        if clip_grad is not None:
            clip_grad_norm_(self.model.parameters(), clip_grad)
        return loss

    def train(self) -> str:
        seed_everything(self.config["loader"].get("seed"))
        self.build_model()
        self.build_optimizer()
        self.build_loss()
        train_loader = self.build_loader()

        print(f"Training {self.model_name} on {len(train_loader.dataset)} samples")
        last_model_path = ""
        for epoch in range(self.config["loader"].get("n_epochs", 1)):
            if epoch <= self.starting_epoch:
                continue
            for batch_idx, batch_data in enumerate(train_loader):
                max_batches = self.config["loader"].get("max_batches")
                if max_batches is not None and batch_idx >= max_batches:
                    break
                start = time.time()
                loss = self.train_batch(batch_data)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.model.detach_states()
                self.loss_function.reset()
                self.total_seen_samples += 1
                if self.config.get("vis", {}).get("verbose", True):
                    print(
                        f"epoch={epoch:04d} batch={batch_idx:05d}/{len(train_loader):05d} "
                        f"loss={loss.item():.6f} time={time.time() - start:.3f}s"
                    )

            self.model.reset_states()
            last_model_path = save_model(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                loss=loss,
                path_results=self.config["loader"].get("checkpoints_path", "checkpoints"),
                model_name=self.model_name,
                total_seen_samples=self.total_seen_samples,
            )
            save_config(self.config, Path(last_model_path).parent / "config.json")
        return last_model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Event Suppressor on DSEC or EVIMO.")
    parser.add_argument("--config", required=True, help="Path to a JSON training config.")
    parser.add_argument("--checkpoint", default="", help="Optional checkpoint to resume/fine-tune from.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    trainer = EventSuppressorTrainer(config=config, checkpoint_path=args.checkpoint)
    model_path = trainer.train()
    print(f"Saved checkpoint: {model_path}")


if __name__ == "__main__":
    main()
