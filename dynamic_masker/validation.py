from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from dynamic_masker.config import get_device, load_config
from dynamic_masker.data import build_validation_sequences
from dynamic_masker.metrics import mask_metrics, nanmean_percent, success_at
from dynamic_masker.models.model_hydra import HydraEVNet
from dynamic_masker.utils.utils import load_model


class EventSuppressorValidator:
    def __init__(self, config: dict, model_path: str):
        self.config = config
        self.model_path = model_path
        self.device = get_device(config.get("loader", {}).get("gpu"))
        self.threshold = float(config.get("eval", {}).get("mask_threshold", 0.5))
        self.model = self._setup_model()

    def _setup_model(self) -> HydraEVNet:
        model = HydraEVNet(
            kwargs=self.config["model"].copy(),
            num_bins=self.config["data"].get("voxel_bins", 2),
            final_w_scale_flow=self.config["custom"].get("final_w_scale_flow", 0.01),
            current_flow_sup=self.config["custom"].get("current_flow_sup", False),
            current_flow_scaling=self.config["loader"].get("event_dt_ms", 50),
        ).to(self.device)
        model = load_model(model, self.device, self.model_path)
        model.eval()
        return model

    def _run_pair(self, data_t0: dict, data_t1: dict) -> tuple[dict, dict]:
        voxel = data_t0["representation"].float().to(self.device)
        dt = data_t1["sampled_dt"]
        if not isinstance(dt, torch.Tensor):
            dt = torch.as_tensor(dt)
        dt = dt.float().to(self.device).view(1, 1)

        with torch.no_grad():
            output = self.model(voxel.unsqueeze(0), dt)
            mask_t0 = output["mask"][-1].squeeze(0)
            mask_t1 = output["future_mask"][-1].squeeze(0)

        metrics_t0 = mask_metrics(
            pred_logits=mask_t0,
            target=data_t0["dynamic_mask"].float().to(self.device),
            event_voxel=voxel,
            threshold=self.threshold,
        )
        metrics_t1 = mask_metrics(
            pred_logits=mask_t1,
            target=data_t1["dynamic_mask"].float().to(self.device),
            event_voxel=voxel,
            threshold=self.threshold,
        )
        return metrics_t0, metrics_t1

    def _evaluate_sequence(self, sequence) -> dict:
        iou_t0, miou_t0, piou_t0, sr_t0 = [], [], [], []
        iou_t1, miou_t1, piou_t1, sr_t1 = [], [], [], []

        max_samples = self.config.get("eval", {}).get("max_samples")
        num_pairs = len(sequence) - 1
        if max_samples is not None:
            num_pairs = min(num_pairs, int(max_samples))

        for index in tqdm(range(num_pairs), leave=False):
            data_t0 = sequence[index]
            data_t1 = sequence[index + 1]
            metrics_t0, metrics_t1 = self._run_pair(data_t0, data_t1)

            iou_t0.append(metrics_t0["IoU"])
            miou_t0.append(metrics_t0["mIoU"])
            piou_t0.append(metrics_t0["pIoU"])
            sr_t0.append(success_at(metrics_t0["IoU"][0]))

            iou_t1.append(metrics_t1["IoU"])
            miou_t1.append(metrics_t1["mIoU"])
            piou_t1.append(metrics_t1["pIoU"])
            sr_t1.append(success_at(metrics_t1["IoU"][0]))

        return {
            "IoU/t0": nanmean_percent(iou_t0),
            "mIoU/t0": nanmean_percent(miou_t0),
            "pIoU/t0": nanmean_percent(piou_t0),
            "SR@0.5/t0": nanmean_percent(sr_t0),
            "IoU/t1": nanmean_percent(iou_t1),
            "mIoU/t1": nanmean_percent(miou_t1),
            "pIoU/t1": nanmean_percent(piou_t1),
            "SR@0.5/t1": nanmean_percent(sr_t1),
        }

    @staticmethod
    def _sequence_id(sequence) -> str:
        if hasattr(sequence, "sequence_id"):
            return str(sequence.sequence_id)
        first = sequence[0]
        return str(first.get("sequence_id", "sequence"))

    @staticmethod
    def _aggregate(results: dict[str, dict]) -> dict:
        keys = next(iter(results.values())).keys()
        return {
            key: nanmean_percent([np.asarray(stats[key], dtype=float) / 100.0 for stats in results.values()])
            for key in keys
        }

    def validate(self, save_path: str | Path) -> dict:
        sequences = build_validation_sequences(self.config)
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        results = {}
        max_sequences = self.config.get("eval", {}).get("max_sequences")
        if max_sequences is not None:
            sequences = sequences[: int(max_sequences)]

        for sequence in tqdm(sequences):
            sequence_id = self._sequence_id(sequence)
            results[sequence_id] = self._evaluate_sequence(sequence)
            self.model.reset_states()
            self._write(results, save_path / "results.json")

        results["test/total"] = self._aggregate(results)
        self._write(results, save_path / "results.json")
        return results

    @staticmethod
    def _write(results: dict, path: Path) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
            handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Event Suppressor on EVIMO or EED.")
    parser.add_argument("--config", required=True, help="Path to a JSON validation config.")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint to evaluate.")
    parser.add_argument("--output", required=True, help="Directory where results.json is written.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validator = EventSuppressorValidator(load_config(args.config), args.checkpoint)
    results = validator.validate(args.output)
    print(json.dumps(results["test/total"], indent=2))


if __name__ == "__main__":
    main()
