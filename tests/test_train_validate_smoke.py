import json
from pathlib import Path

import torch

from dynamic_masker.training import EventSuppressorTrainer
from dynamic_masker.validation import EventSuppressorValidator


def _base_config(tmp_path):
    return {
        "data": {
            "dataset": "synthetic",
            "voxel_bins": 2,
            "scales_loss": 1,
            "passes_loss": 2,
        },
        "model": {
            "final_w_scale": 1.0,
            "num_encoders": 2,
            "num_residual_blocks": 1,
        },
        "custom": {
            "final_w_scale_flow": 0.01,
            "current_flow_sup": False,
        },
        "loss": {
            "name": "SmokeLoss",
            "clip_grad": None,
            "bce_weight": 1.0,
            "dice_weight": 0.0,
            "class_weights": 1.0,
            "lambda_future_mask": 1.0,
            "warping": "Iterative",
            "iterative_mode": "two",
            "round_ts": False,
            "flow_spat_smooth_weight": None,
            "flow_temp_smooth_weight": None,
        },
        "optimizer": {"lr": 1e-4},
        "loader": {
            "checkpoints_path": str(tmp_path / "checkpoints"),
            "n_epochs": 1,
            "batch_size": 1,
            "n_workers": 0,
            "resolution": [32, 32],
            "gpu": None,
            "seed": 1,
            "event_dt_ms": 50,
            "optical_flow_dt_ms": 100,
            "shuffle": False,
            "max_batches": 1,
        },
        "eval": {
            "mask_threshold": 0.5,
            "max_sequences": 1,
            "max_samples": 1,
        },
        "vis": {"verbose": False},
    }


def _sample(offset=0.0):
    mask = torch.zeros(1, 32, 32)
    mask[:, 8:24, 8:24] = 1.0
    return {
        "representation": torch.full((2, 32, 32), offset + 1.0),
        "dynamic_mask": mask,
        "sampled_dt": torch.tensor([0.05]),
        "event_list": torch.zeros(2, 4),
        "polarity_mask": torch.zeros(2, 2),
        "d_event_list": torch.zeros(2, 4),
        "d_polarity_mask": torch.zeros(2, 2),
    }


class SyntheticTrainDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return [_sample(0.0), _sample(1.0)]


class SyntheticValidationSequence:
    sequence_id = "synthetic_sequence"

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return _sample(float(index))


class SmokeLoss(torch.nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.loss = torch.tensor(0.0, device=device)

    def update(self, pred_mask_t0, gt_mask_t0, pred_mask_t1, gt_mask_t1, **kwargs):
        self.loss = self.loss + torch.nn.functional.binary_cross_entropy_with_logits(
            pred_mask_t0[-1], gt_mask_t0
        )
        self.loss = self.loss + torch.nn.functional.binary_cross_entropy_with_logits(
            pred_mask_t1[-1], gt_mask_t1
        )

    def forward(self):
        return self.loss

    def reset(self):
        self.loss = torch.tensor(0.0, device=self.device)


def test_training_loop_writes_checkpoint(monkeypatch, tmp_path):
    import dynamic_masker.training as training

    monkeypatch.setattr(training, "build_train_dataset", lambda config: SyntheticTrainDataset())
    monkeypatch.setitem(training.LOSS_REGISTRY, "SmokeLoss", SmokeLoss)

    trainer = EventSuppressorTrainer(_base_config(tmp_path))
    checkpoint_path = Path(trainer.train())

    assert checkpoint_path.is_file()
    assert checkpoint_path.name == "model_epoch_0.pth"
    assert (checkpoint_path.parent / "config.json").is_file()
    assert trainer.total_seen_samples == 1


def test_validation_loop_writes_results(monkeypatch, tmp_path):
    import dynamic_masker.validation as validation

    monkeypatch.setattr(
        validation,
        "build_validation_sequences",
        lambda config: [SyntheticValidationSequence()],
    )

    config = _base_config(tmp_path)
    model = validation.HydraEVNet(
        kwargs=config["model"].copy(),
        num_bins=config["data"]["voxel_bins"],
        final_w_scale_flow=config["custom"]["final_w_scale_flow"],
        current_flow_scaling=config["loader"]["event_dt_ms"],
    )
    checkpoint_path = tmp_path / "model.pth"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    output_dir = tmp_path / "validation"
    validator = EventSuppressorValidator(config, str(checkpoint_path))
    results = validator.validate(output_dir)

    results_path = output_dir / "results.json"
    assert results_path.is_file()
    stored_results = json.loads(results_path.read_text())
    assert "synthetic_sequence" in stored_results
    assert "test/total" in stored_results
    for key in ["IoU/t0", "mIoU/t0", "pIoU/t0", "IoU/t1", "mIoU/t1", "pIoU/t1"]:
        assert key in results["test/total"]
