import json
from pathlib import Path

import torch

from evsup.config import load_config
from evsup.metrics import mask_metrics, success_at


def test_public_configs_are_loadable():
    config_dir = Path("evsup/configs")
    for path in [
        config_dir / "train_dsec.json",
        config_dir / "train_evimo.json",
        config_dir / "validate_evimo.json",
        config_dir / "validate_eed.json",
    ]:
        config = load_config(path)
        assert config["data"]["dataset"]
        json.dumps(config)


def test_mask_metrics_perfect_prediction():
    target = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    logits = torch.tensor([[[20.0, -20.0], [-20.0, 20.0]]])
    voxel = torch.ones(2, 2, 2)

    metrics = mask_metrics(logits, target, voxel)

    assert metrics["IoU"] == [1.0, 1.0]
    assert metrics["mIoU"] == 1.0
    assert metrics["pIoU"] == 1.0
    assert success_at(metrics["IoU"][0]) == 1.0
