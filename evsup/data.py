from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import torch


def ensure_evloader_importable(repo_root: str | Path | None = None) -> None:
    root = Path(repo_root or Path(__file__).resolve().parents[1])
    ev_loader = root / "ev-loader"
    if ev_loader.is_dir() and str(ev_loader) not in sys.path:
        sys.path.insert(0, str(ev_loader))


def _concat_or_raise(datasets: list[torch.utils.data.Dataset], split_path: Path) -> torch.utils.data.Dataset:
    if not datasets:
        raise FileNotFoundError(f"No loadable sequences were found in {split_path}")
    return torch.utils.data.ConcatDataset(datasets)


def _h5_files(split_path: Path) -> Iterable[Path]:
    yield from sorted(split_path.glob("*/*.h5"))
    yield from sorted(split_path.glob("*/*.hdf5"))


def build_dsec_train_dataset(config: dict) -> torch.utils.data.Dataset:
    ensure_evloader_importable()
    from evloader.DSEC_dataloader.HydraSequence import HydraSequence

    data = config["data"]
    loader = config["loader"]
    train_path = Path(data["path"]) / "train"
    if not train_path.is_dir():
        raise FileNotFoundError(f"DSEC train split not found: {train_path}")

    datasets = []
    for sequence_path in sorted(p for p in train_path.iterdir() if p.is_dir()):
        datasets.append(
            HydraSequence(
                seq_path=sequence_path,
                mode="train",
                delta_t_ms=loader.get("event_dt_ms", 50),
                num_bins=data.get("voxel_bins", 2),
                sequence_len=data.get("sequence_len", 5),
                representation=data.get("representation", "voxel"),
                max_num_grad_events=loader.get("max_num_grad_events"),
                dt=data.get("dt_ms", [100, 100]),
                augment=loader.get("augment", []),
                augment_prob=loader.get("augment_prob", []),
                multiple_batches=loader.get("batch_size", 1) > 1,
            )
        )
    return _concat_or_raise(datasets, train_path)


def build_evimo_train_dataset(config: dict) -> torch.utils.data.Dataset:
    ensure_evloader_importable()
    from evloader.EVIMOv1_dataloader.EVIMOSequenceRandByNumber import EVIMOSequenceRandByNumber

    data = config["data"]
    loader = config["loader"]
    train_path = Path(data["path"]) / "train"
    if not train_path.is_dir():
        raise FileNotFoundError(f"EVIMO train split not found: {train_path}")

    datasets = []
    for h5_path in _h5_files(train_path):
        datasets.append(
            EVIMOSequenceRandByNumber(
                h5_path=str(h5_path),
                window_ms=loader.get("event_dt_ms", 50),
                num_bins=data.get("voxel_bins", 2),
                sequence_len=data.get("sequence_len", 5),
                batch_size=loader.get("batch_size", 1),
                augment=loader.get("augment", []),
                augment_prob=loader.get("augment_prob", []),
            )
        )
    return _concat_or_raise(datasets, train_path)


def build_evimo_validation_sequences(config: dict) -> list[torch.utils.data.Dataset]:
    ensure_evloader_importable()
    from evloader.EVIMOv1_dataloader.EVIMOTestSequence import EVIMOTestSequence

    data = config["data"]
    split = data.get("split", "test")
    split_path = Path(data["path"]) / split
    if not split_path.is_dir():
        raise FileNotFoundError(f"EVIMO {split} split not found: {split_path}")

    sequences = [
        EVIMOTestSequence(
            h5_path=str(h5_path),
            window_ms=config["loader"].get("event_dt_ms", 50),
            num_bins=data.get("voxel_bins", 2),
            representation=data.get("representation", "voxel"),
        )
        for h5_path in _h5_files(split_path)
    ]
    if not sequences:
        raise FileNotFoundError(f"No EVIMO .h5/.hdf5 sequences found in {split_path}")
    return sequences


def build_eed_validation_sequences(config: dict) -> list[torch.utils.data.Dataset]:
    ensure_evloader_importable()
    try:
        from evloader.EED_dataloader import EEDSequence  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "EED validation is configured, but this ev-loader checkout does not provide "
            "an EED_dataloader/EEDSequence implementation. Please add the EED loader to "
            "ev-loader or share its expected API."
        ) from exc

    data = config["data"]
    split_path = Path(data["path"]) / data.get("split", "test")
    return [EEDSequence(path) for path in sorted(split_path.iterdir()) if path.is_dir()]


def build_train_dataset(config: dict) -> torch.utils.data.Dataset:
    dataset_name = config["data"]["dataset"].lower()
    if dataset_name == "dsec":
        return build_dsec_train_dataset(config)
    if dataset_name == "evimo":
        return build_evimo_train_dataset(config)
    raise ValueError(f"Unsupported training dataset: {dataset_name}")


def build_validation_sequences(config: dict) -> list[torch.utils.data.Dataset]:
    dataset_name = config["data"]["dataset"].lower()
    if dataset_name == "evimo":
        return build_evimo_validation_sequences(config)
    if dataset_name == "eed":
        return build_eed_validation_sequences(config)
    raise ValueError(f"Unsupported validation dataset: {dataset_name}")
