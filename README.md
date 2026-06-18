# Motion-aware Event Suppression for Event Cameras

[![Event Suppressor method overview](assets/method_new.png)](https://www.youtube.com/watch?v=mnY45jmwDC0)

This is the official PyTorch implementation of the RSS 2026 paper
[**Motion-aware Event Suppression for Event Cameras**](https://arxiv.org/abs/2602.23204).

## Citation

If you use any part of this code or datasets accompanying the paper please consider citing the following:

```bibtex
@inproceedings{Pellerito2026Suppression,
  title={Motion-aware Event Suppression for Event Cameras},
  author={Pellerito, Roberto and Messikommer, Nico and Cioffi, Giovanni and Cannici, Marco and Scaramuzza, Davide},
  booktitle={Robotics: Science and Systems 2026},
  year={2026}
}
```

## Info

This repository contains the essential training and validation code for dynamic object mask prediction from event-camera data. The public release focuses on:

- training on **DSEC**;
- training on **EVIMO v1**;
- validation on **EVIMO v1** at the current instant `t0` and future instant `t1`;
- validation entry point for **EED** at `t0` and `t1`.

Data loading is delegated to the external repository checked out at `ev-loader/`. The current `ev-loader` copy contains DSEC and EVIMO loaders. It does not currently expose an EED loader, so EED validation raises an explicit error until an `evloader.EED_dataloader.EEDSequence` implementation is added.

## Installation

Create a minimal conda environment and install the Python packages with `pip`:

```bash
conda create -n evsup python=3.10 -y
conda activate evsup
export PYTHONNOUSERSITE=1
```

Install PyTorch. NVIDIA drivers are backward-compatible with older CUDA runtimes, so a machine reporting CUDA 13.x through `nvidia-smi` can run the CUDA 12.1 PyTorch wheels. For CUDA-capable machines:

```bash
python -m pip install --no-cache-dir \
  torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121 \
  --extra-index-url https://pypi.org/simple
```

For CPU-only machines:

```bash
python -m pip install --no-cache-dir \
  torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple
```

Then install Event Suppressor:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip install pytest
```

PyTorch is intentionally not listed in `requirements.txt` because the correct wheel depends on your CUDA/CPU setup.

If importing PyTorch fails with `ImportError: libcudnn.so.9`, user-site packages are likely leaking into the conda environment. Keep `PYTHONNOUSERSITE=1` set and repair the PyTorch stack with:

```bash
python -m pip install --force-reinstall --no-cache-dir \
  torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121 \
  --extra-index-url https://pypi.org/simple

python -m pip show torch nvidia-cudnn-cu12 | grep -E 'Name|Version|Location'
```

The `Location` lines should point inside `$CONDA_PREFIX/lib/python3.10/site-packages`, not `~/.local/lib/python3.10/site-packages`.

Do not install `ev-loader` with `pip install -e ./ev-loader` unless you also want all of its optional loader and visualization dependencies. This repository imports `ev-loader` directly from the checked-out `./ev-loader` folder.

After installation, run:

```bash
python -m pytest -q
python train.py --help
python validate.py --help
```

## Repository Layout

```text
evsup/
  configs/
    train_dsec.json        # DSEC training config
    train_evimo.json       # EVIMO training config
    validate_evimo.json    # EVIMO t0/t1 validation config
    validate_eed.json      # EED t0/t1 validation config
  models/                  # Event Suppressor / Hydra recurrent U-Net
  loss/                    # Mask and event-warping losses
  data.py                  # Dataset builders backed by ev-loader
  training.py              # Training loop
  validation.py            # Validation loop
ev-loader/                 # External event-data loader repository
train.py                   # CLI wrapper
validate.py                # CLI wrapper
tests/                     # Public smoke/unit tests
```

## Dataset Structure

Set `data.path` in the JSON configs to the dataset root.

DSEC:

```text
DSEC/
  train/
    zurich_city_00_a/
    ...
  test/ or validation/
    ...
```

EVIMO v1 after conversion to HDF5:

```text
EVIMO1/
  train/
    box/
      seq_00.h5
      ...
  test/
    box/
      seq_00.h5
      ...
```

EED expected structure:

```text
EED/
  test/
    <sequence directories or files expected by the EED loader>
```

The EED structure depends on the missing `ev-loader` EED loader. Add that loader to `ev-loader/evloader/EED_dataloader` and keep the public validation command unchanged.

## Training

Edit the dataset path in the config first:

```json
"data": {
  "dataset": "evimo",
  "path": "/path/to/EVIMO1"
}
```

Train on EVIMO:

```bash
python train.py --config evsup/configs/train_evimo.json
```

Train on DSEC:

```bash
python train.py --config evsup/configs/train_dsec.json
```

Resume or fine-tune from a checkpoint:

```bash
python train.py \
  --config evsup/configs/train_evimo.json \
  --checkpoint checkpoints/EventSuppressor_EVIMO_<timestamp>/model_epoch_10.pth
```

Checkpoints are written under `loader.checkpoints_path`.

## Validation

Validate EVIMO at current and future instants:

```bash
python validate.py \
  --config evsup/configs/validate_evimo.json \
  --checkpoint checkpoints/EventSuppressor_EVIMO_<timestamp>/model_epoch_49.pth \
  --output results/evimo_model_epoch_49
```

Validate EED after adding the EED loader to `ev-loader`:

```bash
python validate.py \
  --config evsup/configs/validate_eed.json \
  --checkpoint checkpoints/EventSuppressor_EVIMO_<timestamp>/model_epoch_49.pth \
  --output results/eed_model_epoch_49
```

Validation writes `results.json` with per-sequence and aggregate metrics:

- `IoU/t0`, `mIoU/t0`, `pIoU/t0`, `SR@0.5/t0`;
- `IoU/t1`, `mIoU/t1`, `pIoU/t1`, `SR@0.5/t1`.

For short smoke runs, configs may include:

- `loader.max_batches`: stop training after this many batches per epoch;
- `eval.max_sequences`: validate only the first N sequences;
- `eval.max_samples`: validate only the first N pairs per sequence.

## Tests

```bash
python -m pytest -q
```

The tests cover public config loading, metric computation, public module imports, and the explicit EED-loader error.
