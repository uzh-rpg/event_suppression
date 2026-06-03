# Motion-aware Event Suppression for Event Cameras

[![Event Suppressor method overview](assets/method_new.png)](https://rpg.ifi.uzh.ch/event_suppression/)

Official public code for **Event Suppressor**, associated with the paper
[arXiv:2602.23204](https://arxiv.org/abs/2602.23204).

This repository contains the essential training and validation code for dynamic object mask prediction from event-camera data. The public release focuses on:

- training on **DSEC**;
- training on **EVIMO v1**;
- validation on **EVIMO v1** at the current instant `t0` and future instant `t1`;
- validation entry point for **EED** at `t0` and `t1`.

Data loading is delegated to the external repository checked out at `ev-loader/`. The current `ev-loader` copy contains DSEC and EVIMO loaders. It does not currently expose an EED loader, so EED validation raises an explicit error until an `evloader.EED_dataloader.EEDSequence` implementation is added.

## Installation

Using conda:

```bash
conda env create -f environment.yml
conda activate event-suppressor
```

Using venv:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Repository Layout

```text
dynamic_masker/
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
python train.py --config dynamic_masker/configs/train_evimo.json
```

Train on DSEC:

```bash
python train.py --config dynamic_masker/configs/train_dsec.json
```

Resume or fine-tune from a checkpoint:

```bash
python train.py \
  --config dynamic_masker/configs/train_evimo.json \
  --checkpoint checkpoints/EventSuppressor_EVIMO_<timestamp>/model_epoch_10.pth
```

Checkpoints are written under `loader.checkpoints_path`.

## Validation

Validate EVIMO at current and future instants:

```bash
python validate.py \
  --config dynamic_masker/configs/validate_evimo.json \
  --checkpoint checkpoints/EventSuppressor_EVIMO_<timestamp>/model_epoch_49.pth \
  --output results/evimo_model_epoch_49
```

Validate EED after adding the EED loader to `ev-loader`:

```bash
python validate.py \
  --config dynamic_masker/configs/validate_eed.json \
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
pytest -q
```

The tests cover public config loading, metric computation, public module imports, and the explicit EED-loader error.

## Citation

```bibtex
@article{dynamicmasker2026,
  title={Event Suppressor},
  author={Pellerito, Roberto and collaborators},
  journal={arXiv preprint arXiv:2602.23204},
  year={2026}
}
```
