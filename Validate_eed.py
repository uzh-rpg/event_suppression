import json
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from evlicious import Events, RenderingType
from tqdm import tqdm

from Validate import open_config_json
from Validate_hydra import ValidateHydra


class ValidateEED(ValidateHydra):
    def __init__(self, config, model_path):
        super().__init__(config, model_path)

    def _get_dataset(self):
        dataset = self.dataset_provider.get_eed_dataset(sequence_len=1, batch_size=1, filter_events_to_dynamic_mask=False)
        return dataset.datasets

    def _aggregate_results(self, results):
        total = lambda key: np.nanmean([results[seq][key] for seq in results if seq != "test/total"], axis=0).tolist()
        return {
            "IoU/t0": total("IoU/t0"),
            "mIoU/t0": total("mIoU/t0"),
            "pIoU/t0": total("pIoU/t0"),
            "IoU/t1": total("IoU/t1"),
            "mIoU/t1": total("mIoU/t1"),
            "pIoU/t1": total("pIoU/t1"),
            "SR@0.5/t0": total("SR@0.5/t0"),
            "SR@0.5/t1": total("SR@0.5/t1"),
        }

    @staticmethod
    def _success_from_iou(iou_value, threshold=0.5):
        if np.isnan(iou_value):
            return np.nan
        return float(iou_value > threshold)

    def _mask_threshold(self):
        return float(self.config.get("eval", {}).get("mask_threshold", 0.5))

    @staticmethod
    def _events_from_index(sequence, idx):
        id0 = sequence.start_ev_ind[idx]
        id1 = sequence.end_ev_ind[idx]
        if id1 <= id0:
            return np.zeros((0, 4), dtype=np.float32)

        timestamps_us = (sequence.events_t[id0:id1] - sequence.events_t[id0]) * 1e6
        return np.stack(
            [
                sequence.events_x[id0:id1].astype(np.int64),
                sequence.events_y[id0:id1].astype(np.int64),
                timestamps_us.astype(np.int64),
                sequence.events_p[id0:id1].astype(np.int8),
            ],
            axis=1,
        )

    @staticmethod
    def _to_rgb(image):
        image = np.asarray(image)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        image = image.astype(np.float32)
        if image.max() > 1.0:
            image /= 255.0
        return np.clip(image, 0.0, 1.0)

    @staticmethod
    def _mask_to_bool(mask, threshold=0.5):
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        mask = np.asarray(mask)
        if mask.ndim == 3:
            mask = mask[0]
        return mask > threshold

    @staticmethod
    def _prediction_to_bool(pred, threshold=0.5):
        pred_mask = torch.sigmoid(pred).detach().cpu()
        if pred_mask.ndim == 3:
            pred_mask = pred_mask[0]
        return pred_mask.numpy() > threshold

    def _iou_metrics(self, pred, target, event_voxel):
        mask_threshold = self._mask_threshold()

        pred_probs = torch.sigmoid(pred)
        pred_mask_1 = pred_probs > mask_threshold
        target_mask_1 = target > mask_threshold

        pred_mask_0 = ~pred_mask_1
        target_mask_0 = ~target_mask_1

        iou_1 = self.iou_metric.iou(pred_mask_1, target_mask_1)
        iou_0 = self.iou_metric.iou(pred_mask_0, target_mask_0)
        miou = torch.nanmean(torch.stack([iou_0, iou_1])).item()

        event_mask = torch.any(event_voxel != 0, dim=0)
        piou = self.iou_metric.piou(pred_mask_1, target_mask_1, event_mask).item()

        return {
            "IoU": [iou_1.item(), iou_0.item()],
            "mIoU": miou,
            "pIoU": piou,
            "IoU%": [iou_1.item() * 100, iou_0.item() * 100],
            "mIoU%": miou * 100,
            "pIoU%": piou * 100,
        }

    @staticmethod
    def _overlay_mask(base_image, mask, color, alpha=0.35):
        base_rgb = ValidateEED._to_rgb(base_image)
        mask = ValidateEED._mask_to_bool(mask)
        color = np.asarray(color, dtype=np.float32)

        overlay = base_rgb.copy()
        overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color
        return np.clip(overlay, 0.0, 1.0)

    @staticmethod
    def _sample_spatial_shape(*items):
        for item in items:
            if isinstance(item, torch.Tensor):
                shape = tuple(item.shape)
            else:
                shape = tuple(np.asarray(item).shape)

            if len(shape) >= 2:
                return int(shape[-2]), int(shape[-1])

        raise ValueError("Could not infer the spatial shape for EED visualization.")

    @staticmethod
    def _render_events(events, height, width):
        canvas = np.ones((height, width, 3), dtype=np.float32)
        if len(events) == 0:
            return canvas

        events = np.asarray(events)
        x = events[:, 0].astype(np.int64)
        y = events[:, 1].astype(np.int64)
        t = events[:, 2].astype(np.int64)
        p = events[:, 3].astype(np.int8)

        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        x = x[valid]
        y = y[valid]
        t = t[valid]
        p = p[valid]
        if len(x) == 0:
            return canvas

        ev_window = Events(
            x=x.astype(np.uint16),
            y=y.astype(np.uint16),
            t=t.astype(np.int64),
            p=p.astype(np.int8),
            width=width,
            height=height,
        )
        rendered = ev_window.render(rendering_type=RenderingType.RED_BLUE_NO_OVERLAP)
        return ValidateEED._to_rgb(rendered)

    @staticmethod
    def _comparison_image(pred_mask, gt_mask):
        pred_mask = ValidateEED._mask_to_bool(pred_mask)
        gt_mask = ValidateEED._mask_to_bool(gt_mask)

        comparison = np.ones((*pred_mask.shape, 3), dtype=np.float32)
        comparison[pred_mask & gt_mask] = np.array([0.15, 0.75, 0.15], dtype=np.float32)
        comparison[pred_mask & ~gt_mask] = np.array([1.0, 0.65, 0.0], dtype=np.float32)
        comparison[~pred_mask & gt_mask] = np.array([0.85, 0.15, 0.15], dtype=np.float32)
        return comparison

    def _plot_alignment(self, sequence, data_t0, data_t1, mask_t0, mask_t1, plot_path, ind, iou_t0, iou_t1):
        plot_dir = Path(plot_path) / "alignment"
        plot_dir.mkdir(parents=True, exist_ok=True)

        mask_threshold = self._mask_threshold()
        gt_t0 = self._mask_to_bool(data_t0["dynamic_mask"], threshold=mask_threshold)
        gt_t1 = self._mask_to_bool(data_t1["dynamic_mask"], threshold=mask_threshold)
        pred_t0 = self._prediction_to_bool(mask_t0, threshold=mask_threshold)
        pred_t1 = self._prediction_to_bool(mask_t1, threshold=mask_threshold)

        height, width = self._sample_spatial_shape(data_t0["dynamic_mask"], data_t0["representation"])
        target_events = self._events_from_index(sequence, ind)
        eval_events = self._events_from_index(sequence, ind + 1)

        target_events_image = self._render_events(target_events, height, width)
        eval_events_image = self._render_events(eval_events, height, width)

        target_events_gt = self._overlay_mask(target_events_image, gt_t0, color=(0.0, 0.85, 0.2))
        target_events_pred_t0 = self._overlay_mask(target_events_image, pred_t0, color=(1.0, 0.55, 0.0))
        eval_events_gt = self._overlay_mask(eval_events_image, gt_t1, color=(0.0, 0.85, 0.2))
        eval_events_pred_t1 = self._overlay_mask(eval_events_image, pred_t1, color=(0.15, 0.35, 1.0))
        comparison_t0 = self._comparison_image(pred_t0, gt_t0)
        comparison_t1 = self._comparison_image(pred_t1, gt_t1)

        target_bbox_ts = float(sequence.bbox_timestamps[ind])
        future_bbox_ts = float(sequence.bbox_timestamps[ind + 1])
        dt_ms = (future_bbox_ts - target_bbox_ts) * 1000.0
        iou_t0_text = "nan" if np.isnan(iou_t0) else f"{iou_t0 * 100:.2f}%"
        iou_t1_text = "nan" if np.isnan(iou_t1) else f"{iou_t1 * 100:.2f}%"

        fig, axes = plt.subplots(2, 4, figsize=(24, 12))
        axes = axes.ravel()

        panels = [
            (target_events_image, f"Target events to model\n{len(target_events)} events | {height}x{width}"),
            (target_events_gt, "Target events + GT t0"),
            (target_events_pred_t0, f"Target events + pred t0\nIoU={iou_t0_text}"),
            (comparison_t0, "Pred t0 vs GT t0"),
            (eval_events_image, f"Eval events\n{len(eval_events)} events | delta={dt_ms:.2f} ms"),
            (eval_events_gt, "Eval events + GT t1"),
            (eval_events_pred_t1, f"Eval events + pred t1\nIoU={iou_t1_text}"),
            (comparison_t1, "Pred t1 vs GT t1"),
        ]

        for ax, (image, title) in zip(axes, panels):
            ax.imshow(image)
            ax.set_title(title)
            ax.axis("off")

        fig.text(0.5, 0.01, "green=TP, orange=FP, red=FN", ha="center", fontsize=10)

        fig.suptitle(
            (
                f"{sequence.sequence_id} | sample={ind:06d} | "
                f"t0={target_bbox_ts:.6f} | t1={future_bbox_ts:.6f} | "
                f"mask_thr={mask_threshold:.2f}"
            ),
            fontsize=14,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(plot_dir / f"{str(ind).zfill(6)}.png", dpi=200)
        plt.close(fig)

    def _evaluate_sequence(self, sequence, plot_path):
        plot_path_segmentation = plot_path / "segmentation"
        plot_path_segmentation.mkdir(parents=True, exist_ok=True)

        iou_t0_sets, miou_t0_sets, piou_t0_sets = [], [], []
        iou_t1_sets, miou_t1_sets, piou_t1_sets = [], [], []
        success_t0_sets, success_t1_sets = [], []

        self.model = self.model.to(self.device)

        for ind in tqdm(range(len(sequence) - 1)):
            data_t0 = sequence.get_single_item(ind)
            data_t1 = sequence.get_single_item(ind + 1)

            voxel = data_t0["representation"].to(self.device)
            dt = data_t1["sampled_dt"].to(self.device)
            with torch.no_grad():
                output = self.model(x=voxel.unsqueeze(0), dt=dt)
                mask_t0 = output["mask"][-1].squeeze(0)
                mask_t1 = output["future_mask"][-1].squeeze(0)

                gt_mask_t0 = data_t0["dynamic_mask"].to(self.device)
                gt_mask_t1 = data_t1["dynamic_mask"].to(self.device)

            metrics_t0 = self._iou_metrics(mask_t0, gt_mask_t0, voxel)
            metrics_t1 = self._iou_metrics(mask_t1, gt_mask_t1, voxel)

            iou_dyn_t0 = metrics_t0["IoU"][0]
            iou_dyn_t1 = metrics_t1["IoU"][0]

            if self.config["vis"]["plot"]:
                self._plot_alignment(
                    sequence=sequence,
                    data_t0=data_t0,
                    data_t1=data_t1,
                    mask_t0=mask_t0,
                    mask_t1=mask_t1,
                    plot_path=plot_path_segmentation,
                    ind=ind,
                    iou_t0=iou_dyn_t0,
                    iou_t1=iou_dyn_t1,
                )

            iou_t0_sets.append(metrics_t0["IoU"])
            miou_t0_sets.append(metrics_t0["mIoU"])
            piou_t0_sets.append(metrics_t0["pIoU"])
            iou_t1_sets.append(metrics_t1["IoU"])
            miou_t1_sets.append(metrics_t1["mIoU"])
            piou_t1_sets.append(metrics_t1["pIoU"])
            success_t0_sets.append(self._success_from_iou(iou_dyn_t0))
            success_t1_sets.append(self._success_from_iou(iou_dyn_t1))

        return {
            "IoU/t0": np.nanmean(np.array(iou_t0_sets) * 100, axis=0).tolist(),
            "mIoU/t0": np.nanmean(np.array(miou_t0_sets) * 100).tolist(),
            "pIoU/t0": np.nanmean(np.array(piou_t0_sets) * 100).tolist(),
            "IoU/t1": np.nanmean(np.array(iou_t1_sets) * 100, axis=0).tolist(),
            "mIoU/t1": np.nanmean(np.array(miou_t1_sets) * 100).tolist(),
            "pIoU/t1": np.nanmean(np.array(piou_t1_sets) * 100).tolist(),
            "SR@0.5/t0": np.nanmean(np.array(success_t0_sets) * 100).tolist(),
            "SR@0.5/t1": np.nanmean(np.array(success_t1_sets) * 100).tolist(),
        }

    def validate_model(self, save_path):
        self.dataset = self._get_dataset()
        self.model = self.setup_model(self.model_path)
        self.model.eval()

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        results = {}

        for sequence in tqdm(self.dataset):
            sequence_id = sequence.sequence_id
            save_sequence_path = save_path / sequence_id
            save_sequence_path.mkdir(exist_ok=True)
            plot_path = save_sequence_path / f"plots_{sequence_id}"
            plot_path.mkdir(exist_ok=True)

            stats = self._evaluate_sequence(sequence, plot_path)
            results[sequence_id] = stats
            logs = self._aggregate_logs(stats, sequence_id)

            self._write_json(results, save_path / "results.json")

            if self.config["vis"]["verbose"]:
                self._print_stats(sequence_id, stats)

            self.model.reset_states()

        aggregated = self._aggregate_results(results)
        logs = self._add_tot_to_logs(logs, aggregated)

        results["test/total"] = aggregated
        self._write_json(results, save_path / "results.json")
        return results


def validate_models(config_path, models_dir, save_path):
    config = open_config_json(config_path)
    models_dir = Path(models_dir)
    general_save_path = Path(save_path, models_dir.stem)
    if not general_save_path.exists():
        general_save_path.mkdir(parents=True)

    all_results = {}
    for model_path in sorted(models_dir.iterdir()):
        print(f"Validating model: {model_path}")
        model_save_path = Path(general_save_path, model_path.stem)
        validator = ValidateEED(config=config, model_path=model_path)
        model_results = validator.validate_model(save_path=str(model_save_path))
        all_results[model_path.stem] = model_results["test/total"]
        print(f"Model {model_path} validated and results saved at {model_save_path}")

    with open(Path(general_save_path, "all_results.json"), "w") as json_file:
        json.dump(all_results, json_file, indent=4)


if __name__ == "__main__":
    config = open_config_json("dynamic_masker/configs/validate_EED.json")

    validator = ValidateEED(
        config=config,
        model_path="checkpoints/checkpoints_prerebuttal/Herm_2025-04-15_15-49-03/model_epoch_11.pth",
    )
    validator.validate_model(
        save_path="dynamic_masker/results/EED_Herm_2025-04-15_15-49-03/model_epoch_11"
    )
