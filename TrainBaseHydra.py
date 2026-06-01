import os
import json
import time
import torch
import random
import numpy as np
import wandb
from datetime import datetime
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

from dynamic_masker.loss.HydraLoss import HydraLoss
from dynamic_masker.models.model_hydra import HydraEVNet
from dynamic_masker.utils.utils import save_model, load_model, load_optimizer_epoch_seen_samples
from dynamic_masker.models.model_util import initialize_weights, label_smoothing
from dynamic_masker.utils.train_log import train_log_hydra, plot_segmentation_masks
from dynamic_masker.utils.visualization import Visualization
from dynamic_masker.configs.utils import get_device
from suppressor.DSEC_dataloader.provider import DatasetProvider
from Validate_hydra import ValidateHydra

# Set the seed before creating the DataLoader
torch.manual_seed(42)  # For CPU & CUDA
np.random.seed(42)  # Numpy seed
random.seed(42)  # Python seed

class TrainBaseHydra:
    def __init__(self, config_path: str, checkpoint_path: str = ""):
        self.config = self.load_config(config_path)
        self.checkpoint_path = checkpoint_path
        self.device = get_device(gpu_num=self.config["loader"]["gpu"])
        self.config["loader"]["device"] = self.device
        self.model_name = self.init_model_name()
        self.model = None
        self.optimizer = None
        self.loss_function = None
        self.total_seen_samples = 0
        self.starting_epoch = 0
        self.run = None
        self.rank = 0

    @staticmethod
    def seed_worker(worker_id):
        """Ensure each worker has a deterministic seed"""
        seed = torch.initial_seed() % (2**32)  # Get unique seed for each worker
        np.random.seed(seed)
        random.seed(seed)

    def load_config(self, path):
        return json.load(open(path, "r"))

    def init_model_name(self):
        if self.checkpoint_path:
            return os.path.basename(os.path.dirname(self.checkpoint_path))
        else:
            return "Hydra_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def setup_logging(self):
        if self.config["vis"]["log_wandb"]:
            self.run = wandb.init(
                project="events-to-dynamic-mask",
                config=self.config,
                name=self.model_name,
            )

    def build_model(self):
        model_config = self.config["model"]
        model = HydraEVNet(
            kwargs=model_config,
            num_bins=self.config["data"]["voxel_bins"],
            final_w_scale_flow=self.config["custom"]["final_w_scale_flow"],
            current_flow_sup=self.config["custom"]["current_flow_sup"],
            current_flow_scaling=self.config["loader"]["event_dt_ms"],
        ).to(self.device)

        if self.checkpoint_path:
            model = load_model(model, self.device, self.checkpoint_path)
        else:
            model.apply(initialize_weights)

        model.train()
        self.model = model
        return model

    def build_optimizer(self):
        optimizer_cls = eval(self.config["optimizer"]["name"])
        self.optimizer = optimizer_cls(self.model.parameters(), lr=self.config["optimizer"]["lr"])
        self.optimizer, self.starting_epoch, self.total_seen_samples = load_optimizer_epoch_seen_samples(
            self.optimizer, self.device, self.checkpoint_path
        )

    def build_loss_function(self):
        self.loss_function = HydraLoss(self.config, device=self.device)

    def build_dataloader(self):
        provider = DatasetProvider(
            self.config["data"]["path"],
            num_bins=self.config["data"]["voxel_bins"],
            representation=self.config["data"]["representation"],
            delta_t_ms=self.config["loader"]["event_dt_ms"]
        )
        train_dataset = provider.get_hydra_train_dataset(
            sequence_len=self.config["data"]["sequence_len"],
            max_num_grad_events=self.config["loader"]["max_num_grad_events"],
            dt=self.config["data"]["dt_ms"],
            augment=self.config["loader"]["augment"],
            augment_prob=self.config["loader"]["augment_prob"]
        )
        return torch.utils.data.DataLoader(
            dataset=train_dataset,
            drop_last=True,
            batch_size=self.config["loader"]["batch_size"],
            shuffle=self.config["loader"]["shuffle"],
            num_workers=self.config["loader"]["n_workers"],
            prefetch_factor=self.config["loader"]["prefetch_factor"],
            worker_init_fn=self.seed_worker,
            pin_memory=True
        )

    def train(self):
        self.setup_logging()
        self.build_model()
        self.build_optimizer()
        self.build_loss_function()
        train_loader = self.build_dataloader()

        print("Training started")
        print("Dataset size:", len(train_loader.dataset))

        for epoch in range(self.config["loader"]["n_epochs"]):
            if epoch <= self.starting_epoch:
                continue
            self.train_epoch(epoch, train_loader)

    def train_epoch(self, epoch, train_loader):
        for batch_idx, batch_data in enumerate(train_loader):
            it_start_time = time.time()

            loss_sequence = self.train_batch(batch_data)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.model.detach_states()
            self.loss_function.reset()
            self.total_seen_samples += 1

            if self.config["vis"]["verbose"] and self.rank == 0:
                it_time = time.time() - it_start_time
                print(
                    "Train time {:.6f}s Epoch: {:04d} [{:03d}/{:03d} ({:03d}%)] Loss: {:.6f}".format(
                        it_time,
                        epoch,
                        batch_idx,
                        len(train_loader),
                        int(100 * batch_idx / len(train_loader)),
                        loss_sequence.item(),
                    )
                )

            # if self.config["vis"]["log_wandb"]:
            #     self.log_wandb(batch_data, loss_sequence)

            # break  # remove or adjust depending on epoch size

        self.model.reset_states()
        model_dir = save_model(
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            loss=loss_sequence,
            path_results=self.config["loader"]["checkpoints_path"],
            model_name=self.model_name,
            total_seen_samples=self.total_seen_samples
        )

        self.run_validation(model_dir, epoch)

    def train_batch(self, batch_data):
        loss_sequence = torch.tensor(0.0).to(self.device)

        for data_ind in range(len(batch_data) - 1):
            data_t0 = batch_data[data_ind]
            data_t1 = batch_data[data_ind + 1]

            event_voxel = data_t0["representation"].to(self.device)
            dt = torch.tensor([data_t1["sampled_dt"]]).float().to(self.device) if len(data_t1["sampled_dt"]) == 1 else data_t1["sampled_dt"].float().to(self.device)

            outputs = self.model(event_voxel, dt)

            # Load ground truths and masks
            flow_gt_t0 = data_t0["forward_flow_gt"].to(self.device) * (self.config["loader"]["event_dt_ms"] / self.config["loader"]["optical_flow_dt_ms"])
            flow_gt_t1 = flow_gt_t0 * dt.item() / 100 if len(data_t0["sampled_dt"]) == 1 else data_t1["forward_flow_gt"].to(self.device) * dt.view(dt.shape[0], 1, 1, 1) / 100

            # Preprocess GT
            dynamic_mask_t0_gt = label_smoothing(data_t0["dynamic_mask"].to(self.device))
            dynamic_mask_t1_gt = label_smoothing(data_t1["dynamic_mask"].to(self.device))

            self.loss_function.update(
                pred_mask_t0=outputs["mask"],
                gt_mask_t0=dynamic_mask_t0_gt,
                pred_mask_t1=outputs["future_mask"],
                gt_mask_t1=dynamic_mask_t1_gt,
                pred_flow_t0=outputs["flow_t0"],
                gt_flow_t0=flow_gt_t0,
                mask_invalid_flows_t0=data_t0["has_flow"].to(self.device),
                pred_flow_t1=outputs["flow"],
                gt_flow_t1=flow_gt_t1,
                mask_invalid_flows_t1=data_t1["has_flow"].to(self.device),
                event_list=data_t1["event_list"].float().to(self.device).detach(),
                pol_mask=data_t1["polarity_mask"].float().to(self.device).detach(),
                d_event_list=data_t1["d_event_list"].float().to(self.device).detach(),
                d_pol_mask=data_t1["d_polarity_mask"].float().to(self.device).detach(),
            )

        loss = self.loss_function()
        loss.backward()

        if self.config["loss"]["clip_grad"] is not None:
            clip_grad_norm_(self.model.parameters(), self.config["loss"]["clip_grad"])

        return loss

    def log_wandb(self, mask, dynamic_mask_t0_gt, future_mask, dynamic_mask_t1_gt, event_voxel, flow_t1, flow_gt_t1=None):
        logging_dict = train_log_hydra(mask_pred_t0=mask[-1],
                        mask_target_t0=dynamic_mask_t0_gt,
                        mask_pred_t1=future_mask[-1],
                        mask_target_t1=dynamic_mask_t1_gt,
                        event_voxel=event_voxel,
                        config=self.config,
                        flow_pred_t1=flow_t1[-1],
                        flow_target_t1=flow_gt_t1
                        )
        if self.config["vis"]["plot"]:
            # Debugging images
            if flow_gt_t1 is not None:
                gt_flow_to_plot = flow_gt_t1[0].permute(1,2, 0).detach().cpu().numpy()
                rendered_gt_flow = Visualization.flow_to_image(gt_flow_to_plot)
                gt_flow_wandb_img = wandb.Image(rendered_gt_flow)
                logging_dict.update({"images/flow/t1/gt_total_flow": gt_flow_wandb_img})

            
            flow_to_plot = flow_t1[-1][0].permute(1,2, 0).detach().cpu().numpy()
            rendered_flow = Visualization.flow_to_image(flow_to_plot)
            pred_flow_wandb_img = wandb.Image(rendered_flow)
            logging_dict.update({"images/flow/t1/pred_total_flow": pred_flow_wandb_img})

            mask_t0_wandb_img = plot_segmentation_masks(
                pred=mask[-1].cpu(), smooth_target=dynamic_mask_t0_gt.cpu(), event_voxel=event_voxel.cpu())
            mask_t1_wandb_img = plot_segmentation_masks(
                pred=future_mask[-1].cpu(), smooth_target=dynamic_mask_t1_gt.cpu(), event_voxel=event_voxel.cpu())
            logging_dict.update({"images/mask/t0": mask_t0_wandb_img})
            logging_dict.update({"images/mask/t1": mask_t1_wandb_img})

        self.run.log(logging_dict,step=self.total_seen_samples)

    def run_validation(self, model_dir, epoch):
        validator = ValidateHydra(config=self.config, model_path=model_dir)
        save_path = f"results/{self.model_name}/model_epoch_{epoch}"
        results = validator.validate_model(save_path=save_path)

        if self.config["vis"]["log_wandb"] and self.run:
            self.run.log(results, step=self.total_seen_samples)

    def finish(self):
        if self.run:
            wandb.finish()


if __name__ == "__main__":
    trainer = TrainBaseHydra(config_path="configs/train_hydra.json", checkpoint_path="")
    trainer.train()
    trainer.finish()
