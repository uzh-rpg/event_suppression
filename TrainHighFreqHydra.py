"""
Train using this command:
CUDA_VISIBLE_DEVICES=4,5,6,0 torchrun --nproc_per_node=4 --master_port=29513 TrainHighFreqHydra.py
"""
import os
import time
import torch
import random
import numpy as np
import wandb
from datetime import datetime
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from TrainBaseHydra import TrainBaseHydra
from dynamic_masker.models.model_util import label_smoothing
from suppressor.DSEC_dataloader.provider import DatasetProvider

from dynamic_masker.utils.utils import save_model
from dynamic_masker.utils.visualization import Visualization
from dynamic_masker.utils.train_log import log_future_hydra_masks, plot_segmentation_masks

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class TrainHighFreqHydra(TrainBaseHydra):
    def __init__(self, config_path: str, checkpoint_path: str = "", distributed: bool = True):
        super().__init__(config_path, checkpoint_path)
        self.init_distributed()
        self.distributed = distributed

    def init_distributed(self):
        self.distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
        if self.distributed:
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend='nccl', init_method='env://')
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def init_model_name(self):
        if self.checkpoint_path:
            return os.path.basename(os.path.dirname(self.checkpoint_path))
        else:
            return "HighFreq_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def build_dataloader(self):
        provider = DatasetProvider(
            self.config["data"]["path"],
            num_bins=self.config["data"]["voxel_bins"],
            representation=self.config["data"]["representation"],
            delta_t_ms=self.config["loader"]["event_dt_ms"]
        )
        train_dataset = provider.get_high_freq_hydra_train_dataset(
            sequence_len=self.config["data"]["sequence_len"],
            max_num_grad_events=self.config["loader"]["max_num_grad_events"],
            dt=self.config["data"]["dt_ms"],
            augment=self.config["loader"]["augment"],
            augment_prob=self.config["loader"]["augment_prob"],
            multiple_batches=self.config["loader"]["batch_size"] > 1
        )

        sampler = DistributedSampler(
            train_dataset, num_replicas=self.world_size, rank=self.rank,
            shuffle=self.config["loader"]["shuffle"]
        ) if self.distributed else None

        return torch.utils.data.DataLoader(
            dataset=train_dataset,
            sampler=sampler,
            drop_last=True,
            batch_size=self.config["loader"]["batch_size"],
            num_workers=self.config["loader"]["n_workers"],
            prefetch_factor=self.config["loader"]["prefetch_factor"],
            worker_init_fn=self.seed_worker,
            pin_memory=True
        )

    def build_model(self):
        super().build_model()
        if self.distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
        self.model.train()

    def train_batch(self, batch_data):
        for data_ind in range(len(batch_data) - 1):
            data = batch_data[data_ind]
            event_voxel = data["representation"].to(self.device)
            if len(data["sampled_dt"]) == 1 :
                dt = torch.tensor([data["sampled_dt"]]).float().to(self.device)
            else:
                dt = data["sampled_dt"].float().to(self.device)
            outputs = self.model(event_voxel, dt)

            dynamic_mask_gt = label_smoothing(data["dynamic_mask"].to(self.device))
            self.loss_function.update(
                pred_mask_t1=outputs["future_mask"],
                gt_mask_t1=dynamic_mask_gt,
                pred_flow_t1=outputs["flow"],
                event_list=data["event_list"].float().to(self.device).detach(),
                pol_mask=data["polarity_mask"].float().to(self.device).detach(),
                d_event_list=data["d_event_list"].float().to(self.device).detach(),
                d_pol_mask=data["d_polarity_mask"].float().to(self.device).detach(),
            )

        loss = self.loss_function()
        loss.backward()

        if self.config["loss"]["clip_grad"] is not None:
            clip_grad_norm_(self.model.parameters(), self.config["loss"]["clip_grad"])

        if self.config["vis"]["log_wandb"] and self.rank == 0:
            self.log_wandb(
                future_mask=outputs["future_mask"],
                dynamic_mask_t1_gt=dynamic_mask_gt,
                event_voxel=event_voxel,
                flow_t1=outputs["flow"]
            )

        return loss
    
    def log_wandb(self, future_mask, dynamic_mask_t1_gt, event_voxel, flow_t1):
        logging_dict = log_future_hydra_masks(
            mask_pred_t1=future_mask[-1], 
            mask_target_t1=dynamic_mask_t1_gt, 
            event_voxel=event_voxel, 
            config=self.config
            )
        
        if self.config["vis"]["plot"]:
            flow_to_plot = flow_t1[-1][0].permute(1,2, 0).detach().cpu().numpy()
            rendered_flow = Visualization.flow_to_image(flow_to_plot)
            pred_flow_wandb_img = wandb.Image(rendered_flow)
            logging_dict.update({"images/flow/t1/pred_total_flow": pred_flow_wandb_img})

            mask_t1_wandb_img = plot_segmentation_masks(
                pred=future_mask[-1].cpu(), smooth_target=dynamic_mask_t1_gt.cpu(), event_voxel=event_voxel.cpu())
            logging_dict.update({"images/mask/t1": mask_t1_wandb_img})

        self.run.log(logging_dict,step=self.total_seen_samples)

    def train_epoch(self, epoch, train_loader):
        if self.distributed:
            train_loader.sampler.set_epoch(epoch)

        for batch_idx, batch_data in enumerate(train_loader):
            start_time = time.time()
            loss_sequence = self.train_batch(batch_data)
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.distributed:
                self.model.module.detach_states()
            else:
                self.model.detach_states()

            self.loss_function.reset()
            self.total_seen_samples += 1

            if self.config["vis"]["verbose"] and self.rank == 0:
                it_time = time.time() - start_time
                print(
                    f"Train time {it_time:.6f}s Epoch: {epoch:04d} [{batch_idx:03d}/{len(train_loader):03d} ({100 * batch_idx / len(train_loader):03.0f}%)] Loss: {loss_sequence.item():.6f}"
                )

        if self.distributed:
            self.model.module.reset_states()
        else:
            self.model.reset_states()

        if self.rank == 0:
            model_dir = save_model(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                loss=loss_sequence,
                path_results=self.config["loader"]["checkpoints_path"],
                model_name=self.model_name,
                total_seen_samples=self.total_seen_samples
            )

            # self.run_validation(model_dir, epoch)

    def train(self):
        if self.rank == 0:
            self.setup_logging()
        self.build_model()
        self.build_optimizer()
        self.build_loss_function()
        train_loader = self.build_dataloader()

        print(f"Training started, Dataset size: {len(train_loader.dataset)}")

        for epoch in range(self.config["loader"]["n_epochs"]):
            if epoch <= self.starting_epoch:
                continue
            self.train_epoch(epoch, train_loader)

    def finish(self):
        if self.distributed:
            dist.destroy_process_group()
        super().finish()

if __name__ == "__main__":
    trainer = TrainHighFreqHydra(
        config_path="dynamic_masker/configs/train_high_freq.json",
        checkpoint_path="checkpoints/HighFreq_2025-06-24_14-59-34/model_epoch_1.pth",
        distributed=True
    )
    trainer.train()
    trainer.finish()
