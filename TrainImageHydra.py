"""
Launch a distributed training using images and events as input.
Run using -> torchrun --nproc_per_node=8 your_training_script.py
Example -> CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 TrainImageHydra.py --master_port=29513
From Roberto Pellerito -> rpellerito@ifi.uzh.ch
"""

import os
import torch
from time import time
from datetime import datetime
from torch.nn.utils import clip_grad_norm_

from TrainBaseHydra import TrainBaseHydra
from dynamic_masker.models.model_util import label_smoothing, initialize_weights
from dynamic_masker.models.model_hydra import HydraEVNet
from dynamic_masker.utils.utils import load_model, save_model
from Validate_image_hydra import ValidateImageHydra
from suppressor.DSEC_dataloader.provider import DatasetProvider

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


class TrainImageHydra(TrainBaseHydra):
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
            augment_prob=self.config["loader"]["augment_prob"],
            multiple_batches= True if self.config["loader"]["batch_size"] > 1 else False,
        )

        if self.distributed:
            sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=self.config["loader"]["shuffle"]
            )
        else:
            sampler = None
        
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

    def init_model_name(self):
        if self.checkpoint_path:
            return os.path.basename(os.path.dirname(self.checkpoint_path))
        else:
            return "ImageHydra_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def build_model(self):
        model_config = self.config["model"]
        model = HydraEVNet(
            kwargs=model_config,
            num_bins=self.config["data"]["voxel_bins"] + 3,  # +3 for image channels
            final_w_scale_flow=self.config["custom"]["final_w_scale_flow"],
            current_flow_sup=self.config["custom"]["current_flow_sup"],
            current_flow_scaling=self.config["loader"]["event_dt_ms"],
        ).to(self.device)

        if self.checkpoint_path:
            model = load_model(model, self.device, self.checkpoint_path)
        else:
            model.apply(initialize_weights)

        model.train()
        
        if self.distributed:
            self.model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank)
        else:
            self.model = model

    
    def run_validation(self, model_dir, epoch):
        validator = ValidateImageHydra(config=self.config, model_path=model_dir)
        save_path = f"results/{self.model_name}/model_epoch_{epoch}"
        results = validator.validate_model(save_path=save_path)

        if self.config["vis"]["log_wandb"] and self.run:
            self.run.log(results, step=self.total_seen_samples)

    def train_batch(self, batch_data):
        for data_ind in range(len(batch_data) - 1):
            data_t0 = batch_data[data_ind]
            data_t1 = batch_data[data_ind + 1]

            event_voxel = data_t0["representation"].to(self.device)
            frame = data_t0['frame'].to(self.device)
            dt = torch.tensor([data_t1["sampled_dt"]]).float().to(self.device) if len(data_t1["sampled_dt"]) == 1 else data_t1["sampled_dt"].float().to(self.device)

            input_ = torch.cat((event_voxel, frame), dim=1)
            outputs = self.model(input_, dt)
            flow_t1 = outputs["flow"]
            flow_t0 = outputs["flow_t0"]
            mask = outputs["mask"]
            future_mask = outputs["future_mask"]

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
            
        if self.config["vis"]["log_wandb"] and self.rank == 0:
            self.log_wandb(mask, dynamic_mask_t0_gt, future_mask, dynamic_mask_t1_gt, event_voxel, flow_t1, flow_gt_t1)

        return loss

    def train_epoch(self, epoch, train_loader):
        for batch_idx, batch_data in enumerate(train_loader):
            it_start_time = time()
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
                it_time = time() - it_start_time
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

        print("Training started")
        print("Dataset size:", len(train_loader.dataset))

        for epoch in range(self.config["loader"]["n_epochs"]):
            if epoch <= self.starting_epoch:
                continue
            self.train_epoch(epoch, train_loader)

# if __name__ == "__main__":
#     trainer = TrainImageHydra(config_path="dynamic_masker/configs/train_image_hydra.json", checkpoint_path="")
#     trainer.train()
#     trainer.finish()
    
if __name__ == "__main__":
    trainer = TrainImageHydra(
        config_path="dynamic_masker/configs/train_image_hydra.json", 
        checkpoint_path="checkpoints/ImageHydra_2025-06-11_09-49-15/model_epoch_21.pth", 
        distributed=True
        )
    trainer.train()
    trainer.finish()
    if dist.is_initialized():
        dist.destroy_process_group()

