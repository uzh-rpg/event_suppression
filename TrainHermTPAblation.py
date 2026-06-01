"""
Launch a distributed training using only events as input (EVIMO dataset).
Run using -> torchrun --nproc_per_node=[NUM_GPUS] your_training_script.py
Example -> CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 TrainHermTPAblation.py --master_port=29514
"""

import os
import torch
from datetime import datetime
from torch.nn.utils import clip_grad_norm_

from suppressor.DSEC_dataloader.provider import DatasetProvider
from dynamic_masker.models.model_util import initialize_weights, label_smoothing
from dynamic_masker.utils.train_log import train_log_hydra
from dynamic_masker.models.model_hydra import HydraEVNet
from dynamic_masker.utils.utils import load_model

from TrainHermDistributed import TrainHermDistributed  # Assuming TrainHerm inherits from TrainBaseHydra

# --- NEW: Imports for distributed training ---
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


class TrainHermTPAblation(TrainHermDistributed):
    """
    A training class for the 'Herm' configuration modified for multi-GPU training.
    It uses the EVIMO dataset.
    This class is to train the model to ablate the different temporal horizons.
    """
    # --- NEW: Added __init__ to handle distributed setup ---
    def __init__(self, config_path: str, checkpoint_path: str = "", distributed: bool = True):
        super().__init__(config_path, checkpoint_path, distributed)

    def init_model_name(self):
        """
        Overrides the base method to provide a more specific name for this model's runs.
        """
        if self.checkpoint_path:
            return os.path.basename(os.path.dirname(self.checkpoint_path))
        else:
            return "Herm_Tp_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def build_dataloader(self):
        """
        MODIFIED: Uses DistributedSampler to partition the dataset across GPUs.
        """
        print(f"[{self.rank}] Building dataloader for EVIMO dataset...")
        provider = DatasetProvider(
            dataset_path=self.config["data"]["path"],
            representation=self.config["data"]["representation"],
            num_bins=self.config["data"]["voxel_bins"],
            delta_t_ms=self.config["loader"]["event_dt_ms"]
        )
        train_dataset = provider.get_evimo_train_dataset_by_varying_event_window(
            sequence_len=self.config["data"]["sequence_len"], 
            batch_size=self.config["loader"]["batch_size"]
        )

        # --- MODIFICATION: Create a DistributedSampler ---
        if self.distributed:
            sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=self.config["loader"]["shuffle"]
            )
        else:
            sampler = None
        
        # When using a sampler, shuffle in DataLoader must be False (or None).
        # The sampler handles the shuffling.
        is_shuffle = self.config["loader"]["shuffle"] if sampler is None else False

        return torch.utils.data.DataLoader(
            dataset=train_dataset,
            sampler=sampler,
            drop_last=True,
            batch_size=self.config["loader"]["batch_size"],
            shuffle=is_shuffle,
            num_workers=self.config["loader"]["n_workers"],
            prefetch_factor=self.config["loader"]["prefetch_factor"],
            worker_init_fn=self.seed_worker,
            pin_memory=True
        )

    def train_batch(self, batch_data):
        for data_ind in range(len(batch_data) - 1):
            data_t0 = batch_data[data_ind]
            data_t1 = batch_data[data_ind + 1]

            event_voxel = data_t0["representation"].to(self.device)
            dt = torch.tensor([data_t1["sampled_dt"]]).float().to(self.device) if len(data_t1["sampled_dt"]) == 1 else data_t1["sampled_dt"].float().to(self.device)

            # transform dt to be in milliseconds and integer type to save memory
            dt = (dt * 1000.0).to(torch.int8)
            
            outputs = self.model(event_voxel, dt)

            dynamic_mask_t0_gt = label_smoothing(data_t0["dynamic_mask"].to(self.device))
            dynamic_mask_t1_gt = label_smoothing(data_t1["dynamic_mask"].to(self.device))

            self.loss_function.update(
                pred_mask_t0=outputs["mask"],
                gt_mask_t0=dynamic_mask_t0_gt,
                pred_mask_t1=outputs["future_mask"],
                gt_mask_t1=dynamic_mask_t1_gt,
                pred_flow_t0=outputs["flow_t0"],
                pred_flow_t1=outputs["flow"],
                event_list=data_t1["event_list"].float().to(self.device).detach(),
                pol_mask=data_t1["polarity_mask"].float().to(self.device).detach(),
                d_event_list=data_t1["d_event_list"].float().to(self.device).detach(),
                d_pol_mask=data_t1["d_polarity_mask"].float().to(self.device).detach(),
            )

        loss = self.loss_function()
        loss.backward()

        if self.config["loss"]["clip_grad"] is not None:
            clip_grad_norm_(self.model.parameters(), self.config["loss"]["clip_grad"])
            
        # Logging should be done only on the main process
        if self.config["vis"]["log_wandb"] and self.rank == 0:
            self.log_wandb(
                mask=outputs["mask"], 
                dynamic_mask_t0_gt=dynamic_mask_t0_gt, 
                future_mask=outputs["future_mask"], 
                dynamic_mask_t1_gt=dynamic_mask_t1_gt, 
                event_voxel=event_voxel, 
                flow_t1=outputs["flow"],
                dt=dt,
            )

        return loss
    
    def log_wandb(self,
                  dt,
                  mask, 
                  dynamic_mask_t0_gt, 
                  future_mask, 
                  dynamic_mask_t1_gt, 
                  event_voxel, 
                  flow_t1, 
                  flow_gt_t1=None
        ):
        logging_dict = train_log_hydra(mask_pred_t0=mask[-1],
                        mask_target_t0=dynamic_mask_t0_gt,
                        mask_pred_t1=future_mask[-1],
                        mask_target_t1=dynamic_mask_t1_gt,
                        event_voxel=event_voxel,
                        config=self.config,
                        flow_pred_t1=flow_t1[-1],
                        flow_target_t1=flow_gt_t1
                        )
        logging_dict["train/dt"] = dt
        self.run.log(logging_dict,step=self.total_seen_samples)

    
if __name__ == "__main__":
    print("\nInstantiating the new trainer for EVIMO dataset...")
    herm_trainer = TrainHermTPAblation(
        config_path="dynamic_masker/configs/train_herm_ablation_TP_ablation.json", 
        checkpoint_path="",
        distributed=True # This flag can be controlled by a command-line argument if needed
    )

    print(f"[{herm_trainer.rank}] Trainer created. Model name will be: {herm_trainer.model_name}")
    
    # The `train` method from the base class will now call the overridden methods
    # (train_epoch, build_model, build_dataloader, etc.) in this class.
    herm_trainer.train()
    
    # Wait for all processes to finish before cleaning up
    if dist.is_initialized():
        dist.barrier()
        
    herm_trainer.finish()

    # --- NEW: Clean up the distributed environment ---
    if dist.is_initialized():
        print(f"[{herm_trainer.rank}] Destroying process group.")
        dist.destroy_process_group()