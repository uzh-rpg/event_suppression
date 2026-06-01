"""
Launch a distributed training using only events as input (EVIMO dataset).
Run using -> torchrun --nproc_per_node=[NUM_GPUS] your_training_script.py
Example -> CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 TrainHermDistributed.py --master_port=29514
"""

import os
import torch
from datetime import datetime
from torch.nn.utils import clip_grad_norm_
from time import time

from suppressor.DSEC_dataloader.provider import DatasetProvider
from dynamic_masker.models.model_hydra import HydraEVNet
from dynamic_masker.utils.utils import load_model, save_model
from dynamic_masker.models.model_util import initialize_weights, label_smoothing

from TrainHerm import TrainHerm  # Assuming TrainHerm inherits from TrainBaseHydra

# --- NEW: Imports for distributed training ---
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


class TrainHermDistributed(TrainHerm):
    """
    A training class for the 'Herm' configuration modified for multi-GPU training.
    It uses the EVIMO dataset.
    """
    # --- NEW: Added __init__ to handle distributed setup ---
    def __init__(self, config_path: str, checkpoint_path: str = "", distributed: bool = True):
        # We need to call the base class's __init__ first. 
        # Since TrainHerm's __init__ is not shown, we assume it calls its parent, TrainBaseHydra.
        # This setup assumes TrainBaseHydra handles the core config loading.
        super().__init__(config_path, checkpoint_path)
        self.init_distributed()
        self.distributed = distributed

    # --- NEW: Method to initialize distributed environment ---
    def init_distributed(self):
        # Check if the environment variables for distributed training are set
        self.distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ

        if self.distributed:
            # These are set by torchrun
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.local_rank = int(os.environ["LOCAL_RANK"])
            
            # Pin the process to a specific GPU
            torch.cuda.set_device(self.local_rank)
            
            # Initialize the process group for communication
            dist.init_process_group(backend='nccl', init_method='env://')
            print(f"Initialized process {self.rank}/{self.world_size} on GPU {self.local_rank}")
        else:
            # Fallback for single-GPU or CPU training
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def init_model_name(self):
        """
        Overrides the base method to provide a more specific name for this model's runs.
        """
        if self.checkpoint_path:
            return os.path.basename(os.path.dirname(self.checkpoint_path))
        else:
            return "Herm_XXS_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    def train(self):
        if self.rank == 0:  # <--- THIS IS THE CRITICAL GUARD
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

        # --- MODIFICATION: Wrap model with DDP if in distributed mode ---
        if self.distributed:
            self.model = DDP(
                model, 
                device_ids=[self.local_rank], 
                output_device=self.local_rank,
                find_unused_parameters=True  # <--- ADD THIS ARGUMENT
            )
        else:
            self.model = model

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
        train_dataset = provider.get_evimo_train_dataset(
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
        # This method remains largely the same, as DDP is transparent during the forward pass.
        for data_ind in range(len(batch_data) - 1):
            data_t0 = batch_data[data_ind]
            data_t1 = batch_data[data_ind + 1]

            event_voxel = data_t0["representation"].to(self.device)
            dt = torch.tensor([data_t1["sampled_dt"]]).float().to(self.device) if len(data_t1["sampled_dt"]) == 1 else data_t1["sampled_dt"].float().to(self.device)

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
                outputs["mask"], 
                dynamic_mask_t0_gt, 
                outputs["future_mask"], 
                dynamic_mask_t1_gt, 
                event_voxel, 
                outputs["flow"], 
                )

        return loss
    
    # --- NEW: train_epoch method adapted for distributed training ---
    # This mirrors the logic from your reference script.
    # It's important to have this to handle state resets and checkpointing correctly.
    def train_epoch(self, epoch, train_loader):
        # Set epoch for the sampler, which is important for proper shuffling across epochs
        if self.distributed and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        for batch_idx, batch_data in enumerate(train_loader):
            it_start_time = time()
            loss_sequence = self.train_batch(batch_data)
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # --- MODIFICATION: Access original model through `.module` ---
            # When using DDP, the original model is wrapped. To call its methods,
            # you need to access it via the `.module` attribute.
            # Assuming `detach_states` is a method on HydraEVNetNoAtten.
            if self.distributed:
                if hasattr(self.model.module, 'detach_states'):
                    self.model.module.detach_states()
            else:
                if hasattr(self.model, 'detach_states'):
                    self.model.detach_states()

            self.loss_function.reset()
            self.total_seen_samples += 1
                
            # --- MODIFICATION: Only print from the main process ---
            if self.config["vis"]["verbose"] and self.rank == 0:
                it_time = time() - it_start_time
                print(
                    "Train time {:.6f}s Epoch: {:04d} [{:03d}/{:03d} ({:03d}%)] Loss: {:.6f}".format(
                        it_time, epoch, batch_idx, len(train_loader),
                        int(100 * batch_idx / len(train_loader)), loss_sequence.item()
                    )
                )
        
        # Reset states at the end of the epoch
        if self.distributed:
            if hasattr(self.model.module, 'reset_states'):
                self.model.module.reset_states()
        else:
            if hasattr(self.model, 'reset_states'):
                self.model.reset_states()

        # --- MODIFICATION: Only save model from the main process ---
        if self.rank == 0:
            save_model(
                model=self.model, # DDP automatically saves the underlying module's state_dict
                optimizer=self.optimizer,
                epoch=epoch,
                loss=loss_sequence,
                path_results=self.config["loader"]["checkpoints_path"],
                model_name=self.model_name,
                total_seen_samples=self.total_seen_samples
            )

    
if __name__ == "__main__":
    print("\nInstantiating the new trainer for EVIMO dataset...")
    herm_trainer = TrainHermDistributed(
        config_path="dynamic_masker/configs/train_herm_ablation_XXS.json", 
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