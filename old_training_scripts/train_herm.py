import os
import time
import json
import torch
import wandb
import random
import argparse
import numpy as np
from torch.optim import AdamW
from datetime import datetime

from dynamic_masker.loss.HermLoss import HermLoss

from dynamic_masker.utils.utils import save_model, load_optimizer_epoch_seen_samples
from dynamic_masker.models.model_util import initialize_weights, label_smoothing
from dynamic_masker.models.model_hydra import HydraEVNet
from dynamic_masker.utils.visualization import Visualization
from dynamic_masker.configs.utils import get_device

from suppressor.DSEC_dataloader.provider import DatasetProvider
from dynamic_masker.utils.train_log import train_log_herm, plot_segmentation_masks
from dynamic_masker.utils.utils import load_model, get_scaled_lr_scheduler

from Validate_herm import ValidateHerm

def seed_worker(worker_id):
    """Ensure each worker has a deterministic seed"""
    seed = torch.initial_seed() % (2**32)  # Get unique seed for each worker
    np.random.seed(seed)
    random.seed(seed)

# Set the seed before creating the DataLoader
torch.manual_seed(42)  # For CPU & CUDA
np.random.seed(42)  # Numpy seed
random.seed(42)  # Python seed

def train(config_path, checkpoint_path=""):
    """
    Main function of the training pipeline for event-based optical flow estimation.
    """

    # configs
    config = json.load(open(config_path, 'r'))

    # initialize settings
    device = get_device(gpu_num=config["loader"]["gpu"])
    config["loader"]["device"] = device
    
    # data loader
    num_bins = config["data"]["voxel_bins"]
    event_dt_ms = config["loader"]["event_dt_ms"]
    # NOTE: we need this ratio to scale the optical flow GT
    dataset_provider = DatasetProvider(
        dataset_path=config["data"]["path"], 
        representation=config["data"]["representation"], 
        num_bins=num_bins, 
        delta_t_ms=event_dt_ms
        )
    train_dataset = dataset_provider.get_evimo_train_dataset(
        sequence_len=config["data"]["sequence_len"], batch_size=config["loader"]["batch_size"])
    
    train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=config["loader"]["batch_size"],
            shuffle=config["loader"]["shuffle"],
            num_workers=config["loader"]["n_workers"],
            prefetch_factor=config["loader"]["prefetch_factor"],
            worker_init_fn=seed_worker,
            pin_memory=True,
            drop_last=True,
            )

    # model initialization and settings (load model from scratch)
    model = HydraEVNet(
        kwargs=config["model"].copy(), 
        num_bins=num_bins,
        final_w_scale_flow=config["custom"]["final_w_scale_flow"],
        current_flow_sup=config["custom"]["current_flow_sup"],
        current_flow_scaling=event_dt_ms
        )
    model = model.to(device)
    print("Model initialized on device: ", device)
    
    loaded_from_checkpoint = False
    if checkpoint_path:
        model = load_model(model, device, model_dir=checkpoint_path)
        loaded_from_checkpoint = True
    else:
        print("Loading model from scratch")
        model.apply(initialize_weights)
    model.train()

    # loss functions
    loss_function = HermLoss(config, device=device)

    # optimizers
    optimizer = eval(config["optimizer"]["name"])(model.parameters(), lr=config["optimizer"]["lr"])
    optimizer, starting_epoch, total_seen_samples = load_optimizer_epoch_seen_samples(
        optimizer, device, model_dir=checkpoint_path)
    optimizer.zero_grad()

    scheduler = get_scaled_lr_scheduler(
        optimizer=optimizer, config=config, steps_per_epoch=len(train_loader)
        )

    # Init wandb
    if loaded_from_checkpoint:
        model_name = os.path.basename(os.path.dirname(checkpoint_path))
    else:
        model_name = "Herm_"
        model_name += datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if config["vis"]["log_wandb"]:
        run = wandb.init(project="events-to-dynamic-mask", config=config, name=model_name)

    print("Training started")
    print("Dataset size: ", len(train_dataset))

    # Epochs loop
    for epoch in range(config["loader"]["n_epochs"]):
        if epoch <= starting_epoch:
            print(f"Skipping epoch {epoch} as it is less/= than starting epoch {starting_epoch}")
            continue
        print(f"Epoch {epoch} started")
        for batch_idx, batch_data in enumerate(train_loader):
            it_start_time = time.time()
            loss_sequence = torch.tensor(0.0).to(device)
            for data_ind in range(len(batch_data)-1):
                data_t0 = batch_data[data_ind]
                data_t1 = batch_data[data_ind+1]
                    
                event_voxel = data_t0["representation"].to(device)
                # TODO we need to linearly interpolate flow and mask and sample random dt
                if len(data_t1['sampled_dt']) == 1:
                    dt = torch.tensor([data_t1['sampled_dt']]).float().to(device)
                else:
                    dt = data_t1['sampled_dt'].float().to(device)
                
                # if the batch is not full, skip the batch entirely
                if event_voxel.shape[0] != config["loader"]["batch_size"]:
                    print("Skipping batch as it is not full")
                    continue
                
                x = model(event_voxel, dt)
                flow_t1 = x["flow"]
                mask = x["mask"]
                future_mask = x["future_mask"]

                # loss computation
                # NOTE: since optical flow is in px and calculated for 100ms
                # we need to scale it to the input time which is 50 ms
                # |(t_-1)->start --- (t_0)->current --- (t_1)->future|
                
                dynamic_mask_t0_gt = data_t0["dynamic_mask"].to(device)
                dynamic_mask_t1_gt = data_t1["dynamic_mask"].to(device)

                dynamic_mask_t0_gt = label_smoothing(dynamic_mask_t0_gt)
                dynamic_mask_t1_gt = label_smoothing(dynamic_mask_t1_gt)

                event_list = data_t1["event_list"].float().to(device).detach()
                pol_mask = data_t1["polarity_mask"].float().to(device).detach()
                d_event_list = data_t1["d_event_list"].float().to(device).detach()
                d_pol_mask = data_t1["d_polarity_mask"].float().to(device).detach()

                loss_function.update(pred_mask_t0=mask, gt_mask_t0=dynamic_mask_t0_gt,
                                     pred_mask_t1=future_mask, gt_mask_t1=dynamic_mask_t1_gt,
                                     pred_flow_t1=flow_t1,
                                     event_list=event_list, pol_mask=pol_mask, 
                                     d_event_list=d_event_list, d_pol_mask=d_pol_mask
                                     )

            # loss computation and backpropagation
            loss_sequence = loss_function()
            loss_sequence.backward()

            if config["loss"]["clip_grad"] is not None:
                torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), config["loss"]["clip_grad"])
            
            optimizer.step()
            optimizer.zero_grad()
            
            # reset the states after a batch
            model.detach_states()
            loss_function.reset()
            total_seen_samples += 1

            if config["vis"]["verbose"]:
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
            
            if config["vis"]["log_wandb"]:
                logging_dict = train_log_herm(
                                mask_pred_t0=mask[-1],
                                mask_target_t0=dynamic_mask_t0_gt,
                                mask_pred_t1=future_mask[-1],
                                mask_target_t1=dynamic_mask_t1_gt,
                                event_voxel=event_voxel,
                                config=config
                                )
                if config["vis"]["plot_wandb"]:
                    # Debugging images
                    flow_to_plot = flow_t1[-1][0].permute(1,2, 0).detach().cpu().numpy()
                    rendered_flow = Visualization.flow_to_image(flow_to_plot)
                    pred_flow_wandb_img = wandb.Image(rendered_flow)
                    logging_dict.update({"images/flow/t1/pred_total_flow": pred_flow_wandb_img})

                    mask_t0_wandb_img = plot_segmentation_masks(
                        pred=mask[-1].detach().cpu(), 
                        smooth_target=dynamic_mask_t0_gt.detach().cpu(), 
                        event_voxel=event_voxel.detach().cpu())
                    mask_t1_wandb_img = plot_segmentation_masks(
                        pred=future_mask[-1].detach().cpu(), 
                        smooth_target=dynamic_mask_t1_gt.detach().cpu(), 
                        event_voxel=event_voxel.detach().cpu())
                    logging_dict.update({"images/mask/t0": mask_t0_wandb_img})
                    logging_dict.update({"images/mask/t1": mask_t1_wandb_img})

                run.log(logging_dict,step=total_seen_samples)

        model.reset_states()
        scheduler.step()
        optimizer.zero_grad()
        model_dir = save_model(
            model=model, 
            optimizer=optimizer, 
            epoch=epoch, 
            loss=loss_sequence, 
            path_results=config["loader"]["checkpoints_path"],
            model_name=model_name,
            total_seen_samples=total_seen_samples
            )
        validator = ValidateHerm(config=config, model_path=model_dir)
        save_path = "results" + f"/{model_name}/model_epoch_{epoch}"
        results = validator.validate_model(save_path=save_path)
        if config["vis"]["log_wandb"]:
            run.log(results, step=total_seen_samples)
        
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/train_herm.json",
        help="training configuration",
    )
    parser.add_argument(
        "--path_cache",
        default="",
        help="location of the cache version of the formatted dataset",
    )
    # add checkpoint path to eventually resume training
    parser.add_argument(
        "--checkpoint",
        default="",
        help="location of the checkpoint to resume training",
    )
    args = parser.parse_args()

    # launch training
    train(config_path=args.config, checkpoint_path=args.checkpoint)
