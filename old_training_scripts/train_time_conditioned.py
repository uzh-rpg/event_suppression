import json
import torch
import wandb
import random
import argparse
import numpy as np
from io import BytesIO
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from matplotlib import pyplot as plt
from datetime import datetime

from dynamic_masker.loss.L1Loss import L1Loss
from dynamic_masker.loss.flow import Iterative

from dynamic_masker.utils.utils import load_model, save_model, load_optimizer_epoch_seen_samples
from dynamic_masker.models.model_util import initialize_weights, label_smoothing
from dynamic_masker.models.model_time_conditioned import TimeCondRecEVFlowNet
from dynamic_masker.utils.visualization import Visualization
from dynamic_masker.configs.utils import get_device

from suppressor.DSEC_dataloader.provider import DatasetProvider
from suppressor.utils.utils import render_events

def seed_worker(worker_id):
    """Ensure each worker has a deterministic seed"""
    seed = torch.initial_seed() % (2**32)  # Get unique seed for each worker
    np.random.seed(seed)
    random.seed(seed)

# Set the seed before creating the DataLoader
torch.manual_seed(42)  # For CPU & CUDA
np.random.seed(42)  # Numpy seed
random.seed(42)  # Python seed

def train(config_path):
    """
    Main function of the training pipeline for event-based optical flow estimation.
    """

    # configs
    config = json.load(open(config_path, 'r'))

    # initialize settings
    device = get_device(gpu_num=config["loader"]["gpu"])
    config["loader"]["device"] = device

    # visualization tool
    if config["vis"]["enabled"]:
        pass
    
    # data loader
    num_bins = config["data"]["voxel_bins"]
    dsec_dir = config["data"]["path"]
    batch_size=config["loader"]["batch_size"]
    dataset_provider = DatasetProvider(dsec_dir, num_bins=num_bins, representation=config["data"]["representation"])
    train_dataset = dataset_provider.get_flow_train_dataset(
        sequence_len=config["data"]["passes_loss"], # passes_loss corresponds to sequence len
        max_num_grad_events=config["data"]["max_num_grad_events"],
        dt = config["data"]["dt_ms"]
        )
    train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            drop_last=True,
            batch_size=batch_size,
            shuffle=True,
            num_workers=config["loader"]["n_workers"],
            worker_init_fn=seed_worker,
            pin_memory=True
            )

    # model initialization and settings (load model from scratch)
    model_config = config["model"].copy()
    model = TimeCondRecEVFlowNet(model_config, num_bins, key="future_OF")
    model = model.to(device)
    print("WARNING: Always loading the model from scratch")
    model.apply(initialize_weights)
    model.train()

    # loss functions
    loss_function = eval(config["loss"]["warping"])(config=config, device=device)

    # optimizers
    optimizer = eval(config["optimizer"]["name"])(model.parameters(), lr=config["optimizer"]["lr"])
    optimizer, epoch, loss0 = load_optimizer_epoch_seen_samples(optimizer, device)
    optimizer.zero_grad()

    # simulation variables
    total_seen_samples = 0

    # Init wandb
    model_name = "TimeCond_"
    # model_name = "TamingCM_"
    model_name += datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if config["vis"]["log_wandb"]:
        run = wandb.init(project="events-to-dynamic-mask", config=config, name=model_name)

    print("Training started")
    print("Dataset size: ", len(train_dataset))

    # dataloader loop
    for epoch in range(config["loader"]["n_epochs"]):
        for batch_idx, batch_data in enumerate(train_loader):
            for data in batch_data:
                event_voxel = data["representation"]["left"].to(device)
                dt = torch.tensor([data["sampled_dt"]]).float().to(device)

                # forward pass (flow in px/input_time)
                x = model(event_voxel, dt)
                for i in range(len(x["future_OF"])):
                    x["future_OF"][i] = x["future_OF"][i] * config["loss"]["flow_scaling"]

                loss_function.update(
                        flow_list=x["future_OF"],
                        event_list=data["event_list"].float().to(device).detach(),
                        pol_mask=data["polarity_mask"].float().to(device).detach(),
                        d_event_list=data["d_event_list"].float().to(device).detach(),
                        d_pol_mask=data["d_polarity_mask"].float().to(device).detach()
                    )

            # loss computation and backpropagation
            loss = loss_function()
            loss.backward()

            if config["loss"]["clip_grad"] is not None:
                torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), config["loss"]["clip_grad"])
            
            optimizer.step()
            optimizer.zero_grad()
            
            # reset the states after a batch
            model.detach_states()
            loss_function.reset()
            total_seen_samples += 1

            if config["vis"]["verbose"]:
                print(
                    "Train Epoch: {:04d} [{:03d}/{:03d} ({:03d}%)] Loss: {:.6f}".format(
                        epoch,
                        batch_idx,
                        len(train_loader),
                        int(100 * batch_idx / len(train_loader)),
                        loss.item(),
                    )
                )
            
            if config["vis"]["log_wandb"]:
                plt.clf()
                # Debugging images
                rendered_events = plot_events_from_data(data, config)

                flow_to_plot = x["future_OF"][-1][0].permute(1,2, 0).detach().cpu().numpy()
                rendered_flow = Visualization.flow_to_image(flow_to_plot)

                fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                axs[0].imshow(rendered_events)
                axs[0].axis("off")
                axs[0].set_title("Rendered Events")
                axs[1].imshow(rendered_flow)
                axs[1].axis("off")
                axs[1].set_title("Predicted Optical Flow")
                plt.tight_layout()

                events_rgb = wandb.Image(fig)

                run.log(
                        {   
                            "train/sample_loss": loss.item(),
                            "images/event_for_loss_compute": events_rgb
                        },
                        step=total_seen_samples
                    )


        model.reset_states()
        loss_function.reset()
        optimizer.zero_grad()
        save_model(
            model=model, 
            optimizer=optimizer, 
            epoch=epoch, 
            loss=loss, 
            path_results=config["loader"]["checkpoints_path"],
            model_name=model_name
            )
    wandb.finish()
  
def plot_events_from_data(data, config):
    t_ = data["event_list"][:,0]
    y_ = data["event_list"][:,1]
    x_ = data["event_list"][:,2]
    p_ = data["event_list"][:,3]
    width, height = config['loader']['resolution']
    mask_ev_outside = (x_ < height) & (x_ > 0) & (y_ < width) & (y_ > 0)
    x_ = x_[mask_ev_outside]
    y_ = y_[mask_ev_outside]
    t_ = t_[mask_ev_outside]
    p_ = p_[mask_ev_outside]
    rendered_events = render_events(torch.stack([x_, y_, t_, p_], dim=1), image_shape=(height, width))
    return rendered_events


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/train_time_cond.json",
        help="training configuration",
    )
    parser.add_argument(
        "--path_cache",
        default="",
        help="location of the cache version of the formatted dataset",
    )
    args = parser.parse_args()

    # launch training
    train(config_path=args.config)
