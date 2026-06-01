import json
import torch
import wandb
import argparse
from torch.optim import Adam, AdamW
from matplotlib import pyplot as plt
from datetime import datetime

from dynamic_masker.loss.BCEDiceLoss import BCEDiceLoss
from dynamic_masker.loss.FocalLoss import FocalLoss

from dynamic_masker.utils.utils import load_model, save_model, load_optimizer_epoch_seen_samples
from dynamic_masker.models.model_util import initialize_weights, label_smoothing
from suppressor.DSEC_dataloader.provider import DatasetProvider
from dynamic_masker.models.model import RecEVFlowNet
from dynamic_masker.configs.utils import get_device

from dynamic_masker.utils.grads_log import debug_training_step
from dynamic_masker.utils.train_log import train_log


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
    train_dataset = dataset_provider.get_recurrent_train_dataset(sequence_len=config["data"]["sequence_len"])
    train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            drop_last=True,
            batch_size=batch_size,
            shuffle=True,
            num_workers=config["loader"]["n_workers"],
            pin_memory=True
            )

    # model initialization and settings (load model from scratch)
    model_config = config["model"].copy()
    model = RecEVFlowNet(model_config, num_bins, key="dynamic_mask")
    model = model.to(device)
    model = load_model(model, device, model_dir="")
    model.apply(initialize_weights)
    model.train()

    # loss functions
    loss_function = eval(config["loss"]["name"])(**config["loss"])

    # optimizers
    optimizer = eval(config["optimizer"]["name"])(model.parameters(), lr=config["optimizer"]["lr"])
    optimizer, epoch, loss0 = load_optimizer_epoch_seen_samples(optimizer, device)
    optimizer.zero_grad()

    # simulation variables
    total_seen_samples = 0

    # Init wandb
    model_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if config["vis"]["log_wandb"]:
        run = wandb.init(project="events-to-dynamic-mask", config=config, name=model_name)

    print("Training started")
    print("Dataset size: ", len(train_dataset))

    # dataloader loop
    for epoch in range(config["loader"]["n_epochs"]):
        for batch_idx, batch_data in enumerate(train_loader):
            for data in batch_data:
                depth_gt = data["depth_gt"]
                file_index = data["file_index"]
                sequence_id = data["sequence_id"]
                positive_pixels_percentage = data["positive_pixels_percentage"]
                negative_pixels_percentage = data["negative_pixels_percentage"]
                dynamic_mask_gt = data["dynamic_mask_gt"].to(device)
                event_voxel = data["representation"]["left"].to(device)

                x = model(event_voxel)

            # loss computation
            smooth_target = label_smoothing(dynamic_mask_gt)
            loss = loss_function(pred=x["dynamic_mask"][-1], target=smooth_target)
            loss.backward()

            if config["loss"]["clip_grad"] is not None:
                torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), config["loss"]["clip_grad"])
            
            optimizer.step()

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
                train_log(run=run, 
                          pred=x["dynamic_mask"][-1].detach().cpu(), 
                          target=dynamic_mask_gt.detach().cpu(), 
                          event_voxel=event_voxel.detach().cpu(), 
                          smooth_target=smooth_target, 
                          loss=loss, 
                          pos_pixels=positive_pixels_percentage, 
                          neg_pixels=negative_pixels_percentage, 
                          config=config, 
                          total_seen_samples=total_seen_samples
                          )
                debug_training_step(model=model, 
                                    optimizer=optimizer, 
                                    step=total_seen_samples
                                    )

            # reset the states after a batch
            model.reset_states()
            optimizer.zero_grad()
            total_seen_samples += 1

        optimizer.zero_grad()
        model.reset_states()
        save_model(
            model=model, 
            optimizer=optimizer, 
            epoch=epoch, 
            loss=loss, 
            path_results=config["loader"]["checkpoints_path"],
            model_name=model_name
            )
    wandb.finish()
  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/train.json",
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
