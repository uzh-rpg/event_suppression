import os
import torch
from pathlib import Path
from torch.utils.data.dataloader import default_collate
from torch.optim.lr_scheduler import OneCycleLR

from evsup.models.model_hydra import HydraEVNet
from evsup.config import load_config


open_config_json = load_config

def setup_hydra_model(model_path, config_path, device):
    config = open_config_json(config_path)
    model_config = config["model"].copy()
    final_w_scale_flow = config["custom"]["final_w_scale_flow"]
    current_flow_sup = config["custom"]["current_flow_sup"]
    event_dt_ms = config["loader"]["event_dt_ms"]
    num_bins = config["data"]["voxel_bins"]
    model = HydraEVNet(
        kwargs=model_config, 
        num_bins=num_bins,
        final_w_scale_flow=final_w_scale_flow,
        current_flow_sup=current_flow_sup,
        current_flow_scaling=event_dt_ms
        )
    model = model.to(device)
    return load_model(model, device, model_dir=model_path)

def load_model(model, device, model_dir="/checkpoints/data/model.pth"):
    if os.path.isfile(model_dir):
        print(f"Model found in {model_dir} LOADING ...")
        checkpoint = torch.load(model_dir, map_location=device, weights_only=False)["model_state_dict"]

        # Patch input-dependent layers if needed
        for key in checkpoint.keys():
            if key.split(".")[1] == "pooling" and key.split(".")[-1] in ["weight", "weight_f"]:
                model.encoder_unet.pooling = model.encoder_unet.build_pooling(checkpoint[key].shape).to(device)
                model.encoder_unet.get_axonal_delays()

        # Strip "module." prefix if present (from DDP)
        new_state_dict = {}
        for k, v in checkpoint.items():
            new_key = k.replace("module.", "") if k.startswith("module.") else k
            new_state_dict[new_key] = v

        # Load updated state dict
        model.load_state_dict(new_state_dict)
    else:
        print(f"\nNo model found in this dir: {model_dir} LOADING FROM SCRATCH\n")

    return model


def load_optimizer_epoch_seen_samples(optimizer, device, model_dir=""):
    if os.path.isfile(model_dir):
        checkpoint = torch.load(model_dir, map_location=device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch']
        # legacy code to load old models without total_seen_samples
        if 'total_seen_samples' in checkpoint.keys():
            total_seen_samples = checkpoint['total_seen_samples']
        else:
            total_seen_samples = 0
    else:
        starting_epoch = -1
        total_seen_samples = 0
    
    return optimizer, starting_epoch, total_seen_samples

def save_model(model, optimizer, epoch, loss, path_results, model_name, total_seen_samples):
    if not Path(path_results).is_dir():
        os.makedirs(path_results)
    model_dir = path_results + f"/{model_name}/model_epoch_{epoch}.pth"
    if not Path(path_results + f"/{model_name}").is_dir():
        os.makedirs(path_results + f"/{model_name}")
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'total_seen_samples': total_seen_samples,
    }
    torch.save(checkpoint, model_dir)
    return model_dir

def create_model_dir(path_results, runid):
    path_results += runid + "/"
    if not os.path.exists(path_results):
        os.makedirs(path_results)
    print("Results stored at " + path_results + "\n")
    return path_results

def binary_search_array(array, x, left=None, right=None, side="left"):
    left = 0 if left is None else left
    right = len(array) - 1 if right is None else right
    mid = left + (right - left) // 2

    if left > right:
        return left if side == "left" else right

    if array[mid] == x:
        return mid

    if x < array[mid]:
        return binary_search_array(array, x, left=left, right=mid - 1, side=side)

    return binary_search_array(array, x, left=mid + 1, right=right, side=side)

def strict_flow_collate_fn(batch_):
    """
    Collate only if all items have flow or none have flow.
    Otherwise, return None to signal skipping the batch.
    """
    # TODO this collate is SLOW if batch is SMALL

    # Check if batch is empty
    batch = []
    for b in batch_:
        if b is not None:
            batch.append(b)
        else:
            print("[Collate Warning] a part of the batch has None item")
            
    # batch = [b for b in batch if b is not None]
    #TODO remove as this is redundant
    if batch == None or len(batch) == 0:
        print("[Collate Warning] Skipping batch with None item")
        return None

    # Check the forward_flow_gt in each item
    has_flow_flags = [
        all(item_i.get('has_flow')==True for item_i in item)
        for item in batch
    ]

    # Ensure all items in the batch agree: all have flow or all don't
    if all(has_flow_flags) or not any(has_flow_flags):
        return default_collate(batch)

    # Ensure all samples have the same number of frames
    seq_lengths = [len(sample) for sample in batch]
    if len(set(seq_lengths)) != 1:
        print(f"[Collate Warning] Skipping batch with inconsistent lengths: {seq_lengths}")
        return None

    # Mixed batch (some with flow, some without) — skip
    print("[Collate Warning] Skipping batch with mixed flow")
    return None

def custom_collate(batch):
    """
    Collate function for variable-length event data.
    Each item in the batch is a list (sequence) of data dicts.
    So we first want to collate over the outer batch dimension,
    and then manage the inhomogeneous fields.
    """
    seq_len = len(batch[0])  # assuming consistent seq_len across samples

    # Initialize lists to hold the data
    collated = [[] for _ in range(seq_len)]

    for b in batch:
        for i, item in enumerate(b):
            collated[i].append(item)

    # Now, for each time step, collate homogeneous fields and keep lists for inhomogeneous
    output = []
    for timestep_data in collated:
        timestep_output = {}
        keys = timestep_data[0].keys()

        for key in keys:
            items = [d[key] for d in timestep_data]

            if key in ["event_list", "d_event_list", "polarity_mask", "d_polarity_mask"]:
                # Keep these as a list (i.e. variable-length)
                timestep_output[key] = items
            elif isinstance(items[0], torch.Tensor) and items[0].ndim > 0:
                timestep_output[key] = torch.stack(items)
            else:
                timestep_output[key] = items

        output.append(timestep_output)

    return output

def get_scaled_lr_scheduler(
    optimizer, 
    config,
    steps_per_epoch,
    warmup_epochs=2
    ):
    """
    Returns a OneCycleLR scheduler with scaled base LR according to batch size.

    Args:
        optimizer (Optimizer): PyTorch optimizer.
        config (dict): Training configuration containing optimizer and loader settings.
        steps_per_epoch (int): Number of steps per epoch.
        warmup_epochs (int): Number of epochs for warm-up (affects pct_start in OneCycleLR).

    Returns:
        scheduler (OneCycleLR): A PyTorch OneCycleLR scheduler.
    """
    base_lr = config["optimizer"]["lr"]
    base_batch_size = 1 
    current_batch_size = config["loader"]["batch_size"]
    num_epochs = config["loader"]["n_epochs"]

    # Scale LR based on batch size
    scaled_lr = base_lr * current_batch_size / base_batch_size
    for param_group in optimizer.param_groups:
        param_group['lr'] = scaled_lr

    total_steps = steps_per_epoch * num_epochs
    pct_start = warmup_epochs / float(num_epochs)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=scaled_lr,
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs,
        pct_start=pct_start,
        anneal_strategy='cos',  # You can also try 'linear'
        cycle_momentum=False  # Set to False for Adam/AdamW
    )
    return scheduler
