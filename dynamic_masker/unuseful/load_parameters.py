import os
import torch
from unuseful.parser import YAMLParser
from models.model import RecEVFlowNet


def import_params(run_id, mlflow_dir):
    params_dir = f"{mlflow_dir}/{run_id}/params/"
    params = {}

    for param_file in os.listdir(params_dir):
        with open(os.path.join(params_dir, param_file), "r") as f:
            params[param_file] = f.read().strip()
    return params


def import_metrics(run_id, mlflow_dir):
    metrics_dir = f"{mlflow_dir}/{run_id}/metrics/"
    metrics = {}

    for metric_file in os.listdir(metrics_dir):
        with open(os.path.join(metrics_dir, metric_file), "r") as f:
            values = [line.strip().split() for line in f.readlines()]
            metrics[metric_file] = [float(v[1]) for v in values]  # Extract metric values
    return metrics


def extend_params(params, eval_flow_config_path):
    config_parser = YAMLParser(eval_flow_config_path)
    config = config_parser.merge_configs(params)
        
    # configs
    config["loader"]["batch_size"] = 1
    device = config_parser.device
    kwargs = config_parser.loader_kwargs

    # initialize settings
    config["loader"]["device"] = device
    return config, device, kwargs, config_parser


def load_model_weights(model, device, experiment_dir):
    model_dir = experiment_dir + "/artifacts/model/data/model.pth"
    if model_dir[:7] == "file://":
        model_dir = model_dir[7:]

    if os.path.isfile(model_dir):
        model_loaded = torch.load(model_dir, map_location=device).state_dict()

        # check for input-dependent layers
        for key in model_loaded.keys():
            if key.split(".")[1] == "pooling" and key.split(".")[-1] in ["weight", "weight_f"]:
                model.encoder_unet.pooling = model.encoder_unet.build_pooling(model_loaded[key].shape).to(device)
                model.encoder_unet.get_axonal_delays()

        new_params = model.state_dict()
        new_params.update(model_loaded)
        model.load_state_dict(new_params)

        print("Model restored")
    else:
        print("No model found")

    return model, 0

def load_full_model(config, device, experiment_dir):
    num_bins = 2 if config["data"]["voxel"] is None else config["data"]["voxel"]
    model = eval(config["model"]["name"])(config["model"].copy(), num_bins)
    model = model.to(device)
    model, _ = load_model_weights(model, device, experiment_dir)
    model.eval()
    return model

