import numpy as np
import random
import torch
import yaml

def get_device(gpu_num=0):
    """
    Get the device to use in the pipeline.
    """
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(gpu_num) if cuda else "cpu")
    return device

@staticmethod
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def init_seeds(self):
    """
    Initialize random seeds.
    """
    torch.manual_seed(_config["loader"]["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(_config["loader"]["seed"])
        torch.cuda.manual_seed_all(_config["loader"]["seed"])
    np.random.seed(_config["loader"]["seed"])
    random.seed(_config["loader"]["seed"])
