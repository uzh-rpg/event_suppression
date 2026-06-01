import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import time
import numpy as np
from tqdm import tqdm

from Validate import open_config_json
from Validate_herm import ValidateHerm

# Fix CPU threads
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Inference timing utility
def measure_inference_speed(model, device, num_repeats=100, warmup_repeats=10):
    model.eval()
    times = []
    voxel = torch.randn(1, 2, 260, 346).to(device)  # Dummy voxel input
    dt = torch.tensor([0.1]).to(device)  # Dummy dt input

    with torch.no_grad():
        for i in tqdm(range(num_repeats), desc="Measuring inference speed"):

            # Warm-up (to ignore first-run setup overhead)
            if i == 0:
                for _ in range(warmup_repeats):
                    _ = model(x=voxel, dt=dt)

            torch.cuda.synchronize()
            start = time.time()

            _ = model(x=voxel, dt=dt)

            torch.cuda.synchronize()
            end = time.time()

            inference_time = (end - start) * 1000  # ms
            times.append(inference_time)

    mean_time = np.mean(times)
    std_time = np.std(times)
    return mean_time, std_time, times


if __name__ == "__main__":
    config = open_config_json("dynamic_masker/configs/validate_herm.json")
    validator = ValidateHerm(
        config=config,
        model_path="dynamic_masker/checkpoints/Herm_2025-04-15_15-49-03/model_epoch_15.pth"
    )
    device = validator.device
    validator.model = validator.setup_model(validator.model_path)
    validator.model.eval()

    print(f"Using device: {device}")
    mean_time, std_time, all_times = measure_inference_speed(
        model=validator.model,
        # device=device,"""  """
        num_repeats=1000,  # Adjust as needed
        warmup_repeats=10  # Warm-up iterations to ignore initial overhead
    )

    print(f"\nAverage Inference Time: {mean_time:.2f} ms")
    print(f"Standard Deviation: {std_time:.2f} ms")
