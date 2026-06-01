import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sensibility_test import SensibiliyTest
from suppressor.utils.utils import open_config_json


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

if __name__ == "__main__":
    config_path = "dynamic_masker/configs/train_herm.json"  # Path to your config
    model_path = "checkpoints/Herm_2025-04-15_15-49-03/model_epoch_49.pth"  # Path to your model
    save_path = Path("dynamic_masker/results/sensitivity_plots/object_velocity")  # Output path

    config = open_config_json(config_path)
    validator = SensibiliyTest(config=config, model_path=model_path)

    # check if velocities were already measured
    if (save_path / "measured_velocities.txt").exists():
        all_velocities = np.loadtxt(save_path / "measured_velocities.txt")
        print("Measured velocities file already exists. Skipping to avoid redundant computation.")
    else:
        dataset = validator._get_dataset(window_ev_num=50)
        all_velocities = []
        velocities_per_sequence = []

        for sequence in dataset:
            sequence_main_name = Path(sequence.h5_path).parent.name
            velocities = validator.measure_dataset_velocities(sequence, save_path)
            vel = velocities["velocities"]
            all_velocities.extend(vel)
            velocities_per_sequence.append({sequence_main_name: (np.array(vel).min(), np.array(vel).mean(), np.array(vel).max())})

            print(f"Measured velocities for sequence, average velocity: {np.mean(vel):.2f}")

        # Convert velocities to px/s
        all_velocities = (np.array(all_velocities)/100.0)*1e3
        
        prev_seq = ""
        seq_list = {}
        for elem in velocities_per_sequence:
            seq_name = list(elem.keys())[0]
            min_vel, mean_vel, max_vel = elem[seq_name]
            if seq_name != prev_seq:
               seq_list[seq_name] = [mean_vel]
            else:
                seq_list[seq_name].append(mean_vel) 
            prev_seq = seq_name
            

        # Save velocities to a text file
        save_path.mkdir(parents=True, exist_ok=True)
        np.savetxt(save_path / "measured_velocities.txt", all_velocities)
        print(f"Measured velocities saved to {save_path / 'measured_velocities.txt'}")

    # --- Histogram Plot ---
    v_clip = np.clip(all_velocities, a_min=None, a_max=1000)
    bins = np.linspace(v_clip.min(), 1000, 100)

    plt.figure(figsize=(8, 5))
    counts, edges, patches = plt.hist(v_clip, bins=bins, color='gray', edgecolor='black', alpha=0.7)
    plt.xlim(0, 1000)                 # visually cap at 1000
    plt.xlabel("Velocity (px/s)")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.5)

    # Relabel the last tick to show 1000+
    xticks = np.arange(100, 1001, 100)      # 100, 200, ..., 1000
    xticklabels = [("1000+" if np.isclose(t, 1000) else f"{t:g}") for t in xticks]
    plt.xticks(xticks, xticklabels)

    # Save plot
    plt.tight_layout()
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path / "velocity_histogram.png", dpi=300)
    plt.close()
    print(f"Histogram saved to {save_path / 'velocity_histogram.png'}")

