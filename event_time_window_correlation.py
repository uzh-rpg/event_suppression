import numpy as np
from sensibility_test import SensibiliyTest
from suppressor.utils.utils import open_config_json

if __name__ == "__main__":
    config_path = "dynamic_masker/configs/train_herm.json" # Path to your config
    model_path = "checkpoints/Herm_2025-04-15_15-49-03/model_epoch_49.pth" # Path to your model
    save_path = "dynamic_masker/results/sensitivity_plots" # Output path

    try:
        config = open_config_json(config_path)
    except FileNotFoundError:
        print(f"Config file not found at {config_path}. Please ensure it exists.")
        # Create a dummy config for the script to run if needed for placeholder
        config = {"vis": {"plot": False, "verbose": True}, "dataset_params": {"name": "dummy_dataset"},
                  "model_params": {}, "dataloader_params": {}} # Adjust as per ValidateHerm needs
        print("Using a placeholder config.")


    validator = SensibiliyTest(config=config, model_path=model_path)

    event_time_window_sizes = list(np.logspace(start=2, stop=6, num=20).astype(int))
    event_numbers = []
    delta_ts = []
    for window in event_time_window_sizes:
        print(f"Evaluating for event time window size: {window} events")
        dataset = validator._get_dataset(window_ev_num=window, only_sequence="box")
        dataset_delta_ts = []
        for n_, subdataset in enumerate(dataset):
            if n_ >= 1:
                continue  # Limit to first 1 sequences for speed
            for data in subdataset:
                t = (data['events'][-1][2]-data['events'][0][2])*1000  # in milliseconds
                dataset_delta_ts.append(t)
        delta_ts.append(dataset_delta_ts)
        event_numbers.append([window] * len(dataset_delta_ts))
    print("Event time window sizes:", event_time_window_sizes)
    delta_ts = np.array(delta_ts).flatten()
    event_numbers = np.array(event_numbers).flatten()
    SensibiliyTest.correlation_log_plot_general(
        columns=event_numbers,
        rows=delta_ts,
        column_name="Number of Events",
        rows_name="Time Window (ms)",
        title="",
        save_path=f"{save_path}/event_time_window_correlation.png"
    )