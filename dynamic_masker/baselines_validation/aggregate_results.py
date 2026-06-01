from suppressor.utils.aggregate_results import aggregate_results

if __name__ == "__main__":
    # Example usage
    # path = "results/Herm_2025-04-15_15-49-03/current_as_future"
    path = "dynamic_masker/results/Herm_2025-04-15_15-49-03/model_epoch_49"
    aggregate_results(path_to_json=f"{path}/results.json", save_path=f"{path}/aggregated_results.json")