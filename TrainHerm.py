# Assuming all the imports from the base class file are available here
import os
import torch
from datetime import datetime
from suppressor.DSEC_dataloader.provider import DatasetProvider
from TrainBaseHydra import TrainBaseHydra


class TrainHerm(TrainBaseHydra):
    """
    A training class for the 'Herm' configuration that inherits from TrainBaseHydra
    but uses the EVIMO dataset instead of the default DSEC dataset.
    """

    def init_model_name(self):
        """
        Overrides the base method to provide a more specific name for this model's runs.
        """
        if self.checkpoint_path:
            # If loading from a checkpoint, use its existing name
            return os.path.basename(os.path.dirname(self.checkpoint_path))
        else:
            # Otherwise, create a new name prefixed with "Herm_"
            return "Herm_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def build_dataloader(self):
        """
        Overrides the base dataloader to load the EVIMO dataset.

        This method replaces the call to `get_hydra_train_dataset` with
        `get_evimo_train_dataset` as specified.
        """
        print("Building dataloader for EVIMO dataset...")
        provider = DatasetProvider(
            dataset_path=self.config["data"]["path"],
            representation=self.config["data"]["representation"],
            num_bins=self.config["data"]["voxel_bins"],
            delta_t_ms=self.config["loader"]["event_dt_ms"]
        )

        # Get the EVIMO train dataset instead of the Hydra/DSEC one.
        # Note: The 'batch_size' argument is handled by the DataLoader, not the Dataset object itself.
        # This follows standard PyTorch practice.
        train_dataset = provider.get_evimo_train_dataset(
            sequence_len=self.config["data"]["sequence_len"], 
            batch_size=self.config["loader"]["batch_size"]
        )

        # The rest of the DataLoader setup is identical to the base class
        return torch.utils.data.DataLoader(
            dataset=train_dataset,
            drop_last=True,
            batch_size=self.config["loader"]["batch_size"],
            shuffle=self.config["loader"]["shuffle"],
            num_workers=self.config["loader"]["n_workers"],
            prefetch_factor=self.config["loader"]["prefetch_factor"],
            worker_init_fn=self.seed_worker,
            pin_memory=True
        )


if __name__ == "__main__":
    # --- Original Trainer ---
    # print("Instantiating the base trainer for DSEC dataset...")
    # hydra_trainer = TrainBaseHydra(config_path="configs/train_hydra.json", checkpoint_path="")
    # hydra_trainer.train()
    # hydra_trainer.finish()

    # --- New Inherited Trainer ---
    print("\nInstantiating the new trainer for EVIMO dataset...")
    # NOTE: You will likely need a different config file (e.g., "configs/train_herm.json")
    # that points to the correct EVIMO dataset path and may have other different parameters.
    # For demonstration, we use the same config path.
    herm_trainer = TrainHerm(config_path="configs/train_hydra.json", checkpoint_path="")

    # You can now run the training using the exact same methods as the base class.
    # The new dataloader will be used automatically.
    print(f"Trainer created. Model name will be: {herm_trainer.model_name}")
    print("To start training, call: herm_trainer.train()")

    # Example of starting the training (uncomment to run)
    herm_trainer.train()
    herm_trainer.finish()