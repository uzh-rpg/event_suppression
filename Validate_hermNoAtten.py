import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from Validate import open_config_json
from Validate_herm import ValidateHerm
from dynamic_masker.utils.utils import load_model

from dynamic_masker.models.model_hydraNoAtten import HydraEVNetNoAtten


class ValidateHermNoAtten(ValidateHerm):
    def __init__(self, config, model_path):
        super().__init__(config, model_path)

    def setup_model(self, model_path):
        model_config = self.config["model"].copy()
        final_w_scale_flow = self.config["custom"]["final_w_scale_flow"]
        current_flow_sup = self.config["custom"]["current_flow_sup"]
        event_dt_ms = self.config["loader"]["event_dt_ms"]

        model = HydraEVNetNoAtten(
            kwargs=model_config, 
            num_bins=self.num_bins,
            final_w_scale_flow=final_w_scale_flow,
            current_flow_sup=current_flow_sup,
            current_flow_scaling=event_dt_ms
            )
        model = model.to(self.device)
        return load_model(model, self.device, model_dir=model_path)
    

    
if __name__ == "__main__":
    config = open_config_json("dynamic_masker/configs/validate_herm.json")
    validator = ValidateHermNoAtten(
                        config=config,
                        model_path="checkpoints/HermNoAtten_2025-06-19_17-59-22/model_epoch_4.pth"
    )
    # validator.validate_model(
    #     save_path="dynamic_masker/results/HermNoAtten_2025-06-19_17-59-22/model_epoch_4.pth"
    # )
    
    validator.validate_all_models_in_folder(
        model_folder="checkpoints/HermNoAtten_2025-06-19_17-59-22",
        results_folder="results/HermNoAtten_2025-06-19_17-59-22"
    )