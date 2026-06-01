import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch

from Validate import open_config_json
from fvcore.nn import FlopCountAnalysis
from dynamic_masker.models.model_hydra_flops import HydraEVNet



if __name__ == "__main__":
    config = open_config_json("dynamic_masker/configs/validate_herm.json")
    # validator = ValidateHerm(
    #     config=config,
    #     model_path="dynamic_masker/checkpoints/Herm_2025-04-15_15-49-03/model_epoch_15.pth"
    # )
    # validator.model = validator.setup_model(validator.model_path)
    # validator.model.eval().cpu()

    model_config = config["model"].copy()
    final_w_scale_flow = config["custom"]["final_w_scale_flow"]
    current_flow_sup = config["custom"]["current_flow_sup"]
    event_dt_ms = config["loader"]["event_dt_ms"]

    model = HydraEVNet(
        kwargs=model_config, 
        num_bins=2,
        final_w_scale_flow=final_w_scale_flow,
        current_flow_sup=current_flow_sup,
        current_flow_scaling=event_dt_ms
        )
    model = model.eval().cpu()
    voxel = torch.randn(1, 2, 260, 346).cpu()
    dt = torch.tensor([0.1]).cpu()

    flops = FlopCountAnalysis(model, (voxel, dt))
    print("FLOPs: ", flops.total())




