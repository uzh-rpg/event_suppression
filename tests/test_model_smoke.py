import torch

from dynamic_masker.models.model_hydra import HydraEVNet


def test_hydra_model_forward_smoke_cpu():
    model = HydraEVNet(
        kwargs={"final_w_scale": 1.0, "num_encoders": 2, "num_residual_blocks": 1},
        num_bins=2,
        final_w_scale_flow=0.01,
        current_flow_scaling=50,
    )
    model.eval()

    with torch.no_grad():
        outputs = model(torch.randn(1, 2, 32, 32), torch.tensor([[0.05]]))

    assert outputs["mask"][-1].shape == (1, 1, 32, 32)
    assert outputs["future_mask"][-1].shape == (1, 1, 32, 32)
    assert outputs["flow"][-1].shape == (1, 2, 32, 32)
