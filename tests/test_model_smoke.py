import torch

from evsup.models.model_hydra import HydraEVNet


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

    assert set(outputs) == {"mask", "future_mask", "flow", "flow_t0"}
    assert len(outputs["mask"]) == 2
    assert len(outputs["future_mask"]) == 2
    assert len(outputs["flow"]) == 2
    assert len(outputs["flow_t0"]) == 2

    for mask in outputs["mask"]:
        assert mask.shape == (1, 1, 32, 32)
    for future_mask in outputs["future_mask"]:
        assert future_mask.shape == (1, 1, 32, 32)
    for flow in outputs["flow"]:
        assert flow.shape == (1, 2, 32, 32)
    for flow_t0 in outputs["flow_t0"]:
        assert flow_t0.shape == (1, 2, 32, 32)
