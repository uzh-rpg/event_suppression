import torch
import torch.nn.functional as F
from .model import RecEVFlowNet
from .arch_hydra import HydraUnetRecurrent


class HydraEVNet(RecEVFlowNet):
    """
    """

    net_type = HydraUnetRecurrent
    recurrent_block_type = "convgru"
    activations = ["relu", None]

    def __init__(
            self, 
            kwargs, 
            num_bins=2, 
            final_w_scale_flow=1.0, 
            min_size=16, 
            current_flow_sup=False, 
            current_flow_scaling=50
            ):
        # TODO: Hardcoded num_output_channels to 1 for dynamic mask
        kwargs["num_output_channels"] = 1
        self.final_w_scale_flow = final_w_scale_flow
        # TODO Remove current flow sup from kwargs
        self.current_flow_sup = current_flow_sup
        self.current_flow_scaling = current_flow_scaling # in ms 
    
        # Override Kwargs before calling parent constructor
        super().__init__(kwargs=kwargs, num_bins=num_bins, min_size=min_size)
        self.arch = self.net_type(
            self.arch_kwargs, final_w_scale_flow=self.final_w_scale_flow
            )

    def forward(self, x, dt):
        # image padding
        # x = self.image_padder.pad(x).contiguous()
        # forward pass
        dynamic_masks, optical_flows_t0, optical_flows_t1 = self.arch.forward(x, dt)
        return dynamic_masks

