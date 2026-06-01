import torch
from .arch_time_conditioned import TimeCondUnetRecurrent
from .model import RecEVFlowNet


class TimeCondRecEVFlowNet(RecEVFlowNet):
    """
    """

    net_type = TimeCondUnetRecurrent
    recurrent_block_type = "convgru"
    activations = ["relu", None]

    def __init__(self, kwargs, num_bins=2, key="flow", min_size=16):
        # TODO: Hardcoded num_output_channels to 2 for predicting future optical flow
        kwargs["num_output_channels"] = 2
        # Override Kwargs before calling parent constructor
        super().__init__(kwargs=kwargs, num_bins=num_bins, key=key, min_size=min_size)
        self.arch = self.net_type(self.arch_kwargs)

    def forward(self, x, dt):
        # image padding
        x = self.image_padder.pad(x).contiguous()

        # forward pass
        multires_flow = self.arch.forward(x, dt)

        # upsample flow estimates to the original input resolution
        flow_list = []
        for i, flow in enumerate(multires_flow):
            scaling_h = x.shape[2] / flow.shape[2]
            scaling_w = x.shape[3] / flow.shape[3]
            scaling_flow = 2 ** (self.num_encoders - i - 1)
            upflow = scaling_flow * torch.nn.functional.interpolate(
                flow, scale_factor=(scaling_h, scaling_w), mode="bilinear", align_corners=False
            )
            upflow = self.image_padder.unpad(upflow)
            flow_list.append(upflow)

        return {self.key: flow_list}
