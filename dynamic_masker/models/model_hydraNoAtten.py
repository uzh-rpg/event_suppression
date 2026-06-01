import torch
import torch.nn.functional as F
from .model import RecEVFlowNet
from .arch_ablation_no_attention import NoAttenUnetRecurrent



class HydraEVNetNoAtten(RecEVFlowNet):
    """
    """

    net_type = NoAttenUnetRecurrent
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
        
    def upscale_feature_pyramid_(self, pyramid):
        """Upscale the feature pyramid to the original input resolution
        """
        upscaled_data = []
        for i, data in enumerate(pyramid):
            scaling_h = self.h_input / data.shape[2]
            scaling_w = self.w_input / data.shape[3]
            scaling_int = 2 ** (self.num_encoders - i - 1)
            upflow = scaling_int * torch.nn.functional.interpolate(
                data, scale_factor=(scaling_h, scaling_w), mode="bilinear", align_corners=False
            )
            upflow = self.image_padder.unpad(upflow)
            upscaled_data.append(upflow)
        return upscaled_data
    
    def upscale_feature_pyramid(self, pyramid):
        """JIT traceable version to Upscale the feature pyramid to the original input resolution"""
        upscaled_data = []
        for i, data in enumerate(pyramid):
            # Convert to plain Python floats for compatibility
            input_h, input_w = data.shape[2], data.shape[3]
            scaling_h = self.h_input / input_h
            scaling_w = self.w_input / input_w

            # Ensure scaling factors are floats (tracing requires this)
            scaling_h = float(scaling_h)
            scaling_w = float(scaling_w)

            scaling_int = 2 ** (self.num_encoders - i - 1)

            upflow = scaling_int * torch.nn.functional.interpolate(
                data, scale_factor=(scaling_h, scaling_w),
                mode="bilinear", align_corners=False
            )
            upflow = self.image_padder.unpad(upflow)
            upscaled_data.append(upflow)

        return upscaled_data

    
    @staticmethod
    def warp_mask_with_flow(logits, flow):
        """
        logits: Tensor of shape (B, 1, H, W)
        flow: Tensor of shape (B, 2, H, W), in pixels: (dx, dy)

        Returns:
            warped_logits: Tensor of shape (B, 1, H, W)
        """
        B, _, H, W = logits.shape

        # Create normalized mesh grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=logits.device),
            torch.linspace(-1, 1, W, device=logits.device),
            indexing='ij'
        ) # grid of coordinates in the range [-1, 1]

        grid = torch.stack((grid_x, grid_y), dim=-1)  # shape (H, W, 2)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # shape (B, H, W, 2)

        # Normalize flow to [-1, 1] for grid_sample to work
        flow_x = flow[:, 0] * 2 / (W - 1)
        flow_y = flow[:, 1] * 2 / (H - 1)
        flow_norm = torch.stack((flow_x, flow_y), dim=-1) # (B, H, W, 2)

        # Subtract flow from grid (because we are sampling from source)
        sampling_grid = grid + flow_norm

        # Sample using grid_sample
        warped_logits = F.grid_sample(
            logits, sampling_grid, mode='bilinear', padding_mode='zeros', align_corners=True
            )

        return warped_logits

    @staticmethod
    def project_mask_to_future(dynamic_masks, flow_list):
        """
        """
        future_masks = []
        for i, (mask, flow) in enumerate(zip(dynamic_masks, flow_list)):
            future_mask = HydraEVNetNoAtten.warp_mask_with_flow(mask, flow)
            future_masks.append(future_mask)
        return future_masks
    
    def multiply_flow_by_time(self, flow_list, time):
        for i in range(len(flow_list)):
            flow_list[i] = flow_list[i] * time
        return flow_list

    def forward(self, x, dt):
        # image padding
        x = self.image_padder.pad(x).contiguous()
        self.h_input, self.w_input = x.shape[2], x.shape[3]

        # forward pass
        dynamic_masks, optical_flows_t0, optical_flows_t1 = self.arch.forward(x, dt)

        # Scale the flow by the total prediction time as flow is in px/input_time
        # TODO Flow at time t0 is not used, remove its upscaling 
        optical_flows_t0 = self.multiply_flow_by_time(optical_flows_t0, self.current_flow_scaling)
        optical_flows_t1 = self.multiply_flow_by_time(optical_flows_t1, time=dt.view(dt.shape[0], 1, 1, 1))

        # Warp logits with the flow to get future masks
        future_masks = self.project_mask_to_future(dynamic_masks, optical_flows_t1)

        # upsample flow estimates to the original input resolution
        dynamic_masks_upscaled = self.upscale_feature_pyramid(dynamic_masks)
        optical_flows_t0_upscaled = self.upscale_feature_pyramid(optical_flows_t0)
        optical_flows_t1_upscaled = self.upscale_feature_pyramid(optical_flows_t1)
        future_masks_upscaled = self.upscale_feature_pyramid(future_masks)

        outs = {
            "flow": optical_flows_t1_upscaled, 
            "mask": dynamic_masks_upscaled, 
            "future_mask": future_masks_upscaled,
            "flow_t0": optical_flows_t0_upscaled
            }
        return outs


import time
if __name__ == "__main__":

    kwargs = {
        "num_bins": 2,
        "base_channels": 64,
        "num_output_channels": 1,
        "skip_type": "sum",
        "norm": None,
        "recurrent_block_type": "convgru",
        "final_w_scale": 1.0,
        "num_encoders": 4,
        "num_residual_blocks": 2
    }
    final_w_scale_flow = 0.01
    with torch.no_grad():
        hydranet = HydraEVNetNoAtten(kwargs=kwargs, num_bins=2, final_w_scale_flow=final_w_scale_flow)
        hydranet = hydranet.cuda()
        x = torch.randn(8, 2, 480, 640).cuda() # Batch, Channels, Height, Width
        dt = torch.randn(8, 1).cuda() # Batch, Time

        times = []
        for i in range(100):
            torch.cuda.synchronize()
            start = time.time()
            outs = hydranet(x, dt)
            final = time.time() - start
            times.append(final)
            print(f"Time: {final}")
            
        print(f"Average time: {sum(times[1:]) / len(times[1:])}")
