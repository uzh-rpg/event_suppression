import torch
from .arch_hydra import HydraUnetRecurrent

class NoAttenUnetRecurrent(HydraUnetRecurrent):
    def __init__(self, kwargs, num_heads=8, final_w_scale_flow=1.0):
        super().__init__(kwargs, num_heads, final_w_scale_flow)

    def forward(self, x, dt):
        """
        :param 
            x: N x num_input_channels x H x W
            dt: N x num_timestamps
        :return: [N x num_output_channels x H x W for i in range(self.num_encoders)]
        """

        # encoder
        self.blocks = []
        for i, encoder in enumerate(self.encoders):
            x, self.states[i] = encoder(x, self.states[i])
            self.blocks.append(x)

        # residual blocks
        for resblock in self.resblocks:
            x, _ = resblock(x)

        self.embedding = x

        B, C, H_, W_ = x.shape
        t_enc = self.positional_encoding(dt)  # [B, C]
        channels = H_ * W_
        t_enc_ = t_enc.unsqueeze(-2).expand(B, channels, C)  # [B, H*W, C]
        t_enc_ = t_enc_.permute(0, 2, 1)  # [B, C, H*W]
        t_enc_ = t_enc_.view(B, C, H_, W_)  # [B, C, H, W]

        # condition tokens just by injection
        conditioned_x = x + t_enc_  # [B, C, H, W]

        # decoder and multires predictions
        # Using skip connections from encoder blocks (look at blocks.append)
        self.dynamic_masks = self.decoder_dynamic_mask(x)


        # decoder and pyramidal predictions
        optical_flows_t0 = self.decoder_flow(x)
        optical_flows_t1 = self.decoder_flow(conditioned_x)

        return self.dynamic_masks, optical_flows_t0, optical_flows_t1


if __name__ == "__main__":
    kwargs = {
        "num_bins": 15,
        "base_channels": 64,
        "num_output_channels": 1,
        "skip_type": "sum",
        "norm": None,
        "recurrent_block_type": "convgru",
        "final_w_scale": 1.0,
        "num_encoders": 5,
        "num_residual_blocks": 2
    }
    hydranet = HydraUnetRecurrent(kwargs=kwargs)
    hydranet = hydranet.cuda()
    x = torch.randn(8, 15, 128, 128).cuda() # Batch, Channels, Height, Width
    dt = torch.randn(8, 1).cuda() # Batch, Time
    dynamic_masks, optical_flows_t0, optical_flows_t1 = hydranet(x, dt)
    print(dynamic_masks[-1].shape, optical_flows_t1[-1].shape)