import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from .arch import MultiResUNetRecurrent

class HydraUnetRecurrent(MultiResUNetRecurrent):
    def __init__(self, kwargs, num_heads=8, final_w_scale_flow=1.0):
        super().__init__(kwargs)
        self.num_heads = num_heads
        self.dt_encoding_dim = self.max_num_channels
        self.final_w_scale_flow = final_w_scale_flow

        self.embedding_dim = self.max_num_channels
        self.attn_grid = (17, 22)  # <- e.g., (16,16) => L = 256 tokens (constant)
        # self.attn_grid = (10, 10)

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.embedding_dim, 
            num_heads=self.num_heads, 
            batch_first=True)
        
        self.num_output_channels_of = 2
        
        self.decoders_flow = self.build_multires_prediction_decoders_optical_flow()
        self.preds_flow = self.build_multires_prediction_layer_optical_flow()

    def build_multires_prediction_decoders_optical_flow(self):
        decoder_input_sizes = reversed(self.encoder_output_sizes)
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        decoders = nn.ModuleList()
        for i, (input_size, output_size) in enumerate(zip(decoder_input_sizes, decoder_output_sizes)):
            input_size = 2 * input_size if self.skip_type == "concat" else input_size
            prediction_channels = 0 if i == 0 else self.num_output_channels_of
            decoders.append(
                self.up_type(
                    input_size + prediction_channels,
                    output_size,
                    kernel_size=self.kernel_size,
                    activation=self.ff_act,
                    norm=self.norm,
                )
            )
        return decoders

    def build_multires_prediction_layer_optical_flow(self):
        preds = nn.ModuleList()
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        for output_size in decoder_output_sizes:
            preds.append(
                self.ff_type(
                    in_channels=output_size,
                    out_channels=self.num_output_channels_of, 
                    kernel_size=1,
                    activation=self.final_activation, # None by default
                    norm=self.norm,
                    w_scale=self.final_w_scale_flow,
                    bias=self.final_bias,
                )
            )
        return preds

    def positional_encoding(self, delta_t):
        """
        delta_t: Tensor shape [batch_size, 1] (time scalar)
        dim: Embedding dimension
        """
        half_dim = self.dt_encoding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=delta_t.device) * -emb)  # [half_dim]
        emb = delta_t * emb  # broadcasting [batch_size, half_dim]
        pos_enc = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # [batch_size, dim]
        return pos_enc  # [batch_size, dim]

    def time_attention(self, x, dt):
        B, C, H, W = x.shape # [B, C, H, W]

        # embed_dim = H*W
        embed_dim = self.embedding_dim # embedding dim = C
        channels = H * W

        # time encoding
        t_enc = self.positional_encoding(dt)  # [B, C]
        q = t_enc.unsqueeze(-2).expand(B, channels, embed_dim)  # [B, H*W, C]

        # spatial encoding
        k = x.view(B, C, H*W)  # [B, C, H*W]
        v = x.view(B, C, H*W)  # [B, C, H*W]
        k = k.permute(0, 2, 1) # [B, H*W, C]
        v = v.permute(0, 2, 1) # [B, H*W, C]

        # Cross-attention with Multi Head Attention 
        # TODO Use flash attention instead (30% faster)
        conditioned_tokens, _ = self.multihead_attn(query=q, key=k, value=v) # [B, H*W, C]

        # Reshape back to spatial dimensions
        conditioned_tokens = conditioned_tokens.view(B, H, W, C) # [B, H, W, C]
        conditioned_tokens = conditioned_tokens.permute(0, 3, 1, 2) # [B, C, H, W]

        return conditioned_tokens
    
    def time_attention_constant_size(self, x, dt):
        """
        Make attention sequence length constant by:
          1) pooling x to a fixed (Gh, Gw) grid,
          2) building L = Gh*Gw tokens,
          3) conditioning queries with dt,
          4) upsampling the attended map back to (H, W).
        """
        B, C, H, W = x.shape
        Gh, Gw = self.attn_grid
        L = Gh * Gw  # constant sequence length

        # 1) Pool to a fixed grid -> constant #tokens
        xg = F.adaptive_avg_pool2d(x, (Gh, Gw))            # [B, C, Gh, Gw]
        tokens = xg.permute(0, 2, 3, 1).reshape(B, L, C)    # [B, L, C] (batch_first=True)

        # 2) Time encoding -> project to channel dim
        t_enc = self.positional_encoding(dt)                # [B, C]  (C == dt_encoding_dim)

        # 3) Build queries with dt and content (keep sequence length = L)
        #    We broadcast the time code to every token position in the fixed grid.
        q = t_enc.unsqueeze(-2).expand(B, L, C)                  # [B, L, C]
        k = tokens                                               # [B, L, C]
        v = tokens                                               # [B, L, C]

        # Optionally use mixed precision for memory
        # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out, _ = self.multihead_attn(query=q, key=k, value=v)  # [B, L, C]

        # 4) Back to spatial and upsample to original size
        out = out.view(B, Gh, Gw, C).permute(0, 3, 1, 2)       # [B, C, Gh, Gw]
        out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)  # [B, C, H, W]
        return out

    def decoder_dynamic_mask(self, x):
        # decoder and multires predictions
        # Using skip connections from encoder blocks (look at blocks.append)
        predictions = []
        for i, (decoder, pred) in enumerate(zip(self.decoders, self.preds)):
            x = self.skip_fn(x, self.blocks[self.num_encoders - i - 1], mode=self.skip_type)
            if i > 0:
                x = self.skip_fn(predictions[-1], x, mode="concat")
            x = decoder(x)
            predictions.append(pred(x))
        return predictions
    
    def decoder_flow(self, x):
        predictions = []
        for i, (decoder, pred) in enumerate(zip(self.decoders_flow, self.preds_flow)):
            x = self.skip_fn(x, self.blocks[self.num_encoders - i - 1], mode=self.skip_type)
            if i > 0:
                x = self.skip_fn(predictions[-1], x, mode="concat")
            x = decoder(x)
            predictions.append(pred(x))
        return predictions
    
    @property
    def current_flow(self):
        flows = self.decoder_flow(self.embedding)
        return flows

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

        # decoder and multires predictions
        # Using skip connections from encoder blocks (look at blocks.append)
        self.dynamic_masks = self.decoder_dynamic_mask(x)

        # TODO save computation with masked attention with flash attention 2
        # time conditioning for optical flow
        attention_score = self.time_attention_constant_size(x, dt)

        # add attention score to the input to conceptually compose
        # current optical flow with the future optical flow
        # x = x + attention_score

        # decoder and pyramidal predictions
        optical_flows_t0 = self.decoder_flow(x)
        optical_flows_t1 = self.decoder_flow(attention_score)

        return self.dynamic_masks, optical_flows_t0, optical_flows_t1


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
    hydranet = HydraUnetRecurrent(kwargs=kwargs)
    hydranet = hydranet.cuda()
    x = torch.randn(8, 2, 128, 128).cuda() # Batch, Channels, Height, Width
    dt = torch.randn(8, 1).cuda() # Batch, Time
    dynamic_masks, optical_flows = hydranet(x, dt)
    print(dynamic_masks[-1].shape, optical_flows[-1].shape)
