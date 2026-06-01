import torch
import math
import torch.nn as nn

from .arch import MultiResUNetRecurrent

class TimeCondUnetRecurrent(MultiResUNetRecurrent):
    def __init__(self, kwargs, num_heads=8):
        super().__init__(kwargs)
        self.dt_encoding_dim = self.max_num_channels
        self.num_heads = num_heads
        self.attention_dim = self.max_num_channels
        # TODO Hardcoded embedding dimension to 30*40=1200
        self.embedding_dim = 1200
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.embedding_dim, 
            num_heads=self.num_heads, 
            batch_first=True).cuda()

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
        embed_dim = H*W

        # time encoding
        t_enc = self.positional_encoding(dt)  # [B, C]
        q = t_enc.unsqueeze(-1).expand(B, C, embed_dim)  # [B, C, embed_dim]

        # spatial encoding
        k = x.view(B, C, embed_dim)  # [B, C, embed_dim]
        v = x.view(B, C, embed_dim)  # [B, C, embed_dim]

        # Cross-attention with Multi Head Attention 
        # TODO Use flash attention instead (30% faster)
        conditioned_tokens, _ = self.multihead_attn(query=q, key=k, value=v) # [B, C, embed_dim]

        # Reshape back to spatial dimensions
        conditioned_tokens = conditioned_tokens.view(B, C, H, W) # [B, C, H, W]
        return conditioned_tokens


    def forward(self, x, dt):
        """
        :param 
            x: N x num_input_channels x H x W
            dt: N x num_timestamps
        :return: [N x num_output_channels x H x W for i in range(self.num_encoders)]
        """

        # encoder
        blocks = []
        for i, encoder in enumerate(self.encoders):
            x, self.states[i] = encoder(x, self.states[i])
            blocks.append(x)

        # residual blocks
        for resblock in self.resblocks:
            x, _ = resblock(x)

        # time conditioning
        x = self.time_attention(x, dt)

        # decoder and multires predictions
        # Using skip connections from encoder blocks (look at blocks.append)
        predictions = []
        for i, (decoder, pred) in enumerate(zip(self.decoders, self.preds)):
            x = self.skip_fn(x, blocks[self.num_encoders - i - 1], mode=self.skip_type)
            if i > 0:
                x = self.skip_fn(predictions[-1], x, mode="concat")
            x = decoder(x)
            predictions.append(pred(x))

        return predictions
