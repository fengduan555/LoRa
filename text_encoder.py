import torch
import torch.nn as nn

import config


class TagTextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        dim=config.text_dim,
        depth=config.text_depth,
        heads=config.text_heads,
        ffn_ratio=4.0,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.embed.weight, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=int(dim * ffn_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, ids, mask=None):
        if mask is None:
            mask = ids != 0
        safe_mask = mask.clone()
        empty = ~safe_mask.any(dim=1)
        if empty.any():
            safe_mask[empty, 0] = True

        x = self.embed(ids)
        x = self.encoder(x, src_key_padding_mask=~safe_mask)
        x = self.norm(x)

        weights = safe_mask.unsqueeze(-1).to(x.dtype)
        pooled = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        return x, pooled
