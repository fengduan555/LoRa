import math

import torch
import torch.nn as nn
import torch.utils.checkpoint

import config
from model.blocks import DiTBlock


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float().reshape(-1, 1) * freqs.reshape(1, -1)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class DiT(nn.Module):
    def __init__(
        self,
        vocab_size,
        resolution=config.resolution,
        patch_size=config.patch_size,
        in_channels=config.in_channels,
        dim=config.dim,
        depth=config.depth,
        heads=config.heads,
        mlp_ratio=config.mlp_ratio,
        text_dim=config.text_dim,
        grad_checkpoint=config.grad_checkpoint,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.grad_checkpoint = grad_checkpoint
        self.grid = resolution // patch_size
        self.num_patches = self.grid * self.grid

        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )
        self.text_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, dim))
        self.text_pool_proj = nn.Linear(text_dim, dim)

        self.blocks = nn.ModuleList([DiTBlock(dim, heads, mlp_ratio) for _ in range(depth)])

        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.final_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        nn.init.zeros_(self.final_mod[-1].weight)
        nn.init.zeros_(self.final_mod[-1].bias)
        self.final_linear = nn.Linear(dim, patch_size * patch_size * in_channels)
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)

    def unpatchify(self, x):
        p = self.patch_size
        c = self.in_channels
        h = w = self.grid
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    def forward(self, x, t, text_tokens, text_mask=None, text_pool=None):
        x = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed

        context = self.text_proj(text_tokens)
        t_emb = self.time_mlp(timestep_embedding(t, self.time_mlp[0].in_features).to(x.dtype))
        if text_pool is None:
            text_pool = text_tokens.mean(dim=1)
        c = t_emb + self.text_pool_proj(text_pool.to(x.dtype))

        for block in self.blocks:
            if self.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, c, context, text_mask, use_reentrant=False
                )
            else:
                x = block(x, c, context, text_mask)

        shift, scale = self.final_mod(c).chunk(2, dim=1)
        x = self.final_norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.unpatchify(self.final_linear(x))
