"""
DPIM: Dynamic Prompt Interaction Module

Injects DA-CLIP degradation-aware features into the main model via cross-attention.
"""

import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    """Cross-attention module: query from main model, key/value from DA-CLIP patch tokens"""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key_value):
        """
        Args:
            query: [B, N, C]
            key_value: [B, M, C]
        Returns:
            [B, N, C]
        """
        B, N, C = query.shape
        B, M, C = key_value.shape

        q = self.q(query).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(key_value).reshape(B, M, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DPIM(nn.Module):
    """
    Dynamic Prompt Interaction Module

    Injects DA-CLIP patch tokens into each transformer layer of the main model
    via cross-attention in a low-dimensional latent space.
    """

    def __init__(self, num_layers: int, embed_dim: int, latent_dim: int = 64, num_heads: int = 8):
        super().__init__()
        self.num_layers = num_layers

        self.down_proj = nn.ModuleList([
            nn.Linear(embed_dim, latent_dim) for _ in range(num_layers)
        ])
        self.cross_attn = nn.ModuleList([
            CrossAttention(dim=latent_dim, num_heads=num_heads) for _ in range(num_layers)
        ])
        self.up_proj = nn.ModuleList([
            nn.Linear(latent_dim, embed_dim) for _ in range(num_layers)
        ])
        self.scale_factor = nn.Parameter(torch.randn(num_layers, embed_dim) * 0.02)
        self.patch_proj = nn.Linear(embed_dim, latent_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, layer_idx: int, x: torch.Tensor, patch_tokens: torch.Tensor):
        """
        Args:
            layer_idx: Current layer index (0-indexed)
            x: Main model features [B, L, D] (L includes cls token)
            patch_tokens: DA-CLIP patch tokens [B, M, D] (M=49)
        Returns:
            Injected features [B, L, D]
        """
        x_down = self.down_proj[layer_idx](x)
        patch_tokens_low = self.patch_proj(patch_tokens)
        x_attn = self.cross_attn[layer_idx](x_down, patch_tokens_low)
        x_up = self.up_proj[layer_idx](x_attn)
        scale = self.scale_factor[layer_idx].unsqueeze(0).unsqueeze(0)
        return x + x_up * scale
