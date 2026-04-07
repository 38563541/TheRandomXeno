"""
Intra-video relation module.

Applies a single Transformer encoder block within each video's sequence.

Option A: input is frame embeddings  [batch, num_frames, 512]
Option B: input is tuple embeddings  [batch, num_tuples, 1152]

Same module is used for both options — only the input dim changes.
"""

import torch
import torch.nn as nn


class IntraRelation(nn.Module):
    """
    Temporal self-attention within a single video.

    One Transformer encoder block: MHSA → Add&Norm → FFN → Add&Norm.

    Args:
        dim       : int  feature dimension (512 for frame-level, 1152 for tuple-level)
        num_heads : int  number of attention heads (default 8)
    """

    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0, \
            f"IntraRelation: dim ({dim}) must be divisible by num_heads ({num_heads})"

        self.attn  = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn   = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x : [batch, seq_len, dim]
                seq_len = num_frames (Option A) or num_tuples (Option B)

        Returns:
            x_enhanced : [batch, seq_len, dim]
        """
        # Self-attention with residual
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)

        # Feed-forward with residual
        x = self.norm2(x + self.ffn(x))
        return x
