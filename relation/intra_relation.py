"""
Intra-video relation module (skeleton).

Applies temporal self-attention within a single video's sequence.

Used in Stage 2:
    Option A: input is frame embeddings  [batch, num_frames, dim]  (before tuple sampling)
    Option B: input is tuple embeddings  [batch, num_tuples, dim]  (after tuple sampling)

IMPORTANT: each element may only attend to elements from the same video.
Cross-video attention is prevented via masking (see TODO below).
"""

import torch
import torch.nn as nn


class IntraRelation(nn.Module):
    """
    Temporal self-attention within a single video.

    For Option A: applied to frame embeddings BEFORE tuple sampling.
        input / output shape: [batch, num_frames, dim]

    For Option B: applied to tuple embeddings AFTER tuple sampling.
        input / output shape: [batch, num_tuples, dim]

    Each token attends only to tokens from the same video instance.
    (When batch contains multiple videos, masking must prevent cross-video leakage.)

    Args:
        dim       : int  feature dimension
        num_heads : int  number of attention heads (default 8)
    """

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        # TODO: implement multi-head self-attention
        # Suggested: nn.MultiheadAttention or custom implementation
        # self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        # self.norm = nn.LayerNorm(dim)
        # self.ffn = nn.Sequential(
        #     nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        # )
        # self.norm2 = nn.LayerNorm(dim)
        raise NotImplementedError("IntraRelation: implement in Stage 2")

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, dim]
               seq_len = num_frames (Option A) or num_tuples (Option B)

        Returns:
            x_enhanced: [batch, seq_len, dim]
        """
        # TODO: self-attention with residual connection
        # attn_out, _ = self.attn(x, x, x)
        # x = self.norm(x + attn_out)
        # x = self.norm2(x + self.ffn(x))
        # return x
        raise NotImplementedError
