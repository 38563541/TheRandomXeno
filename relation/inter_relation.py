"""
Inter-video relation module.

Applies bidirectional cross-attention between all query videos and all
support videos simultaneously.

    Query attends to ALL support tokens (flattened across videos)
    Support attends to ALL query tokens (flattened across videos)

Option A: input is frame embeddings  [n_videos, num_frames, 512]
Option B: input is tuple embeddings  [n_videos, num_tuples, 1152]

Same module is used for both options — only the input dim changes.
"""

import torch
import torch.nn as nn


class InterRelation(nn.Module):
    """
    Bidirectional cross-attention between query and support.

    Args:
        dim       : int  feature dimension
        num_heads : int  number of attention heads (default 8)
    """

    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0, \
            f"InterRelation: dim ({dim}) must be divisible by num_heads ({num_heads})"

        # Query attends to support
        self.q2s_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_q   = nn.LayerNorm(dim)

        # Support attends to query
        self.s2q_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_s   = nn.LayerNorm(dim)

    def forward(self, query_x, support_x):
        """
        Args:
            query_x   : [n_query,   seq_len, dim]
            support_x : [n_support, seq_len, dim]

        Returns:
            enhanced_query_x   : [n_query,   seq_len, dim]
            enhanced_support_x : [n_support, seq_len, dim]
        """
        nq, Tq, dim   = query_x.shape
        ns, Ts, _     = support_x.shape

        # --- Query attending to ALL support tokens ---
        # Flatten support across videos: [1, ns*Ts, dim] → broadcast to [nq, ns*Ts, dim]
        s_flat = support_x.reshape(1, ns * Ts, dim).expand(nq, -1, -1)
        q_attn, _ = self.q2s_attn(query_x, s_flat, s_flat)
        enhanced_q = self.norm_q(query_x + q_attn)

        # --- Support attending to ALL query tokens ---
        # Flatten query across videos: [1, nq*Tq, dim] → broadcast to [ns, nq*Tq, dim]
        q_flat = query_x.reshape(1, nq * Tq, dim).expand(ns, -1, -1)
        s_attn, _ = self.s2q_attn(support_x, q_flat, q_flat)
        enhanced_s = self.norm_s(support_x + s_attn)

        return enhanced_q, enhanced_s
