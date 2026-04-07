"""
Inter-video relation module (skeleton).

Applies bidirectional cross-attention between query and support sequences.

Used in Stage 2:
    Option A: input is frame embeddings  (before tuple sampling)
    Option B: input is tuple embeddings  (after tuple sampling)

Query attends to support; support attends to query.
Both directions are enhanced before the matching step.
"""

import torch
import torch.nn as nn


class InterRelation(nn.Module):
    """
    Bidirectional cross-attention between query and support.

    For Option A: applied to frame embeddings.
    For Option B: applied to tuple embeddings.

    Query attends to support  -> enhanced query representation.
    Support attends to query  -> enhanced support representation.
    Both enhanced representations are then passed to the matching module.

    Args:
        dim       : int  feature dimension
        num_heads : int  number of attention heads (default 8)
    """

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        # TODO: implement cross-attention (query->support and support->query)
        # self.q2s_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        # self.s2q_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        # self.norm_q = nn.LayerNorm(dim)
        # self.norm_s = nn.LayerNorm(dim)
        raise NotImplementedError("InterRelation: implement in Stage 2")

    def forward(self, query_x, support_x):
        """
        Args:
            query_x   : [n_query, seq_len, dim]
                        seq_len = num_frames (Option A) or num_tuples (Option B)
            support_x : [n_support, seq_len, dim]

        Returns:
            enhanced_query_x   : [n_query, seq_len, dim]
            enhanced_support_x : [n_support, seq_len, dim]

        Note on batching:
            Cross-attention across all (query, support) pairs is memory-intensive.
            Consider chunking or approximations for large n_support.
        """
        # TODO: bidirectional cross-attention with residual connections
        # Query attending to ALL support tokens (flattened):
        #   s_flat: [1, n_support * seq_len, dim]  (broadcast over queries)
        #   q_enhanced: cross_attn(Q=query_x, K=s_flat, V=s_flat)
        # Support attending to ALL query tokens (flattened):
        #   q_flat: [1, n_query * seq_len, dim]  (broadcast over support)
        #   s_enhanced: cross_attn(Q=support_x, K=q_flat, V=q_flat)
        raise NotImplementedError
