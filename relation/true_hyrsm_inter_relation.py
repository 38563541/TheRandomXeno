"""
True HyRSM inter-relation module, faithfully adapted from the original code:
  github.com/alibaba-mmai-research/HyRSM  —  CNN_HyRSM_1shot.get_feats()

The key steps are:
  1. Global avg-pool each video (or tuple-set) down to a single descriptor.
  2. Self-attention across ALL video descriptors (all support classes + every query).
  3. Expand-Concatenate-Conv: broadcast the inter-relation descriptor back to the
     sequence length, concatenate with the original per-frame/tuple features,
     then a 1-D conv reduces [dim*2 → dim].

Input shapes follow our Stage-2 tuple-level convention:
    query_x   : [nq, T, dim]
    support_x : [ns, T, dim]

Because step 2 is computed per query (each query attends to its own descriptor +
all support descriptors), the enhanced support has shape [nq, ns, T, dim].
The module returns:
    enhanced_query   : [nq, T, dim]
    enhanced_support : [nq, ns, T, dim]   — different per query

The caller (stage2_model.py) must loop over queries when computing Hausdorff
distances against the per-query enhanced support.
"""

import torch
import torch.nn as nn


class TrueHyRSMInterRelation(nn.Module):
    """
    HyRSM inter-relation: global-pool → self-attention → expand-concat-conv.

    Args:
        dim       : int  feature dimension
        num_heads : int  attention heads (default 8)
    """

    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0

        # Self-attention among pooled video descriptors (same as temporal_atte in HyRSM)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=0.05, batch_first=True)
        # 1-D conv to merge [dim*2 → dim] (same as layer2 in HyRSM)
        self.conv = nn.Conv1d(dim * 2, dim, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, query_x, support_x):
        """
        Args:
            query_x   : [nq, T, dim]
            support_x : [ns, T, dim]

        Returns:
            enhanced_query   : [nq, T, dim]
            enhanced_support : [nq, ns, T, dim]
        """
        nq, T, dim = query_x.shape
        ns = support_x.shape[0]

        # ── Step 1: global avg-pool ───────────────────────────────────────────
        q_pooled = query_x.mean(dim=1)    # [nq, dim]
        s_pooled = support_x.mean(dim=1)  # [ns, dim]

        # ── Step 2: self-attention per query across all video descriptors ─────
        # For each query i: attend over [support_0, ..., support_ns-1, query_i]
        s_pooled_exp = s_pooled.unsqueeze(0).expand(nq, -1, -1)  # [nq, ns, dim]
        q_pooled_exp = q_pooled.unsqueeze(1)                      # [nq,  1, dim]
        feature_in   = torch.cat([s_pooled_exp, q_pooled_exp], dim=1)  # [nq, ns+1, dim]

        attended, _ = self.attn(feature_in, feature_in, feature_in)
        feature_in   = self.relu(attended)  # [nq, ns+1, dim]

        s_inter = feature_in[:, :ns, :]   # [nq, ns, dim]
        q_inter = feature_in[:, -1, :]    # [nq,     dim]

        # ── Step 3: expand-concatenate-conv ──────────────────────────────────
        # Query
        q_inter_exp  = q_inter.unsqueeze(1).expand(-1, T, -1)       # [nq, T, dim]
        query_cat    = torch.cat([query_x, q_inter_exp], dim=2)      # [nq, T, dim*2]
        # conv expects [batch, channels, length]
        enhanced_query = self.conv(query_cat.permute(0, 2, 1)).permute(0, 2, 1)  # [nq, T, dim]

        # Support (per-query)
        s_x_exp      = support_x.unsqueeze(0).expand(nq, -1, -1, -1)       # [nq, ns, T, dim]
        s_inter_exp  = s_inter.unsqueeze(2).expand(-1, -1, T, -1)           # [nq, ns, T, dim]
        support_cat  = torch.cat([s_x_exp, s_inter_exp], dim=3)             # [nq, ns, T, dim*2]
        support_flat = support_cat.reshape(nq * ns, T, dim * 2)
        enhanced_s_flat = self.conv(support_flat.permute(0, 2, 1)).permute(0, 2, 1)  # [nq*ns, T, dim]
        enhanced_support = enhanced_s_flat.reshape(nq, ns, T, dim)

        return enhanced_query, enhanced_support


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device_list = ["cpu"]
    if torch.cuda.is_available():
        device_list.append("cuda")

    nq, ns, T, dim = 5, 5, 28, 1152

    for device in device_list:
        print(f"\n--- Testing on {device} ---")
        model = TrueHyRSMInterRelation(dim, num_heads=8).to(device)

        q = torch.randn(nq, T, dim, device=device, requires_grad=True)
        s = torch.randn(ns, T, dim, device=device)

        eq, es = model(q, s)

        assert eq.shape == (nq, T, dim),        f"query shape: {eq.shape}"
        assert es.shape == (nq, ns, T, dim),    f"support shape: {es.shape}"
        print(f"  enhanced_query shape  : {eq.shape}  PASSED")
        print(f"  enhanced_support shape: {es.shape}  PASSED")

        loss = eq.sum() + es.sum()
        loss.backward()
        assert q.grad is not None
        print(f"  Gradient flow: PASSED")

    # 5-shot
    print("\n--- 5-shot (ns=25) ---")
    model5 = TrueHyRSMInterRelation(dim)
    q5 = torch.randn(nq, T, dim)
    s5 = torch.randn(25, T, dim)
    eq5, es5 = model5(q5, s5)
    assert eq5.shape == (nq, T, dim)
    assert es5.shape == (nq, 25, T, dim)
    print(f"  5-shot shapes: query {eq5.shape}, support {es5.shape}  PASSED")

    print("\nAll TrueHyRSMInterRelation smoke tests passed.")
