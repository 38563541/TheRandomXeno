"""
HyRSM-style inter-video relation module.

Unlike the global InterRelation (which attends to all support classes at once),
this module is called once per class inside the per-class Hausdorff loop.
The query attends only to the support of the target class, producing a
class-conditioned query representation that is architecturally consistent
with per-class distance computation.

Reference: HyRSM (Wang et al., CVPR 2022) — class-specific cross-attention.

Interface:
    forward(query_x, class_support_x) -> enhanced_query_x
    query_x         : [nq, T, dim]
    class_support_x : [k_shot, T, dim]  — ONE class only
    enhanced_query_x: [nq, T, dim]

Only the query side is updated. Support features are left unchanged so
the same IntraRelation-enhanced class_s is used in both the attention
call and the subsequent Hausdorff distance computation.
"""

import torch
import torch.nn as nn


class HyRSMInterRelation(nn.Module):
    """
    Class-specific unidirectional cross-attention: query attends to one class's support.

    Args:
        dim       : int  feature dimension
        num_heads : int  number of attention heads (default 8)
    """

    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0, \
            f"HyRSMInterRelation: dim ({dim}) must be divisible by num_heads ({num_heads})"

        self.q2s_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_q   = nn.LayerNorm(dim)

    def forward(self, query_x, class_support_x):
        """
        Args:
            query_x         : [nq, T, dim]
            class_support_x : [k_shot, T, dim]  — support for ONE class only

        Returns:
            enhanced_query_x : [nq, T, dim]
        """
        nq, Tq, dim   = query_x.shape
        ks, Ts, _     = class_support_x.shape

        # Flatten class support into a single token sequence, broadcast across queries
        s_flat = class_support_x.reshape(1, ks * Ts, dim).expand(nq, -1, -1)
        q_attn, _ = self.q2s_attn(query_x, s_flat, s_flat)
        enhanced_q = self.norm_q(query_x + q_attn)

        return enhanced_q


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    device_list = ["cpu"]
    if torch.cuda.is_available():
        device_list.append("cuda")

    nq, k_shot, T, dim = 5, 1, 28, 1152
    num_heads = 8

    for device in device_list:
        print(f"\n--- Testing on {device} ---")
        model = HyRSMInterRelation(dim, num_heads).to(device)

        q = torch.randn(nq, T, dim, device=device, requires_grad=True)
        s = torch.randn(k_shot, T, dim, device=device)

        # Shape test
        out = model(q, s)
        assert out.shape == (nq, T, dim), \
            f"Shape mismatch: expected ({nq}, {T}, {dim}), got {out.shape}"
        print(f"  Output shape: {out.shape}  PASSED")

        # Gradient flow test
        loss = out.sum()
        loss.backward()
        assert q.grad is not None, "No gradient on query_x"
        assert q.grad.shape == q.shape, "Gradient shape mismatch"
        print(f"  Gradient flow: PASSED  (grad shape {q.grad.shape})")

    # 5-shot variant
    print("\n--- 5-shot support test (CPU) ---")
    model = HyRSMInterRelation(dim, num_heads)
    q5 = torch.randn(nq, T, dim)
    s5 = torch.randn(5, T, dim)
    out5 = model(q5, s5)
    assert out5.shape == (nq, T, dim), f"5-shot shape failed: {out5.shape}"
    print(f"  5-shot output shape: {out5.shape}  PASSED")

    print("\nAll HyRSMInterRelation smoke tests passed.")
