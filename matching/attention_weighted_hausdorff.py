"""
Attention-weighted Hausdorff matching.

Variant of mean-Hausdorff that up-weights query tuples which have
strong matches (small nearest-neighbour distance) in the support set,
and down-weights query tuples that are "orphaned" (no close support match).

Algorithm for each (query_i, support_j) pair:
    For each query tuple q_t:
        d_t* = min_{s_t'} dist(q_t, s_j_t')    nearest-support distance
    weights = softmax(-d* / tau)                 softmax over query tuples
    score   = sum_t( weights_t * d_t* )          attention-weighted distance

When tau -> 0:  reduces to min distance (only the nearest-match tuple counts).
When tau -> inf: reduces to unweighted mean (all tuples equally weighted).
Typical useful range: tau in [0.01, 1.0].

Interface matches STEP 2 spec:
    attention_weighted_hausdorff(
        query_embeddings,    # [n_query, n_tuples, dim]
        support_embeddings,  # [n_support, n_tuples, dim]
        tau=0.1
    ) -> torch.Tensor        # [n_query, n_support]
"""

import torch


def attention_weighted_hausdorff(query_embeddings, support_embeddings, tau=0.1):
    """
    Attention-weighted Hausdorff distance (instance-based pairwise matrix).

    out[i, j] = attention-weighted distance from query i's tuple set
                to support j's tuple set.

    Args:
        query_embeddings   : [n_query, n_tuples, dim]
        support_embeddings : [n_support, n_tuples, dim]
        tau                : float, softmax temperature
                             smaller -> sharper focus on best-matching tuples
                             larger  -> more uniform weighting (approaches mean-Hausdorff)

    Returns:
        distances: [n_query, n_support]

    Memory:
        Builds [nq, T, ns, T] internally. Same profile as mean_hausdorff().
    """
    nq, T, d = query_embeddings.shape
    ns = support_embeddings.shape[0]

    # --- build full pairwise tuple-distance tensor ---
    q_flat = query_embeddings.reshape(nq * T, d)     # [nq*T, d]
    s_flat = support_embeddings.reshape(ns * T, d)   # [ns*T, d]

    q_norm = (q_flat ** 2).sum(dim=-1, keepdim=True)  # [nq*T, 1]
    s_norm = (s_flat ** 2).sum(dim=-1).unsqueeze(0)   # [1, ns*T]
    dot    = torch.matmul(q_flat, s_flat.t())          # [nq*T, ns*T]
    dist2  = (q_norm + s_norm - 2.0 * dot).clamp(min=0)
    dist2  = dist2.reshape(nq, T, ns, T)              # [nq, T_q, ns, T_s]

    # --- nearest-support distance per query tuple ---
    # For each (i, j) and each query tuple t:
    #   d_t* = min over s_j's T tuples of dist(q_i_t, s_j_t')
    nn_dist = dist2.min(dim=3).values                 # [nq, T_q, ns]

    # --- softmax attention weights over query tuples ---
    # dim=1 is the query-tuple dimension T_q
    weights = torch.softmax(-nn_dist / tau, dim=1)    # [nq, T_q, ns]

    # --- weighted sum over query tuples ---
    score = (weights * nn_dist).sum(dim=1)            # [nq, ns]

    return score


# ---------------------------------------------------------------------------
# Shape-verification tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    n_query, n_support, n_tuples, dim = 5, 25, 10, 512
    q = torch.randn(n_query, n_tuples, dim, device=device)
    s = torch.randn(n_support, n_tuples, dim, device=device)

    # ---- standard tau ----
    out = attention_weighted_hausdorff(q, s, tau=0.1)
    assert out.shape == (n_query, n_support), \
        f"Expected ({n_query},{n_support}), got {out.shape}"
    print(f"attention_weighted_hausdorff shape test PASSED: {out.shape}")

    # ---- distances should be non-negative ----
    assert (out >= 0).all(), "Distances should be non-negative"
    print("Non-negativity check PASSED")

    # ---- tau sensitivity: large tau -> closer to mean-Hausdorff ----
    out_large_tau = attention_weighted_hausdorff(q, s, tau=1e6)
    from matching.mean_hausdorff import mean_hausdorff
    out_mean = mean_hausdorff(q, s, bidirectional=False)
    # With very large tau, weights are ~uniform, so result should approach mean_hausdorff
    max_diff = (out_large_tau - out_mean).abs().max().item()
    assert max_diff < 1e-3, f"Large tau should approach mean_hausdorff, max diff={max_diff:.6f}"
    print(f"Large-tau convergence check PASSED (max diff from mean_hausdorff: {max_diff:.2e})")

    print("\nAll attention_weighted_hausdorff tests passed.")
