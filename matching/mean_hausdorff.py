"""
Mean Hausdorff distance for tuple-level set matching.

Two interfaces are provided:

  mean_hausdorff_bidir(q_set, s_set, chunk_s) -> (nq,)   [POOL-BASED]
    Low-level. Flattens all support instances into one pool.
    A query tuple can match to any support tuple from any shot.
    Used by TRXSetMatching.forward() for per-class computation.
    Matches the research goal: "search against the FULL support tuple pool
    across all K support instances."

  mean_hausdorff(query_embeddings, support_embeddings, bidirectional) -> [nq, ns]   [INSTANCE-BASED]
    Standardized interface. Computes pairwise distance from each query
    video to each support video individually.
    Use for: ablation configs, integration with STEP 3 config system,
    and when per-shot distances are needed (e.g. nearest-neighbour aggregation).
"""

import torch


# ---------------------------------------------------------------------------
# Low-level pool-based (used by TRXSetMatching)
# ---------------------------------------------------------------------------

def mean_hausdorff_bidir(q_set, s_set, chunk_s=64):
    """
    Bidirectional mean-Hausdorff distance: query set vs pooled support set.

    All support instances are flattened into one tuple pool (pool-based).
    A query tuple can match any support tuple from any shot.

    Args:
        q_set  : (nq, Tq, d)  query tuple embeddings
        s_set  : (ns, Ts, d)  support tuple embeddings, flattened to (ns*Ts, d)
        chunk_s: int          chunk size for memory-efficient min-distance scan

    Returns:
        distances: (nq,)   scalar distance per query to the pooled support set
    """
    nq, Tq, d = q_set.shape
    s_flat = s_set.reshape(-1, d)   # (S,) where S = ns * Ts
    S = s_flat.shape[0]
    device = q_set.device
    dtype = q_set.dtype

    # ---------- Q -> S ----------
    min_dist_q = torch.full((nq, Tq), float("inf"), device=device, dtype=dtype)
    q_norm = (q_set ** 2).sum(dim=-1, keepdim=True)   # (nq, Tq, 1)

    for start in range(0, S, chunk_s):
        end = min(start + chunk_s, S)
        s_chunk = s_flat[start:end]                             # (c, d)
        s_norm = (s_chunk ** 2).sum(dim=-1).view(1, 1, -1)     # (1, 1, c)
        qs = torch.matmul(q_set, s_chunk.t())                   # (nq, Tq, c)
        dist2 = (q_norm + s_norm - 2.0 * qs).clamp(min=0)      # (nq, Tq, c)
        min_dist_q = torch.minimum(min_dist_q, dist2.min(dim=-1).values)

    d_q2s = min_dist_q.mean(dim=-1)   # (nq,)

    # ---------- S -> Q ----------
    min_dist_s = torch.full((nq, S), float("inf"), device=device, dtype=dtype)
    q_flat = q_set.reshape(nq * Tq, d)
    q_flat_t = q_flat.t()
    q_flat_norm = (q_flat ** 2).sum(dim=-1).view(1, -1)   # (1, nq*Tq)

    for start in range(0, S, chunk_s):
        end = min(start + chunk_s, S)
        s_chunk = s_flat[start:end]                             # (c, d)
        s_norm = (s_chunk ** 2).sum(dim=-1).view(-1, 1)        # (c, 1)
        sq = torch.matmul(s_chunk, q_flat_t)                    # (c, nq*Tq)
        dist2 = (s_norm + q_flat_norm - 2.0 * sq).clamp(min=0)
        dist2 = dist2.view(end - start, nq, Tq)
        min_over_qtuple = dist2.min(dim=-1).values              # (c, nq)
        min_dist_s[:, start:end] = min_over_qtuple.t()

    d_s2q = min_dist_s.mean(dim=-1)   # (nq,)

    return 0.5 * (d_q2s + d_s2q)


# ---------------------------------------------------------------------------
# Standardized instance-based interface (STEP 2)
# ---------------------------------------------------------------------------

def mean_hausdorff(query_embeddings, support_embeddings, bidirectional=False):
    """
    Pairwise mean-Hausdorff distance matrix (instance-based).

    out[i, j] = Hausdorff distance from query video i's tuple set
                to support video j's tuple set.

    Each video's tuples are compared independently (NOT pooled across shots).
    To aggregate per class: take mean or min over the shot dimension.
    For pool-based matching, use mean_hausdorff_bidir() instead.

    Args:
        query_embeddings   : [n_query, n_tuples, dim]
        support_embeddings : [n_support, n_tuples, dim]
                             n_tuples must be equal for both.
        bidirectional      : bool
            False -> one-directional Q->S  (mean over q-tuples of nearest s-tuple)
            True  -> 0.5 * (Q->S + S->Q)  (symmetric)

    Returns:
        distances: [n_query, n_support]

    Memory note:
        Internally builds [nq, T, ns, T] pairwise distance tensor.
        For nq=5, ns=25, T=28, d=256: ~98 K floats (~0.4 MB).  Fine for RTX 3080 Ti.
        If T or ns grows large, consider chunking (see mean_hausdorff_bidir).
    """
    nq, T, d = query_embeddings.shape
    ns = support_embeddings.shape[0]

    # Reshape to flat tuple lists for efficient batched matmul
    q_flat = query_embeddings.reshape(nq * T, d)    # [nq*T, d]
    s_flat = support_embeddings.reshape(ns * T, d)  # [ns*T, d]

    # Squared L2 distance via expanded dot product:
    # ||q - s||^2 = ||q||^2 + ||s||^2 - 2 q·s
    q_norm = (q_flat ** 2).sum(dim=-1, keepdim=True)   # [nq*T, 1]
    s_norm = (s_flat ** 2).sum(dim=-1).unsqueeze(0)    # [1, ns*T]
    dot    = torch.matmul(q_flat, s_flat.t())           # [nq*T, ns*T]
    dist2  = (q_norm + s_norm - 2.0 * dot).clamp(min=0)
    dist2  = dist2.reshape(nq, T, ns, T)               # [nq, T_q, ns, T_s]

    # Q -> S: for each (i, j), mean over q-tuples of min over s-tuples
    d_q2s = dist2.min(dim=3).values.mean(dim=1)        # [nq, ns]

    if not bidirectional:
        return d_q2s

    # S -> Q: for each (i, j), mean over s-tuples of min over q-tuples
    # dist2.min(dim=1).values -> [nq, ns, T_s]  (min over q-tuple dim)
    # .mean(dim=-1) -> [nq, ns]
    d_s2q = dist2.min(dim=1).values.mean(dim=-1)       # [nq, ns]

    return 0.5 * (d_q2s + d_s2q)


# ---------------------------------------------------------------------------
# Shape-verification tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    # ---- mean_hausdorff_bidir (pool-based) ----
    nq, ns, T, d = 5, 5, 28, 256
    q = torch.randn(nq, T, d, device=device)
    s = torch.randn(ns, T, d, device=device)
    out_pool = mean_hausdorff_bidir(q, s)
    assert out_pool.shape == (nq,), f"Expected ({nq},), got {out_pool.shape}"
    print(f"mean_hausdorff_bidir  shape test PASSED: {out_pool.shape}")

    # ---- mean_hausdorff one-directional ----
    n_query, n_support, n_tuples, dim = 5, 25, 10, 512
    q = torch.randn(n_query, n_tuples, dim, device=device)
    s = torch.randn(n_support, n_tuples, dim, device=device)

    out = mean_hausdorff(q, s, bidirectional=False)
    assert out.shape == (n_query, n_support), f"Expected ({n_query},{n_support}), got {out.shape}"
    print(f"mean_hausdorff (uni)  shape test PASSED: {out.shape}")

    # ---- mean_hausdorff bidirectional ----
    out_bi = mean_hausdorff(q, s, bidirectional=True)
    assert out_bi.shape == (n_query, n_support), f"Expected ({n_query},{n_support}), got {out_bi.shape}"
    print(f"mean_hausdorff (bi)   shape test PASSED: {out_bi.shape}")

    # Symmetry sanity: bidirectional(q,s) should relate to bidirectional(s,q)
    # (not identical because of direction averaging, but distances should be positive)
    assert (out_bi >= 0).all(), "Distances should be non-negative"
    print(f"Non-negativity check  PASSED")

    print("\nAll mean_hausdorff tests passed.")
