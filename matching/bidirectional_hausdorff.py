"""
Bidirectional Hausdorff matching (convenience wrapper).

Symmetric version of mean_hausdorff: scores both directions equally.
    score = 0.5 * (Q->S distance + S->Q distance)

This is simply mean_hausdorff(..., bidirectional=True) exposed as a named
function for clarity in configs and ablation comparisons.

Interface:
    bidirectional_hausdorff(
        query_embeddings,    # [n_query, n_tuples, dim]
        support_embeddings,  # [n_support, n_tuples, dim]
    ) -> torch.Tensor        # [n_query, n_support]
"""

import torch
from matching.mean_hausdorff import mean_hausdorff


def bidirectional_hausdorff(query_embeddings, support_embeddings):
    """
    Symmetric (bidirectional) mean-Hausdorff distance matrix.

    Equivalent to mean_hausdorff(..., bidirectional=True).
    Exposed as a named function for use in configs:
        matching_method: bidirectional_hausdorff

    Args:
        query_embeddings   : [n_query, n_tuples, dim]
        support_embeddings : [n_support, n_tuples, dim]

    Returns:
        distances: [n_query, n_support]
            out[i, j] = 0.5 * (Q_i->S_j + S_j->Q_i) Hausdorff distance
    """
    return mean_hausdorff(query_embeddings, support_embeddings, bidirectional=True)


# ---------------------------------------------------------------------------
# Shape-verification tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    n_query, n_support, n_tuples, dim = 5, 25, 10, 512
    q = torch.randn(n_query, n_tuples, dim, device=device)
    s = torch.randn(n_support, n_tuples, dim, device=device)

    out = bidirectional_hausdorff(q, s)
    assert out.shape == (n_query, n_support), \
        f"Expected ({n_query},{n_support}), got {out.shape}"
    print(f"bidirectional_hausdorff shape test PASSED: {out.shape}")

    # Verify equivalence with mean_hausdorff(bidirectional=True)
    out_ref = mean_hausdorff(q, s, bidirectional=True)
    assert torch.allclose(out, out_ref), "bidirectional_hausdorff must equal mean_hausdorff(bidirectional=True)"
    print("Equivalence with mean_hausdorff(bidirectional=True) PASSED")

    # Non-negativity
    assert (out >= 0).all(), "Distances should be non-negative"
    print("Non-negativity check PASSED")

    print("\nAll bidirectional_hausdorff tests passed.")
