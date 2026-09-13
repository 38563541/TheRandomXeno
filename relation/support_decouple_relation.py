"""
Support-side class decoupling module (tuple level).

Adapted from TEAM (CVPR 2025) Eq. 10-12, translated from
"learnable pattern token × frame features" to "tuple set × tuple set".

Design principles:
    1. Query is NEVER modified — all five class distances remain in a common
       scale so cross-class comparison is preserved.
    2. Decoupling runs once per episode (before the class loop), not once
       per class, so complexity is O(N²) not O(N³).
    3. The cosine gate E_{n,o,m} is computed slot-wise: same frame-pair index
       m denotes the same temporal relationship across all videos, which is
       guaranteed by the fixed enumeration in TRXSetMatching._make_pair_set.

Reference:
    TEAM: Temporal-Aware Episodic Attention Module.  CVPR 2025.
    Our adaptation: class-conditioned attention on tuple embeddings,
    with the decoupled representation replacing (not augmenting) the
    original support set before Hausdorff matching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupportDecoupleRelation(nn.Module):
    """
    Support-side class decoupling at the tuple level (TEAM Eq. 10-12 adaptation).

    For anchor class n and every other class o != n:
        E_{n,o,m} = cos( mean_k S_{n,k,m} , mean_k S_{o,k,m} )   # per-slot scalar
        S~_{n,o}  = S_n + (1 + E) * CA(S_n, C_n, C_n) - E * CA(S_n, C_o, C_o)
        S^_n      = LayerNorm( mean_{o != n}[ S~_{n,o} + MLP(S~_{n,o}) ] )

    where CA(query, key, value) is cross-attention (read) and
    C_n = flatten(S_n) is the class context (all k*T tokens).

    Args:
        dim       : embedding dimension (= trans_linear_out_dim, e.g. 1152)
        num_heads : MHA heads (default 8)
        gate      : bool — use cosine gate E (default True).
                    If False, E is clamped to 1.0 everywhere.
        mode      : "remove" (subtract other-class signal) or
                    "inject"  (add other-class signal, for ablation only)
        dropout   : dropout on attention weights
    """

    def __init__(self, dim: int, num_heads: int = 8, gate: bool = True,
                 mode: str = "remove", dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0, \
            f"dim={dim} must be divisible by num_heads={num_heads}"
        assert mode in ("remove", "inject"), \
            f"mode must be 'remove' or 'inject', got {mode!r}"

        self.gate = gate
        self.mode = mode
        self.dim  = dim

        # Cross-attention: anchor class reads from its own / another class's context
        self.read = nn.MultiheadAttention(
            embed_dim   = dim,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True,    # expects [batch, seq, dim]
        )

        # Per-class residual MLP (point-wise, applied once after averaging over o)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

        # Final layer-norm (applied once per class, after averaging over o)
        self.norm = nn.LayerNorm(dim)

        # LayerScale: zero-init scalar — at init the module is an identity map
        # (D1b Fix 1). alpha stays 0 at start so random initialisation of
        # read/mlp/norm cannot disturb the support set; learned gradually.
        self.alpha = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _class_idx(labels: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Return row indices of support_x belonging to class c."""
        return torch.nonzero(torch.eq(labels, c), as_tuple=False).reshape(-1)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, support_x: torch.Tensor,
                support_labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            support_x      : [ns, T, dim]  — all support tuple embeddings
            support_labels : [ns]          — integer class indices in {0..N-1}

        Returns:
            out            : [ns, T, dim]  — decoupled support embeddings
                             (same layout and dtype as input)
        """
        ns, T, dim = support_x.shape
        classes = torch.unique(support_labels)
        N = classes.numel()

        if N < 2:
            # Nothing to decouple with only one class (shouldn't happen in
            # a well-formed episode, but return gracefully just in case)
            return support_x

        # ── D1b Fix 2: episode-level mean for centred cosine gate ──────────
        # mu_m = average support vector at slot m across ALL classes.
        # Subtracting the shared mean removes the DC component that makes raw
        # cosine nearly constant (~0.987), exposing the class-discriminative
        # residual signal.
        mu = support_x.mean(dim=0)   # [T, dim]

        # ── Pre-compute per-class slot means and grouped tensors ────────────
        idx_of  = {}   # class → row indices into support_x
        S_of    = {}   # class → [k, T, dim]
        mean_of = {}   # class → [T, dim]   slot-aligned means across k shots

        for c in classes:
            k_int       = int(c.item())
            idx         = self._class_idx(support_labels, c)
            S           = support_x[idx]           # [k, T, dim]
            idx_of[k_int]  = idx
            S_of[k_int]    = S
            mean_of[k_int] = S.mean(dim=0)        # [T, dim]

        # ── Per-class decoupling ────────────────────────────────────────────
        out = torch.empty_like(support_x)

        for c_n in classes:
            n   = int(c_n.item())
            S_n = S_of[n]                          # [k, T, dim]
            k   = S_n.shape[0]

            # Build the class-n context: flatten all k*T tokens
            # We broadcast it to match the k query videos
            C_n = S_n.reshape(1, k * T, dim).expand(k, -1, -1)   # [k, k*T, dim]

            # Self-read: how does class n attend to its own context?
            read_self, _ = self.read(S_n, C_n, C_n)   # [k, T, dim]

            # Accumulate cross-class attention (S_n NOT added here — D1b Fix 1)
            acc = None

            for c_o in classes:
                o = int(c_o.item())
                if o == n:
                    continue

                S_o  = S_of[o]                         # [ko, T, dim]
                ko   = S_o.shape[0]
                C_o  = S_o.reshape(1, ko * T, dim).expand(k, -1, -1)

                # Cross-read: how does class n attend to class o's context?
                read_other, _ = self.read(S_n, C_o, C_o)   # [k, T, dim]

                # D1b Fix 2: centred cosine gate
                # E_{n,o,m} = cos( mean_n - mu , mean_o - mu ) clamped to [0,1]
                if self.gate:
                    e = F.cosine_similarity(mean_of[n] - mu, mean_of[o] - mu, dim=-1)
                    e = e.clamp(min=0.0).view(1, T, 1)   # [1, T, 1] broadcastable
                else:
                    e = torch.ones(1, T, 1, device=S_n.device, dtype=S_n.dtype)

                # Accumulate: (1+E)*read_self ± E*read_other  (S_n excluded)
                sign = -1.0 if self.mode == "remove" else 1.0
                s_t = (1.0 + e) * read_self + sign * e * read_other   # [k, T, dim]
                acc = s_t if acc is None else acc + s_t

            # D1b Fix 1: LayerScale residual
            # corr = average of cross-class attention terms + MLP
            # out  = S_n + alpha * LayerNorm(corr)
            # alpha is zero-initialised → identity at start of training
            corr = acc / (N - 1)
            corr = corr + self.mlp(corr)
            out[idx_of[n]] = S_n + self.alpha * self.norm(corr)   # [k, T, dim]

        return out
