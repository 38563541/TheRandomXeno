"""
Stage 1 model: Set-level matching TRX without prototype aggregation.

Key change vs original TRX (trx_original.py):
    Original: aggregate support tuples -> prototype -> distance(query, prototype)
    Stage 1:  keep all support tuples as a set -> Hausdorff distance(query_set, support_set)

Classes:
    TRXSetMatching  - renamed from TemporalCrossTransformerHausdorffPairs
                      pairs-only (temp_set=[2]), bidirectional mean-Hausdorff
    CNN_TRX         - ResNet backbone + TRXSetMatching head

The _mean_hausdorff_bidir computation is extracted to matching/mean_hausdorff.py.
"""

import torch
import torch.nn as nn
import math
from itertools import combinations
from torch.autograd import Variable
import torchvision.models as models

from utils import split_first_dim_linear
from matching.mean_hausdorff import mean_hausdorff_bidir

NUM_SAMPLES = 1


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000, pe_scale_factor=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe_scale_factor = pe_scale_factor
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) * self.pe_scale_factor
        pe[:, 1::2] = torch.cos(position * div_term) * self.pe_scale_factor
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class TRXSetMatching(nn.Module):
    """
    Pairs-only set-level matching module (Stage 1).

    Renamed from TemporalCrossTransformerHausdorffPairs.

    Key idea:
      - Keep query tuples as a set  Q  of size T = C(seq_len, 2)
      - Keep class support tuples as a set  S  of size k_shot * T
      - Distance(Q, S) = bidirectional mean-Hausdorff (from matching/)
      - Logit = -Distance   (no prototype aggregation)

    Args:
        args              : argument namespace with seq_len, trans_linear_in_dim,
                            trans_linear_out_dim, trans_dropout, way
        temporal_set_size : must be 2 (pairs only in Stage 1)
    """

    def __init__(self, args, temporal_set_size=2):
        super().__init__()
        assert temporal_set_size == 2, "Stage 1 is pairs-only. Use temporal_set_size=2."
        self.args = args
        self.temporal_set_size = temporal_set_size

        max_len = int(self.args.seq_len * 1.5)
        self.pe = PositionalEncoding(self.args.trans_linear_in_dim, self.args.trans_dropout, max_len=max_len)

        # Project concatenated pair features (2 * d_in) -> d_out
        self.embed_linear = nn.Linear(
            self.args.trans_linear_in_dim * temporal_set_size,
            self.args.trans_linear_out_dim
        )
        self.norm = nn.LayerNorm(self.args.trans_linear_out_dim)

        # Precompute all pair index combinations: shape (T, 2)
        frame_idxs = list(range(self.args.seq_len))
        pair_list = list(combinations(frame_idxs, temporal_set_size))
        tuples = torch.tensor(pair_list, dtype=torch.long)   # (T, 2)
        self.register_buffer("tuples", tuples, persistent=False)
        self.tuples_len = tuples.shape[0]

        # Chunk size for Hausdorff computation (memory <-> speed trade-off)
        self.support_chunk = 64
        print(f"[INFO] TRXSetMatching: tuples_len={self.tuples_len}, support_chunk={self.support_chunk}")

    @staticmethod
    def _extract_class_indices(labels, which_class):
        class_mask = torch.eq(labels, which_class)
        class_mask_indices = torch.nonzero(class_mask, as_tuple=False)
        return torch.reshape(class_mask_indices, (-1,))

    def _make_pair_set(self, x):
        """
        Build tuple embeddings from frame features.

        Args:
            x: (N, seq_len, d_in)
        Returns:
            z: (N, T, d_out)  where T = C(seq_len, 2)
        """
        device = x.device
        tuples = self.tuples.to(device)   # (T, 2)

        # Add positional encoding
        x = self.pe(x)   # (N, seq_len, d_in)

        # Concatenate features at each pair of time indices
        pair_feats = []
        for p in tuples:                                          # p: (2,)
            xp = torch.index_select(x, dim=-2, index=p)          # (N, 2, d_in)
            xp = xp.reshape(x.shape[0], -1)                      # (N, 2*d_in)
            pair_feats.append(xp)
        pair_feats = torch.stack(pair_feats, dim=1)               # (N, T, 2*d_in)

        # Project and normalize
        z = self.embed_linear(pair_feats)                         # (N, T, d_out)
        z = self.norm(z)
        return z

    def forward(self, support_set, support_labels, queries):
        """
        Args:
            support_set    : (n_support, seq_len, d_in)
            support_labels : (n_support,)  integer class indices in {0..way-1}
            queries        : (n_queries, seq_len, d_in)

        Returns:
            dict with key 'logits': (n_queries, way)
        """
        device = queries.device
        unique_labels = torch.unique(support_labels)

        # Build tuple-embedding sets: no aggregation happens here
        q_set = self._make_pair_set(queries)        # (nq, T, d_out)
        s_set = self._make_pair_set(support_set)    # (ns, T, d_out)

        n_queries = q_set.shape[0]
        all_distances_tensor = torch.zeros(
            n_queries, self.args.way, device=device, dtype=q_set.dtype
        )

        for c in unique_labels:
            idx = self._extract_class_indices(support_labels, c)
            class_s = torch.index_select(s_set, 0, idx)   # (k_shot, T, d_out)

            # Bidirectional mean-Hausdorff: no prototype, direct set distance
            dist = mean_hausdorff_bidir(
                q_set, class_s, chunk_s=self.support_chunk
            )   # (nq,)

            all_distances_tensor[:, c.long()] = -dist   # logit = -distance

        return {"logits": all_distances_tensor}


class CNN_TRX(nn.Module):
    """
    ResNet backbone connected to TRXSetMatching (Stage 1).

    Backbone: ResNet-18/34/50 (pretrained), last avg-pool layer removed.
    Matching: TRXSetMatching (pairs-only bidirectional Hausdorff).
    """
    def __init__(self, args):
        super(CNN_TRX, self).__init__()

        self.train()
        self.args = args

        if self.args.method == "resnet18":
            resnet = models.resnet18(pretrained=True)
        elif self.args.method == "resnet34":
            resnet = models.resnet34(pretrained=True)
        elif self.args.method == "resnet50":
            resnet = models.resnet50(pretrained=True)

        last_layer_idx = -1
        self.resnet = nn.Sequential(*list(resnet.children())[:last_layer_idx])

        print("[INFO] Using TRXSetMatching (pairs-only, bidirectional Hausdorff).")
        self.transformers = nn.ModuleList(
            [TRXSetMatching(args, temporal_set_size=2)]
        )

    def forward(self, context_images, context_labels, target_images):
        context_features = self.resnet(context_images).squeeze()
        target_features = self.resnet(target_images).squeeze()

        dim = int(context_features.shape[1])

        context_features = context_features.reshape(-1, self.args.seq_len, dim)
        target_features = target_features.reshape(-1, self.args.seq_len, dim)

        all_logits = [
            t(context_features, context_labels, target_features)['logits']
            for t in self.transformers
        ]
        all_logits = torch.stack(all_logits, dim=-1)
        sample_logits = torch.mean(all_logits, dim=[-1])

        return_dict = {
            'logits': split_first_dim_linear(
                sample_logits, [NUM_SAMPLES, target_features.shape[0]]
            )
        }
        return return_dict

    def distribute_model(self):
        """Distribute CNN across multiple GPUs."""
        if self.args.num_gpus > 1:
            self.resnet.cuda(0)
            self.resnet = torch.nn.DataParallel(
                self.resnet, device_ids=[i for i in range(0, self.args.num_gpus)]
            )
            self.transformers.cuda(0)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    class ArgsObject:
        def __init__(self):
            self.trans_linear_in_dim = 512
            self.trans_linear_out_dim = 128
            self.way = 5
            self.shot = 1
            self.query_per_class = 5
            self.trans_dropout = 0.1
            self.seq_len = 8
            self.img_size = 84
            self.method = "resnet18"
            self.num_gpus = 1
            self.temp_set = [2]

    args = ArgsObject()
    torch.manual_seed(0)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = CNN_TRX(args).to(device)

    support_imgs = torch.rand(args.way * args.shot * args.seq_len, 3, args.img_size, args.img_size).to(device)
    target_imgs = torch.rand(args.way * args.query_per_class * args.seq_len, 3, args.img_size, args.img_size).to(device)
    support_labels = torch.tensor([0, 1, 2, 3, 4]).to(device)

    out = model(support_imgs, support_labels, target_imgs)
    print(f"CNN_TRX (Stage 1) output logits shape: {out['logits'].shape}")
    assert out['logits'].shape == (NUM_SAMPLES, args.way * args.query_per_class, args.way)
    print("Smoke test passed.")
