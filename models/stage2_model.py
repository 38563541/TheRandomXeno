"""
Stage 2 model: TRXSetMatching + optional IntraRelation / InterRelation.

Two ablation options controlled by 'relation_level' in the YAML config:

    Option A (relation_level="frame"):
        ResNet → [IntraRelation] → [InterRelation] → Tuple Sampling → Hausdorff → Logits
        Relation enriches per-frame representations BEFORE tuples are formed.

    Option B (relation_level="tuple"):
        ResNet → Tuple Sampling → [IntraRelation] → [InterRelation] → Hausdorff → Logits
        Relation enriches per-tuple representations AFTER tuples are formed.
        This is the primary Stage 2 contribution.

Both intra and inter can be independently toggled via:
    use_intra_relation: true/false
    use_inter_relation: true/false
"""

import torch
import torch.nn as nn
import torchvision.models as models

from utils import split_first_dim_linear
from matching.mean_hausdorff import mean_hausdorff_bidir
from models.stage1_model import TRXSetMatching, NUM_SAMPLES
from relation.intra_relation import IntraRelation
from relation.inter_relation import InterRelation


class TRXSetMatchingWithRelation(nn.Module):
    """
    Stage 2: TRXSetMatching with optional relation modules.

    Args:
        args           : argparse namespace
        relation_level : "frame" (Option A) or "tuple" (Option B)
        use_intra      : bool — apply IntraRelation
        use_inter      : bool — apply InterRelation
    """

    def __init__(self, args, relation_level="tuple", use_intra=True, use_inter=True):
        super().__init__()
        assert relation_level in ("frame", "tuple"), \
            f"relation_level must be 'frame' or 'tuple', got {relation_level!r}"

        self.args           = args
        self.relation_level = relation_level
        self.use_intra      = use_intra
        self.use_inter      = use_inter

        # Reuse Stage 1 tuple embedding + distance computation
        self.matching = TRXSetMatching(args, temporal_set_size=2)

        # Relation dim depends on where it is applied
        if relation_level == "frame":
            rel_dim = args.trans_linear_in_dim   # 512 for ResNet18
        else:
            rel_dim = args.trans_linear_out_dim  # 1152

        num_heads = 8

        self.intra = IntraRelation(rel_dim, num_heads) if use_intra else nn.Identity()
        self.inter = InterRelation(rel_dim, num_heads) if use_inter else None

        print(
            f"[INFO] TRXSetMatchingWithRelation: relation_level={relation_level}, "
            f"use_intra={use_intra}, use_inter={use_inter}, rel_dim={rel_dim}"
        )

    @staticmethod
    def _extract_class_indices(labels, which_class):
        class_mask = torch.eq(labels, which_class)
        return torch.reshape(torch.nonzero(class_mask, as_tuple=False), (-1,))

    def forward(self, support_set, support_labels, queries):
        """
        Args:
            support_set    : [n_support, seq_len, d_in]
            support_labels : [n_support,]
            queries        : [n_queries, seq_len, d_in]

        Returns:
            dict with 'logits': [n_queries, way]
        """
        device        = queries.device
        unique_labels = torch.unique(support_labels)

        if self.relation_level == "frame":
            # ── Option A: relation on frame embeddings ──────────────────────
            # IntraRelation: enrich each video's frame sequence independently
            queries     = self.intra(queries)      # [nq, seq_len, d_in]
            support_set = self.intra(support_set)  # [ns, seq_len, d_in]

            # InterRelation: cross-attend across all queries and all support
            if self.inter is not None:
                queries, support_set = self.inter(queries, support_set)

            # Build tuple sets from enhanced frame features
            q_set = self.matching._make_pair_set(queries)      # [nq, T, d_out]
            s_set = self.matching._make_pair_set(support_set)  # [ns, T, d_out]

        else:
            # ── Option B: relation on tuple embeddings ───────────────────────
            # Build tuple sets first (no relation yet)
            q_set = self.matching._make_pair_set(queries)      # [nq, T, d_out]
            s_set = self.matching._make_pair_set(support_set)  # [ns, T, d_out]

            # IntraRelation: enrich each video's tuple sequence
            q_set = self.intra(q_set)  # [nq, T, d_out]
            s_set = self.intra(s_set)  # [ns, T, d_out]

            # InterRelation: cross-attend across all queries and all support
            if self.inter is not None:
                q_set, s_set = self.inter(q_set, s_set)

        # ── Per-class Hausdorff distances ────────────────────────────────────
        n_queries    = q_set.shape[0]
        all_distances = torch.zeros(
            n_queries, self.args.way, device=device, dtype=q_set.dtype
        )

        for c in unique_labels:
            idx     = self._extract_class_indices(support_labels, c)
            class_s = torch.index_select(s_set, 0, idx)   # [k_shot, T, d_out]
            dist    = mean_hausdorff_bidir(q_set, class_s, chunk_s=64)  # [nq,]
            all_distances[:, c.long()] = -dist             # logit = -distance

        return {"logits": all_distances}


class CNN_TRXWithRelation(nn.Module):
    """
    ResNet backbone + TRXSetMatchingWithRelation (Stage 2).

    Reads relation config from args:
        args.relation_level      : "frame" or "tuple"
        args.use_intra_relation  : bool
        args.use_inter_relation  : bool
    """

    def __init__(self, args):
        super().__init__()
        self.train()
        self.args = args

        relation_level = getattr(args, "relation_level",     "tuple")
        use_intra      = getattr(args, "use_intra_relation", True)
        use_inter      = getattr(args, "use_inter_relation", True)

        # Backbone (last avg-pool removed)
        if args.method == "resnet18":
            resnet = models.resnet18(pretrained=True)
        elif args.method == "resnet34":
            resnet = models.resnet34(pretrained=True)
        elif args.method == "resnet50":
            resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])

        self.transformers = nn.ModuleList([
            TRXSetMatchingWithRelation(
                args,
                relation_level=relation_level,
                use_intra=use_intra,
                use_inter=use_inter,
            )
        ])

    def forward(self, context_images, context_labels, target_images):
        context_features = self.resnet(context_images).squeeze()
        target_features  = self.resnet(target_images).squeeze()

        dim = int(context_features.shape[1])
        context_features = context_features.reshape(-1, self.args.seq_len, dim)
        target_features  = target_features.reshape(-1, self.args.seq_len, dim)

        all_logits = [
            t(context_features, context_labels, target_features)['logits']
            for t in self.transformers
        ]
        all_logits    = torch.stack(all_logits, dim=-1)
        sample_logits = torch.mean(all_logits, dim=-1)

        return {
            'logits': split_first_dim_linear(
                sample_logits, [NUM_SAMPLES, target_features.shape[0]]
            )
        }

    def distribute_model(self):
        if self.args.num_gpus > 1:
            self.resnet = torch.nn.DataParallel(
                self.resnet,
                device_ids=list(range(self.args.num_gpus))
            )
            self.transformers.cuda(0)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import types

    args = types.SimpleNamespace(
        trans_linear_in_dim=512,
        trans_linear_out_dim=1152,
        way=5, shot=1, query_per_class=5,
        trans_dropout=0.1,
        seq_len=8,
        img_size=84,
        method="resnet18",
        num_gpus=1,
        temp_set=[2],
        use_intra_relation=True,
        use_inter_relation=True,
        relation_level="tuple",
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    for level in ("frame", "tuple"):
        args.relation_level = level
        model = CNN_TRXWithRelation(args).to(device)

        support = torch.rand(args.way * args.shot * args.seq_len, 3, 84, 84).to(device)
        target  = torch.rand(args.way * args.query_per_class * args.seq_len, 3, 84, 84).to(device)
        labels  = torch.arange(args.way).to(device)

        out = model(support, labels, target)
        expected = (NUM_SAMPLES, args.way * args.query_per_class, args.way)
        assert out['logits'].shape == expected, \
            f"Option {'A' if level == 'frame' else 'B'}: expected {expected}, got {out['logits'].shape}"
        print(f"Option {'A' if level == 'frame' else 'B'} ({level} level): logits shape {out['logits'].shape} ✓")

    print("All smoke tests passed.")
