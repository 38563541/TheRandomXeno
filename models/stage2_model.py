"""
Stage 2 model: TRXSetMatching + optional relation modules.

Skeleton only — to be implemented after Stage 1 experiments are complete.

Two ablation options:
    Option A (relation_level="frame"):
        IntraRelation and InterRelation applied to FRAME embeddings
        BEFORE tuple sampling. Similar to HyRSM approach.

    Option B (relation_level="tuple"):
        IntraRelation and InterRelation applied to TUPLE embeddings
        AFTER tuple sampling. Our main research contribution.

Both options should be independently ablatable via use_intra / use_inter flags.
"""

import torch
import torch.nn as nn

# TODO: uncomment when relation modules are implemented
# from relation.intra_relation import IntraRelation
# from relation.inter_relation import InterRelation
# from models.stage1_model import CNN_TRX, TRXSetMatching, PositionalEncoding, NUM_SAMPLES


class TRXSetMatchingWithRelation(nn.Module):
    """
    Stage 2: TRXSetMatching + optional IntraRelation and InterRelation modules.

    Args:
        args           : namespace with model hyperparameters
        relation_level : "frame" (Option A) or "tuple" (Option B)
        use_intra      : bool — apply IntraRelation (self-attention within each video)
        use_inter      : bool — apply InterRelation (cross-attention query <-> support)

    Pipeline (Option A — frame level):
        ResNet -> [IntraRelation] -> [InterRelation] -> Tuple Sampling
        -> TRXSetMatching -> Prediction

    Pipeline (Option B — tuple level):
        ResNet -> Tuple Sampling -> [IntraRelation] -> [InterRelation]
        -> TRXSetMatching -> Prediction
    """

    def __init__(self, args, relation_level="tuple", use_intra=True, use_inter=True):
        super().__init__()
        assert relation_level in ("frame", "tuple"), \
            f"relation_level must be 'frame' or 'tuple', got {relation_level!r}"
        self.args = args
        self.relation_level = relation_level
        self.use_intra = use_intra
        self.use_inter = use_inter

        # TODO: initialise backbone, relation modules, and matching head
        # self.resnet = ...
        # self.intra = IntraRelation(dim=...) if use_intra else nn.Identity()
        # self.inter = InterRelation(dim=...) if use_inter else nn.Identity()
        # self.matching = TRXSetMatching(args)
        raise NotImplementedError(
            "TRXSetMatchingWithRelation is a skeleton. "
            "Implement after Stage 1 ablation experiments are complete."
        )

    def forward(self, context_images, context_labels, target_images):
        # TODO
        raise NotImplementedError
