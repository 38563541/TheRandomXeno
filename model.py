"""
model.py — thin wrapper for backward compatibility with run.py.

All implementation has moved to models/:
    models/stage1_model.py  — TRXSetMatching (Stage 1, active)
    models/trx_original.py  — TemporalCrossTransformer (original TRX baseline)
    models/stage2_model.py  — TRXSetMatchingWithRelation (Stage 2)

Matching functions are in matching/:
    matching/mean_hausdorff.py
    matching/bidirectional_hausdorff.py
    matching/attention_weighted_hausdorff.py

run.py imports: from model import CNN_TRX

CNN_TRX is a factory function: returns CNN_TRXWithRelation (Stage 2) when
args.use_intra_relation or args.use_inter_relation is set, else the Stage 1 model.
"""

from models.stage1_model import CNN_TRX as _CNN_TRX_Stage1, TRXSetMatching, NUM_SAMPLES  # noqa: F401
from models.stage2_model import CNN_TRXWithRelation as _CNN_TRXWithRelation


def CNN_TRX(args):
    """
    Factory that selects the appropriate model based on args.

    Stage 2 (relation modules enabled):
        args.use_intra_relation=True  OR  args.use_inter_relation=True
        → CNN_TRXWithRelation (models/stage2_model.py)

    Stage 1 (default):
        → CNN_TRX (models/stage1_model.py)
    """
    use_relation = (
        getattr(args, "use_intra_relation", False)
        or getattr(args, "use_inter_relation", False)
    )
    if use_relation:
        return _CNN_TRXWithRelation(args)
    return _CNN_TRX_Stage1(args)


__all__ = ["CNN_TRX", "TRXSetMatching", "NUM_SAMPLES"]
