"""
model.py — thin wrapper for backward compatibility with run.py.

All implementation has moved to models/:
    models/stage1_model.py  — TRXSetMatching (Stage 1, active)
    models/trx_original.py  — TemporalCrossTransformer (original TRX baseline)
    models/stage2_model.py  — TRXSetMatchingWithRelation (Stage 2, skeleton)

Matching functions are in matching/:
    matching/mean_hausdorff.py
    matching/bidirectional_hausdorff.py
    matching/attention_weighted_hausdorff.py

run.py imports: from model import CNN_TRX
"""

from models.stage1_model import CNN_TRX, TRXSetMatching, NUM_SAMPLES  # noqa: F401

__all__ = ["CNN_TRX", "TRXSetMatching", "NUM_SAMPLES"]
