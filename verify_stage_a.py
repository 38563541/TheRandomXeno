"""
Stage A: Verify relation_level behavioral correctness using forward hooks.

Judgment rules:
  relation_level="frame"  -> intra receives shape[1] == 8  (seq_len)
  relation_level="tuple"  -> intra receives shape[1] == 28 (tuple count)
  Both pass -> PASS
  Any fail  -> FAIL (do NOT attempt to fix)
"""
import sys, types, unittest.mock as mock
sys.path.insert(0, "/home/ccwu/Documents/work/trx")

import torch
import argparse

# ── stub torchvision pretrained download ──────────────────────────────────────
import torchvision.models as tv_models
_real_r18 = tv_models.resnet18
def _r18_no_pretrain(**kwargs):
    kwargs.pop("pretrained", None)
    return _real_r18(weights=None, **kwargs)
tv_models.resnet18 = _r18_no_pretrain

from models.stage2_model import TRXSetMatchingWithRelation

SEQ_LEN = 8
D_FRAME  = 512    # ResNet-18 frame dim
D_TUPLE  = 1152   # trans_linear_out_dim
T        = 28     # C(8,2) pairs
WAY      = 5
SHOT     = 1
NQ       = 5

def make_args(relation_level):
    return argparse.Namespace(
        seq_len              = SEQ_LEN,
        trans_linear_in_dim  = D_FRAME,
        trans_linear_out_dim = D_TUPLE,
        trans_dropout        = 0.1,
        way                  = WAY,
        matching             = "bidirectional",
        set_aggregation      = "pool",
        tau                  = 0.1,
        temp_set             = [2],
        relation_level       = relation_level,
        use_intra_relation   = True,
        use_inter_relation   = True,
        inter_style          = "global",
    )


def run_hook_test(relation_level):
    """Build model, register hooks on intra + inter, run forward, return shapes."""
    args  = make_args(relation_level)
    model = TRXSetMatchingWithRelation(
        args,
        relation_level = relation_level,
        use_intra      = True,
        use_inter      = True,
        inter_style    = "global",
    ).eval()

    ns = WAY * SHOT
    nq = WAY * NQ

    if relation_level == "frame":
        # Frame: relation runs before tuple construction → inputs are frame tensors
        support = torch.randn(ns, SEQ_LEN, D_FRAME)
        queries = torch.randn(nq, SEQ_LEN, D_FRAME)
    else:
        # Tuple: relation runs after tuple construction → inputs are tuple tensors
        support = torch.randn(ns, SEQ_LEN, D_FRAME)
        queries = torch.randn(nq, SEQ_LEN, D_FRAME)

    labels = torch.repeat_interleave(torch.arange(WAY), SHOT)

    intra_shapes  = []
    inter_shapes  = []

    h_intra = model.intra.register_forward_hook(
        lambda m, i, o: intra_shapes.append(tuple(i[0].shape)))
    # inter receives (q, s) as positional args — capture query input
    if model.inter is not None:
        h_inter = model.inter.register_forward_hook(
            lambda m, i, o: inter_shapes.append(tuple(i[0].shape)))
    else:
        h_inter = None

    with torch.no_grad():
        model(support, labels, queries)

    h_intra.remove()
    if h_inter:
        h_inter.remove()

    return intra_shapes, inter_shapes


def check(label, actual_dim1, expected_dim1):
    ok = (actual_dim1 == expected_dim1)
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}: got shape[1]={actual_dim1}, expected={expected_dim1}")
    return ok


def main():
    results = {}

    print("\n=== Stage A: relation_level hook test ===\n")

    for rl, expected_dim1 in [("frame", SEQ_LEN), ("tuple", T)]:
        print(f"-- relation_level={rl!r} --")
        try:
            intra_shapes, inter_shapes = run_hook_test(rl)
            print(f"   intra hook fired {len(intra_shapes)} time(s): {intra_shapes}")
            print(f"   inter hook fired {len(inter_shapes)} time(s): {inter_shapes}")

            if not intra_shapes:
                print(f"  [FAIL] intra hook never fired (Identity path?)")
                results[rl] = False
                continue

            actual = intra_shapes[0][1]
            ok = check(f"intra input dim1 ({rl})", actual, expected_dim1)

            # Also check inter if present
            inter_ok = True
            if inter_shapes:
                inter_actual = inter_shapes[0][1]
                inter_ok = check(f"inter input dim1 ({rl})", inter_actual, expected_dim1)

            results[rl] = ok and inter_ok

        except Exception as e:
            print(f"  [FAIL] Exception: {e}")
            import traceback; traceback.print_exc()
            results[rl] = False
        print()

    all_pass = all(results.values())
    verdict = "PASS" if all_pass else "FAIL"
    print(f"\n=== Stage A result: {verdict} ===")
    for rl, ok in results.items():
        print(f"  relation_level={rl!r}: {'PASS' if ok else 'FAIL'}")

    return all_pass, results


if __name__ == "__main__":
    ok, details = main()
    sys.exit(0 if ok else 1)
