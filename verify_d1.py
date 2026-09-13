"""
D1 wiring verification — four tests must ALL pass before training is launched.

V1  Shape:        model with inter_style="decouple" produces logits (nq, way).
V2  Query intact: q_set is identical across two forward passes with different supports.
V3  Not no-op:    s_set with decouple != s_set without inter (use_inter_relation=False).
V4  Co-scale:     5 class distances are similar in magnitude; permuting support
                  only permutes the distances, values unchanged.

Any FAIL → record and exit(1) without attempting to fix.
"""
import sys, types, argparse, torch
sys.path.insert(0, "/home/ccwu/Documents/work/trx")

# ── torchvision stub (no pretrained download) ────────────────────────────────
import torchvision.models as tv_models
_real_r18 = tv_models.resnet18
tv_models.resnet18 = lambda **kw: _real_r18(weights=None, **kw)

from models.stage2_model import TRXSetMatchingWithRelation

# ── constants ─────────────────────────────────────────────────────────────────
SEQ_LEN = 8
D_FRAME  = 512
D_TUPLE  = 1152
T        = 28       # C(8,2)
WAY      = 5
SHOT     = 1
NQ       = 5

def make_args(**extra):
    a = argparse.Namespace(
        seq_len              = SEQ_LEN,
        trans_linear_in_dim  = D_FRAME,
        trans_linear_out_dim = D_TUPLE,
        trans_dropout        = 0.1,
        way                  = WAY,
        matching             = "bidirectional",
        set_aggregation      = "pool",
        tau                  = 0.1,
        temp_set             = [2],
        decouple_gate        = True,
        decouple_mode        = "remove",
    )
    for k, v in extra.items():
        setattr(a, k, v)
    return a

def build(relation_level="tuple", use_intra=True, use_inter=True,
          inter_style="decouple", **extra):
    args = make_args(**extra)
    m = TRXSetMatchingWithRelation(
        args,
        relation_level = relation_level,
        use_intra      = use_intra,
        use_inter      = use_inter,
        inter_style    = inter_style,
    ).eval()
    return m

def make_episode(seed=0):
    torch.manual_seed(seed)
    ns = WAY * SHOT
    nq = WAY * NQ
    support = torch.randn(ns, SEQ_LEN, D_FRAME)
    queries = torch.randn(nq, SEQ_LEN, D_FRAME)
    labels  = torch.repeat_interleave(torch.arange(WAY), SHOT)
    return support, labels, queries


PASS_COUNT = 0
FAIL_COUNT = 0

def check(name, ok, detail=""):
    global PASS_COUNT, FAIL_COUNT
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f"  — {detail}" if detail else ""))
    if ok:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    return ok


# ════════════════════════════════════════════════════════════════════════════════
print("\n=== D1 wiring verification ===\n")

# ── V1: Shape ─────────────────────────────────────────────────────────────────
print("V1 — shape")
try:
    model = build()
    sup, lbl, qry = make_episode(0)
    with torch.no_grad():
        out = model(sup, lbl, qry)
    logits = out["logits"]
    expected = (NQ * WAY, WAY)
    check("logits shape == (nq, way)", tuple(logits.shape) == expected,
          f"got {tuple(logits.shape)}, expected {expected}")
except Exception as e:
    check("V1 no exception", False, str(e))
    import traceback; traceback.print_exc()
print()


# ── V2: Query intact ───────────────────────────────────────────────────────────
print("V2 — query unchanged across different supports")
try:
    model = build()
    model.eval()

    # In tuple mode, self.intra fires twice:
    #   1st call: input shape [nq, T, d]  — query set  (nq = WAY*NQ = 25)
    #   2nd call: input shape [ns, T, d]  — support set (ns = WAY*SHOT = 5)
    # We capture only the q_set by filtering on the batch size.
    NQ_TOTAL = WAY * NQ   # 25
    captured_q = []
    def _hook_q(module, inp, out):
        # output of intra on q_set: shape [nq, T, d]
        if out.shape[0] == NQ_TOTAL:
            captured_q.append(out.detach().clone())

    h = model.intra.register_forward_hook(_hook_q)

    sup1, lbl1, qry = make_episode(1)
    sup2, lbl2, _   = make_episode(2)   # different support, SAME query

    with torch.no_grad():
        model(sup1, lbl1, qry)
        if not captured_q:
            raise RuntimeError("hook did not capture q_set in run 1")
        q_from_run1 = captured_q[-1].clone()

        captured_q.clear()
        model(sup2, lbl2, qry)
        if not captured_q:
            raise RuntimeError("hook did not capture q_set in run 2")
        q_from_run2 = captured_q[-1].clone()

    h.remove()

    close = torch.allclose(q_from_run1, q_from_run2, atol=1e-5)
    max_diff = (q_from_run1 - q_from_run2).abs().max().item()
    check("q_set allclose with different supports", close,
          f"max_diff={max_diff:.2e}")
except Exception as e:
    check("V2 no exception", False, str(e))
    import traceback; traceback.print_exc()
print()


# ── V3: Not a no-op ────────────────────────────────────────────────────────────
print("V3 — decouple changes s_set (not a no-op)")
try:
    model_decouple  = build(inter_style="decouple")
    model_no_inter  = build(use_inter=False)

    # Clone weights from decouple to no_inter (matching, intra only)
    no_inter_sd = model_no_inter.state_dict()
    dec_sd      = model_decouple.state_dict()
    for k in no_inter_sd:
        if k in dec_sd and dec_sd[k].shape == no_inter_sd[k].shape:
            no_inter_sd[k] = dec_sd[k].clone()
    model_no_inter.load_state_dict(no_inter_sd, strict=False)

    sup, lbl, qry = make_episode(3)

    # Capture s_set after intra (and after decouple, if present)
    s_decouple_cap = []
    s_no_inter_cap = []

    def _hook_s_dec(m, i, o):
        s_decouple_cap.append(o.detach().clone())
    def _hook_s_noi(m, i, o):
        s_no_inter_cap.append(o.detach().clone())

    # For decouple model: hook on inter (SupportDecoupleRelation)
    h_dec = model_decouple.inter.register_forward_hook(
        lambda m, i, o: s_decouple_cap.append(o.detach().clone()))
    # For no-inter model: hook on intra (last thing that touches s_set)
    h_noi = model_no_inter.intra.register_forward_hook(
        lambda m, i, o: s_no_inter_cap.append(i[0].detach().clone()))

    with torch.no_grad():
        model_decouple(sup, lbl, qry)
        model_no_inter(sup, lbl, qry)

    h_dec.remove()
    h_noi.remove()

    if s_decouple_cap and s_no_inter_cap:
        s_dec = s_decouple_cap[-1]
        s_noi = s_no_inter_cap[-1]
        same  = torch.allclose(s_dec, s_noi, atol=1e-5)
        max_diff = (s_dec - s_noi).abs().max().item()
        check("s_set modified by decouple (not same as no_inter)",
              not same, f"max_diff={max_diff:.4f}")
    else:
        check("hooks fired", False,
              f"dec_cap={len(s_decouple_cap)}, noi_cap={len(s_no_inter_cap)}")
except Exception as e:
    check("V3 no exception", False, str(e))
    import traceback; traceback.print_exc()
print()


# ── V4: Co-scale + permutation invariance ──────────────────────────────────────
print("V4 — distances co-scale and permute correctly")
try:
    model = build()
    sup, lbl, qry = make_episode(4)

    with torch.no_grad():
        logits_orig = model(sup, lbl, qry)["logits"]  # [nq, way]

    # Permute support: swap class 0 and class 1
    perm      = torch.tensor([1, 0, 2, 3, 4])
    perm_lbl  = perm[lbl]                             # remap class indices

    with torch.no_grad():
        logits_perm = model(sup, perm_lbl, qry)["logits"]

    # The distances to class 0 in orig should equal distances to class 1 in perm
    # (both use the same support videos, just with swapped labels)
    dist_0_orig = -logits_orig[:, 0]   # negate to get distance from logit
    dist_1_perm = -logits_perm[:, 1]

    close_perm = torch.allclose(dist_0_orig, dist_1_perm, atol=1e-4)
    perm_diff  = (dist_0_orig - dist_1_perm).abs().max().item()

    # Check co-scale: all 5 distances same order of magnitude
    # Use first query's absolute distances; log-range < 3 decades = co-scale
    dists_abs = -logits_orig[0]   # [way] (distances are negated logits)
    log_range = (dists_abs.abs().max() / (dists_abs.abs().min() + 1e-8)).log10().item()
    co_scale  = log_range < 3.0

    check("permuting support permutes distances (perm diff < 1e-3)",
          perm_diff < 1e-3, f"max_diff={perm_diff:.2e}")
    check("all 5 class distances co-scale (log-range < 3 decades)",
          co_scale, f"log_range={log_range:.2f}")
except Exception as e:
    check("V4 no exception", False, str(e))
    import traceback; traceback.print_exc()
print()


# ════════════════════════════════════════════════════════════════════════════════
print(f"=== D1 result: {PASS_COUNT} PASS, {FAIL_COUNT} FAIL ===")
if FAIL_COUNT > 0:
    print("  FAIL detected — do NOT launch training. Switch to Route 2.")
    sys.exit(1)
else:
    print("  All 4 checks passed — ready to launch Route 1.")
    sys.exit(0)
