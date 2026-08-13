"""
verify_stage2_dispatch.py — 驗證 Stage 2 接上距離函數分派後行為正確。

要在有 torch 的機器上跑：
    python verify_stage2_dispatch.py

四項檢查，全部印 PASS/FAIL：

  V1  回歸等價：新的預設路徑（bidirectional + pool）與舊的 mean_hausdorff_bidir
      在數值上完全相同 → 既有的所有 relation module 結果不受影響
  V2  分派真的生效：改 matching / set_aggregation 會讓 logits 改變
      （這正是先前失效的地方——參數讀進來了但模型沒去查）
  V3  三種 inter_style × 兩種 set_aggregation 的 forward 形狀與梯度都正常
  V4  1-shot 時 pool 與 instance 等價（數學上必然，同時是實作正確性的檢查）
"""

import sys, os, types, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from matching.mean_hausdorff import mean_hausdorff_bidir
from models.stage2_model import TRXSetMatchingWithRelation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

WAY, SEQ, D_IN, D_OUT = 5, 8, 512, 1152
_passed, _failed = [], []


def check(name, ok, detail=""):
    (_passed if ok else _failed).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def make_args(matching="bidirectional", set_aggregation="pool", tau=0.1, shot=1):
    return types.SimpleNamespace(
        trans_linear_in_dim=D_IN, trans_linear_out_dim=D_OUT,
        way=WAY, shot=shot, query_per_class=5, trans_dropout=0.1,
        seq_len=SEQ, img_size=224, method="resnet18", num_gpus=1, temp_set=[2],
        matching=matching, set_aggregation=set_aggregation, tau=tau,
    )


def make_batch(shot=1, nq=10):
    ns = WAY * shot
    s = torch.randn(ns, SEQ, D_IN, device=DEVICE)
    q = torch.randn(nq, SEQ, D_IN, device=DEVICE)
    lab = torch.arange(WAY, device=DEVICE).repeat_interleave(shot)
    return s, lab, q


# ───────────────────────────────────────────────────────────────────────────
print("\nV1 — 回歸等價：預設路徑是否與舊的寫死實作完全相同")
print("     （若這一項 FAIL，代表既有結果會被改動，必須停下來）")
# ───────────────────────────────────────────────────────────────────────────
for shot, style in itertools.product((1, 5), ("global", "true_hyrsm", "hyrsm")):
    torch.manual_seed(42)
    args = make_args(shot=shot)
    m = TRXSetMatchingWithRelation(args, relation_level="tuple",
                                   use_intra=True, use_inter=True,
                                   inter_style=style).to(DEVICE).eval()
    s, lab, q = make_batch(shot=shot)
    with torch.no_grad():
        new = m(s, lab, q)["logits"]

    # 用舊的實作手動重算同一件事
    def old_dist(qs, cs):
        return mean_hausdorff_bidir(qs, cs, chunk_s=64)
    orig = m._class_distance
    m._class_distance = lambda a, b: old_dist(a, b)
    with torch.no_grad():
        old = m(s, lab, q)["logits"]
    m._class_distance = orig

    diff = (new - old).abs().max().item()
    check(f"shot={shot} inter_style={style:<11s}", diff < 1e-5, f"最大差 {diff:.2e}")

# ───────────────────────────────────────────────────────────────────────────
print("\nV2 — 分派真的生效：換設定必須讓 logits 改變")
print("     （先前的缺陷正是『參數讀進來了但模型沒去查』）")
# ───────────────────────────────────────────────────────────────────────────
torch.manual_seed(7)
s, lab, q = make_batch(shot=5)
base_logits, base_state = None, None
for mode in ("bidirectional", "unidirectional", "attention_weighted"):
    torch.manual_seed(7)
    m = TRXSetMatchingWithRelation(make_args(matching=mode, shot=5),
                                   relation_level="tuple", use_intra=True,
                                   use_inter=True, inter_style="global").to(DEVICE).eval()
    if base_state is None:
        base_state = m.state_dict()
    else:
        m.load_state_dict(base_state)      # 同一組權重，只換距離函數
    with torch.no_grad():
        lg = m(s, lab, q)["logits"]
    if mode == "bidirectional":
        base_logits = lg
    else:
        d = (lg - base_logits).abs().max().item()
        check(f"matching={mode:<20s} 有生效", d > 1e-4, f"與雙向的最大差 {d:.3f}")

torch.manual_seed(7)
m = TRXSetMatchingWithRelation(make_args(set_aggregation="instance", shot=5),
                               relation_level="tuple", use_intra=True,
                               use_inter=True, inter_style="global").to(DEVICE).eval()
m.load_state_dict(base_state)
with torch.no_grad():
    lg_inst = m(s, lab, q)["logits"]
d = (lg_inst - base_logits).abs().max().item()
check("set_aggregation=instance 有生效（5-shot）", d > 1e-4, f"與 pool 的最大差 {d:.3f}")

# ───────────────────────────────────────────────────────────────────────────
print("\nV3 — 形狀與梯度：三種 inter_style × 兩種 set_aggregation")
# ───────────────────────────────────────────────────────────────────────────
for style, agg in itertools.product(("global", "true_hyrsm", "hyrsm"), ("pool", "instance")):
    args = make_args(set_aggregation=agg, shot=5)
    m = TRXSetMatchingWithRelation(args, relation_level="tuple", use_intra=True,
                                   use_inter=True, inter_style=style).to(DEVICE)
    s, lab, q = make_batch(shot=5)
    s.requires_grad_(True)
    out = m(s, lab, q)["logits"]
    ok_shape = tuple(out.shape) == (q.shape[0], WAY)
    out.sum().backward()
    ok_grad = s.grad is not None and torch.isfinite(s.grad).all().item()
    check(f"{style:<11s} + {agg:<9s}", ok_shape and ok_grad,
          f"logits {tuple(out.shape)}, 梯度 {'正常' if ok_grad else '異常'}")

# ───────────────────────────────────────────────────────────────────────────
print("\nV4 — 1-shot 時 pool 與 instance 必須等價（數學上必然）")
# ───────────────────────────────────────────────────────────────────────────
torch.manual_seed(11)
s, lab, q = make_batch(shot=1)
ref = None
for agg in ("pool", "instance"):
    torch.manual_seed(11)
    m = TRXSetMatchingWithRelation(make_args(set_aggregation=agg, shot=1),
                                   relation_level="tuple", use_intra=True,
                                   use_inter=False).to(DEVICE).eval()
    if ref is None:
        ref_state = m.state_dict()
    else:
        m.load_state_dict(ref_state)
    with torch.no_grad():
        lg = m(s, lab, q)["logits"]
    if ref is None:
        ref = lg
    else:
        d = (lg - ref).abs().max().item()
        check("1-shot pool ≡ instance", d < 1e-4, f"最大差 {d:.2e}")

# ───────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 66)
print(f"通過 {len(_passed)} 項，失敗 {len(_failed)} 項")
if _failed:
    print("失敗項目：")
    for f in _failed:
        print("   -", f)
    print("\n⚠ V1 若失敗代表既有結果會被改動，先不要跑任何正式實驗。")
    sys.exit(1)
print("✅ 全部通過。Stage 2 已正確接上距離函數分派，且預設行為與先前完全一致。")
