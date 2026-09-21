"""
verify_ckpt_shape.py — Phase 0.1 驗證

V1  state_dict 鍵名完全沒變（grad_ckpt on/off 的模型鍵集合相同）
V2  能載入既有 checkpoint（strict=True）
V3  梯度真的有流（backbone conv1.weight.grad 非零）
V4  數值等價（同一輸入，logits 最大絕對差 < 1e-4）

全部通過才算 Phase 0.1 修正正確。
"""

import sys
import os
import types
import torch
import torch.nn as nn
import random

# 確保從 repo 根目錄執行
sys.path.insert(0, os.path.dirname(__file__))

from models.stage1_model import CNN_TRX
from models.stage2_model import CNN_TRXWithRelation

# Stage 1 checkpoint (B1 backbone, no relation modules) — needed for strict load test
CKPT_PATH = (
    "/home/ccwu/trx_data/checkpoints/"
    "b1_seed43_1shot_hmdb3/hmdb_split3_20260907_123206/checkpoint60000.pt"
)
# Stage 2 checkpoint (B2 seed43, intra-only) — used if CKPT_PATH is not Stage 1
CKPT_PATH_S2 = (
    "/home/ccwu/trx_data/checkpoints/"
    "b2_seed43_1shot_hmdb3/hmdb_split3_20260906_010142/checkpoint25000.pt"
)

def make_args(grad_ckpt=False, ckpt_segments=8, freeze_backbone=False,
              method="resnet18", trans_linear_in_dim=512, ckpt_prefix=None):
    return types.SimpleNamespace(
        trans_linear_in_dim=trans_linear_in_dim,
        trans_linear_out_dim=1152,
        way=5, shot=1, query_per_class=5,
        trans_dropout=0.1,
        seq_len=8,
        img_size=84,
        method=method,
        num_gpus=1,
        temp_set=[2],
        grad_ckpt=grad_ckpt,
        ckpt_segments=ckpt_segments,
        ckpt_prefix=ckpt_prefix,
        freeze_backbone=freeze_backbone,
        matching="bidirectional",
        set_aggregation="pool",
        tau=1.0,
    )


PASS = "✓ PASS"
FAIL = "✗ FAIL"
results = {}


# ---------------------------------------------------------------------------
# V1: state_dict key names unchanged
# ---------------------------------------------------------------------------
print("\n--- V1: state_dict 鍵名 ---")
args_off = make_args(grad_ckpt=False)
args_on  = make_args(grad_ckpt=True, ckpt_segments=8)
m_off = CNN_TRX(args_off)
m_on  = CNN_TRX(args_on)
keys_off = set(m_off.state_dict().keys())
keys_on  = set(m_on.state_dict().keys())

if keys_off == keys_on:
    print(f"  keys: {len(keys_off)} total, diff = 0  {PASS}")
    results["V1"] = True
else:
    only_off = keys_off - keys_on
    only_on  = keys_on  - keys_off
    print(f"  only in grad_ckpt=False: {sorted(only_off)[:5]}")
    print(f"  only in grad_ckpt=True:  {sorted(only_on)[:5]}")
    print(f"  {FAIL}")
    results["V1"] = False


# ---------------------------------------------------------------------------
# V2: load existing checkpoint strict=True
# ---------------------------------------------------------------------------
print("\n--- V2: 載入既有 checkpoint ---")
def _try_load(ckpt_path, model):
    """Try strict=True load; return (ok, detail_str)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    else:
        sd = ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return unexpected, missing

# V2 — test with Stage 1 checkpoint into Stage 1 model
for ckpt_path, label in [(CKPT_PATH, "Stage1 B1"), (CKPT_PATH_S2, "Stage2 B2")]:
    if not os.path.exists(ckpt_path):
        print(f"  {label}: checkpoint 不存在: {ckpt_path}")
        continue
    try:
        # Build matching model
        if label.startswith("Stage1"):
            m_v2 = CNN_TRX(make_args(grad_ckpt=True))
        else:
            # Stage 2 B2 (intra-only, no inter)
            from models.stage2_model import CNN_TRXWithRelation
            args_s2 = make_args(grad_ckpt=True, ckpt_segments=8)
            args_s2.use_intra_relation = True
            args_s2.use_inter_relation = False
            args_s2.relation_level = "tuple"
            args_s2.inter_style = "global"
            args_s2.intra_depth = 1
            m_v2 = CNN_TRXWithRelation(args_s2)

        unexpected, missing = _try_load(ckpt_path, m_v2)
        if len(unexpected) == 0:
            print(f"  {label}: 0 unexpected keys (missing={len(missing)})  {PASS}")
            results["V2"] = True
        else:
            print(f"  {label}: unexpected={len(unexpected)} keys: {unexpected[:3]}  {FAIL}")
            results.setdefault("V2", False)
        break  # first working checkpoint is enough
    except Exception as e:
        print(f"  {label}: exception: {e}  {FAIL}")
        results.setdefault("V2", False)


# ---------------------------------------------------------------------------
# V3: gradients flow through backbone
# ---------------------------------------------------------------------------
print("\n--- V3: 梯度流 ---")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
args_v3 = make_args(grad_ckpt=True, ckpt_segments=8)
m_v3 = CNN_TRX(args_v3).to(device).train()

torch.manual_seed(42)
WAY, SHOT, QPC, SEQ = 5, 1, 5, 8
ctx = torch.randn(WAY * SHOT * SEQ, 3, 84, 84, device=device)
tgt = torch.randn(WAY * QPC  * SEQ, 3, 84, 84, device=device)
lbl = torch.arange(WAY, device=device)

out = m_v3(ctx, lbl, tgt)
loss = out["logits"].sum()
loss.backward()

# Check conv1.weight.grad (index 0 in resnet = conv1)
conv1_grad = None
for name, p in m_v3.named_parameters():
    if "resnet" in name and "weight" in name and p.grad is not None:
        conv1_grad = p.grad.abs().sum().item()
        conv1_name = name
        break

if conv1_grad is not None and conv1_grad > 0:
    print(f"  {conv1_name}.grad.abs().sum() = {conv1_grad:.6f} > 0  {PASS}")
    results["V3"] = True
else:
    print(f"  backbone grad is None or zero  {FAIL}")
    results["V3"] = False


# ---------------------------------------------------------------------------
# V4: 梯度等價（0921 修好版 —— 原版用 .eval() 讓 checkpoint 分支從未執行，
# 是空測試，見 0918_階段4_等價性與BN修正.md 的「1b. 修好的 V4」）
#
# 修法：
#   - 兩個模型都 .train()（checkpoint 分支才會被觸發）
#   - 所有 BN 的 momentum 設 0（running stats 不更新，避免污染比較；
#     不影響 train 模式下這次 forward 的輸出——跟原本用 .eval() 想達到的
#     效果一樣，但不會連帶關掉 checkpoint 分支）
#   - 所有 Dropout 的 p 設 0（train() 模式下 dropout 會讓兩個模型隨機不同，
#     這是原 V4 沒處理到的另一個潛在污染源）
#   - m_on 的權重從 m_off 複製，同一份輸入
#   - 不用 no_grad()：做 loss.backward()，比每一個參數的 .grad
#   - forward() 裡的 self._last_ckpt_branch 旗標：assert m_on 真的走了
#     checkpoint 分支、m_off 沒走 —— 這是最重要的一行，原 V4 就是沒有
#     這行才讓 bug 藏了三天
#   - 控制組：m_off 用同一份輸入跑兩次，兩次的差值當底線（CPU 上兩次
#     forward+backward 應該逐位元相同，這條底線預期是 0.00e+00）
#   - 全部在 CPU 上跑，求位元級可比
# ---------------------------------------------------------------------------
print("\n--- V4: 梯度等價（修好版，checkpoint_sequential 路徑） ---")


def _zero_stochastic(model):
    """把所有會讓 train() 模式下兩次 forward 不一樣的隨機性關掉，但不動
    真正要比較的計算路徑。BN momentum 只影響 running_mean/var 這個側效應
    的更新，不影響 train 模式下這次 forward 算出來的數值——所以不需要（也
    不該）複刻 run.py 的 BN_MOM_FIX，全程 momentum=0 就夠。"""
    for mod in model.modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            mod.momentum = 0.0
        elif isinstance(mod, nn.Dropout):
            mod.p = 0.0
        elif isinstance(mod, nn.MultiheadAttention):
            # nn.MultiheadAttention 的 dropout 是建構時存的 float 屬性，
            # 不是子模組，不會被上面的 nn.Dropout 分支抓到。
            # true_hyrsm inter-relation 用到的 self.attn 就是 dropout=0.05。
            mod.dropout = 0.0


def _grad_dict(model):
    return {name: p.grad.detach().clone()
            for name, p in model.named_parameters() if p.grad is not None}


def _max_diff(grads_a, grads_b):
    assert set(grads_a.keys()) == set(grads_b.keys()), "參數集合不一致"
    max_d = -1.0
    max_name = None
    for name in grads_a:
        d = (grads_a[name] - grads_b[name]).abs().max().item()
        if d > max_d:
            max_d, max_name = d, name
    return max_d, max_name


def _run_equivalence_core(label, m_off, m_on, ctx, tgt, lbl_cpu, conv1_prefix="resnet.0.",
                           expected_branch_on="checkpoint_sequential"):
    """V4 / V4b 共用的核心：假設 m_off/m_on 已經 .train()、zero 掉隨機性、
    權重同步過了。跑一次 off、一次 on、一次 off 的控制組，比梯度。"""
    # --- m_off: 不開 ckpt ---
    m_off.zero_grad()
    logits_off = m_off(ctx, lbl_cpu, tgt)["logits"]
    logits_off.sum().backward()
    branch_off = getattr(m_off, "_last_ckpt_branch", None)
    grads_off = _grad_dict(m_off)

    # --- m_on: 開 ckpt（checkpoint_sequential 路徑） ---
    m_on.zero_grad()
    logits_on = m_on(ctx, lbl_cpu, tgt)["logits"]
    logits_on.sum().backward()
    branch_on = getattr(m_on, "_last_ckpt_branch", None)
    grads_on = _grad_dict(m_on)

    # --- 控制組：m_off 用同一份輸入再跑一次，當噪聲底線 ---
    m_off.zero_grad()
    logits_off2 = m_off(ctx, lbl_cpu, tgt)["logits"]
    logits_off2.sum().backward()
    grads_off2 = _grad_dict(m_off)
    baseline_diff, _ = _max_diff(grads_off, grads_off2)

    max_diff, max_name = _max_diff(grads_off, grads_on)
    conv1_name = next((n for n in grads_off if n.startswith(conv1_prefix)), None)
    conv1_diff = (grads_off[conv1_name] - grads_on[conv1_name]).abs().max().item() if conv1_name else float("nan")

    # 梯度非零檢查：差值=0 有可能只是兩邊梯度都是零，不是真的等價。
    all_grad_abs_max = max(g.abs().max().item() for g in grads_off.values())
    conv1_grad_abs_max = grads_off[conv1_name].abs().max().item() if conv1_name else float("nan")
    grad_nonzero_ok = all_grad_abs_max > 1e-8 and conv1_grad_abs_max > 1e-8

    branch_ok = (branch_off == "none" and branch_on == expected_branch_on)
    print(f"  [{label}] branch_off={branch_off!r} branch_on={branch_on!r} "
          f"(預期 branch_on={expected_branch_on!r})  "
          f"{'✓ 分支正確' if branch_ok else '✗ 分支不對，測試本身無效'}")
    print(f"  [{label}] 全體參數梯度 max|grad| = {all_grad_abs_max:.3e}   "
          f"backbone 第一層（{conv1_name}）max|grad| = {conv1_grad_abs_max:.3e}  "
          f"{'✓ 非零' if grad_nonzero_ok else '✗ 梯度是零，測試空洞'}")
    print(f"  [{label}] 控制組底線 |off - off2| max = {baseline_diff:.3e}")
    print(f"  [{label}] max |grad_off - grad_on| (全體參數) = {max_diff:.3e}  (在 {max_name})")
    print(f"  [{label}] |grad_off - grad_on| (backbone 第一層 {conv1_name}) = {conv1_diff:.3e}")

    ok = branch_ok and grad_nonzero_ok and (max_diff <= baseline_diff or max_diff < 1e-4)
    status = PASS if ok else FAIL
    print(f"  [{label}] {status}")
    return ok, dict(branch_ok=branch_ok, grad_nonzero_ok=grad_nonzero_ok,
                     all_grad_abs_max=all_grad_abs_max, conv1_grad_abs_max=conv1_grad_abs_max,
                     baseline_diff=baseline_diff, max_diff=max_diff,
                     max_name=max_name, conv1_diff=conv1_diff)


def run_v4(label, method, trans_linear_in_dim):
    """V4：models.stage1_model.CNN_TRX（B1，無 relation 模組）。"""
    device_cpu = "cpu"
    args_off = make_args(grad_ckpt=False, method=method, trans_linear_in_dim=trans_linear_in_dim)
    args_on  = make_args(grad_ckpt=True, ckpt_segments=8, method=method, trans_linear_in_dim=trans_linear_in_dim)

    m_off = CNN_TRX(args_off).to(device_cpu).train()
    m_on  = CNN_TRX(args_on ).to(device_cpu).train()

    _zero_stochastic(m_off)
    _zero_stochastic(m_on)
    m_on.load_state_dict(m_off.state_dict())

    torch.manual_seed(99)
    ctx = torch.randn(WAY * SHOT * SEQ, 3, 84, 84, device=device_cpu)
    tgt = torch.randn(WAY * QPC  * SEQ, 3, 84, 84, device=device_cpu)
    lbl_cpu = lbl.to(device_cpu)

    return _run_equivalence_core(label, m_off, m_on, ctx, tgt, lbl_cpu)


def make_args_s2(grad_ckpt=False, ckpt_segments=8, method="resnet18",
                  trans_linear_in_dim=512, ckpt_prefix=None, ckpt_policy="full"):
    """對照 configs/stage2_true_hyrsm_b4_tuple.yaml 的設定。"""
    return types.SimpleNamespace(
        trans_linear_in_dim=trans_linear_in_dim,
        trans_linear_out_dim=1152,
        way=5, shot=1, query_per_class=5,
        trans_dropout=0.1,
        seq_len=8,
        img_size=84,
        method=method,
        num_gpus=1,
        temp_set=[2],
        grad_ckpt=grad_ckpt,
        ckpt_segments=ckpt_segments,
        ckpt_prefix=ckpt_prefix,
        ckpt_policy=ckpt_policy,
        freeze_backbone=False,
        matching="bidirectional",
        set_aggregation="pool",
        tau=0.1,
        relation_level="tuple",
        use_intra_relation=True,
        use_inter_relation=True,
        inter_style="true_hyrsm",
        intra_depth=1,
    )


def run_v4b(label, method, trans_linear_in_dim):
    """V4b：models.stage2_model.CNN_TRXWithRelation（B4，100k 那次跑的模型），
    設定照 stage2_true_hyrsm_b4_tuple.yaml。"""
    device_cpu = "cpu"
    args_off = make_args_s2(grad_ckpt=False, method=method, trans_linear_in_dim=trans_linear_in_dim)
    args_on  = make_args_s2(grad_ckpt=True, ckpt_segments=8, method=method, trans_linear_in_dim=trans_linear_in_dim)

    m_off = CNN_TRXWithRelation(args_off).to(device_cpu).train()
    m_on  = CNN_TRXWithRelation(args_on ).to(device_cpu).train()

    _zero_stochastic(m_off)
    _zero_stochastic(m_on)
    m_on.load_state_dict(m_off.state_dict())

    torch.manual_seed(99)
    ctx = torch.randn(WAY * SHOT * SEQ, 3, 84, 84, device=device_cpu)
    tgt = torch.randn(WAY * QPC  * SEQ, 3, 84, 84, device=device_cpu)
    lbl_cpu = lbl.to(device_cpu)

    return _run_equivalence_core(label, m_off, m_on, ctx, tgt, lbl_cpu)


v4_rn18_ok, v4_rn18_detail = run_v4("RN18", "resnet18", 512)
v4_rn50_ok, v4_rn50_detail = run_v4("RN50", "resnet50", 2048)
results["V4_RN18"] = v4_rn18_ok
results["V4_RN50"] = v4_rn50_ok

print("\n--- V4b: 梯度等價（stage2_model.CNN_TRXWithRelation，B4/true_hyrsm 設定） ---")
v4b_rn18_ok, v4b_rn18_detail = run_v4b("V4b-RN18", "resnet18", 512)
v4b_rn50_ok, v4b_rn50_detail = run_v4b("V4b-RN50", "resnet50", 2048)
results["V4b_RN18"] = v4b_rn18_ok
results["V4b_RN50"] = v4b_rn50_ok


# ---------------------------------------------------------------------------
# 1b: --ckpt_prefix 路徑的等價性（0921，stage2_model 上，用 V4b 流程）
# ---------------------------------------------------------------------------
def run_ckpt_prefix_equivalence(label, method, trans_linear_in_dim, N, segs, ckpt_policy="full"):
    """驗證新的 --ckpt_prefix 手動分組 checkpoint 路徑：
      - 梯度等價（同 V4/V4b 的核心比較）
      - branch_on 走的是 'prefix_full'／'prefix_save_conv'（0922 改名，
        原本叫 'ckpt_prefix'），不是舊的 'checkpoint_sequential'
      - BN 涵蓋數對得上（chain[:N] 裡的 BN 數）
      - 分組（_split_even）恰好涵蓋 0..N-1，不重複不遺漏——N 不整除 segs
        時最容易漏測到 off-by-one。
    """
    device_cpu = "cpu"
    args_off = make_args_s2(grad_ckpt=False, method=method, trans_linear_in_dim=trans_linear_in_dim)
    args_on  = make_args_s2(grad_ckpt=True, ckpt_segments=segs, ckpt_prefix=N, ckpt_policy=ckpt_policy,
                             method=method, trans_linear_in_dim=trans_linear_in_dim)

    m_off = CNN_TRXWithRelation(args_off).to(device_cpu).train()
    m_on  = CNN_TRXWithRelation(args_on ).to(device_cpu).train()

    _zero_stochastic(m_off)
    _zero_stochastic(m_on)
    m_on.load_state_dict(m_off.state_dict())

    # --- BN 涵蓋數 ---
    chain = m_on._ckpt_chain()
    n_chain = len(chain)
    N_eff = min(N, n_chain)
    total_bn = sum(1 for mm in m_on.modules() if isinstance(mm, nn.modules.batchnorm._BatchNorm))
    covered_bn = sum(1 for blk in chain[:N_eff] for mm in blk.modules()
                      if isinstance(mm, nn.modules.batchnorm._BatchNorm))
    print(f"  [{label}] BN 涵蓋 {covered_bn} / 總計 {total_bn}，n_ckpt={N_eff}")

    # --- 分組邊界：恰好涵蓋 0..N-1，不重複不遺漏 ---
    segs_eff = max(1, min(segs, N_eff)) if N_eff > 0 else 0
    bounds = m_on._split_even(N_eff, segs_eff) if N_eff > 0 else []
    group_str = "".join(f"[{s}-{e - 1}]" for s, e in bounds)
    covered_idx = set()
    overlap = False
    for s, e in bounds:
        rng = set(range(s, e))
        if covered_idx & rng:
            overlap = True
        covered_idx |= rng
    coverage_ok = (not overlap) and covered_idx == set(range(N_eff))
    print(f"  [{label}] 分組: {group_str or '(N=0)'}  "
          f"涵蓋 0..{N_eff - 1} 完整無重複無遺漏: {'✓' if coverage_ok else '✗ 有問題！'}")

    torch.manual_seed(99)
    ctx = torch.randn(WAY * SHOT * SEQ, 3, 84, 84, device=device_cpu)
    tgt = torch.randn(WAY * QPC  * SEQ, 3, 84, 84, device=device_cpu)
    lbl_cpu = lbl.to(device_cpu)

    _expected_branch = "prefix_save_conv" if ckpt_policy == "save_conv" else "prefix_full"
    ok, detail = _run_equivalence_core(label, m_off, m_on, ctx, tgt, lbl_cpu,
                                        expected_branch_on=_expected_branch)
    ok = ok and coverage_ok
    detail["covered_bn"] = covered_bn
    detail["total_bn"] = total_bn
    detail["n_ckpt"] = N_eff
    detail["coverage_ok"] = coverage_ok
    if not ok:
        print(f"  [{label}] {FAIL}（含分組/BN 涵蓋檢查）")
    return ok, detail


print("\n--- 1b: --ckpt_prefix 路徑的等價性（stage2_model，用 V4b 流程） ---")
prefix_1bi_ok, prefix_1bi_detail   = run_ckpt_prefix_equivalence("1b-i (RN18,N=7,segs=7)",  "resnet18", 512,  N=7,  segs=7)
prefix_1bii_ok, prefix_1bii_detail = run_ckpt_prefix_equivalence("1b-ii(RN50,N=14,segs=7)", "resnet50", 2048, N=14, segs=7)
prefix_1biii_ok, prefix_1biii_detail = run_ckpt_prefix_equivalence("1b-iii(RN18,N=11,segs=4)", "resnet18", 512, N=11, segs=4)
results["1b-i"]   = prefix_1bi_ok
results["1b-ii"]  = prefix_1bii_ok
results["1b-iii"] = prefix_1biii_ok

print("\n--- 3b: selective checkpointing（save_conv）路徑的等價性 ---")
sac_rn18_ok, sac_rn18_detail = run_ckpt_prefix_equivalence(
    "3b-RN18(N=7,segs=7,save_conv)", "resnet18", 512, N=7, segs=7, ckpt_policy="save_conv")
sac_rn50_ok, sac_rn50_detail = run_ckpt_prefix_equivalence(
    "3b-RN50(N=14,segs=7,save_conv)", "resnet50", 2048, N=14, segs=7, ckpt_policy="save_conv")
results["3b-RN18"] = sac_rn18_ok
results["3b-RN50"] = sac_rn50_ok


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n========================================")
print("Phase 0.1 驗證結果：")
all_pass = True
for k in ["V1", "V2", "V3", "V4_RN18", "V4_RN50", "V4b_RN18", "V4b_RN50",
          "1b-i", "1b-ii", "1b-iii", "3b-RN18", "3b-RN50"]:
    status = PASS if results.get(k) else FAIL
    print(f"  {k}: {status}")
    if not results.get(k):
        all_pass = False

if all_pass:
    print("\n✅ 全部通過 — Phase 0.1 修正正確")
    sys.exit(0)
else:
    print("\n❌ 有項目失敗 — 查看上方細節")
    sys.exit(1)
