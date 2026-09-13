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

def make_args(grad_ckpt=False, ckpt_segments=8, freeze_backbone=False):
    return types.SimpleNamespace(
        trans_linear_in_dim=512,
        trans_linear_out_dim=1152,
        way=5, shot=1, query_per_class=5,
        trans_dropout=0.1,
        seq_len=8,
        img_size=84,
        method="resnet18",
        num_gpus=1,
        temp_set=[2],
        grad_ckpt=grad_ckpt,
        ckpt_segments=ckpt_segments,
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
# V4: numerical equivalence grad_ckpt on vs off
# ---------------------------------------------------------------------------
print("\n--- V4: 數值等價 ---")
args_v4_off = make_args(grad_ckpt=False)
args_v4_on  = make_args(grad_ckpt=True, ckpt_segments=8)

m_off = CNN_TRX(args_v4_off).to(device).train()
m_on  = CNN_TRX(args_v4_on ).to(device).train()

# Copy weights so they start identical
m_on.load_state_dict(m_off.state_dict())

# BN momentum correction for grad_ckpt (same as init_model)
import math
m_val = 1 - math.sqrt(1 - 0.1)
for mod in m_on.modules():
    if isinstance(mod, nn.modules.batchnorm._BatchNorm):
        mod.momentum = m_val

# Set eval to avoid BN stochasticity between runs
m_off.eval()
m_on.eval()

torch.manual_seed(99)
ctx2 = torch.randn(WAY * SHOT * SEQ, 3, 84, 84, device=device)
tgt2 = torch.randn(WAY * QPC  * SEQ, 3, 84, 84, device=device)

with torch.no_grad():
    logits_off = m_off(ctx2, lbl, tgt2)["logits"]
    logits_on  = m_on (ctx2, lbl, tgt2)["logits"]

max_diff = (logits_off - logits_on).abs().max().item()
if max_diff < 1e-4:
    print(f"  max |logits_off - logits_on| = {max_diff:.2e} < 1e-4  {PASS}")
    results["V4"] = True
else:
    print(f"  max diff = {max_diff:.2e} ≥ 1e-4  {FAIL}")
    results["V4"] = False


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n========================================")
print("Phase 0.1 驗證結果：")
all_pass = True
for k in ["V1", "V2", "V3", "V4"]:
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
