"""
verify_matching.py — 四項正確性驗證，全部印出 PASS/FAIL。

V1 — 回歸等價：pool_hausdorff(bidir) ≈ mean_hausdorff_bidir
V2 — 1-shot 等價：pool ≈ instance (K=1)
V3 — 5-shot 不等價 + 方向性：pool < instance (K>1)
V4 — 形狀與梯度：12 種組合

最後跑 250-iter 短訓練確認 loss 下降 + [INFO] 行正確。
"""

import sys, os, subprocess, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from matching.mean_hausdorff import (
    mean_hausdorff_bidir,
    pool_hausdorff,
    instance_class_distance,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

_passed = []
_failed = []

def _result(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    msg = f"  [{tag}] {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    (_passed if ok else _failed).append(name)


# ===========================================================================
# V1 — 回歸等價
# ===========================================================================
print("\n" + "=" * 60)
print("V1 — 回歸等價: pool_hausdorff(bidir) ≈ mean_hausdorff_bidir")
print("=" * 60)

# 1-shot
torch.manual_seed(0)
q = torch.randn(5, 28, 1152, device=DEVICE)
s = torch.randn(1, 28, 1152, device=DEVICE)
ref  = mean_hausdorff_bidir(q, s)
got  = pool_hausdorff(q, s, mode="bidirectional")
ok   = torch.allclose(got, ref, atol=1e-5)
_result("1-shot: pool_bidir ≈ mean_hausdorff_bidir", ok,
        f"max_diff={( got - ref).abs().max().item():.2e}")

# 5-shot
torch.manual_seed(1)
q5 = torch.randn(5, 28, 1152, device=DEVICE)
s5 = torch.randn(5, 28, 1152, device=DEVICE)
ref5 = mean_hausdorff_bidir(q5, s5)
got5 = pool_hausdorff(q5, s5, mode="bidirectional")
ok5  = torch.allclose(got5, ref5, atol=1e-5)
_result("5-shot: pool_bidir ≈ mean_hausdorff_bidir", ok5,
        f"max_diff={(got5 - ref5).abs().max().item():.2e}")


# ===========================================================================
# V2 — 1-shot 時 pool 與 instance 等價
# ===========================================================================
print("\n" + "=" * 60)
print("V2 — 1-shot 等價: pool ≈ instance (K=1)")
print("=" * 60)

torch.manual_seed(2)
q  = torch.randn(5, 28, 1152, device=DEVICE)
s1 = torch.randn(1, 28, 1152, device=DEVICE)   # K=1

for mode in ("unidirectional", "bidirectional"):
    p = pool_hausdorff(q, s1, mode=mode)
    i = instance_class_distance(q, s1, mode=mode)
    diff = (p - i).abs().max().item()
    ok = diff < 1e-4
    _result(f"K=1, mode={mode}: pool ≈ instance", ok,
            f"max_diff={diff:.2e}")


# ===========================================================================
# V3 — 5-shot 時兩者不等價，且方向性正確 (pool <= instance)
# ===========================================================================
print("\n" + "=" * 60)
print("V3 — 5-shot 不等價 + 方向性: pool ≤ instance (K>1)")
print("=" * 60)

torch.manual_seed(3)
q   = torch.randn(5, 28, 1152, device=DEVICE)
s5  = torch.randn(5, 28, 1152, device=DEVICE)   # K=5

for mode in ("unidirectional", "bidirectional"):
    p5 = pool_hausdorff(q, s5, mode=mode)
    i5 = instance_class_distance(q, s5, mode=mode)
    diff = (p5 - i5).abs().mean().item()
    not_equal = diff > 1e-4
    direction_ok = (p5 <= i5 + 1e-4).all().item()   # pool ≤ instance

    _result(f"K=5, mode={mode}: not equal", not_equal,
            f"mean_diff={diff:.4f}")
    _result(f"K=5, mode={mode}: pool ≤ instance", direction_ok,
            f"pool_mean={p5.mean().item():.4f}, inst_mean={i5.mean().item():.4f}")

    if not direction_ok:
        print("    ⚠️  方向性錯誤：pool 應 ≤ instance，請停下來檢查實作")
        sys.exit(1)


# ===========================================================================
# V4 — 形狀與梯度 (12 組合)
# ===========================================================================
print("\n" + "=" * 60)
print("V4 — 形狀與梯度 (3 mode × 2 agg × 2 shot)")
print("=" * 60)

MODES = ["unidirectional", "bidirectional", "attention_weighted"]
AGGS  = ["pool", "instance"]
SHOTS = [1, 5]
nq    = 5
T, d  = 28, 1152

for shot in SHOTS:
    for agg in AGGS:
        for mode in MODES:
            torch.manual_seed(42)
            q = torch.randn(nq, T, d, device=DEVICE, requires_grad=True)
            s = torch.randn(shot, T, d, device=DEVICE)

            if agg == "pool":
                out = pool_hausdorff(q, s, mode=mode)
            else:
                out = instance_class_distance(q, s, mode=mode)

            shape_ok = out.shape == (nq,)
            out.sum().backward()
            grad_ok  = q.grad is not None and not torch.isnan(q.grad).any()

            label = f"shot={shot}, agg={agg}, mode={mode}"
            _result(f"shape (nq,): {label}", shape_ok, f"got {out.shape}")
            _result(f"gradient ok: {label}", grad_ok)


# ===========================================================================
# 短訓練確認
# ===========================================================================
print("\n" + "=" * 60)
print("短訓練 (250 iter) — stage1_bidirectional + new keys")
print("=" * 60)

import tempfile, shutil, yaml as _yaml
PYTHON = "/home/ccwu/miniconda3/envs/trx/bin/python"
ckpt_dir = tempfile.mkdtemp(prefix="verify_ckpt_")

# Load base config and inject new keys into a temporary YAML file
base_cfg_path = os.path.join(os.path.dirname(__file__), "configs/stage1_bidirectional.yaml")
with open(base_cfg_path) as f:
    base_cfg = _yaml.safe_load(f) or {}
base_cfg["matching"]        = "bidirectional"
base_cfg["set_aggregation"] = "pool"
tmp_cfg_path = os.path.join(ckpt_dir, "verify_smoke.yaml")
with open(tmp_cfg_path, "w") as f:
    _yaml.dump(base_cfg, f)

cmd = [
    PYTHON, "-u", "run.py",
    "--config", tmp_cfg_path,
    "--dataset", "hmdb", "--split", "3",
    "--scratch", os.path.expanduser("~/Documents/work/trx/trx_data"),
    "-c", ckpt_dir,
    "--training_iterations", "250",
    "--tasks_per_batch", "1",
    "--num_test_tasks", "10",
    "--test_iters", "250",
    "--print_freq", "50",
    "--save_freq", "9999",
    "--num_workers", "2",
]

print(f"  Running: {' '.join(cmd[-10:])}")
result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
output = result.stdout + result.stderr

# Check [INFO] matching line
has_info = "[INFO] matching=bidirectional, set_aggregation=pool" in output
_result("[INFO] matching/set_aggregation 印出正確", has_info)

# Check [WARN] for legacy keys (stage1_bidirectional.yaml has matching_method + bidirectional)
has_warn = "[WARN]" in output
_result("[WARN] legacy key 偵測到", has_warn)

# Extract first and last loss
losses = [float(l.split("Train Loss: ")[1].split(",")[0])
          for l in output.splitlines() if "Train Loss:" in l]
if len(losses) >= 2:
    loss_drop = losses[-1] < losses[0]
    _result(f"loss 下降 ({losses[0]:.4f} → {losses[-1]:.4f})", loss_drop)
else:
    _result("loss 可讀取", False, f"found {len(losses)} loss lines")

if result.returncode != 0 and "Train Loss:" not in output:
    print("\n  ⚠️  訓練輸出（尾部）:")
    for line in output.splitlines()[-20:]:
        print(f"    {line}")

shutil.rmtree(ckpt_dir, ignore_errors=True)


# ===========================================================================
# 總結
# ===========================================================================
print("\n" + "=" * 60)
total  = len(_passed) + len(_failed)
print(f"結果：{len(_passed)}/{total} PASS，{len(_failed)} FAIL")
if _failed:
    print("  FAIL 項目：")
    for f in _failed:
        print(f"    - {f}")
    print("=" * 60)
    sys.exit(1)
else:
    print("  全部通過 ✓")
    print("=" * 60)
    sys.exit(0)
