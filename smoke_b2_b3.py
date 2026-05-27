"""
Smoke test for B2 (intra only) and B3 (inter only) configs.
Verifies: model loads, loss decreases, correct modules fire, param counts sane.
"""
import subprocess, sys, types, torch, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── param count helper ────────────────────────────────────────────────────────
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ── load model for each config and check param counts ────────────────────────
import yaml
from model import CNN_TRX

def make_args(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if "backbone" in cfg:
        cfg["method"] = cfg.pop("backbone")
    return types.SimpleNamespace(
        trans_linear_in_dim=512, trans_linear_out_dim=cfg.get("trans_linear_out_dim", 1152),
        way=5, shot=1, query_per_class=5, trans_dropout=0.1,
        seq_len=8, img_size=224, method=cfg.get("method","resnet18"),
        num_gpus=1, temp_set=cfg.get("temp_set",[2]), num_samples=cfg.get("num_samples",1),
        use_intra_relation=cfg.get("use_intra_relation", False),
        use_inter_relation=cfg.get("use_inter_relation", False),
        relation_level=cfg.get("relation_level", "tuple"),
    )

print("=" * 60)
print("STEP (d): Parameter count check")
print("=" * 60)
args_full = make_args("configs/stage2_option_b.yaml")
args_b2   = make_args("configs/stage2_b2_intra_only.yaml")
args_b3   = make_args("configs/stage2_b3_inter_only.yaml")

m_full = CNN_TRX(args_full)
m_b2   = CNN_TRX(args_b2)
m_b3   = CNN_TRX(args_b3)

n_full = count_params(m_full)
n_b2   = count_params(m_b2)
n_b3   = count_params(m_b3)

print(f"  Full (intra+inter) : {n_full:,}")
print(f"  B2   (intra only)  : {n_b2:,}  diff={n_b2-n_full:+,}")
print(f"  B3   (inter only)  : {n_b3:,}  diff={n_b3-n_full:+,}")

assert n_b2 < n_full, "B2 should have fewer params than full (no InterRelation)"
assert n_b3 < n_full, "B3 should have fewer params than full (no IntraRelation)"
print("  ✓ Both ablations have fewer params than full model")

# ── enable smoke logging ──────────────────────────────────────────────────────
for t in m_b2.transformers:  t._smoke_log = True
for t in m_b3.transformers:  t._smoke_log = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m_b2.to(device).eval()
m_b3.to(device).eval()

print()
print("=" * 60)
print("STEP (c): Module call verification (one forward pass each)")
print("=" * 60)
support = torch.rand(5 * 1 * 8, 3, 224, 224, device=device)
target  = torch.rand(5 * 5 * 8, 3, 224, 224, device=device)
labels  = torch.arange(5, device=device)

print("  B2 forward:")
with torch.no_grad():
    out_b2 = m_b2(support, labels, target)
print(f"  B2 logits shape: {out_b2['logits'].shape}")

print("  B3 forward:")
with torch.no_grad():
    out_b3 = m_b3(support, labels, target)
print(f"  B3 logits shape: {out_b3['logits'].shape}")

# ── short training runs via subprocess ───────────────────────────────────────
print()
print("=" * 60)
print("STEP (a/b): Short training runs (200 iter) for B2 and B3")
print("=" * 60)

smoke_args_common = [
    "--dataset", "hmdb", "--split", "3",
    "--shot", "1",
    "--scratch", os.path.expanduser("~/Documents/work/trx/trx_data"),
    "--training_iterations", "200",
    "--tasks_per_batch", "1",
    "--num_test_tasks", "10",
    "--test_iters", "200",
    "--print_freq", "50",
    "--save_freq", "9999",
    "--num_workers", "4",
]

for name, config, ckpt in [
    ("B2 (intra only)", "configs/stage2_b2_intra_only.yaml",
     os.path.expanduser("~/trx_data/checkpoints/smoke_b2")),
    ("B3 (inter only)", "configs/stage2_b3_inter_only.yaml",
     os.path.expanduser("~/trx_data/checkpoints/smoke_b3")),
]:
    print(f"\n--- {name} ---")
    cmd = [sys.executable, "run.py", "--config", config, "-c", ckpt] + smoke_args_common
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr

    # extract loss lines
    loss_lines = [l for l in output.splitlines() if "Train Loss" in l]
    if loss_lines:
        print(f"  First loss: {loss_lines[0].split('Train Loss:')[1].strip().split(',')[0]}")
        print(f"  Last  loss: {loss_lines[-1].split('Train Loss:')[1].strip().split(',')[0]}")
        first = float(loss_lines[0].split("Train Loss:")[1].strip().split(",")[0])
        last  = float(loss_lines[-1].split("Train Loss:")[1].strip().split(",")[0])
        if last < first:
            print(f"  ✓ Loss decreased ({first:.4f} → {last:.4f})")
        else:
            print(f"  ✗ Loss did NOT decrease ({first:.4f} → {last:.4f})")
    else:
        print("  ✗ No Train Loss lines found in output")
        print(output[-2000:])

    # check for test accuracy
    acc_lines = [l for l in output.splitlines() if "hmdb:" in l or "accuracy" in l.lower()]
    if acc_lines:
        print(f"  Test: {acc_lines[0]}")
        print(f"  ✓ Test evaluation ran")
    else:
        print("  ✗ No test accuracy found")

    if result.returncode != 0:
        print(f"  ✗ Process exited with code {result.returncode}")
        print(output[-1000:])
    else:
        print(f"  ✓ Process exited cleanly")

print()
print("=" * 60)
print("Smoke test complete.")
print("=" * 60)
