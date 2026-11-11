"""
verify_fixed_test_1111.py — 乙5（以及乙6 加上 val 之後）的驗收：評估用固定 episode。

  F1  同一個 B1 checkpoint（abl_bidir_hmdb3 25k）評估兩次，accuracy 完全相同
  F2  B1 25k 與 B4 s42 25k 各評估一次，兩份 npz 的 episode 識別資訊完全相同（同一批題目）
  F3  評估前後，訓練用的 random／np.random／torch／torch.cuda 亂數狀態都沒有改變
  F4  （有 val 時）val 也一樣：同一顆評兩次相同、B1/B4 題目相同、亂數狀態不變

用法（要 GPU 與資料）：
  python scripts/verify_fixed_test_1111.py [--num_test_tasks 200] [--num_val_tasks 200]
輸出：stdout（由呼叫端導到 logs/1111_verify_fixed_test.log）
"""
import os, sys, random, argparse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, REPO)

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--num_test_tasks", type=int, default=200)
ap.add_argument("--num_val_tasks", type=int, default=200)
cli = ap.parse_args()

C = os.path.expanduser("~/trx_data/checkpoints")
OUT = os.path.join(C, "1111_verify_fixed_test")
SCRATCH = os.path.join(REPO, "trx_data")
B1 = ("configs/stage1_bidirectional.yaml", f"{C}/abl_bidir_hmdb3/hmdb_split3_20260407_220041/checkpoint25000.pt")
B4 = ("configs/stage2_true_hyrsm_b4_tuple.yaml", f"{C}/true_hyrsm_b4_1shot/hmdb_split3_20260526_202630/checkpoint25000.pt")

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)
    if not ok:
        fails.append(name)


def make_learner(cfg, tag):
    import run
    sys.argv = ["run.py", "--config", cfg, "--dataset", "hmdb", "--split", "3",
                "--scratch", SCRATCH, "-c", os.path.join(OUT, tag),
                "--num_test_tasks", str(cli.num_test_tasks), "--num_workers", "0"]
    if hasattr(run.Learner, "_has_val"):
        sys.argv += ["--num_val_tasks", str(cli.num_val_tasks)]
    return run, run.Learner()


def load(learner, ckpt):
    sd = torch.load(ckpt, map_location=learner.device, weights_only=False)["model_state_dict"]
    learner.model.load_state_dict(sd, strict=True)


def rng_snapshot():
    st = {"random": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state().clone()}
    if torch.cuda.is_available():
        st["cuda"] = [s.clone() for s in torch.cuda.get_rng_state_all()]
    return st


def rng_equal(a, b):
    out = {}
    out["random"] = a["random"] == b["random"]
    out["np"] = all(np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y for x, y in zip(a["np"], b["np"]))
    out["torch"] = torch.equal(a["torch"], b["torch"])
    if "cuda" in a:
        out["cuda"] = all(torch.equal(x, y) for x, y in zip(a["cuda"], b["cuda"]))
    return out


def npz(learner, split, it):
    return np.load(os.path.join(learner.checkpoint_dir, f"eval_{split}_{it}.npz"))


def same_episodes(a, b):
    keys = ["batch_class_list", "real_target_labels", "support_names", "query_names"]
    return {k: (a[k].shape == b[k].shape and bool(np.all(a[k] == b[k]))) for k in keys}


def splits_of(learner):
    sp = ["test"]
    if getattr(learner, "_has_val", lambda: False)():
        sp = ["val", "test"]
    return sp


print(f"num_test_tasks={cli.num_test_tasks}, num_val_tasks={cli.num_val_tasks}", flush=True)

# ── B1：評兩次 + 亂數狀態 ─────────────────────────────────────────────────
run, L1 = make_learner(B1[0], "B1")
load(L1, B1[1])
SPLITS = splits_of(L1)
print(f"評估的 split：{SPLITS}", flush=True)
# 模擬訓練中：先把亂數推到任意狀態，再看評估前後是否改變
random.random(); np.random.rand(); torch.rand(1)
res1 = {}
for sp in SPLITS:
    before = rng_snapshot()
    a1 = L1.evaluate(None, sp, 25000)[L1.args.dataset]["accuracy"]
    after = rng_snapshot()
    eq = rng_equal(before, after)
    check(f"F3 {sp}: 評估前後亂數狀態不變", all(eq.values()), str(eq))
    n1 = dict(npz(L1, sp, 25000))
    a2 = L1.evaluate(None, sp, 25000)[L1.args.dataset]["accuracy"]
    n2 = dict(npz(L1, sp, 25000))
    check(f"F1 {sp}: B1 25k 評兩次 accuracy 完全相同", a1 == a2, f"{a1!r} vs {a2!r}")
    check(f"F1 {sp}: B1 兩次逐題 accuracy 完全相同", np.array_equal(n1["accuracy"], n2["accuracy"]))
    print(f"    {sp}: episodes={len(n1['accuracy'])}  support_names 形狀={n1['support_names'].shape}  "
          f"例：{list(n1['support_names'][0])[:2]}", flush=True)
    res1[sp] = (a1, n1)
b1_dir = L1.checkpoint_dir
del L1
torch.cuda.empty_cache()

# ── B4：同一批題目 ────────────────────────────────────────────────────────
run, L4 = make_learner(B4[0], "B4")
load(L4, B4[1])
for sp in SPLITS:
    a4 = L4.evaluate(None, sp, 25000)[L4.args.dataset]["accuracy"]
    n4 = dict(npz(L4, sp, 25000))
    se = same_episodes(res1[sp][1], n4)
    check(f"F2 {sp}: B1 與 B4 的 episode 識別資訊完全相同", all(se.values()), str(se))
    print(f"    {sp}: B1 25k acc={res1[sp][0]:.3f}  B4 s42 25k acc={a4:.3f}", flush=True)

print(f"\nnpz 位置：{b1_dir}、{L4.checkpoint_dir}", flush=True)
print(f"\n結果：{'全部通過' if not fails else '失敗 ' + str(fails)}", flush=True)
sys.exit(1 if fails else 0)
