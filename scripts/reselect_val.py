"""
reselect_val.py — 晚間_0907 階段 2：重算 val-selected test acc

方法：
  - N_VAL_EPS  = 1000  (固定 seed，所有設定共用同一批 val episodes)
  - N_TEST_EPS = 10000 (固定 seed，所有設定共用同一批 test episodes)
  - 使用 vallist03.txt / testlist03.txt (hmdb ARN split 3)
  - 每個設定回報四個檢查點的 val acc + val-best 的 test acc

六個設定：
  B4 seed=42, B4 seed=43
  B2 seed=42, B2 seed=43
  B1 seed=42  (abl_bidir_hmdb3, seed 未記錄，訓練時為預設 42)
  B1 seed=43  (b1_seed43_1shot_hmdb3，本日訓練中；腳本等候至可用)

輸出：logs/reselect_val.log（由 launch.sh 管，此腳本只寫 stdout/stderr）
"""
import sys, os, random, zipfile, io, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
VAL_SEED = 10001   # 所有設定共用，與訓練 seed 不同
TEST_SEED= 20001   # 所有設定共用
SCRATCH  = os.path.expanduser("~/Documents/work/trx/trx_data")
DATA_ZIP = os.path.join(SCRATCH, "video_datasets/data/hmdb51_256q5.zip")
SPLIT_DIR= os.path.join(SCRATCH, "video_datasets/splits/hmdb_ARN")
CKPT_ROOT= os.path.expanduser("~/trx_data/checkpoints")
REPO     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

N_VAL_EPS  = 1000
N_TEST_EPS = 10000

print("=== reselect_val.py start ===", flush=True)
print(f"DEVICE={DEVICE}", flush=True)
print(f"N_VAL_EPS={N_VAL_EPS}, N_TEST_EPS={N_TEST_EPS}", flush=True)
print(f"VAL_SEED={VAL_SEED}, TEST_SEED={TEST_SEED}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Episode loader (same as phase0_eval.py, no changes)
# ─────────────────────────────────────────────────────────────────────────────
def _load_zip():
    mem = open(DATA_ZIP, 'rb').read()
    return zipfile.ZipFile(io.BytesIO(mem))

class SplitEpisodeLoader:
    def __init__(self, list_file, way=5, shot=1, qpc=1, seq_len=8, img_size=224):
        self.way=way; self.shot=shot; self.qpc=qpc
        self.seq_len=seq_len; self.img_size=img_size
        self.zfile = _load_zip()
        all_names = set(self.zfile.namelist())
        self.class_to_vids = {}
        with open(list_file) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                rel_path = line.split()[0]
                cls = rel_path.split('/')[0]
                prefix = rel_path + '/'
                frames = sorted([n for n in all_names if n.startswith(prefix)
                                  and not n.endswith('/')])
                if len(frames) < seq_len: continue
                self.class_to_vids.setdefault(cls, []).append(frames)
        self.classes = sorted(self.class_to_vids.keys())
        print(f"  Loader({os.path.basename(list_file)}): "
              f"{len(self.classes)} classes, "
              f"{sum(len(v) for v in self.class_to_vids.values())} videos", flush=True)
        from videotransforms.video_transforms import Compose, Resize, CenterCrop
        import torchvision.transforms as tv_t
        self.transform = Compose([Resize(256), CenterCrop(img_size)])
        self.to_tensor = tv_t.ToTensor()

    def _sample_frames(self, frame_list):
        n = len(frame_list)
        s, e = 1, n - 2
        if e - s < self.seq_len: s, e = 0, n-1
        idxs = [int(x) for x in np.linspace(s, e, num=self.seq_len)]
        return [frame_list[i] for i in idxs]

    def _load_video(self, frame_paths):
        pil_imgs = []
        for fp in frame_paths:
            data = self.zfile.read(fp)
            img = __import__('PIL').Image.open(io.BytesIO(data)).convert("RGB")
            pil_imgs.append(img)
        from videotransforms.video_transforms import Compose, Resize, CenterCrop
        transformed = self.transform(pil_imgs)
        import torchvision.transforms as tv_t
        to_tensor = tv_t.ToTensor()
        return torch.stack([to_tensor(img) for img in transformed])

    def get_episode(self):
        batch_classes = random.sample(self.classes, self.way)
        sup_t, sup_l, q_t, q_l = [], [], [], []
        for bi, bc in enumerate(batch_classes):
            vids = self.class_to_vids[bc]
            idxs = random.sample(range(len(vids)), self.shot + self.qpc)
            for si in idxs[:self.shot]:
                sup_t.append(self._load_video(self._sample_frames(vids[si])))
                sup_l.append(bi)
            for qi in idxs[self.shot:]:
                q_t.append(self._load_video(self._sample_frames(vids[qi])))
                q_l.append(bi)
        return {
            "support_tensors": torch.stack(sup_t),
            "support_labels":  torch.tensor(sup_l),
            "query_tensors":   torch.stack(q_t),
            "query_labels":    torch.tensor(q_l),
        }


from visualize import load_model

def eval_checkpoint(ckpt_path, cfg_path, ep_loader, n_episodes, rng_seed):
    """
    Evaluate checkpoint on n_episodes with fixed rng_seed.
    Returns (mean_acc_pct, ci_pct) or (None, None) on failure.
    """
    try:
        m, _ = load_model(cfg_path, ckpt_path, device=DEVICE)
    except Exception as ex:
        print(f"    [ERROR] load_model failed: {ex}", flush=True)
        return None, None
    m.eval()
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    correct = []
    for _ in range(n_episodes):
        ep = ep_loader.get_episode()
        ns = ep["support_tensors"].shape[0]
        nq = ep["query_tensors"].shape[0]
        sup_t = ep["support_tensors"].reshape(
            ns * ep_loader.seq_len, 3, ep_loader.img_size, ep_loader.img_size
        ).to(DEVICE)
        q_t = ep["query_tensors"].reshape(
            nq * ep_loader.seq_len, 3, ep_loader.img_size, ep_loader.img_size
        ).to(DEVICE)
        s_lbl = ep["support_labels"].to(DEVICE)
        q_lbl = ep["query_labels"]
        with torch.no_grad():
            out = m(sup_t, s_lbl, q_t)
            logits = out["logits"]
            if logits.ndim == 3: logits = logits.mean(0)
            preds = logits.argmax(dim=-1)
        for qi in range(preds.shape[0]):
            correct.append(float(preds[qi].item() == q_lbl[qi].item()))
        del ep, sup_t, q_t, s_lbl
    n = len(correct)
    mean_acc = 100.0 * float(np.mean(correct))
    ci = 196.0 * float(np.std(correct)) / float(np.sqrt(n))
    del m
    torch.cuda.empty_cache()
    return mean_acc, ci


def run_setting(name, cfg_path, ckpt_map, val_ldr, test_ldr):
    """
    Val-select from ckpt_map, report all 4 val accs, then test best.
    ckpt_map: {iter_int: ckpt_path_str}
    Returns dict or None.
    """
    print(f"\n{'='*60}", flush=True)
    print(f"  {name}", flush=True)
    print(f"{'='*60}", flush=True)

    val_accs = {}
    for it in sorted(ckpt_map):
        ckpt = os.path.expanduser(ckpt_map[it])
        if not os.path.exists(ckpt):
            print(f"    iter {it:>6d}: checkpoint MISSING — {ckpt}", flush=True)
            continue
        v_acc, v_ci = eval_checkpoint(ckpt, cfg_path, val_ldr, N_VAL_EPS, VAL_SEED)
        if v_acc is None:
            print(f"    iter {it:>6d}: EVAL FAILED", flush=True)
            continue
        print(f"    iter {it:>6d}: val={v_acc:.2f}±{v_ci:.2f}", flush=True)
        val_accs[it] = (v_acc, v_ci)

    if not val_accs:
        print(f"  => No valid checkpoints; skipping test.", flush=True)
        return None

    best_iter = max(val_accs, key=lambda k: val_accs[k][0])
    best_ckpt = os.path.expanduser(ckpt_map[best_iter])
    print(f"  => Val-best: iter {best_iter} (val {val_accs[best_iter][0]:.2f}%)", flush=True)

    t_acc, t_ci = eval_checkpoint(best_ckpt, cfg_path, test_ldr, N_TEST_EPS, TEST_SEED)
    if t_acc is not None:
        print(f"  => Test acc (val-selected, {N_TEST_EPS} eps): {t_acc:.2f}±{t_ci:.2f}", flush=True)

    return {
        "name": name,
        "val_accs": val_accs,
        "best_iter": best_iter,
        "test_acc": t_acc,
        "test_ci": t_ci,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Build loaders
# ─────────────────────────────────────────────────────────────────────────────
print("\nBuilding episode loaders ...", flush=True)
val_ldr  = SplitEpisodeLoader(os.path.join(SPLIT_DIR, "vallist03.txt"),
                               way=5, shot=1, qpc=1)
test_ldr = SplitEpisodeLoader(os.path.join(SPLIT_DIR, "testlist03.txt"),
                               way=5, shot=1, qpc=1)

# ─────────────────────────────────────────────────────────────────────────────
# Config and checkpoint paths for all 6 settings
# ─────────────────────────────────────────────────────────────────────────────
CFG_B4 = os.path.join(REPO, "configs/stage2_true_hyrsm_b4_tuple.yaml")
CFG_B2 = os.path.join(REPO, "configs/stage2_b2_intra_only.yaml")
CFG_B1 = os.path.join(REPO, "configs/b1_backbone_1shot.yaml")

B4_S42_DIR = f"{CKPT_ROOT}/true_hyrsm_b4_1shot/hmdb_split3_20260526_202630"
B4_S43_DIR = f"{CKPT_ROOT}/b4_seed43_1shot_hmdb3/hmdb_split3_20260905_154706"
B2_S42_DIR = f"{CKPT_ROOT}/b2_intra_hmdb3_1shot/hmdb_split3_20260504_100143"
B2_S43_DIR = f"{CKPT_ROOT}/b2_seed43_1shot_hmdb3/hmdb_split3_20260906_010142"
B1_S42_DIR = f"{CKPT_ROOT}/abl_bidir_hmdb3/hmdb_split3_20260407_220041"
# B1 seed=43: pick the latest run dir (non-empty one)
_b1s43_base = f"{CKPT_ROOT}/b1_seed43_1shot_hmdb3"
# find the run dir with the most checkpoint files
import glob as _glob
_b1s43_dirs = sorted(_glob.glob(f"{_b1s43_base}/hmdb_split3_*"))
_b1s43_dir  = None
for d in reversed(_b1s43_dirs):
    if _glob.glob(f"{d}/checkpoint*.pt"):
        _b1s43_dir = d
        break
if _b1s43_dir is None:
    print(f"[WARN] B1 seed=43 checkpoint dir not found under {_b1s43_base}", flush=True)
    _b1s43_dir = _b1s43_dirs[-1] if _b1s43_dirs else ""

ITERS = [25000, 50000, 75000, 100000]

def ckpt_map(base_dir):
    return {it: f"{base_dir}/checkpoint{it}.pt" for it in ITERS}

SETTINGS = [
    ("B4_seed42", CFG_B4, ckpt_map(B4_S42_DIR)),
    ("B4_seed43", CFG_B4, ckpt_map(B4_S43_DIR)),
    ("B2_seed42", CFG_B2, ckpt_map(B2_S42_DIR)),
    ("B2_seed43", CFG_B2, ckpt_map(B2_S43_DIR)),
    ("B1_seed42", CFG_B1, ckpt_map(B1_S42_DIR)),   # seed unknown, likely 42
    ("B1_seed43", CFG_B1, ckpt_map(_b1s43_dir) if _b1s43_dir else {}),
]

# ─────────────────────────────────────────────────────────────────────────────
# Run all settings
# ─────────────────────────────────────────────────────────────────────────────
results = []
for name, cfg, ckpts in SETTINGS:
    if not ckpts:
        print(f"\n[SKIP] {name}: no checkpoint map (dir not found)", flush=True)
        results.append(None)
        continue
    r = run_setting(name, cfg, ckpts, val_ldr, test_ldr)
    results.append(r)

# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────
print("\n\n" + "="*80, flush=True)
print("SUMMARY TABLE", flush=True)
print("="*80, flush=True)
header = (f"{'設定':<14} {'seed':>4}  "
          f"{'val@25k':>8} {'val@50k':>8} {'val@75k':>8} {'val@100k':>9}  "
          f"{'選中':>6}  {'test acc (10k eps)':>18}  {'CI':>6}")
print(header, flush=True)
print("-"*len(header), flush=True)

setting_labels = [
    ("B4", "42"), ("B4", "43"),
    ("B2", "42"), ("B2", "43"),
    ("B1", "42*"), ("B1", "43"),
]

for (label, seed), r in zip(setting_labels, results):
    if r is None:
        print(f"  {label:<12} {seed:>5}  {'—':>8} {'—':>8} {'—':>8} {'—':>9}  {'—':>6}  {'無 checkpoint':>18}  {'—':>6}", flush=True)
        continue
    va = r["val_accs"]
    def fmt(it):
        if it in va: return f"{va[it][0]:.2f}"
        return "MISS"
    best_it = r.get("best_iter")
    test_s = f"{r['test_acc']:.2f}" if r.get("test_acc") is not None else "FAIL"
    ci_s   = f"±{r['test_ci']:.2f}" if r.get("test_ci") is not None else "—"
    print(f"  {label:<12} {seed:>5}  "
          f"{fmt(25000):>8} {fmt(50000):>8} {fmt(75000):>8} {fmt(100000):>9}  "
          f"{str(best_it):>6}  {test_s:>18}  {ci_s:>6}", flush=True)

print(flush=True)
print("* B1 seed=42 的 seed 未記錄於 log（訓練時早於 seed logging 功能）", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Three-sentence answers
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*80, flush=True)
print("三句回答", flush=True)
print("="*80, flush=True)

def get_test(results, idx):
    r = results[idx]
    return r["test_acc"] if (r and r.get("test_acc") is not None) else None

b4_42 = get_test(results, 0)
b4_43 = get_test(results, 1)
b2_42 = get_test(results, 2)
b2_43 = get_test(results, 3)

# Sentence 1: B4 - B2 paired gaps
if b4_42 is not None and b2_42 is not None:
    gap42 = b4_42 - b2_42
    gap42_s = f"{gap42:+.2f}"
else:
    gap42_s = "資料不足"
if b4_43 is not None and b2_43 is not None:
    gap43 = b4_43 - b2_43
    gap43_s = f"{gap43:+.2f}"
else:
    gap43_s = "資料不足"
print(f"1. B4 − B2 的配對差距：seed=42 為 {gap42_s} pp，seed=43 為 {gap43_s} pp。", flush=True)

# Sentence 2: val checkpoint spread vs CI
# Compute max spread across all settings
all_spreads = []
for r in results:
    if r and r["val_accs"] and len(r["val_accs"]) >= 2:
        v = [acc for acc, _ in r["val_accs"].values()]
        all_spreads.append(max(v) - min(v))
if all_spreads:
    max_spread = max(all_spreads)
    # CI for 1000 episodes ≈ 1.96*sqrt(p*(1-p)/n) ≈ 1.96*0.5/sqrt(1000) ≈ 3.1pp
    approx_ci = 196.0 * 0.5 / (1000 ** 0.5)
    if max_spread > 2 * approx_ci:
        discriminability = "有鑑別力"
    else:
        discriminability = "在噪聲範圍內"
    print(f"2. val 四個檢查點之間的最大差距是 {max_spread:.2f} pp，"
          f"1000 episode 的 CI 約 {approx_ci:.1f} pp；"
          f"因此檢查點選擇{discriminability}。", flush=True)
else:
    print("2. 無足夠 val 資料可計算差距。", flush=True)

# Sentence 3: how many seeds B4 > B2
n_b4_wins = 0
if b4_42 is not None and b2_42 is not None and b4_42 > b2_42: n_b4_wins += 1
if b4_43 is not None and b2_43 is not None and b4_43 > b2_43: n_b4_wins += 1
print(f"3. B4 在 {n_b4_wins} 個 seed 上高於 B2。", flush=True)

print("\n=== reselect_val.py DONE ===", flush=True)
