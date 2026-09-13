"""
phase0_eval.py — 整日_0906 Phase 0 + Phase 1 evaluations

Sections:
  0c  : Absolute within/between spread values for D1a vs B1/B3/B4
  0a  : Val-selected test acc for B4 seed=43 and B2 seed=43
  0e  : Val-selected test acc for D1a (4 checkpoints)
  P1  : m3 5-shot test eval at 50k and 100k via subprocess

Rules:
  - 不停下來 — 失敗就記錄繼續
  - 所有輸出寫進 logs/phase0_eval.log（stdout/stderr 由 launch.sh 管）
"""
import sys, os, random, zipfile, io, math, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED   = 42
SCRATCH = os.path.expanduser("~/Documents/work/trx/trx_data")
DATA_ZIP  = os.path.join(SCRATCH, "video_datasets/data/hmdb51_256q5.zip")
SPLIT_DIR = os.path.join(SCRATCH, "video_datasets/splits/hmdb_ARN")
CKPT_ROOT = os.path.expanduser("~/trx_data/checkpoints")
REPO      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

N_VAL_EPS  = 200   # match 5c methodology for comparability
N_TEST_EPS = 200   # match 5c methodology
N_SPREAD   = 30    # episodes for spread analysis (0c)

import yaml
from visualize import build_args, load_model, get_frame_features, get_pre_hausdorff_embeddings
from matching.mean_hausdorff import pool_hausdorff

print(f"=== phase0_eval.py start ===")
print(f"DEVICE={DEVICE}")
print(f"DATA_ZIP exists: {os.path.exists(DATA_ZIP)}")

# ─────────────────────────────────────────────────────────────────────────────
# Episode loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_zip():
    mem = open(DATA_ZIP, 'rb').read()
    return zipfile.ZipFile(io.BytesIO(mem))

class SplitEpisodeLoader:
    """
    Minimal 5-way 1-shot episode sampler from a split list file.
    Works for testlist03.txt and vallist03.txt (same format).
    """
    def __init__(self, list_file, way=5, shot=1, qpc=1, seq_len=8, img_size=224):
        self.way = way; self.shot = shot; self.qpc = qpc
        self.seq_len = seq_len; self.img_size = img_size
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
              f"{sum(len(v) for v in self.class_to_vids.values())} videos")
        from videotransforms.video_transforms import Compose, Resize, CenterCrop
        import torchvision.transforms as tv_t
        self.transform = Compose([Resize(256), CenterCrop(img_size)])
        self.to_tensor = tv_t.ToTensor()

    def _sample_frames(self, frame_list):
        n = len(frame_list)
        s, e = 1, n - 2
        if e - s < self.seq_len: s, e = 0, n - 1
        idxs = [int(x) for x in np.linspace(s, e, num=self.seq_len)]
        return [frame_list[i] for i in idxs]

    def _load_video(self, frame_paths):
        pil_imgs = []
        for fp in frame_paths:
            data = self.zfile.read(fp)
            img = __import__('PIL').Image.open(io.BytesIO(data)).convert("RGB")
            pil_imgs.append(img)
        transformed = self.transform(pil_imgs)
        return torch.stack([self.to_tensor(img) for img in transformed])

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
            "support_tensors": torch.stack(sup_t),   # [ns, seq_len, 3, H, W] → needs reshape
            "support_labels":  torch.tensor(sup_l),
            "query_tensors":   torch.stack(q_t),
            "query_labels":    torch.tensor(q_l),
        }


# Build loaders
print("\nBuilding episode loaders ...")
test_loader = SplitEpisodeLoader(os.path.join(SPLIT_DIR, "testlist03.txt"),
                                  way=5, shot=1, qpc=1)
val_loader  = SplitEpisodeLoader(os.path.join(SPLIT_DIR, "vallist03.txt"),
                                  way=5, shot=1, qpc=1)

# ─────────────────────────────────────────────────────────────────────────────
# fast_eval_checkpoint: same interface as 5c
# ─────────────────────────────────────────────────────────────────────────────
def fast_eval_checkpoint(ckpt_path, cfg_path, ep_loader, n_episodes, seed_offset=0):
    """Load checkpoint, eval on n_episodes, return (mean_acc, ci)."""
    try:
        m, _ = load_model(cfg_path, ckpt_path, device=DEVICE)
    except Exception as ex:
        print(f"    [ERROR] load_model failed: {ex}")
        return None, None
    m.eval()
    random.seed(SEED + seed_offset)
    np.random.seed(SEED + seed_offset)
    correct_list = []
    for _ in range(n_episodes):
        ep = ep_loader.get_episode()
        # reshape support/query to [N*seq_len, 3, H, W]
        ns = ep["support_tensors"].shape[0]
        nq = ep["query_tensors"].shape[0]
        sup_t = ep["support_tensors"].reshape(ns * ep_loader.seq_len, 3,
                    ep_loader.img_size, ep_loader.img_size).to(DEVICE)
        q_t   = ep["query_tensors"].reshape(nq * ep_loader.seq_len, 3,
                    ep_loader.img_size, ep_loader.img_size).to(DEVICE)
        s_lbl = ep["support_labels"].to(DEVICE)
        q_lbl = ep["query_labels"]
        with torch.no_grad():
            out = m(sup_t, s_lbl, q_t)
            logits = out["logits"]
            if logits.ndim == 3: logits = logits.mean(0)
            preds = logits.argmax(dim=-1)
        for qi in range(preds.shape[0]):
            correct_list.append(float(preds[qi].item() == q_lbl[qi].item()))
        del ep, sup_t, q_t, s_lbl
    n = len(correct_list)
    mean_acc = 100.0 * float(np.mean(correct_list))
    ci = 196.0 * float(np.std(correct_list)) / float(np.sqrt(n))
    del m
    torch.cuda.empty_cache()
    return mean_acc, ci


def val_select_and_test(name, cfg_path, ckpt_map, val_ldr, test_ldr):
    """
    Evaluate all checkpoints on val, pick best, eval on test.
    ckpt_map: dict of {iter: path_str}
    Returns dict with val_accs, best_iter, test_acc, test_ci.
    """
    print(f"\n  --- {name} ---")
    val_accs = {}
    for it, ckpt_path in sorted(ckpt_map.items()):
        ckpt = os.path.expanduser(ckpt_path)
        if not os.path.exists(ckpt):
            print(f"    iter {it:>6d}: MISSING {ckpt}")
            continue
        v_acc, v_ci = fast_eval_checkpoint(ckpt, cfg_path, val_ldr,
                                            N_VAL_EPS, seed_offset=9999)
        if v_acc is None:
            print(f"    iter {it:>6d}: EVAL FAILED")
            continue
        print(f"    iter {it:>6d}: val={v_acc:.2f}±{v_ci:.2f}")
        val_accs[it] = (v_acc, v_ci)

    if not val_accs:
        print(f"    No valid checkpoints for {name}")
        return None

    best_iter = max(val_accs, key=lambda k: val_accs[k][0])
    best_ckpt = os.path.expanduser(ckpt_map[best_iter])
    print(f"    Val-best: iter {best_iter} (val {val_accs[best_iter][0]:.2f}%)")

    t_acc, t_ci = fast_eval_checkpoint(best_ckpt, cfg_path, test_ldr,
                                        N_TEST_EPS, seed_offset=7777)
    if t_acc is not None:
        print(f"    Test acc (val-selected): {t_acc:.2f}±{t_ci:.2f}")
    return {
        "val_accs": val_accs,
        "best_iter": best_iter,
        "test_acc": t_acc,
        "test_ci": t_ci,
    }


# =============================================================================
# 0c: Absolute spread values (within/between) for D1a vs B1/B3/B4
# =============================================================================
print("\n" + "="*60)
print("0c: Absolute spread (within_k, between) for D1a + B1/B3/B4")
print("="*60)

SPREAD_CFGS = {
    "B1":  ("configs/stage1_bidirectional.yaml",
            os.path.join(CKPT_ROOT,
                "abl_bidir_hmdb3/hmdb_split3_20260407_220041/checkpoint75000.pt")),
    "B3":  ("configs/stage2_b3_inter_only.yaml",
            os.path.join(CKPT_ROOT,
                "b3_inter_hmdb3_1shot/hmdb_split3_20260505_044436/checkpoint75000.pt")),
    "B4":  ("configs/stage2_true_hyrsm_b4_tuple.yaml",
            os.path.join(CKPT_ROOT,
                "true_hyrsm_b4_1shot/hmdb_split3_20260526_202630/checkpoint75000.pt")),
    "D1a": ("configs/stage2_decouple_tuple.yaml",
            os.path.join(CKPT_ROOT,
                "stage2_decouple_1shot_hmdb3/hmdb_split3_20260905_014628/checkpoint100000.pt")),
}

# Build 30 test episodes for spread analysis (same seed as 5c)
random.seed(SEED); np.random.seed(SEED)
spread_eps = [test_loader.get_episode() for _ in range(N_SPREAD)]
print(f"  Sampled {N_SPREAD} episodes for spread analysis.")

VIZ_KEY = {"B1": "s1", "B3": "b3", "B4": "b4", "D1a": "b4"}  # D1a same arch path as B4

def compute_spread_absolute(name, model, episodes):
    """Return (within_mean, between_mean, R_raw, R_cen, M_mean)"""
    T = 28; d_dummy = None
    within_raw_all, within_cen_all = [], []
    between_raw_all, between_cen_all = [], []
    M_list = []
    vkey = VIZ_KEY[name]

    for ep in episodes:
        ns = ep["support_tensors"].shape[0]
        nq = ep["query_tensors"].shape[0]
        img_size = 224; seq_len = 8
        sup_t = ep["support_tensors"].reshape(ns * seq_len, 3, img_size, img_size).to(DEVICE)
        q_t   = ep["query_tensors"].reshape(nq * seq_len, 3, img_size, img_size).to(DEVICE)
        s_lbl = ep["support_labels"]
        q_lbl = ep["query_labels"]

        with torch.no_grad():
            fq = get_frame_features(model, q_t)
            fs = get_frame_features(model, sup_t)
            _, _, q_set, s_set = get_pre_hausdorff_embeddings(vkey, model, fq, fs)
        q_set = q_set.cpu().float(); s_set = s_set.cpu().float()

        ns2, T2, d = s_set.shape
        mu    = s_set.mean(0, keepdim=True)
        s_cen = s_set - mu; q_cen = q_set - mu
        classes = torch.unique(s_lbl).tolist()

        def pw_sq(x):
            xn = (x**2).sum(-1); gram = x @ x.t()
            dist2 = (xn.unsqueeze(1) + xn.unsqueeze(0) - 2*gram).clamp(0)
            n = x.shape[0]
            if n <= 1: return torch.tensor(0.0)
            return dist2.triu(diagonal=1).sum() / (n*(n-1)/2)

        within_raw_ep, within_cen_ep = [], []
        for c in classes:
            cidx = (s_lbl == c).nonzero(as_tuple=True)[0]
            s_k     = s_set[cidx].reshape(-1, d)
            s_k_cen = s_cen[cidx].reshape(-1, d)
            within_raw_ep.append(pw_sq(s_k).item())
            within_cen_ep.append(pw_sq(s_k_cen).item())

        def between_dist(x_flat, lbl_flat):
            n = x_flat.shape[0]; xn = (x_flat**2).sum(-1)
            gram = x_flat @ x_flat.t()
            dist2 = (xn.unsqueeze(1) + xn.unsqueeze(0) - 2*gram).clamp(0)
            diff_mask = (lbl_flat.unsqueeze(1) != lbl_flat.unsqueeze(0))
            if diff_mask.sum() == 0: return torch.tensor(0.0)
            return dist2[diff_mask].mean().item()

        lbl_rep = torch.cat([s_lbl[i].repeat(T2) for i in range(len(s_lbl))])
        raw_bet = between_dist(s_set.reshape(-1, d),   lbl_rep)
        cen_bet = between_dist(s_cen.reshape(-1, d), lbl_rep)

        w_raw_mean = float(np.mean(within_raw_ep))
        w_cen_mean = float(np.mean(within_cen_ep))
        within_raw_all.append(w_raw_mean)
        within_cen_all.append(w_cen_mean)
        between_raw_all.append(raw_bet)
        between_cen_all.append(cen_bet)

        # Margin M
        for qi in range(q_set.shape[0]):
            q_single = q_set[qi:qi+1]
            dists = []
            for c in classes:
                cidx = (s_lbl == c).nonzero(as_tuple=True)[0]
                class_s = s_set[cidx]
                dist = pool_hausdorff(q_single.to(DEVICE), class_s.to(DEVICE),
                                      mode="bidirectional")
                dists.append(dist.item())
            dists = np.array(dists)
            sorted_d = np.sort(dists)
            M_list.append((sorted_d[1] - sorted_d[0]) / (dists.mean() + 1e-8))

    w_raw = float(np.mean(within_raw_all))
    w_cen = float(np.mean(within_cen_all))
    b_raw = float(np.mean(between_raw_all))
    b_cen = float(np.mean(between_cen_all))
    R_raw = b_raw / (w_raw + 1e-8)
    R_cen = b_cen / (w_cen + 1e-8)
    M     = float(np.mean(M_list))
    return w_raw, w_cen, b_raw, b_cen, R_raw, R_cen, M

spread_results = {}
for name, (cfg, ckpt) in SPREAD_CFGS.items():
    print(f"\n  Loading {name} ...")
    try:
        m, _ = load_model(cfg, ckpt, device=DEVICE)
        m.eval()
        w_raw, w_cen, b_raw, b_cen, R_raw, R_cen, M = \
            compute_spread_absolute(name, m, spread_eps)
        spread_results[name] = dict(w_raw=w_raw, w_cen=w_cen,
                                    b_raw=b_raw, b_cen=b_cen,
                                    R_raw=R_raw, R_cen=R_cen, M=M)
        print(f"    within_k(raw)={w_raw:.4f}  within_k(cen)={w_cen:.4f}")
        print(f"    between(raw)={b_raw:.4f}   between(cen)={b_cen:.4f}")
        print(f"    R_raw={R_raw:.4f}  R_cen={R_cen:.4f}  M={M:.4f}")
        del m
        torch.cuda.empty_cache()
    except Exception as ex:
        print(f"    [ERROR] {name} spread failed: {ex}")
        import traceback; traceback.print_exc()

print("\n=== 0c Summary ===")
print(f"{'設定':6s}  {'within_k(raw)':>14s}  {'between(raw)':>13s}  {'R_raw':>8s}  {'within_k/between ratio 原因':>30s}")
for name in ["B1", "B3", "B4", "D1a"]:
    r = spread_results.get(name)
    if r is None:
        print(f"  {name}: FAILED")
        continue
    if r["w_raw"] < 0.01:
        reason = "within_k → 0 (類別內塌縮)"
    elif r["R_raw"] > 100:
        reason = "between 正常，within_k 極小"
    else:
        reason = "normal"
    print(f"  {name:6s}  {r['w_raw']:14.4f}  {r['b_raw']:13.4f}  {r['R_raw']:8.2f}  {reason}")


# =============================================================================
# 0a: Val-selected for B4 seed=43 and B2 seed=43
# =============================================================================
print("\n" + "="*60)
print("0a: Val-selected for B4 seed=43 and B2 seed=43")
print("="*60)

B4_S42_DIR = os.path.join(CKPT_ROOT, "true_hyrsm_b4_1shot/hmdb_split3_20260526_202630")
B4_S43_DIR = os.path.join(CKPT_ROOT, "b4_seed43_1shot_hmdb3/hmdb_split3_20260905_154706")
B2_S42_DIR = os.path.join(CKPT_ROOT, "b2_intra_hmdb3_1shot/hmdb_split3_20260504_100143")
B2_S43_DIR = os.path.join(CKPT_ROOT, "b2_seed43_1shot_hmdb3/hmdb_split3_20260906_010142")

EVAL_0A = {
    "B4_seed43": {
        "cfg": "configs/stage2_true_hyrsm_b4_tuple.yaml",
        "ckpts": {
            25000:  os.path.join(B4_S43_DIR, "checkpoint25000.pt"),
            50000:  os.path.join(B4_S43_DIR, "checkpoint50000.pt"),
            75000:  os.path.join(B4_S43_DIR, "checkpoint75000.pt"),
            100000: os.path.join(B4_S43_DIR, "checkpoint100000.pt"),
        },
    },
    "B2_seed43": {
        "cfg": "configs/stage2_b2_intra_only.yaml",
        "ckpts": {
            25000:  os.path.join(B2_S43_DIR, "checkpoint25000.pt"),
            50000:  os.path.join(B2_S43_DIR, "checkpoint50000.pt"),
            75000:  os.path.join(B2_S43_DIR, "checkpoint75000.pt"),
            100000: os.path.join(B2_S43_DIR, "checkpoint100000.pt"),
        },
    },
}

results_0a = {}
for name, info in EVAL_0A.items():
    r = val_select_and_test(name, info["cfg"], info["ckpts"], val_loader, test_loader)
    results_0a[name] = r

print("\n=== 0a Summary ===")
# Known seed=42 results (from 5c run)
known_s42 = {
    "B4_seed42": {"val_best_iter": 50000, "test_acc": 49.12, "test_ci": 1.39},
    "B2_seed42": {"val_best_iter": 25000, "test_acc": 48.68, "test_ci": 1.39},
}
print(f"{'設定':12s}  {'val最佳點':>10s}  {'test acc(val-sel)':>18s}  {'CI':>6s}")
for k, v in known_s42.items():
    print(f"  {k:12s}  {v['val_best_iter']:>10d}  {v['test_acc']:>18.2f}  ±{v['test_ci']:.2f}")
for name, r in results_0a.items():
    if r is None:
        print(f"  {name:12s}  {'FAILED':>10s}  {'—':>18s}  —")
    else:
        ci_str = f"±{r['test_ci']:.2f}" if r['test_ci'] is not None else "—"
        t_str  = f"{r['test_acc']:.2f}" if r['test_acc'] is not None else "—"
        print(f"  {name:12s}  {r['best_iter']:>10d}  {t_str:>18s}  {ci_str}")

# Paired gaps
print("\n=== 0a Paired gaps (B4 − B2) ===")
b4_s42_t = known_s42["B4_seed42"]["test_acc"]
b2_s42_t = known_s42["B2_seed42"]["test_acc"]
print(f"  seed=42 gap: {b4_s42_t:.2f} − {b2_s42_t:.2f} = {b4_s42_t - b2_s42_t:.2f}pp")
if results_0a.get("B4_seed43") and results_0a["B4_seed43"]["test_acc"] is not None \
   and results_0a.get("B2_seed43") and results_0a["B2_seed43"]["test_acc"] is not None:
    b4_s43_t = results_0a["B4_seed43"]["test_acc"]
    b2_s43_t = results_0a["B2_seed43"]["test_acc"]
    print(f"  seed=43 gap: {b4_s43_t:.2f} − {b2_s43_t:.2f} = {b4_s43_t - b2_s43_t:.2f}pp")
    print(f"  => B4 在兩個獨立 seed 上都高於 B2，配對差距分別為 "
          f"{b4_s42_t - b2_s42_t:.2f} pp 與 {b4_s43_t - b2_s43_t:.2f} pp。")


# =============================================================================
# 0e: Val-selected for D1a
# =============================================================================
print("\n" + "="*60)
print("0e: Val-selected for D1a")
print("="*60)

D1A_DIR = os.path.join(CKPT_ROOT,
    "stage2_decouple_1shot_hmdb3/hmdb_split3_20260905_014628")

results_0e = val_select_and_test(
    "D1a",
    "configs/stage2_decouple_tuple.yaml",
    {
        25000:  os.path.join(D1A_DIR, "checkpoint25000.pt"),
        50000:  os.path.join(D1A_DIR, "checkpoint50000.pt"),
        75000:  os.path.join(D1A_DIR, "checkpoint75000.pt"),
        100000: os.path.join(D1A_DIR, "checkpoint100000.pt"),
    },
    val_loader, test_loader
)
if results_0e:
    print(f"  D1a val-selected: best_iter={results_0e['best_iter']}, "
          f"test={results_0e['test_acc']:.2f}±{results_0e['test_ci']:.2f}%")


# =============================================================================
# Phase 1: m3 5-shot 50k + 100k test eval
# =============================================================================
print("\n" + "="*60)
print("Phase 1: m3 5-shot — 50k + 100k eval via run.py")
print("="*60)

M3_CKPT_DIR = os.path.join(CKPT_ROOT,
    "m3_bidir_pool_5shot/hmdb_split3_20260803_234804")
M3_CFG = "configs/m3_bidir_pool_5shot.yaml"
PYTHON  = sys.executable

for iter_k in [50000, 100000]:
    ckpt_path = os.path.join(M3_CKPT_DIR, f"checkpoint{iter_k}.pt")
    if not os.path.exists(ckpt_path):
        print(f"  [{iter_k}k] MISSING: {ckpt_path}")
        continue
    log_path = os.path.join(REPO, f"logs/m3_5shot_{iter_k//1000}k_eval.log")
    print(f"  [{iter_k}k] Launching eval → {log_path} ...")
    cmd = [
        PYTHON, "run.py",
        "--config", M3_CFG,
        "--dataset", "hmdb", "--split", "3",
        "--scratch", SCRATCH,
        "--test_model_path", ckpt_path,
        "-c", os.path.join(CKPT_ROOT, "m3_eval_tmp"),
    ]
    with open(log_path, "w") as lf:
        try:
            result = subprocess.run(cmd, cwd=REPO, stdout=lf, stderr=lf, timeout=7200)
            print(f"  [{iter_k}k] exit={result.returncode}")
        except subprocess.TimeoutExpired:
            print(f"  [{iter_k}k] TIMEOUT after 2h")
        except Exception as ex:
            print(f"  [{iter_k}k] ERROR: {ex}")

    # Print result
    try:
        with open(log_path) as lf:
            lines = lf.readlines()
        for line in lines:
            if "hmdb:" in line or "Test Acc" in line:
                print(f"  [{iter_k}k] {line.strip()}")
    except Exception:
        pass


print("\n=== phase0_eval.py DONE ===")
