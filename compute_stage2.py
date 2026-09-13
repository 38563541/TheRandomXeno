"""
Stage 2 (2026-09-05): Evaluation batch
========================================
2a: Spread R (raw + centred) and decision margin M — B1, B3, B4
2b: Bidirectional vs Unidirectional on B1 75k (same-checkpoint, off-distribution eval)
2c: Validation-set checkpoint selection (vallist03.txt) for B1, B2, B3, B4

All evaluation uses existing checkpoints only — no retraining.
"""
import sys, os, random, zipfile, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visualize import (
    CHECKPOINTS, MODEL_LABELS, DEVICE, SEED, OUT_DIR,
    SEQ_LEN, WAY, SHOT, N_QUERY,
    load_model, get_frame_features, get_pre_hausdorff_embeddings,
    EpisodeLoader, build_args,
)
from matching.mean_hausdorff import pool_hausdorff

FIGS_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
SCRATCH    = os.path.expanduser("~/Documents/work/trx/trx_data")
DATA_ZIP   = os.path.join(SCRATCH, "video_datasets/data/hmdb51_256q5.zip")
SPLIT_DIR  = os.path.join(SCRATCH, "video_datasets/splits/hmdb_ARN")
os.makedirs(FIGS_DIR, exist_ok=True)

# ── Reproducibility ─────────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Memory note: each episode ≈ 145 MB (30 vid × 8 frames × 224×224×3 float32).
# With 15 GB RAM and 2.8 GB zip-in-memory, safe limit is ≈ 40 episodes in RAM
# at once.  N_TEST_TASKS for 2c is handled lazily (one episode at a time).
N_EPISODES = 30      # test episodes for 2a/2b (pre-loaded)
N_VAL_EPS  = 200     # val episodes for 2c (generated lazily, 1 at a time)

# ── Load all needed models ────────────────────────────────────────────────────
print("=" * 60)
print("Loading models B1, B3, B4 ...")
print("=" * 60)
model_cfgs = {
    "B1": ("configs/stage1_bidirectional.yaml",
           os.path.expanduser("~/trx_data/checkpoints/abl_bidir_hmdb3/"
                              "hmdb_split3_20260407_220041/checkpoint75000.pt")),
    "B3": ("configs/stage2_b3_inter_only.yaml",
           os.path.expanduser("~/trx_data/checkpoints/b3_inter_hmdb3_1shot/"
                              "hmdb_split3_20260505_044436/checkpoint75000.pt")),
    "B4": ("configs/stage2_true_hyrsm_b4_tuple.yaml",
           os.path.expanduser("~/trx_data/checkpoints/true_hyrsm_b4_1shot/"
                              "hmdb_split3_20260526_202630/checkpoint75000.pt")),
}
# map to visualize.py's key names
VIZ_KEY = {"B1": "s1", "B3": "b3", "B4": "b4"}

loaded_models = {}
for label, (cfg, ckpt) in model_cfgs.items():
    print(f"  {label} ...")
    m, a = load_model(cfg, ckpt, device=DEVICE)
    m.eval()
    loaded_models[label] = (m, a, cfg)

# ── Episode loader (test split) ────────────────────────────────────────────────
print("\nSampling episodes ...")
first_cfg = list(model_cfgs.values())[0][0]
loader = EpisodeLoader(build_args(first_cfg))
episodes = [loader.get_episode() for _ in range(N_EPISODES)]
print(f"  Sampled {N_EPISODES} episodes.")


# ==============================================================================
# Helper: extract s_set for one episode under one model
# ==============================================================================
def get_sets(label, model, ep):
    vkey = VIZ_KEY[label]
    sup_t = ep["support_tensors"].to(DEVICE)
    q_t   = ep["query_tensors"].to(DEVICE)
    with torch.no_grad():
        fq = get_frame_features(model, q_t)
        fs = get_frame_features(model, sup_t)
        _, _, q_set, s_set = get_pre_hausdorff_embeddings(vkey, model, fq, fs)
    return q_set.cpu().float(), s_set.cpu().float()


# ==============================================================================
# 2a: Spread R (raw + centred) and decision margin M
# ==============================================================================
print("\n" + "=" * 60)
print("2a: Computing spread R (raw + centred) and margin M ...")
print("=" * 60)

def compute_R_and_M(label, model, episodes):
    """
    R_raw, R_cen: spread ratio
    M: decision margin across all episodes
    """
    T = 28
    R_raw_list = []
    R_cen_list = []
    M_list     = []

    for ep in episodes:
        q_set, s_set = get_sets(label, model, ep)
        s_lbl = ep["support_labels"]
        q_lbl = ep["query_labels"]
        ns, T_, d = s_set.shape

        # ── Centred version ───────────────────────────────────────────────────
        mu    = s_set.mean(0, keepdim=True)   # [1, T, d]
        s_cen = s_set - mu
        q_cen = q_set - mu

        classes = torch.unique(s_lbl).tolist()

        def spread(s, s_cen_ep):
            # within_k: mean pairwise sq L2 among T tuples of class k
            # For 1-shot: class_s is [1, T, d] → T tuples, pairwise among tuples
            within_list_raw, within_list_cen = [], []
            for c in classes:
                cidx = (s_lbl == c).nonzero(as_tuple=True)[0]
                s_k     = s[cidx].reshape(-1, d)     # [k*T, d] → here k=1 → [T, d]
                s_k_cen = s_cen_ep[cidx].reshape(-1, d)

                # pairwise sq L2
                def pw_sq(x):
                    xn = (x ** 2).sum(-1)
                    gram = x @ x.t()
                    dist2 = (xn.unsqueeze(1) + xn.unsqueeze(0) - 2 * gram).clamp(0)
                    n = x.shape[0]
                    if n <= 1:
                        return torch.tensor(0.0)
                    # upper triangle only, no diagonal
                    triu = dist2.triu(diagonal=1)
                    cnt = n * (n - 1) / 2
                    return triu.sum() / cnt

                within_list_raw.append(pw_sq(s_k).item())
                within_list_cen.append(pw_sq(s_k_cen).item())

            # between: mean sq L2 between tuples of different classes
            def between_dist(x, labels):
                """Mean sq L2 between all (i,j) where label[i] != label[j]"""
                x_flat = x.reshape(-1, d)
                n = x_flat.shape[0]
                xn = (x_flat ** 2).sum(-1)
                gram = x_flat @ x_flat.t()
                dist2 = (xn.unsqueeze(1) + xn.unsqueeze(0) - 2 * gram).clamp(0)
                lbl_expanded = labels.repeat_interleave(T if x.ndim == 3 else 1)
                diff_mask = (lbl_expanded.unsqueeze(1) != lbl_expanded.unsqueeze(0))
                if diff_mask.sum() == 0:
                    return torch.tensor(0.0)
                return dist2[diff_mask].mean().item()

            # Build per-class repeated labels for between
            s_lbl_rep = torch.cat([s_lbl[i].repeat(T) for i in range(len(s_lbl))])
            raw_between = between_dist(s.reshape(-1, d),   s_lbl_rep)
            cen_between = between_dist(s_cen_ep.reshape(-1, d), s_lbl_rep)

            R_raw = raw_between / (np.mean(within_list_raw) + 1e-8)
            R_cen = cen_between / (np.mean(within_list_cen) + 1e-8)
            return R_raw, R_cen

        R_raw_ep, R_cen_ep = spread(s_set, s_cen)
        R_raw_list.append(R_raw_ep)
        R_cen_list.append(R_cen_ep)

        # ── Decision margin M ─────────────────────────────────────────────────
        # For each query video, compute distances to all 5 classes
        # using the model's actual distance function (pool_hausdorff, bidir)
        vkey = VIZ_KEY[label]
        s_lbl_t = ep["support_labels"]
        q_lbl_t = ep["query_labels"]

        for qi in range(q_set.shape[0]):
            q_single = q_set[qi:qi+1]    # [1, T, d]
            dists = []
            for c in classes:
                cidx = (s_lbl_t == c).nonzero(as_tuple=True)[0]
                class_s = s_set[cidx]   # [k, T, d]
                dist = pool_hausdorff(q_single.to(DEVICE), class_s.to(DEVICE),
                                      mode="bidirectional")
                dists.append(dist.item())
            dists = np.array(dists)
            sorted_d = np.sort(dists)
            mean_d = np.mean(dists)
            m_ep = (sorted_d[1] - sorted_d[0]) / (mean_d + 1e-8)
            M_list.append(m_ep)

    return float(np.mean(R_raw_list)), float(np.mean(R_cen_list)), float(np.mean(M_list))

results_2a = {}
known_acc = {"B1": 47.49, "B3": 39.01, "B4": 50.26}

for label, (model, args, cfg) in loaded_models.items():
    print(f"\n  {label} ...")
    R_raw, R_cen, M = compute_R_and_M(label, model, episodes)  # use all N_EPISODES
    results_2a[label] = {"R_raw": R_raw, "R_cen": R_cen, "M": M,
                          "acc": known_acc[label]}
    print(f"    R(raw)={R_raw:.4f}  R(cen)={R_cen:.4f}  M={M:.4f}")

print("\n=== 2a Results ===")
print(f"  {'設定':8s}  {'1-shot acc':>10s}  {'R(原始)':>9s}  {'R(中心化)':>11s}  {'M':>6s}")
for lbl in ["B1", "B3", "B4"]:
    r = results_2a[lbl]
    print(f"  {lbl:8s}  {r['acc']:>10.2f}  {r['R_raw']:>9.4f}  {r['R_cen']:>11.4f}  {r['M']:>6.4f}")

# Check ordering consistency
acc_order = sorted(["B1","B3","B4"], key=lambda x: results_2a[x]["acc"])
r_raw_order = sorted(["B1","B3","B4"], key=lambda x: results_2a[x]["R_raw"])
r_cen_order = sorted(["B1","B3","B4"], key=lambda x: results_2a[x]["R_cen"])
m_order = sorted(["B1","B3","B4"], key=lambda x: results_2a[x]["M"])
print(f"\n  acc order: {acc_order}")
print(f"  R_raw order: {r_raw_order}  → 與準確率 {'一致' if r_raw_order == acc_order else '不一致'}")
print(f"  R_cen order: {r_cen_order}  → 與準確率 {'一致' if r_cen_order == acc_order else '不一致'}")
print(f"  M order: {m_order}    → 與準確率 {'一致' if m_order == acc_order else '不一致'}")

# ── fig9: R vs accuracy scatter ────────────────────────────────────────────────
print("\nPlotting fig9_spread_vs_acc.svg ...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor("white")
colors_map = {"B1": "#5b7fb5", "B3": "#e07b54", "B4": "#4caf88"}

for ax_i, (ax, key, title) in enumerate(zip(axes,
    ["R_raw", "R_cen"],
    ["(a) 原始 R vs 準確率", "(b) 中心化 R vs 準確率"])):
    for lbl in ["B1", "B3", "B4"]:
        r = results_2a[lbl]
        ax.scatter(r[key], r["acc"], color=colors_map[lbl], s=80, zorder=3)
        ax.annotate(lbl, (r[key], r["acc"]), textcoords="offset points",
                    xytext=(6, 3), fontsize=9, color=colors_map[lbl])
    ax.set_xlabel("R (集合分散度)", fontsize=10, color="#666")
    ax.set_ylabel("1-shot 準確率 (%)", fontsize=10, color="#666")
    ax.set_title(title, fontsize=11, color="#111", pad=8)
    ax.tick_params(labelsize=8, labelcolor="#333")
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_color("#ddd")

fig.suptitle("Figure 9: Support-set spread vs 1-shot accuracy (B1/B3/B4)",
             fontsize=12, color="#111", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, "fig9_spread_vs_acc.svg"), bbox_inches="tight", dpi=150)
fig.savefig(os.path.join(FIGS_DIR, "fig9_spread_vs_acc.png"), bbox_inches="tight", dpi=150)
plt.close(fig)
print("  Saved: figs/fig9_spread_vs_acc.svg / .png")


# ==============================================================================
# 2b: Bidirectional vs Unidirectional (same B1 75k checkpoint)
# ==============================================================================
print("\n" + "=" * 60)
print("2b: Bidirectional vs Unidirectional on B1 75k ...")
print("=" * 60)

def eval_mode_simple(model, episodes, mode):
    """Evaluate episodes with given pool_hausdorff mode. Single pass, returns acc ± CI."""
    accs = []
    for ep in episodes:
        q_set, s_set = get_sets("B1", model, ep)
        s_lbl = ep["support_labels"]
        q_lbl = ep["query_labels"]
        classes = torch.unique(s_lbl).tolist()

        for qi in range(q_set.shape[0]):
            q_single = q_set[qi:qi+1].to(DEVICE)
            dists = []
            for c in classes:
                cidx = (s_lbl == c).nonzero(as_tuple=True)[0]
                class_s = s_set[cidx].to(DEVICE)
                d = pool_hausdorff(q_single, class_s, mode=mode)
                dists.append(d.item())
            pred_class = classes[int(np.argmin(dists))]
            accs.append(float(pred_class == q_lbl[qi].item()))

    n = len(accs)
    mean_acc = 100.0 * float(np.mean(accs))
    ci = 196.0 * float(np.std(accs)) / float(np.sqrt(n))
    return mean_acc, ci

b1_model = loaded_models["B1"][0]
N_2B = len(episodes)  # use all N_EPISODES
print(f"  Evaluating on {N_2B} episodes ...")

acc_bidir, ci_bidir = eval_mode_simple(b1_model, episodes[:N_2B], "bidirectional")
print(f"  Bidirectional: {acc_bidir:.2f} ± {ci_bidir:.2f}")
acc_uni,   ci_uni   = eval_mode_simple(b1_model, episodes[:N_2B], "unidirectional")
print(f"  Unidirectional: {acc_uni:.2f} ± {ci_uni:.2f}")

print("\n=== 2b Results ===")
print(f"  {'距離函數':20s}  {'集合聚合':>10s}  {'shot':>4s}  {'準確率 ± CI':>14s}")
print(f"  {'雙向 (trained)':20s}  {'pool':>10s}  {'1':>4s}  {acc_bidir:.2f} ± {ci_bidir:.2f}")
print(f"  {'單向 Q→S':20s}  {'pool':>10s}  {'1':>4s}  {acc_uni:.2f} ± {ci_uni:.2f}")
print("  ⚠️  模型以雙向訓練，換單向屬 off-distribution eval，與 §8.4 一致處理")


# ==============================================================================
# 2c: Validation-set checkpoint selection
# ==============================================================================
print("\n" + "=" * 60)
print("2c: Validation-set checkpoint selection ...")
print("=" * 60)

# Build a val-set episode loader
# vallist03.txt has format: "class/videoname" (no label number in some formats)
# We need to read these and create a Split from the zip file

import zipfile as zf_module

class ValEpisodeLoader:
    """
    Minimal episode sampler reading vallist03.txt from the zip.
    Builds a class→video_paths mapping, then samples 5-way 1-shot episodes.
    """
    def __init__(self, args, n_per_class=None):
        self.args = args
        self.seq_len = args.seq_len if hasattr(args, 'seq_len') else SEQ_LEN
        self.img_size = getattr(args, 'img_size', 224)
        self.way  = WAY
        self.shot = SHOT
        self.qpc  = getattr(args, 'query_per_class', N_QUERY)

        # Read vallist
        val_file = os.path.join(SPLIT_DIR, "vallist03.txt")
        self.class_to_vids = {}  # class_name → list of zipfile paths

        # Load zip index
        mem = open(DATA_ZIP, 'rb').read()
        self.zfile = zf_module.ZipFile(io.BytesIO(mem))
        all_names = set(self.zfile.namelist())

        # Parse vallist
        with open(val_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                rel_path = parts[0]   # e.g. "cartwheel/some_video"
                cls = rel_path.split('/')[0]
                vid = rel_path.split('/')[-1] if '/' in rel_path else rel_path

                # Find all frames for this video in the zip
                prefix = rel_path + '/'
                frames = sorted([n for n in all_names if n.startswith(prefix)
                                  and not n.endswith('/')])
                if len(frames) < self.seq_len:
                    continue  # skip short videos

                if cls not in self.class_to_vids:
                    self.class_to_vids[cls] = []
                self.class_to_vids[cls].append(frames)

        self.classes = sorted(self.class_to_vids.keys())
        print(f"  Val split: {len(self.classes)} classes, "
              f"{sum(len(v) for v in self.class_to_vids.values())} videos")

        # transforms
        from videotransforms.video_transforms import Compose, Resize, CenterCrop
        self.transform = Compose([Resize(256), CenterCrop(self.img_size)])
        import torchvision.transforms as tv_t
        self.to_tensor = tv_t.ToTensor()

    def _sample_frames(self, frame_list):
        """Uniform frame sampling (test mode)."""
        n = len(frame_list)
        start, end = 1, n - 2
        if end - start < self.seq_len:
            start, end = 0, n - 1
        idxs = [int(x) for x in np.linspace(start, end, num=self.seq_len)]
        return [frame_list[i] for i in idxs]

    def _load_video(self, frame_paths):
        """Load and transform frames → [seq_len, 3, H, W]."""
        pil_imgs = []
        for fp in frame_paths:
            data = self.zfile.read(fp)
            img = __import__('PIL').Image.open(io.BytesIO(data)).convert("RGB")
            pil_imgs.append(img)
        transformed = self.transform(pil_imgs)
        tensors = [self.to_tensor(img) for img in transformed]
        return torch.stack(tensors)   # [seq_len, 3, H, W]

    def get_episode(self):
        """Sample one 5-way 1-shot episode."""
        batch_classes = random.sample(self.classes, self.way)
        support_tensors, support_labels = [], []
        query_tensors,   query_labels   = [], []

        for bi, bc in enumerate(batch_classes):
            vids = self.class_to_vids[bc]
            idxs = random.sample(range(len(vids)), self.shot + self.qpc)
            for si in idxs[:self.shot]:
                frames = self._sample_frames(vids[si])
                t = self._load_video(frames)
                support_tensors.append(t)
                support_labels.append(bi)
            for qi in idxs[self.shot:]:
                frames = self._sample_frames(vids[qi])
                t = self._load_video(frames)
                query_tensors.append(t)
                query_labels.append(bi)

        return {
            "support_tensors": torch.cat(support_tensors, dim=0),   # [ns*seq_len, 3, H, W]
            "support_labels":  torch.tensor(support_labels),
            "query_tensors":   torch.cat(query_tensors, dim=0),
            "query_labels":    torch.tensor(query_labels),
        }


val_loader_args = build_args("configs/stage1_bidirectional.yaml")
val_loader = ValEpisodeLoader(val_loader_args)
print(f"  Val loader ready. Will generate {N_VAL_EPS} episodes lazily per checkpoint eval.")

def fast_eval_checkpoint(ckpt_path, cfg_path, viz_key, ep_loader, n_episodes, seed_offset=9999):
    """Load checkpoint, eval using full model forward, return acc ± CI.
    Generates episodes lazily (one at a time) to avoid OOM.
    Resets random seed at start so each call evaluates the same episodes."""
    from visualize import load_model as lm
    m, _ = lm(cfg_path, ckpt_path, device=DEVICE)
    m.eval()

    # Deterministic episodes: same seed → same sequence every call
    random.seed(SEED + seed_offset)
    np.random.seed(SEED + seed_offset)

    correct_list = []
    for _ in range(n_episodes):
        ep = ep_loader.get_episode()   # generate one episode, discard after eval
        sup_t = ep["support_tensors"].to(DEVICE)   # [ns*seq_len, 3, H, W]
        q_t   = ep["query_tensors"].to(DEVICE)     # [nq*seq_len, 3, H, W]
        s_lbl = ep["support_labels"].to(DEVICE)
        q_lbl = ep["query_labels"]

        with torch.no_grad():
            out = m(sup_t, s_lbl, q_t)
            logits = out["logits"]   # [NUM_SAMPLES, nq, way] or [nq, way]
            if logits.ndim == 3:
                logits = logits.mean(0)   # [nq, way]
            preds = logits.argmax(dim=-1)   # [nq]

        for qi in range(preds.shape[0]):
            pred_class_idx = preds[qi].item()
            gt_class_idx   = q_lbl[qi].item()
            correct_list.append(float(pred_class_idx == gt_class_idx))

        # Explicit del to help GC (images can be large)
        del ep, sup_t, q_t, s_lbl

    n = len(correct_list)
    mean_acc = 100.0 * float(np.mean(correct_list))
    ci = 196.0 * float(np.std(correct_list)) / float(np.sqrt(n))
    del m
    torch.cuda.empty_cache()
    return mean_acc, ci


# Checkpoint maps for 2c
# (name, config, viz_key, {iter: ckpt_path}, current_best_acc)
CKPT_2C = {
    "B1": {
        "cfg": "configs/stage1_bidirectional.yaml",
        "vkey": "s1",
        "ckpts": {
            25000:  "~/trx_data/checkpoints/abl_bidir_hmdb3/hmdb_split3_20260407_220041/checkpoint25000.pt",
            50000:  "~/trx_data/checkpoints/abl_bidir_hmdb3/hmdb_split3_20260407_220041/checkpoint50000.pt",
            75000:  "~/trx_data/checkpoints/abl_bidir_hmdb3/hmdb_split3_20260407_220041/checkpoint75000.pt",
            100000: "~/trx_data/checkpoints/abl_bidir_hmdb3/hmdb_split3_20260407_220041/checkpoint100000.pt",
        },
        "test_acc": 47.49,
    },
    "B2": {
        "cfg": "configs/stage2_b2_intra_only.yaml",
        "vkey": "b3",   # both B2 and B3 use TRXSetMatchingWithRelation, same viz path
        "ckpts": {
            25000:  "~/trx_data/checkpoints/b2_intra_hmdb3_1shot/hmdb_split3_20260504_100143/checkpoint25000.pt",
            50000:  "~/trx_data/checkpoints/b2_intra_hmdb3_1shot/hmdb_split3_20260504_100143/checkpoint50000.pt",
            75000:  "~/trx_data/checkpoints/b2_intra_hmdb3_1shot/hmdb_split3_20260504_100143/checkpoint75000.pt",
            100000: "~/trx_data/checkpoints/b2_intra_hmdb3_1shot/hmdb_split3_20260504_100143/checkpoint100000.pt",
        },
        "test_acc": 47.1,
    },
    "B3": {
        "cfg": "configs/stage2_b3_inter_only.yaml",
        "vkey": "b3",
        "ckpts": {
            25000:  "~/trx_data/checkpoints/b3_inter_hmdb3_1shot/hmdb_split3_20260505_044436/checkpoint25000.pt",
            50000:  "~/trx_data/checkpoints/b3_inter_hmdb3_1shot/hmdb_split3_20260505_044436/checkpoint50000.pt",
            75000:  "~/trx_data/checkpoints/b3_inter_hmdb3_1shot/hmdb_split3_20260505_044436/checkpoint75000.pt",
            100000: "~/trx_data/checkpoints/b3_inter_hmdb3_1shot/hmdb_split3_20260505_044436/checkpoint100000.pt",
        },
        "test_acc": 39.01,
    },
    "B4": {
        "cfg": "configs/stage2_true_hyrsm_b4_tuple.yaml",
        "vkey": "b4",
        "ckpts": {
            25000:  "~/trx_data/checkpoints/true_hyrsm_b4_1shot/hmdb_split3_20260526_202630/checkpoint25000.pt",
            50000:  "~/trx_data/checkpoints/true_hyrsm_b4_1shot/hmdb_split3_20260526_202630/checkpoint50000.pt",
            75000:  "~/trx_data/checkpoints/true_hyrsm_b4_1shot/hmdb_split3_20260526_202630/checkpoint75000.pt",
            100000: "~/trx_data/checkpoints/true_hyrsm_b4_1shot/hmdb_split3_20260526_202630/checkpoint100000.pt",
        },
        "test_acc": 50.26,
    },
}

results_2c = {}
for name, info in CKPT_2C.items():
    cfg_path = info["cfg"]
    vkey = info["vkey"]
    print(f"\n  {name} ...")
    val_accs = {}
    for it, ckpt_rel in sorted(info["ckpts"].items()):
        ckpt = os.path.expanduser(ckpt_rel)
        if not os.path.exists(ckpt):
            print(f"    {it}: NOT FOUND, skipping")
            continue
        # Lazy val eval: generate N_VAL_EPS episodes on-the-fly (seed=SEED+9999)
        v_acc, v_ci = fast_eval_checkpoint(ckpt, cfg_path, vkey,
                                           val_loader, N_VAL_EPS, seed_offset=9999)
        print(f"    iter {it:>6d}: val={v_acc:.2f}±{v_ci:.2f}")
        val_accs[it] = (v_acc, v_ci)

    if not val_accs:
        print(f"    No checkpoints found for {name}")
        results_2c[name] = None
        continue

    # Pick val-best
    best_iter = max(val_accs, key=lambda k: val_accs[k][0])
    best_ckpt = os.path.expanduser(info["ckpts"][best_iter])
    print(f"    Val-best: iter {best_iter} (val {val_accs[best_iter][0]:.2f}%)")

    # Re-eval on test set with val-best checkpoint (lazy, uses test split loader)
    test_acc, test_ci = fast_eval_checkpoint(best_ckpt, cfg_path, vkey,
                                             loader, N_VAL_EPS, seed_offset=7777)
    print(f"    Test acc (val-selected): {test_acc:.2f}±{test_ci:.2f}")

    results_2c[name] = {
        "val_accs": val_accs,
        "best_iter": best_iter,
        "test_acc_val_selected": test_acc,
        "test_ci_val_selected": test_ci,
        "test_acc_current": info["test_acc"],
        "diff": test_acc - info["test_acc"],
    }

print("\n=== 2c Results ===")
print(f"  {'設定':6s}  {'測試集選點（現行）':>18s}  {'驗證集選點':>10s}  {'差':>6s}")
for name in ["B1", "B2", "B3", "B4"]:
    r = results_2c.get(name)
    if r is None:
        print(f"  {name:6s}  {'N/A':>18s}  {'N/A':>10s}  {'N/A':>6s}")
    else:
        diff_sign = "+" if r['diff'] >= 0 else ""
        print(f"  {name:6s}  {r['test_acc_current']:>18.2f}  "
              f"{r['test_acc_val_selected']:>10.2f}  "
              f"{diff_sign}{r['diff']:>5.2f}")

# Check ranking change
cur_order = sorted(["B1","B2","B3","B4"],
                   key=lambda x: (results_2c[x] or {}).get("test_acc_current", 0))
val_order = sorted(["B1","B2","B3","B4"],
                   key=lambda x: (results_2c[x] or {}).get("test_acc_val_selected", 0))
print(f"\n  測試集選點排序: {cur_order}")
print(f"  驗證集選點排序: {val_order}")
print(f"  排序改變：{'是 ⚠️' if val_order != cur_order else '否 ✅（結論穩健）'}")


# ==============================================================================
# Final machine-readable summary
# ==============================================================================
print("\n" + "=" * 60)
print("STAGE 2 COMPLETE SUMMARY")
print("=" * 60)
print("\n2a:")
for lbl in ["B1", "B3", "B4"]:
    r = results_2a[lbl]
    print(f"  {lbl}: acc={r['acc']:.2f}  R_raw={r['R_raw']:.4f}  R_cen={r['R_cen']:.4f}  M={r['M']:.4f}")
print(f"  R_raw vs acc 排序: {'一致' if r_raw_order==acc_order else '不一致'}")
print(f"  R_cen vs acc 排序: {'一致' if r_cen_order==acc_order else '不一致'}")
print(f"  M vs acc 排序: {'一致' if m_order==acc_order else '不一致'}")

print("\n2b:")
print(f"  雙向 pool 1-shot: {acc_bidir:.2f} ± {ci_bidir:.2f}%")
print(f"  單向 pool 1-shot: {acc_uni:.2f} ± {ci_uni:.2f}%")

print("\n2c:")
for name in ["B1", "B2", "B3", "B4"]:
    r = results_2c.get(name)
    if r:
        print(f"  {name}: test_selected={r['test_acc_current']:.2f}  "
              f"val_selected={r['test_acc_val_selected']:.2f}  "
              f"diff={'+' if r['diff']>=0 else ''}{r['diff']:.2f}  "
              f"val_best_iter={r['best_iter']}")
print(f"  排序改變: {'是 ⚠️' if val_order != cur_order else '否 ✅'}")
