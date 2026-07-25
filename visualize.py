"""
visualize.py — Visualization script for TRX tuple-level set matching research.

Produces three figures comparing S1 / B3 / B4 without modifying any existing module.
Embeddings are extracted by calling public sub-module methods directly (no hooks,
no monkey-patching).

Usage:
    /home/ccwu/miniconda3/envs/trx/bin/python visualize.py

Outputs in reports/viz/:
    fig1_tsne.png / .pdf
    fig2_prediction_panel.png / .pdf
    fig3_heatmap.png / .pdf
    fig1_episodes.json
    fig2_cases.json
    fig3_episode.json
"""

import os, sys, json, random, types, warnings
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
import video_reader
from model import CNN_TRX

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Constants ─────────────────────────────────────────────────────────────────

SCRATCH = os.path.expanduser("~/Documents/work/trx/trx_data")
OUT_DIR = "reports/viz"
SEED    = 42
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINTS = {
    "s1": (
        "configs/stage1_bidirectional.yaml",
        os.path.expanduser(
            "~/trx_data/checkpoints/abl_bidir_hmdb3/"
            "hmdb_split3_20260407_220041/checkpoint50000.pt"
        ),
    ),
    "b3": (
        "configs/stage2_b3_inter_only.yaml",
        os.path.expanduser(
            "~/trx_data/checkpoints/b3_inter_hmdb3_1shot/"
            "hmdb_split3_20260505_044436/checkpoint75000.pt"
        ),
    ),
    "b4": (
        "configs/stage2_true_hyrsm_b4_tuple.yaml",
        os.path.expanduser(
            "~/trx_data/checkpoints/true_hyrsm_b4_1shot/"
            "hmdb_split3_20260526_202630/checkpoint75000.pt"
        ),
    ),
}

MODEL_LABELS = {
    "s1": "S1 — No Relation",
    "b3": "B3 — Token-level Inter",
    "b4": "B4 — Intra + Pool-level Inter",
}
WAY        = 5
SHOT       = 1
N_QUERY    = 5      # queries per class for visualization episodes
SEQ_LEN    = 8
IMG_SIZE   = 224
THUMB_SIZE = 112    # pixels for thumbnail frames in figs 2/3

# ── Model utilities ───────────────────────────────────────────────────────────

def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def build_args(config_path, dataset="hmdb", split=3):
    """Build a minimal args namespace from a YAML config + fixed defaults."""
    cfg = _load_yaml(config_path)
    args = types.SimpleNamespace(
        # required by CNN_TRX / VideoDataset
        method              = cfg.get("backbone", "resnet18"),
        trans_linear_in_dim = 512,
        trans_linear_out_dim= cfg.get("trans_linear_out_dim", 1152),
        trans_dropout       = cfg.get("trans_dropout", 0.1),
        seq_len             = cfg.get("seq_len", SEQ_LEN),
        img_size            = cfg.get("img_size", IMG_SIZE),
        temp_set            = cfg.get("temp_set", [2]),
        num_samples         = cfg.get("num_samples", 1),
        way                 = cfg.get("way", WAY),
        shot                = cfg.get("shot", SHOT),
        query_per_class     = N_QUERY,
        query_per_class_test= 1,
        num_gpus            = 1,
        # relation config
        use_intra_relation  = cfg.get("use_intra_relation", False),
        use_inter_relation  = cfg.get("use_inter_relation", False),
        relation_level      = cfg.get("relation_level", "tuple"),
        inter_style         = cfg.get("inter_style", "global"),
        # data paths
        dataset             = dataset,
        split               = split,
        scratch             = SCRATCH,
        path                = os.path.join(SCRATCH, "video_datasets/data/hmdb51_256q5.zip"),
        traintestlist       = os.path.join(SCRATCH, "video_datasets/splits/hmdb_ARN"),
        num_workers         = 0,
        debug_loader        = False,
    )
    return args


def load_model(config_path, checkpoint_path, device=DEVICE):
    """Build model from config, load checkpoint, return in eval mode."""
    args  = build_args(config_path)
    model = CNN_TRX(args).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Loaded {checkpoint_path.split('/')[-1]}  "
          f"(iter {ckpt.get('iteration', '?')})")
    return model, args


# ── Embedding extraction ──────────────────────────────────────────────────────

@torch.no_grad()
def get_frame_features(model, images, device=DEVICE):
    """
    Run ResNet backbone only.
    images : [N * seq_len, 3, H, W]
    returns: [N, seq_len, d_in]
    """
    feats = model.resnet(images.to(device))   # [N*seq_len, d, 1, 1]
    feats = feats.squeeze(-1).squeeze(-1)      # [N*seq_len, d]
    seq_len = model.args.seq_len if hasattr(model, "args") else SEQ_LEN
    d = feats.shape[1]
    return feats.reshape(-1, seq_len, d)       # [N, seq_len, d]


@torch.no_grad()
def get_pre_hausdorff_embeddings(model_key, model, frame_feats_q, frame_feats_s):
    """
    Extract tuple embeddings just before Hausdorff matching — without modifying
    any existing module. Calls public sub-module methods directly.

    Returns
    -------
    q_emb : [nq, d]   — one vector per query video (mean over T tuples)
    s_emb : [ns, d]   — one vector per support video (mean over T tuples)
    q_set : [nq, T, d] — full tuple set (for fig 3 heatmap)
    s_set : [ns, T, d] — full tuple set (for fig 3 heatmap, pre-inter for B4)
    """
    if model_key == "s1":
        tm = model.transformers[0]           # TRXSetMatching
        q_set = tm._make_pair_set(frame_feats_q)   # [nq, T, d]
        s_set = tm._make_pair_set(frame_feats_s)   # [ns, T, d]

    else:
        tm = model.transformers[0]           # TRXSetMatchingWithRelation
        q_set = tm.matching._make_pair_set(frame_feats_q)   # [nq, T, d]
        s_set = tm.matching._make_pair_set(frame_feats_s)   # [ns, T, d]

        # IntraRelation (if present — Identity when disabled)
        q_set = tm.intra(q_set)
        s_set = tm.intra(s_set)

        # InterRelation
        if tm.inter is not None:
            if tm.inter_style == "true_hyrsm":
                q_set, s_set_pq = tm.inter(q_set, s_set)   # [nq, T, d], [nq, ns, T, d]
                # Average over query dim to get per-video support embeddings
                s_set = s_set_pq.mean(0)                    # [ns, T, d]
            else:  # global inter (B3)
                q_set, s_set = tm.inter(q_set, s_set)       # [nq, T, d], [ns, T, d]

    q_emb = q_set.mean(dim=1)   # [nq, d]
    s_emb = s_set.mean(dim=1)   # [ns, d]
    return q_emb, s_emb, q_set, s_set


# ── Data utilities ────────────────────────────────────────────────────────────

class EpisodeLoader:
    """
    Wraps VideoDataset to produce reproducible few-shot episodes.
    Returns both model-ready tensors AND raw PIL frame lists.
    """

    def __init__(self, args):
        self.args = args
        self.vd   = video_reader.VideoDataset(args)
        self.vd.train = False   # use test split

    def _sample_frames_test(self, paths, seq_len):
        """Uniform sampling matching VideoDataset test-mode behaviour."""
        n = len(paths)
        start, end = 1, n - 2
        if end - start < seq_len:
            start, end = 0, n - 1
        idxs = [int(f) for f in np.linspace(start, end, num=seq_len)]
        return idxs

    def get_episode(self, rng_state=None):
        """
        Sample one 5-way 1-shot episode from the test split.
        Returns a dict with:
            support_tensors   : [5*seq_len, 3, H, W]
            support_labels    : [5] int 0..4
            query_tensors     : [n_query*5, 3, seq_len, H, W]  -- flattened
            query_labels      : int tensor
            batch_class_ids   : list of 5 class int IDs
            class_names       : list of 5 str
            support_raw_frames: list of 5 lists (each: seq_len PIL images)
            query_raw_frames  : list of n_q*5 lists (each: seq_len PIL images)
            support_vid_ids   : list of 5 int
            query_vid_ids     : list of n_q*5 int
        """
        vd = self.vd
        split = vd.test_split
        classes = split.get_unique_classes()

        batch_classes = random.sample(classes, WAY)

        support_tensors    = []
        support_labels     = []
        support_raw_frames = []
        support_vid_ids    = []

        query_tensors      = []
        query_labels       = []
        query_raw_frames   = []
        query_vid_ids      = []

        n_query = self.args.query_per_class

        for bi, bc in enumerate(batch_classes):
            n_total = split.get_num_videos_for_class(bc)
            idxs = random.sample(range(n_total), SHOT + n_query)

            for si in idxs[:SHOT]:
                paths, vid_id = split.get_rand_vid(bc, si)
                frame_idxs    = self._sample_frames_test(paths, SEQ_LEN)
                # tensor — transform["test"] expects a list of PIL images
                pil_imgs = [vd.read_single_image(paths[i]) for i in frame_idxs]
                transformed = vd.transform["test"](pil_imgs)   # returns list of PIL
                imgs = [vd.tensor_transform(img) for img in transformed]
                support_tensors.append(torch.stack(imgs))   # [seq_len, 3, H, W]
                support_labels.append(bi)
                # raw PIL (resized for display, no model transform)
                raw = [vd.read_single_image(paths[i]).convert("RGB")
                       .resize((THUMB_SIZE, THUMB_SIZE)) for i in frame_idxs]
                support_raw_frames.append(raw)
                support_vid_ids.append(vid_id)

            for qi in idxs[SHOT:]:
                paths, vid_id = split.get_rand_vid(bc, qi)
                frame_idxs    = self._sample_frames_test(paths, SEQ_LEN)
                pil_imgs   = [vd.read_single_image(paths[i]) for i in frame_idxs]
                transformed = vd.transform["test"](pil_imgs)
                imgs = [vd.tensor_transform(img) for img in transformed]
                query_tensors.append(torch.stack(imgs))
                query_labels.append(bi)
                raw = [vd.read_single_image(paths[i]).convert("RGB")
                       .resize((THUMB_SIZE, THUMB_SIZE)) for i in frame_idxs]
                query_raw_frames.append(raw)
                query_vid_ids.append(vid_id)

        # Flatten to model-compatible tensors
        # support: [ns*seq_len, 3, H, W]
        sup_t = torch.cat([t for t in support_tensors], dim=0)
        # query:  [nq*seq_len, 3, H, W]
        q_t   = torch.cat([t for t in query_tensors], dim=0)

        class_names = [vd.class_folders[bc] for bc in batch_classes]

        return {
            "support_tensors"   : sup_t,
            "support_labels"    : torch.tensor(support_labels, dtype=torch.long),
            "query_tensors"     : q_t,
            "query_labels"      : torch.tensor(query_labels,   dtype=torch.long),
            "batch_class_ids"   : batch_classes,
            "class_names"       : class_names,
            "support_raw_frames": support_raw_frames,
            "query_raw_frames"  : query_raw_frames,
            "support_vid_ids"   : support_vid_ids,
            "query_vid_ids"     : query_vid_ids,
        }


# ── Metrics ───────────────────────────────────────────────────────────────────

def linear_cka(X, Y):
    """
    Linear CKA between X [n, d] and Y [n, d] (same n, same class ordering).
    Uses class-mean representations so query/support sizes don't need to match.
    """
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    num   = (X @ Y.T).pow(2).sum()
    denom = ((X @ X.T).pow(2).sum() * (Y @ Y.T).pow(2).sum()).sqrt()
    return float(num / (denom + 1e-8))


def compute_metrics(q_emb, s_emb, q_labels, s_labels, way=WAY):
    """
    CKA: between class-mean query and class-mean support (shape [way, d] each).
    Silhouette: pooled query+support embeddings vs class labels.
    """
    q_np = q_emb.cpu().float().numpy()
    s_np = s_emb.cpu().float().numpy()
    ql   = q_labels.cpu().numpy()
    sl   = s_labels.cpu().numpy()

    # Class means [way, d]
    q_means = np.stack([q_np[ql == c].mean(0) for c in range(way)])
    s_means = np.stack([s_np[sl == c].mean(0) for c in range(way)])

    q_t = torch.from_numpy(q_means)
    s_t = torch.from_numpy(s_means)
    cka = linear_cka(q_t, s_t)

    # Silhouette on pooled Q+S
    all_emb = np.concatenate([q_np, s_np], axis=0)
    all_lbl = np.concatenate([ql,   sl  ], axis=0)
    sil = silhouette_score(all_emb, all_lbl) if len(set(all_lbl)) > 1 else float("nan")

    return cka, sil


# ── Figure 1: t-SNE triple panel ──────────────────────────────────────────────

def fig1_tsne(models, loader, n_episodes=15, out_dir=OUT_DIR):
    """
    Collect embeddings from n_episodes, run t-SNE, plot triple panel.
    Annotates CKA and silhouette score per panel.
    """
    print("\n[Fig 1] Collecting embeddings ...")

    # Collect episodes once, share across models
    episodes = []
    episode_meta = []
    for ep in range(n_episodes):
        ep_data = loader.get_episode()
        episodes.append(ep_data)
        episode_meta.append({
            "ep": ep,
            "class_names"    : ep_data["class_names"],
            "batch_class_ids": ep_data["batch_class_ids"],
            "support_vid_ids": ep_data["support_vid_ids"],
            "query_vid_ids"  : ep_data["query_vid_ids"],
        })

    with open(os.path.join(out_dir, "fig1_episodes.json"), "w") as f:
        json.dump(episode_meta, f, indent=2)

    # For each model, extract embeddings
    all_data = {}
    for key, (model, args) in models.items():
        print(f"  Extracting for {key} ...")
        q_embs, s_embs, q_lbls, s_lbls = [], [], [], []

        for ep_data in episodes:
            sup_t   = ep_data["support_tensors"].to(DEVICE)
            q_t     = ep_data["query_tensors"].to(DEVICE)
            sup_lbl = ep_data["support_labels"]
            q_lbl   = ep_data["query_labels"]

            frame_q = get_frame_features(model, q_t)
            frame_s = get_frame_features(model, sup_t)

            q_emb, s_emb, _, _ = get_pre_hausdorff_embeddings(key, model, frame_q, frame_s)

            q_embs.append(q_emb.cpu())
            s_embs.append(s_emb.cpu())
            q_lbls.append(q_lbl)
            s_lbls.append(sup_lbl)

        all_q = torch.cat(q_embs, 0).float().numpy()   # [n_ep*nq*way, d]
        all_s = torch.cat(s_embs, 0).float().numpy()   # [n_ep*way,    d]
        all_q_lbl = torch.cat(q_lbls, 0).numpy()
        all_s_lbl = torch.cat(s_lbls, 0).numpy()

        cka, sil = compute_metrics(
            torch.from_numpy(all_q), torch.from_numpy(all_s),
            torch.from_numpy(all_q_lbl), torch.from_numpy(all_s_lbl),
        )
        print(f"    CKA={cka:.3f}  silhouette={sil:.3f}")

        # t-SNE on pooled Q+S
        all_emb = np.concatenate([all_q, all_s], axis=0)
        all_lbl = np.concatenate([all_q_lbl, all_s_lbl], axis=0)
        is_query = np.array([True]  * len(all_q) + [False] * len(all_s))

        tsne = TSNE(n_components=2, perplexity=30, max_iter=1000,
                    random_state=SEED, init="pca")
        coords = tsne.fit_transform(all_emb)

        all_data[key] = dict(
            coords=coords, lbl=all_lbl, is_query=is_query, cka=cka, sil=sil
        )

    # ── Plot ──────────────────────────────────────────────────────────────────
    CLASS_COLORS = plt.cm.tab10.colors[:WAY]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("t-SNE of Pre-Hausdorff Tuple Embeddings  "
                 "(shape=circle:query / x:support)", fontsize=13)

    for ax, key in zip(axes, ["s1", "b3", "b4"]):
        d = all_data[key]
        coords, lbl, isq = d["coords"], d["lbl"], d["is_query"]

        for ci in range(WAY):
            mask_qs = (lbl == ci) & isq
            mask_ss = (lbl == ci) & ~isq
            c = CLASS_COLORS[ci]
            ax.scatter(coords[mask_qs, 0], coords[mask_qs, 1],
                       c=[c], marker="o", s=30, edgecolors="black",
                       linewidths=0.6, alpha=0.85, label=f"cls {ci}")
            ax.scatter(coords[mask_ss, 0], coords[mask_ss, 1],
                       c=[c], marker="x", s=60, linewidths=1.5, alpha=0.85)

        ax.set_title(MODEL_LABELS[key], fontsize=11, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.02, 0.02,
                f"CKA={d['cka']:.3f}   Silhouette={d['sil']:.3f}",
                transform=ax.transAxes, fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # Legend: markers only
    query_patch   = plt.Line2D([0], [0], marker="o", color="gray", ls="None",
                               markersize=7, markeredgecolor="black", label="Query")
    support_patch = plt.Line2D([0], [0], marker="x", color="gray", ls="None",
                               markersize=9, markeredgewidth=1.5, label="Support")
    axes[1].legend(handles=[query_patch, support_patch],
                   loc="upper right", fontsize=8)

    plt.tight_layout()
    _save(fig, "fig1_tsne", out_dir)
    plt.close(fig)

    metrics = {k: {"cka": v["cka"], "silhouette": v["sil"]} for k, v in all_data.items()}
    return metrics


# ── Figure 2: Prediction panel ────────────────────────────────────────────────

def fig2_prediction_panel(models, loader, n_episodes=200, out_dir=OUT_DIR):
    """
    Run n_episodes, compare S1 and B4, pick 4 illustrative cases.
    """
    print("\n[Fig 2] Running prediction episodes ...")

    model_s1, args_s1 = models["s1"]
    model_b4, args_b4 = models["b4"]

    cases = {"s1_wrong_b4_right": [], "both_right": [], "both_wrong": []}
    all_meta = []

    for ep in range(n_episodes):
        ep_data = loader.get_episode()

        sup_t   = ep_data["support_tensors"].to(DEVICE)
        q_t     = ep_data["query_tensors"].to(DEVICE)
        sup_lbl = ep_data["support_labels"].to(DEVICE)
        q_lbl   = ep_data["query_labels"]

        # S1 forward
        frame_q_s1 = get_frame_features(model_s1, q_t)
        frame_s_s1 = get_frame_features(model_s1, sup_t)
        q_emb_s1, s_emb_s1, q_set_s1, s_set_s1 = get_pre_hausdorff_embeddings(
            "s1", model_s1, frame_q_s1, frame_s_s1)

        # B4 forward
        frame_q_b4 = get_frame_features(model_b4, q_t)
        frame_s_b4 = get_frame_features(model_b4, sup_t)
        q_emb_b4, s_emb_b4, q_set_b4, s_set_b4 = get_pre_hausdorff_embeddings(
            "b4", model_b4, frame_q_b4, frame_s_b4)

        # Use first query video only for simplicity
        q_idx = 0
        gt    = int(q_lbl[q_idx])

        # Compute distances to each class (using mean L2 to class mean)
        def class_dists(q_e, s_e, s_lbls):
            dists = []
            for ci in range(WAY):
                mask  = (s_lbls == ci)
                c_emb = s_e[mask].mean(0)
                d     = float((q_e[q_idx] - c_emb).pow(2).sum().sqrt())
                dists.append(d)
            return dists

        dists_s1 = class_dists(q_emb_s1, s_emb_s1, sup_lbl.cpu())
        dists_b4 = class_dists(q_emb_b4, s_emb_b4, sup_lbl.cpu())

        pred_s1 = int(np.argmin(dists_s1))
        pred_b4 = int(np.argmin(dists_b4))

        s1_correct = pred_s1 == gt
        b4_correct = pred_b4 == gt

        case_info = {
            "ep"          : ep,
            "gt"          : gt,
            "class_names" : ep_data["class_names"],
            "pred_s1"     : pred_s1,
            "pred_b4"     : pred_b4,
            "dists_s1"    : dists_s1,
            "dists_b4"    : dists_b4,
            "s1_correct"  : s1_correct,
            "b4_correct"  : b4_correct,
            "support_vid_ids": ep_data["support_vid_ids"],
            "query_vid_ids"  : [ep_data["query_vid_ids"][q_idx]],
        }
        all_meta.append(case_info)

        if not s1_correct and b4_correct:
            cases["s1_wrong_b4_right"].append((ep_data, case_info))
        elif s1_correct and b4_correct:
            cases["both_right"].append((ep_data, case_info))
        elif not s1_correct and not b4_correct:
            cases["both_wrong"].append((ep_data, case_info))

        # Stop early if we have enough
        if (len(cases["s1_wrong_b4_right"]) >= 2 and
                len(cases["both_right"]) >= 1 and
                len(cases["both_wrong"]) >= 1):
            print(f"  Found all cases at episode {ep+1}")
            break

    # Select up to target cases
    selected = []
    selected += cases["s1_wrong_b4_right"][:2]
    selected += cases["both_right"][:1]
    selected += cases["both_wrong"][:1]

    if len(selected) < 4:
        print(f"  Warning: only {len(selected)}/4 cases found in {n_episodes} episodes")

    # ── Save JSON ──────────────────────────────────────────────────────────────
    save_meta = []
    for ep_data, ci in selected:
        save_meta.append({k: v for k, v in ci.items()
                          if not isinstance(v, list) or k in
                          ("class_names", "dists_s1", "dists_b4",
                           "support_vid_ids", "query_vid_ids")})
    with open(os.path.join(out_dir, "fig2_cases.json"), "w") as f:
        json.dump(save_meta, f, indent=2)

    # ── Plot ──────────────────────────────────────────────────────────────────
    if not selected:
        print("  No cases found — skipping fig2")
        return

    n_cases = len(selected)
    # Layout: one row per case
    # Each row: 5 support thumbnails (3 frames each) | query (8 frames) | text
    n_sup_frames  = 3    # frames shown per support video
    n_qry_frames  = 8

    col_width = 1.2   # inches per thumbnail column
    fig_w = (WAY * n_sup_frames + n_qry_frames + 2) * col_width
    fig_h = n_cases * 2.5

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle("Figure 2: Prediction Comparison  (S1 vs B4)  —  "
                 "●=correct  ✗=wrong", fontsize=11, y=1.01)

    gs = GridSpec(n_cases, 1, figure=fig, hspace=0.6)

    for row_i, (ep_data, ci) in enumerate(selected):
        ax_row = fig.add_subplot(gs[row_i])
        ax_row.axis("off")

        # Case title
        cat = ("S1✗ B4✓" if not ci["s1_correct"] and ci["b4_correct"] else
               "S1✓ B4✓" if ci["s1_correct"] and ci["b4_correct"] else "S1✗ B4✗")
        gt_name = ci["class_names"][ci["gt"]]
        ax_row.set_title(f"[{cat}]  GT: {gt_name}", fontsize=9, loc="left",
                         pad=2, fontweight="bold")

        # Use inset axes for each thumbnail
        n_total_cols = WAY * n_sup_frames + 1 + n_qry_frames   # +1 gap
        col_w_frac = 1.0 / n_total_cols
        row_h = 1.0 / n_cases
        row_y = 1.0 - (row_i + 1) * row_h

        col = 0
        # Support videos
        for si in range(WAY):
            raw_frames = ep_data["support_raw_frames"][si]
            # Pick 3 evenly-spaced frames
            pick_idxs = [0, SEQ_LEN // 2, SEQ_LEN - 1]
            for fi_idx, fi in enumerate(pick_idxs):
                x0 = col * col_w_frac
                ax_img = fig.add_axes([x0, row_y + 0.05 * row_h,
                                       col_w_frac * 0.92, row_h * 0.65])
                ax_img.imshow(raw_frames[fi])
                ax_img.axis("off")
                if fi_idx == 1:
                    lbl_color = "green" if si == ci["gt"] else "black"
                    ax_img.set_title(ci["class_names"][si][:12],
                                     fontsize=5.5, color=lbl_color, pad=1)
                col += 1

        col += 1   # gap before query
        # Query frames
        raw_q = ep_data["query_raw_frames"][0]   # first query
        for fi in range(n_qry_frames):
            x0 = col * col_w_frac
            ax_img = fig.add_axes([x0, row_y + 0.05 * row_h,
                                   col_w_frac * 0.92, row_h * 0.65])
            ax_img.imshow(raw_q[fi])
            ax_img.axis("off")
            if fi == n_qry_frames // 2:
                ax_img.set_title("Query", fontsize=6, pad=1, color="navy")
            col += 1

        # Text block: prediction + distances
        x_text = (col - 0.5) * col_w_frac
        def fmt_dists(dists, pred, gt, names):
            lines = []
            order = np.argsort(dists)
            for rank, ci2 in enumerate(order[:3]):
                arrow = "→" if rank == 0 else " "
                mark  = "✓" if ci2 == gt else " "
                lines.append(f"{arrow}{mark} {names[ci2][:10]}: {dists[ci2]:.2f}")
            return "\n".join(lines)

        txt  = (f"S1 pred: {ci['class_names'][ci['pred_s1']][:12]}\n"
                f"{fmt_dists(ci['dists_s1'], ci['pred_s1'], ci['gt'], ci['class_names'])}\n\n"
                f"B4 pred: {ci['class_names'][ci['pred_b4']][:12]}\n"
                f"{fmt_dists(ci['dists_b4'], ci['pred_b4'], ci['gt'], ci['class_names'])}")

        ax_row.text(1.01, 0.5, txt, transform=ax_row.transAxes,
                    fontsize=5.5, va="center", ha="left",
                    fontfamily="monospace",
                    bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.8))

    plt.tight_layout()
    _save(fig, "fig2_prediction_panel", out_dir)
    plt.close(fig)


# ── Figure 3: Hausdorff heatmap ───────────────────────────────────────────────

def fig3_heatmap(models, loader, n_search=30, out_dir=OUT_DIR):
    """
    Find 1-2 correctly classified B4 episodes, plot query×support
    tuple distance heatmap with bidirectional min pairs + frame thumbnails.
    """
    print("\n[Fig 3] Searching for correct B4 episodes ...")

    model_b4, args_b4 = models["b4"]
    tm = model_b4.transformers[0]
    # Tuple index pairs: (frame_i, frame_j) for each tuple — shape [T, 2]
    tuple_pairs = tm.matching.tuples.cpu().numpy()   # [T, 2]

    good_eps = []

    for ep in range(n_search):
        ep_data = loader.get_episode()
        sup_t   = ep_data["support_tensors"].to(DEVICE)
        q_t     = ep_data["query_tensors"].to(DEVICE)
        sup_lbl = ep_data["support_labels"].to(DEVICE)

        frame_q = get_frame_features(model_b4, q_t)
        frame_s = get_frame_features(model_b4, sup_t)
        q_emb, s_emb, q_set, s_set = get_pre_hausdorff_embeddings(
            "b4", model_b4, frame_q, frame_s)

        # Check if first query is correctly classified
        gt = 0   # query_label for first query is always 0 for first class
        q_idx = 0

        dists = []
        for ci in range(WAY):
            mask  = (sup_lbl.cpu() == ci)
            c_emb = s_emb[mask].mean(0)
            d     = float((q_emb[q_idx] - c_emb).pow(2).sum().sqrt())
            dists.append(d)

        pred = int(np.argmin(dists))
        if pred == gt:
            good_eps.append((ep_data, q_set, s_set, sup_lbl.cpu()))
            print(f"  Found correct episode at ep {ep+1}  (class: {ep_data['class_names'][0]})")
            if len(good_eps) >= 2:
                break

    if not good_eps:
        print("  No correct episodes found — skipping fig3")
        return

    ep_data, q_set, s_set, sup_lbl = good_eps[0]

    # Save JSON
    with open(os.path.join(out_dir, "fig3_episode.json"), "w") as f:
        json.dump({
            "class_names"    : ep_data["class_names"],
            "batch_class_ids": ep_data["batch_class_ids"],
            "support_vid_ids": ep_data["support_vid_ids"],
            "query_vid_ids"  : ep_data["query_vid_ids"],
        }, f, indent=2)

    # Compute distance matrix: q_set[0] [T, d] vs s_set[class0 vids] [k, T, d]
    qi = 0
    gt_cls = 0
    mask_gt = (sup_lbl == gt_cls)
    s_gt = s_set[mask_gt]      # [k_shot, T, d] — for 1-shot, k=1

    q_tuples = q_set[qi].cpu().float()   # [T, d]
    s_tuples = s_gt.reshape(-1, s_gt.shape[-1]).cpu().float()   # [T, d]

    # Pairwise L2 distances
    T_q, T_s = q_tuples.shape[0], s_tuples.shape[0]
    dist_mat = torch.cdist(q_tuples.unsqueeze(0),
                           s_tuples.unsqueeze(0)).squeeze(0).numpy()   # [T_q, T_s]

    # Bidirectional min pairs
    q_to_s_min = dist_mat.argmin(axis=1)   # for each q-tuple, closest s-tuple
    s_to_q_min = dist_mat.argmin(axis=0)   # for each s-tuple, closest q-tuple

    bidir_pairs = set()
    for qi2, si2 in enumerate(q_to_s_min):
        if s_to_q_min[si2] == qi2:
            bidir_pairs.add((qi2, si2))

    # Top-3 mutual pairs by distance
    bidir_list = sorted(bidir_pairs, key=lambda p: dist_mat[p[0], p[1]])[:3]

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 8))
    gs  = GridSpec(2, 2, figure=fig, width_ratios=[2, 1], height_ratios=[1, 1],
                   hspace=0.4, wspace=0.3)

    # --- Heatmap ---
    ax_heat = fig.add_subplot(gs[:, 0])
    im = ax_heat.imshow(dist_mat, aspect="auto", cmap="viridis_r", origin="upper")
    plt.colorbar(im, ax=ax_heat, shrink=0.6, label="L2 distance")
    ax_heat.set_xlabel("Support tuples (frame_i, frame_j)")
    ax_heat.set_ylabel("Query tuples (frame_i, frame_j)")
    ax_heat.set_title(f"B4 Hausdorff Distance Matrix\n"
                      f"Query video vs. '{ep_data['class_names'][gt_cls]}' support",
                      fontsize=10)

    # Mark bidirectional pairs
    for (qi2, si2) in bidir_pairs:
        rect = mpatches.Rectangle(
            (si2 - 0.5, qi2 - 0.5), 1, 1,
            linewidth=1.5, edgecolor="red", facecolor="none"
        )
        ax_heat.add_patch(rect)

    # Annotate axes with top matched pair labels
    for rank, (qi2, si2) in enumerate(bidir_list):
        qf1, qf2 = tuple_pairs[qi2]
        sf1, sf2 = tuple_pairs[si2]
        ax_heat.annotate(f"#{rank+1}", xy=(si2, qi2), fontsize=6,
                         color="white", ha="center", va="center")

    ax_heat.text(0.01, -0.06,
                 "Red boxes = bidirectional min-distance pairs",
                 transform=ax_heat.transAxes, fontsize=7.5, color="red")

    # --- Top matched pairs frame thumbnails ---
    n_pairs_show = min(3, len(bidir_list))

    for rank in range(n_pairs_show):
        qi2, si2 = bidir_list[rank]
        qf1, qf2 = int(tuple_pairs[qi2][0]), int(tuple_pairs[qi2][1])
        sf1, sf2 = int(tuple_pairs[si2][0]), int(tuple_pairs[si2][1])

        ax_pair = fig.add_subplot(gs[rank if rank < 2 else 1, 1])
        ax_pair.axis("off")
        ax_pair.set_title(
            f"Pair #{rank+1}  dist={dist_mat[qi2, si2]:.2f}\n"
            f"Q-tuple=(f{qf1},f{qf2})  S-tuple=(f{sf1},f{sf2})",
            fontsize=7
        )

        # Get the 4 thumbnail frames: q_f1, q_f2, s_f1, s_f2
        q_raw = ep_data["query_raw_frames"][0]   # list of SEQ_LEN PIL images
        s_raw = ep_data["support_raw_frames"][0] # list of SEQ_LEN PIL images

        frames = [q_raw[qf1], q_raw[qf2], s_raw[sf1], s_raw[sf2]]
        labels_f = [f"Q-f{qf1}", f"Q-f{qf2}", f"S-f{sf1}", f"S-f{sf2}"]

        for fi, (frame, lbl_f) in enumerate(zip(frames, labels_f)):
            x0 = fi * 0.24 + 0.02
            ax_f = fig.add_axes([
                ax_pair.get_position().x0 + fi * (ax_pair.get_position().width / 4),
                ax_pair.get_position().y0,
                ax_pair.get_position().width / 4 * 0.88,
                ax_pair.get_position().height * 0.75,
            ])
            ax_f.imshow(frame)
            ax_f.axis("off")
            ax_f.set_title(lbl_f, fontsize=6, pad=1)

    _save(fig, "fig3_heatmap", out_dir)
    plt.close(fig)


# ── Save helper ───────────────────────────────────────────────────────────────

def _save(fig, name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, f"{name}.png")
    pdf_path = os.path.join(out_dir, f"{name}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"  Saved → {png_path}")


# ── Step 0 load check ────────────────────────────────────────────────────────

def verify_checkpoints():
    """Minimal load + dummy forward test for all three models."""
    print("\n[Step 0] Checkpoint verification ...")

    dummy_args = build_args("configs/stage1_bidirectional.yaml")
    # seq_len=8, way=5, shot=1, query_per_class=5
    ns  = WAY * SHOT
    nq  = WAY * N_QUERY
    sl  = SEQ_LEN

    for key, (cfg, ckpt) in CHECKPOINTS.items():
        print(f"\n  [{key.upper()}]  config={cfg}")
        model, args = load_model(cfg, ckpt)

        sup_imgs = torch.rand(ns * sl, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
        q_imgs   = torch.rand(nq * sl, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
        sup_lbl  = torch.arange(WAY).repeat_interleave(SHOT).to(DEVICE)

        frame_q = get_frame_features(model, q_imgs)
        frame_s = get_frame_features(model, sup_imgs)

        q_emb, s_emb, q_set, s_set = get_pre_hausdorff_embeddings(
            key, model, frame_q, frame_s)

        print(f"    q_set  : {q_set.shape}   (nq={nq}, T={q_set.shape[1]}, d={q_set.shape[2]})")
        print(f"    s_set  : {s_set.shape}   (ns={ns})")
        print(f"    q_emb  : {q_emb.shape}")
        print(f"    s_emb  : {s_emb.shape}")
        print(f"    OK ✓")

    print("\n  All checkpoints verified.\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    os.makedirs(OUT_DIR, exist_ok=True)

    # Step 0: verify checkpoints
    verify_checkpoints()

    # Load all models
    print("[Loading models ...]")
    models = {}
    for key, (cfg, ckpt) in CHECKPOINTS.items():
        model, args = load_model(cfg, ckpt)
        models[key] = (model, args)

    # Shared data loader (uses B4 args, but data is the same for all)
    loader_args = build_args("configs/stage2_true_hyrsm_b4_tuple.yaml")
    loader = EpisodeLoader(loader_args)

    # Figure 1
    metrics = fig1_tsne(models, loader, n_episodes=15)
    print("\n  [Fig 1 Metrics]")
    for key, m in metrics.items():
        print(f"    {key}: CKA={m['cka']:.3f}  Silhouette={m['silhouette']:.3f}")

    # Figure 2
    fig2_prediction_panel(models, loader, n_episodes=200)

    # Figure 3
    fig3_heatmap(models, loader, n_search=30)

    # List outputs
    print("\n[Done] Output files:")
    for f in sorted(os.listdir(OUT_DIR)):
        fpath = os.path.join(OUT_DIR, f)
        size = os.path.getsize(fpath)
        print(f"  reports/viz/{f}  ({size//1024} KB)")


if __name__ == "__main__":
    main()
