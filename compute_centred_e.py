"""
Stage 1 (2026-09-05): Centred-E diagnostic
============================================
Computes raw AND centred versions of E_diff and E_same for B1 (abl_bidir).

E_diff = cosine_similarity(class_n_support_mean_m, class_o_support_mean_m)   n≠o
E_same = cosine_similarity(class_n_support_mean_m, class_n_query_mean_m)     same-class

Centred = subtract per-episode per-slot mean over all support videos first:
    μ_m = mean over all ns support videos at slot m
    ŝ = s - μ_m   (and same μ_m applied to query)

Judgment (centred E_diff):
  centred E_diff mean < 0.5  AND  (E_same - E_diff) > 0.10  → GO
  centred E_diff mean >= 0.90                               → NO-GO
  otherwise                                                 → NO-GO
"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from visualize import (
    CHECKPOINTS, MODEL_LABELS, DEVICE, SEED, OUT_DIR,
    SEQ_LEN, WAY, SHOT, N_QUERY,
    load_model, get_frame_features, get_pre_hausdorff_embeddings,
    EpisodeLoader, build_args,
)

FIGS_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
N_EPISODES = 15
T          = 28    # C(8,2) tuple slots
os.makedirs(FIGS_DIR, exist_ok=True)

# ── Reproducibility ────────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ── Load B1 (abl_bidir) ───────────────────────────────────────────────────────
print("Loading B1 model (abl_bidir) ...")
# Use 75k checkpoint (best known)
cfg_path  = "configs/stage1_bidirectional.yaml"
ckpt_path = os.path.expanduser(
    "~/trx_data/checkpoints/abl_bidir_hmdb3/"
    "hmdb_split3_20260407_220041/checkpoint75000.pt"
)
model, args = load_model(cfg_path, ckpt_path, device=DEVICE)
model.eval()

# ── Episode loader ─────────────────────────────────────────────────────────────
loader_args = build_args(cfg_path)
loader = EpisodeLoader(loader_args)
print(f"Sampling {N_EPISODES} episodes (seed={SEED}) ...")
episodes = []
for _ in range(N_EPISODES):
    episodes.append(loader.get_episode())
print("  Done.")

# ── Storage ────────────────────────────────────────────────────────────────────
# Each entry is a flat array of scalars (one per slot per class-pair/class)
raw_diff_all  = []
raw_same_all  = []
cen_diff_all  = []
cen_same_all  = []

# For heatmap: [n_pairs_total, T]  —  raw and centred
raw_diff_mat  = []   # rows = class pairs across all episodes
cen_diff_mat  = []

print("\nComputing E_diff and E_same (raw + centred) ...")
for ep_idx, ep_data in enumerate(episodes):
    sup_t  = ep_data["support_tensors"].to(DEVICE)   # [ns*seq_len, 3, H, W]
    q_t    = ep_data["query_tensors"].to(DEVICE)     # [nq*seq_len, 3, H, W]
    s_lbl  = ep_data["support_labels"]               # [ns]  int 0..4
    q_lbl  = ep_data["query_labels"]                 # [nq]  int 0..4

    with torch.no_grad():
        frame_q = get_frame_features(model, q_t)    # [nq, seq_len, d]
        frame_s = get_frame_features(model, sup_t)  # [ns, seq_len, d]

        _, _, q_set, s_set = get_pre_hausdorff_embeddings(
            "s1", model, frame_q, frame_s
        )   # q_set [nq, T, d], s_set [ns, T, d]

    q_set = q_set.cpu().float()   # [nq, T, d]
    s_set = s_set.cpu().float()   # [ns, T, d]
    ns, _, d = s_set.shape

    # ── Common per-slot mean over ALL support videos ─────────────────────────
    mu = s_set.mean(dim=0)        # [T, d]

    # centred versions
    s_cen = s_set - mu.unsqueeze(0)   # [ns, T, d]
    q_cen = q_set - mu.unsqueeze(0)   # [nq, T, d]

    classes = torch.unique(s_lbl).tolist()

    # per-class per-slot means (raw and centred)
    s_mean_raw = {}   # class → [T, d]
    s_mean_cen = {}
    q_mean_raw = {}   # class → [T, d]   (mean query per class)
    q_mean_cen = {}

    for c in classes:
        s_mask = (s_lbl == c)
        q_mask = (q_lbl == c)
        s_mean_raw[c] = s_set[s_mask].mean(0)    # [T, d]
        s_mean_cen[c] = s_cen[s_mask].mean(0)
        if q_mask.any():
            q_mean_raw[c] = q_set[q_mask].mean(0)
            q_mean_cen[c] = q_cen[q_mask].mean(0)

    # ── E_diff: inter-class cosine similarity (n ≠ o) ────────────────────────
    for ni, cn in enumerate(classes):
        for oi, co in enumerate(classes):
            if co <= cn:
                continue
            # raw
            e_raw = F.cosine_similarity(
                s_mean_raw[cn], s_mean_raw[co], dim=-1   # [T]
            )
            raw_diff_all.extend(e_raw.tolist())
            raw_diff_mat.append(e_raw.numpy())

            # centred
            e_cen = F.cosine_similarity(
                s_mean_cen[cn], s_mean_cen[co], dim=-1
            )
            cen_diff_all.extend(e_cen.tolist())
            cen_diff_mat.append(e_cen.numpy())

    # ── E_same: same-class support-to-query cosine similarity ─────────────────
    for c in classes:
        if c not in q_mean_raw:
            continue
        # raw
        e_raw = F.cosine_similarity(
            s_mean_raw[c], q_mean_raw[c], dim=-1   # [T]
        )
        raw_same_all.extend(e_raw.tolist())

        # centred
        e_cen = F.cosine_similarity(
            s_mean_cen[c], q_mean_cen[c], dim=-1
        )
        cen_same_all.extend(e_cen.tolist())

# ── Stats ──────────────────────────────────────────────────────────────────────
def stats(arr):
    a = np.array(arr)
    return dict(
        mean=float(np.mean(a)),
        p5=float(np.percentile(a, 5)),
        p95=float(np.percentile(a, 95)),
        std=float(np.std(a)),
    )

rd = stats(raw_diff_all)
rs = stats(raw_same_all)
cd = stats(cen_diff_all)
cs = stats(cen_same_all)

raw_diff_mat = np.stack(raw_diff_mat)   # [n_pairs_total, T]
cen_diff_mat = np.stack(cen_diff_mat)

print("\n=== E Stats ===")
print(f"{'':20s}  {'raw mean':>9}  {'raw 5–95%':>14}  {'cen mean':>9}  {'cen 5–95%':>14}")
print(f"{'E_diff':20s}  {rd['mean']:>9.4f}  [{rd['p5']:>6.4f},{rd['p95']:>6.4f}]  {cd['mean']:>9.4f}  [{cd['p5']:>6.4f},{cd['p95']:>6.4f}]")
print(f"{'E_same':20s}  {rs['mean']:>9.4f}  [{rs['p5']:>6.4f},{rs['p95']:>6.4f}]  {cs['mean']:>9.4f}  [{cs['p5']:>6.4f},{cs['p95']:>6.4f}]")
print(f"{'same−diff':20s}  {rs['mean']-rd['mean']:>9.4f}  {'':14s}  {cs['mean']-cd['mean']:>9.4f}")

# ── Judgment ───────────────────────────────────────────────────────────────────
cen_diff_mean = cd['mean']
cen_gap       = cs['mean'] - cd['mean']

if cen_diff_mean >= 0.90:
    verdict = "NO-GO"
    reason  = f"centred E_diff mean={cen_diff_mean:.4f} >= 0.90 (即使中心化後仍幾乎共線)"
elif cen_diff_mean < 0.5 and cen_gap > 0.10:
    verdict = "GO"
    reason  = (f"centred E_diff mean={cen_diff_mean:.4f} < 0.5  AND  "
               f"(E_same−E_diff)={cen_gap:.4f} > 0.10")
else:
    verdict = "NO-GO"
    reason  = (f"centred E_diff mean={cen_diff_mean:.4f}, gap={cen_gap:.4f}  "
               f"—— 條件不滿足（E_diff<0.5且gap>0.10）")

print(f"\n=== Stage 1 Judgment: {verdict} ===")
print(f"  Reason: {reason}")

# ── Plot 3-panel fig8 ──────────────────────────────────────────────────────────
print("\nPlotting fig8 (3 panels) ...")

HEATMAP_PAIRS = min(15, len(raw_diff_mat))  # up to 15 rows
SLOT_LABELS = [f"({i},{j})" for i in range(8) for j in range(i+1, 8)]

PALETTE_BLUE  = "#5b7fb5"
PALETTE_TEAL  = "#4caf88"
PALETTE_ORG   = "#e07b54"
GREY          = "#999999"

fig = plt.figure(figsize=(18, 6))
fig.patch.set_facecolor("white")

# ── (a): Raw E_diff heatmap ─────────────────────────────────────────────────
ax_a = fig.add_subplot(1, 3, 1)
heat_raw = raw_diff_mat[:HEATMAP_PAIRS]
vmax_raw = max(0.7, float(np.percentile(heat_raw, 95)))
im_a = ax_a.imshow(heat_raw, aspect="auto", cmap="Blues",
                   vmin=-0.2, vmax=vmax_raw, interpolation="nearest")
ax_a.set_xticks(range(0, T, 4))
ax_a.set_xticklabels(SLOT_LABELS[::4], rotation=90, fontsize=5)
ax_a.set_yticks(range(len(heat_raw)))
ax_a.set_yticklabels([f"pair {i+1}" for i in range(len(heat_raw))], fontsize=7)
ax_a.set_xlabel("Tuple slot m", fontsize=9, color="#666")
ax_a.set_ylabel("Class pair", fontsize=9, color="#666")
ax_a.set_title("(a) Raw E_diff  (B1 骨幹)", fontsize=10, color="#111", pad=6)
cb_a = fig.colorbar(im_a, ax=ax_a, shrink=0.8)
cb_a.ax.set_ylabel("cos-sim (raw)", fontsize=7, color="#666")
cb_a.ax.yaxis.set_tick_params(labelcolor="#666")

# ── (b): Centred E_diff heatmap ─────────────────────────────────────────────
ax_b = fig.add_subplot(1, 3, 2)
heat_cen = cen_diff_mat[:HEATMAP_PAIRS]
vlim = max(0.5, float(np.percentile(np.abs(heat_cen), 95)))
im_b = ax_b.imshow(heat_cen, aspect="auto", cmap="RdBu_r",
                   vmin=-vlim, vmax=vlim, interpolation="nearest")
ax_b.set_xticks(range(0, T, 4))
ax_b.set_xticklabels(SLOT_LABELS[::4], rotation=90, fontsize=5)
ax_b.set_yticks(range(len(heat_cen)))
ax_b.set_yticklabels([f"pair {i+1}" for i in range(len(heat_cen))], fontsize=7)
ax_b.set_xlabel("Tuple slot m", fontsize=9, color="#666")
ax_b.set_title("(b) Centred E_diff  (B1 骨幹)\n← 扣掉共同均值後的類別間相似度",
               fontsize=10, color="#111", pad=6)
cb_b = fig.colorbar(im_b, ax=ax_b, shrink=0.8)
cb_b.ax.set_ylabel("cos-sim (centred)", fontsize=7, color="#666")
cb_b.ax.yaxis.set_tick_params(labelcolor="#666")

# ── (c): Box-plot of 4 distributions ────────────────────────────────────────
ax_c = fig.add_subplot(1, 3, 3)
data_box = [raw_diff_all, raw_same_all, cen_diff_all, cen_same_all]
labels_box = ["E_diff\n(raw)", "E_same\n(raw)", "E_diff\n(cen)", "E_same\n(cen)"]
colors_box = [PALETTE_BLUE, PALETTE_TEAL, PALETTE_ORG, "#9b59b6"]
bp = ax_c.boxplot(
    data_box, labels=labels_box,
    patch_artist=True,
    medianprops=dict(color="#111", linewidth=1.5),
    whiskerprops=dict(color=GREY),
    capprops=dict(color=GREY),
    flierprops=dict(marker="o", markersize=2, markerfacecolor="#ccc",
                   markeredgecolor="#ccc", alpha=0.4),
)
for patch, color in zip(bp["boxes"], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Reference lines
ax_c.axhline(0.5,  color="#c44", linewidth=0.9, linestyle="--",
             label="cen E_diff < 0.5 (GO cond.)")
ax_c.axhline(0.0,  color="#888", linewidth=0.6, linestyle=":")
ax_c.set_ylabel("Cosine similarity", fontsize=9, color="#666")
ax_c.set_title("(c) 四個量的分布\n(原始/中心化 × diff/same)",
               fontsize=10, color="#111", pad=6)
ax_c.legend(fontsize=7, loc="upper right")
ax_c.tick_params(axis="x", labelsize=8, labelcolor="#333")
ax_c.tick_params(axis="y", labelsize=8, labelcolor="#333")
ax_c.set_facecolor("white")
for sp in ax_c.spines.values():
    sp.set_color("#ddd")

# Annotate means
for xi, (arr, label) in enumerate(zip(data_box, labels_box), 1):
    mv = float(np.mean(arr))
    ax_c.text(xi, mv + 0.015, f"μ={mv:.2f}",
              ha="center", va="bottom", fontsize=7, color="#333")

# Suptitle with verdict
fig.suptitle(
    f"Figure 8 (updated): Centred vs raw entanglement — B1 骨幹\n"
    f"centred E_diff μ={cd['mean']:.4f}, gap(same−diff)={cen_gap:.4f}  →  {verdict}\n"
    f"({reason})",
    fontsize=10, color="#111", y=1.02
)
fig.tight_layout()

out_svg = os.path.join(FIGS_DIR, "fig8_entanglement.svg")
fig.savefig(out_svg, bbox_inches="tight", dpi=150)
print(f"  Saved: {out_svg}")
out_png = os.path.join(FIGS_DIR, "fig8_entanglement.png")
fig.savefig(out_png, bbox_inches="tight", dpi=150)
print(f"  Saved: {out_png}")
plt.close(fig)

# ── Print machine-readable summary for report ──────────────────────────────────
print("\n=== STAGE 1 SUMMARY (for report) ===")
print(f"raw  E_diff: mean={rd['mean']:.4f}  5%={rd['p5']:.4f}  95%={rd['p95']:.4f}")
print(f"raw  E_same: mean={rs['mean']:.4f}  5%={rs['p5']:.4f}  95%={rs['p95']:.4f}")
print(f"raw  same−diff: {rs['mean']-rd['mean']:.4f}")
print(f"cen  E_diff: mean={cd['mean']:.4f}  5%={cd['p5']:.4f}  95%={cd['p95']:.4f}")
print(f"cen  E_same: mean={cs['mean']:.4f}  5%={cs['p5']:.4f}  95%={cs['p95']:.4f}")
print(f"cen  same−diff: {cen_gap:.4f}")
print(f"VERDICT: {verdict}")
print(f"REASON:  {reason}")
