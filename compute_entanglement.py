"""
Stage B: Entanglement measurement for fig8_entanglement.svg

For each episode, each class pair (n, o), each tuple slot m:
    s̄_{n,m} = mean across k shots of S_{n,k,m}   [d]
    E_{n,o,m} = cosine_similarity(s̄_{n,m}, s̄_{o,m})  scalar

Uses existing visualize.py infrastructure (EpisodeLoader, load_model,
get_pre_hausdorff_embeddings) without modifying any file.

Models: B1 (s1/abl_bidir), B3 (b3_inter), B4 (true_hyrsm)
Episodes: 15 (same seed as fig1: SEED=42)
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

# ── reuse visualize.py constants & utilities ──────────────────────────────────
from visualize import (
    CHECKPOINTS, MODEL_LABELS, DEVICE, SEED, OUT_DIR,
    SEQ_LEN, WAY, SHOT, N_QUERY,
    load_model, get_frame_features, get_pre_hausdorff_embeddings,
    EpisodeLoader, build_args,
)

FIGS_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
N_EPISODES = 15    # same as fig1
T          = 28    # C(8,2) tuple slots
PALETTE    = "#5b7fb5"

os.makedirs(FIGS_DIR, exist_ok=True)

# ── Reproducibility ───────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ── Load models ───────────────────────────────────────────────────────────────
print("Loading models ...")
models_dict = {}
for key, (cfg_path, ckpt_path) in CHECKPOINTS.items():
    if not os.path.exists(ckpt_path):
        print(f"  [SKIP] {key}: checkpoint not found at {ckpt_path}")
        continue
    print(f"  {key} ...")
    model, args = load_model(cfg_path, ckpt_path, device=DEVICE)
    models_dict[key] = (model, args)

if not models_dict:
    print("[FATAL] No models loaded — cannot compute entanglement")
    sys.exit(1)


# ── Episode loader ────────────────────────────────────────────────────────────
# Use B4 args for loader (dataset path); data is same for all models
first_key  = list(models_dict.keys())[0]
loader_args = build_args(CHECKPOINTS[first_key][0])
loader = EpisodeLoader(loader_args)

print(f"\nSampling {N_EPISODES} episodes (seed={SEED}) ...")
episodes = []
for _ in range(N_EPISODES):
    episodes.append(loader.get_episode())
print(f"  Done.")


# ── Compute E per model ───────────────────────────────────────────────────────
# E_all[key] = list of (E_{n,o,m}) flat arrays across all episodes & pairs
E_all     = {key: [] for key in models_dict}
# slot_E[key] = [N_EPISODES × C(WAY,2), T] for heatmap
slot_E_all = {key: [] for key in models_dict}

# For heatmap: pick 10 class pairs from first N_EPISODES episodes (all pairs ×
# first episodes until we have 10). Actually use all pairs, clip to 10 for display.
HEATMAP_PAIRS = 10

print("\nComputing entanglement per model ...")
for key, (model, args) in models_dict.items():
    print(f"  {key} ...")
    model.eval()

    for ep_idx, ep_data in enumerate(episodes):
        sup_t   = ep_data["support_tensors"].to(DEVICE)  # [ns*seq_len, 3, H, W]
        q_t     = ep_data["query_tensors"].to(DEVICE)

        with torch.no_grad():
            frame_q = get_frame_features(model, q_t)   # [nq, seq_len, d]
            frame_s = get_frame_features(model, sup_t) # [ns, seq_len, d]

            _, _, _, s_set = get_pre_hausdorff_embeddings(
                key, model, frame_q, frame_s
            )  # s_set: [ns, T, d]

        s_set = s_set.cpu().float()   # [ns, T, d]

        labels    = ep_data["support_labels"]  # [ns] integers 0..4 (1-shot → ns=5)
        classes   = torch.unique(labels).tolist()

        # Per-class per-slot mean: s̄[class_idx, slot, dim]
        n_classes = len(classes)
        d         = s_set.shape[-1]
        s_mean    = torch.zeros(n_classes, T, d)   # [N, T, d]

        for ci, c in enumerate(classes):
            mask        = (labels == c)
            s_mean[ci]  = s_set[mask].mean(0)     # [T, d]  (mean over k shots)

        # E_{n,o,m} for all class pairs and slots
        for ni in range(n_classes):
            for oi in range(ni + 1, n_classes):
                # [T] cosine similarity per slot
                e_m = F.cosine_similarity(
                    s_mean[ni],   # [T, d]
                    s_mean[oi],   # [T, d]
                    dim=-1
                )  # [T]
                E_all[key].append(e_m.numpy())
                slot_E_all[key].append(e_m.numpy())  # [T]


# ── Aggregate stats ───────────────────────────────────────────────────────────
print("\n=== Entanglement Stats ===")
stats = {}
for key in models_dict:
    e_flat = np.concatenate(E_all[key])   # all E values for this model
    mean_e    = float(np.mean(e_flat))
    p5, p95   = float(np.percentile(e_flat, 5)), float(np.percentile(e_flat, 95))
    frac_gt05 = float(np.mean(e_flat > 0.5))
    stats[key] = {
        "mean": mean_e, "p5": p5, "p95": p95, "frac_gt05": frac_gt05,
        "e_flat": e_flat,
        "slot_matrix": np.stack(slot_E_all[key]),  # [n_pairs_total, T]
    }
    print(f"  {key:4s}: mean E={mean_e:.3f}  5%={p5:.3f}  95%={p95:.3f}  "
          f"E>0.5 slots={frac_gt05:.1%}")


# ── Judgment ─────────────────────────────────────────────────────────────────
# Use B1 (s1/骨幹) for go/no-go
judge_key  = "s1"
if judge_key not in stats:
    # fall back to first available
    judge_key = list(stats.keys())[0]

mean_e_b1  = stats[judge_key]["mean"]
slot_mat   = stats[judge_key]["slot_matrix"]   # [n_pairs, T]
slot_means = slot_mat.mean(axis=0)             # [T] — per-slot mean across pairs/episodes
has_structure = (slot_means.max() - slot_means.min()) > 0.10  # visible variation across slots

if mean_e_b1 >= 0.30:
    verdict = "GO"
    reason  = f"mean E={mean_e_b1:.3f} >= 0.30"
elif mean_e_b1 >= 0.15 and has_structure:
    verdict = "GO"
    reason  = f"mean E={mean_e_b1:.3f} in [0.15,0.30) but slot structure detected (range={slot_means.max()-slot_means.min():.3f})"
elif mean_e_b1 >= 0.15:
    verdict = "NO-GO"
    reason  = f"mean E={mean_e_b1:.3f} in [0.15,0.30) but no slot structure"
else:
    verdict = "NO-GO"
    reason  = f"mean E={mean_e_b1:.3f} < 0.15"

print(f"\n=== Stage B Judgment: {verdict} ===")
print(f"  Reason: {reason}")


# ── Plot fig8_entanglement.svg ────────────────────────────────────────────────
print("\nPlotting fig8_entanglement.svg ...")

SLOT_LABELS = [f"({i},{j})" for i in range(8) for j in range(i+1, 8)]  # 28 labels

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("white")

# ── Panel (a): Heatmap of E for B1, first HEATMAP_PAIRS pairs ────────────────
ax_heat = axes[0]

if "s1" in stats:
    heat_data = stats["s1"]["slot_matrix"][:HEATMAP_PAIRS]  # [10, 28]
else:
    heat_data = list(stats.values())[0]["slot_matrix"][:HEATMAP_PAIRS]

im = ax_heat.imshow(
    heat_data, aspect="auto", cmap="Blues",
    vmin=0.0, vmax=max(0.6, heat_data.max()),
    interpolation="nearest"
)
ax_heat.set_xticks(range(T))
ax_heat.set_xticklabels(SLOT_LABELS, rotation=90, fontsize=5)
ax_heat.set_yticks(range(len(heat_data)))
ax_heat.set_yticklabels([f"pair {i+1}" for i in range(len(heat_data))], fontsize=8)
ax_heat.set_xlabel("Tuple slot m  (frame pair)", fontsize=9, color="#666")
ax_heat.set_ylabel("Class pair (n, o)", fontsize=9, color="#666")
ax_heat.set_title("(a) Slot-level entanglement E — Backbone B1",
                  fontsize=10, color="#111", pad=8)
cbar = fig.colorbar(im, ax=ax_heat, shrink=0.8)
cbar.ax.set_ylabel("cosine similarity", fontsize=8, color="#666")
cbar.ax.yaxis.set_tick_params(labelcolor="#666")

# ── Panel (b): Box-plot of E distribution per model ──────────────────────────
ax_box = axes[1]

model_order = [k for k in ["s1", "b3", "b4"] if k in stats]
labels_box  = [MODEL_LABELS.get(k, k) for k in model_order]
data_box    = [stats[k]["e_flat"] for k in model_order]

colors = ["#5b7fb5", "#e07b54", "#4caf88"][:len(model_order)]
bp = ax_box.boxplot(
    data_box, labels=labels_box,
    patch_artist=True,
    medianprops=dict(color="#111", linewidth=1.5),
    whiskerprops=dict(color="#999"),
    capprops=dict(color="#999"),
    flierprops=dict(marker="o", markersize=2, markerfacecolor="#bbb",
                   markeredgecolor="#bbb", alpha=0.4),
    notch=False,
)
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax_box.axhline(0.30, color="#c44", linewidth=0.8, linestyle="--", label="GO threshold (0.30)")
ax_box.axhline(0.15, color="#c84", linewidth=0.8, linestyle=":",  label="lower bound (0.15)")
ax_box.set_ylabel("Cosine similarity E", fontsize=9, color="#666")
ax_box.set_title("(b) Entanglement distribution per model",
                 fontsize=10, color="#111", pad=8)
ax_box.legend(fontsize=7, loc="upper right")
ax_box.tick_params(axis="x", labelsize=8, labelcolor="#333")
ax_box.tick_params(axis="y", labelsize=8, labelcolor="#333")
ax_box.set_facecolor("white")
for sp in ax_box.spines.values():
    sp.set_color("#ddd")

# Annotate mean values
for xi, k in enumerate(model_order, 1):
    mean_v = stats[k]["mean"]
    ax_box.text(xi, mean_v + 0.02, f"μ={mean_v:.2f}",
                ha="center", va="bottom", fontsize=7, color="#333")

fig.suptitle(
    f"Figure 8: Support-set entanglement across tuple slots\n"
    f"B1 mean E={stats.get('s1', list(stats.values())[0])['mean']:.3f}  "
    f"→  {verdict} ({reason})",
    fontsize=11, color="#111", y=1.01
)
fig.tight_layout()

out_svg = os.path.join(FIGS_DIR, "fig8_entanglement.svg")
fig.savefig(out_svg, bbox_inches="tight", dpi=150)
print(f"  Saved: {out_svg}")
out_png = os.path.join(FIGS_DIR, "fig8_entanglement.png")
fig.savefig(out_png, bbox_inches="tight", dpi=150)
print(f"  Saved: {out_png}")
plt.close(fig)


# ── Print final summary for report ────────────────────────────────────────────
print("\n=== STAGE B SUMMARY ===")
for key in model_order:
    s = stats[key]
    label = MODEL_LABELS.get(key, key)
    print(f"  {label}:")
    print(f"    mean E = {s['mean']:.4f},  5%={s['p5']:.4f},  95%={s['p95']:.4f}")
    print(f"    E>0.5 fraction: {s['frac_gt05']:.1%}")
print(f"\n  Stage B verdict: {verdict}")
print(f"  Reason: {reason}")
print(f"  Slot structure (B1 slot range): {slot_means.max()-slot_means.min():.4f}")
