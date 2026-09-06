"""
Generate/update figs for 夜間_0906:
  fig9_spread_vs_acc.svg  — 四個點（B1/B3/B4/D1a），log-scale x 軸
  fig10_val_vs_test_selection.svg — 配對圖（5c 結果）
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

# Register and use Noto Serif CJK for Chinese labels
_cjk_fonts = fm.findSystemFonts(fontext="otf") + fm.findSystemFonts(fontext="ttf")
_noto_cjk = [f for f in _cjk_fonts if "NotoSerifCJK" in f or "NotoSansCJK" in f]
if _noto_cjk:
    _prop = fm.FontProperties(fname=_noto_cjk[0])
    matplotlib.rcParams["font.family"] = _prop.get_name()
    try:
        fm.fontManager.addfont(_noto_cjk[0])
    except Exception:
        pass

# ─── palette (matches existing figs) ────────────────────────────────────────
C_B1  = "#5b7fb5"
C_B3  = "#e07b54"
C_B4  = "#4caf88"
C_D1a = "#9b59b6"   # purple for decouple
LABEL_COLOR = "#111111"
ANNOT_COLOR = "#666666"
BG = "#ffffff"

# ─── Data ───────────────────────────────────────────────────────────────────
data = {
    "B1":  dict(acc=47.49, R_raw=4.9123,   R_cen=5.9111,  M=0.158,  c=C_B1,  m="o"),
    "B3":  dict(acc=39.01, R_raw=7.5786,   R_cen=9.2347,  M=0.001,  c=C_B3,  m="s"),
    "B4":  dict(acc=50.26, R_raw=7.9465,   R_cen=9.7134,  M=0.177,  c=C_B4,  m="^"),
    "D1a": dict(acc=34.18, R_raw=1164.27,  R_cen=1509.52, M=0.016,  c=C_D1a, m="D"),
}

# ═══════════════════════════════════════════════════════════════════════════
# Fig 9 — Spread R vs 1-shot accuracy (two panels: raw, centred)
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
fig.patch.set_facecolor(BG)

for ax, r_key, panel, title in [
    (axes[0], "R_raw", "(a)", "(a) 原始 R vs 準確率"),
    (axes[1], "R_cen", "(b)", "(b) 中心化 R vs 準確率"),
]:
    ax.set_facecolor(BG)
    ax.set_xscale("log")

    for label, d in data.items():
        ax.scatter(d[r_key], d["acc"],
                   color=d["c"], marker=d["m"], s=70, zorder=5,
                   linewidths=0.8, edgecolors=d["c"])
        # label offset: push D1a label down-left to avoid axis edge
        dx, dy = 0.05, -1.5
        if label == "D1a":
            dx, dy = -0.02, 1.5   # above the point
        if label == "B4":
            dx, dy = 0.03, -1.5
        ax.annotate(label,
                    xy=(d[r_key], d["acc"]),
                    xytext=(d[r_key] * (1 + dx), d["acc"] + dy),
                    color=d["c"], fontsize=8, ha="left", va="center",
                    fontweight="bold")

    ax.set_xlabel("R （集合分散度）", color=ANNOT_COLOR, fontsize=9)
    ax.set_ylabel("1-shot 準確率 (%)", color=ANNOT_COLOR, fontsize=9)
    ax.set_title(title, color=LABEL_COLOR, fontsize=10, pad=6)
    ax.tick_params(colors="#333333", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#dddddd")
    ax.yaxis.grid(True, color="#eeeeee", zorder=0)
    ax.set_xlim(left=3.5)
    ax.set_ylim(30, 53)

    # annotate: "R 排序 ≠ 準確率排序"
    ax.text(0.97, 0.05,
            "R 排序 ≠ 準確率排序",
            transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.5, color=ANNOT_COLOR,
            style="italic")

fig.suptitle(
    "Figure 9: Support-set spread vs 1-shot accuracy (B1/B3/B4/D1a)",
    color=LABEL_COLOR, fontsize=11, y=1.01)
fig.tight_layout()
fig.savefig("figs/fig9_spread_vs_acc.svg",
            bbox_inches="tight", facecolor=BG)
print("fig9 saved")

# ═══════════════════════════════════════════════════════════════════════════
# Fig 10 — Val-selected vs Test-selected (5c paired plot)
# ═══════════════════════════════════════════════════════════════════════════
# 5c data: test-selected (at val-best checkpoint) vs val-selected (200 ep est)
# Note: B2 test-selected in 5c was 47.10 (75k checkpoint, wrong);
#       corrected value is 48.41 (25k).  We show BOTH: 47.10 as dashed,
#       48.41 as solid correction marker.
models = ["B1", "B2", "B4"]
colors = {"B1": C_B1, "B2": C_B3, "B4": C_B4}

test_sel = {"B1": 47.49, "B2": 47.10, "B4": 50.26}   # from 5c
val_sel  = {"B1": 47.08, "B2": 48.68, "B4": 49.12}   # from 5c

fig2, ax2 = plt.subplots(figsize=(5.5, 4.5))
fig2.patch.set_facecolor(BG)
ax2.set_facecolor(BG)

x_test, x_val = 0, 1
ax2.set_xlim(-0.4, 1.4)

for m in models:
    c = colors[m]
    t, v = test_sel[m], val_sel[m]
    ax2.plot([x_test, x_val], [t, v],
             color=c, linewidth=1.4, zorder=3)
    ax2.scatter([x_test, x_val], [t, v],
                color=c, s=55, zorder=4, marker="o")
    # label at test end
    ax2.annotate(m, xy=(x_test, t),
                 xytext=(x_test - 0.06, t),
                 color=c, fontsize=9, fontweight="bold",
                 ha="right", va="center")

# Correction marker for B2 (48.41)
ax2.scatter([x_test], [48.41], color=C_B3, s=55, zorder=5,
            marker="*", linewidths=0.5, edgecolors="white")
ax2.annotate("B2 修正\n(48.41)", xy=(x_test, 48.41),
             xytext=(x_test + 0.18, 48.41),
             color=C_B3, fontsize=7, ha="left", va="center",
             arrowprops=dict(arrowstyle="-", color=C_B3, lw=0.8))

ax2.set_xticks([x_test, x_val])
ax2.set_xticklabels(["test 選點", "val 選點"], fontsize=10, color="#333333")
ax2.set_ylabel("1-shot 準確率 (%)", color=ANNOT_COLOR, fontsize=9)
ax2.set_ylim(44, 53)
ax2.tick_params(colors="#333333", labelsize=8)
for spine in ax2.spines.values():
    spine.set_edgecolor("#dddddd")
ax2.yaxis.grid(True, color="#eeeeee", zorder=0)
ax2.set_title(
    "Figure 10: 5c val 選點 vs test 選點（B1/B2/B4）",
    color=LABEL_COLOR, fontsize=10, pad=6)

# note
ax2.text(0.98, 0.05,
         "★ B2 test 修正值（47.10→48.41）",
         transform=ax2.transAxes,
         ha="right", va="bottom", fontsize=7, color=ANNOT_COLOR,
         style="italic")

fig2.tight_layout()
fig2.savefig("figs/fig10_val_vs_test_selection.svg",
             bbox_inches="tight", facecolor=BG)
print("fig10 saved")
