#!/usr/bin/env python3
"""
scripts/plot_tradeoff_1005.py — figs/lst_tradeoff_rn50.png：RN50 的時間 × 準確率（點旁標 alloc GB）。

x = ms/iter（logs/lst_cost_1005.csv，fp32），y = 50k accuracy（summarize_1005.get），
B1 與 B4 兩個 panel（共用 y 軸範圍，同一個量）。E0 全微調不開 checkpoint 會 OOM，沒有點，寫在圖說。
顏色依 arm 固定（不依名次）；每個點都直接標字＋不同 marker，不靠顏色辨識。
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from summarize_1005 import get  # noqa: E402

COLOR = {"E": "#2a78d6", "A": "#eb6834", "B": "#1baf7a", "C": "#eda100", "D": "#e87ba4"}
MARK = {"E": "s", "A": "o", "B": "D", "C": "^", "D": "v"}
NAME = {"E": "E1 全微調+ckpt", "A": "A 凍結", "B": "B top adapter", "C": "C LST", "D": "D 只解凍 layer4"}
COST_TAG = {("E", "B4"): "RN50_E1", ("E", "B1"): "RN50_E1_B1"}


def main():
    cost = {r["tag"]: r for r in csv.DictReader(open(os.path.join(REPO, "logs", "lst_cost_1005.csv")))}
    from matplotlib import font_manager
    for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
        if os.path.exists(f):
            font_manager.fontManager.addfont(f)
    plt.rcParams.update({"font.family": ["Noto Sans CJK JP", "DejaVu Sans"],
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    pts_all = []
    for ax, head in zip(axes, ("B1", "B4")):
        pts = []
        for arm in ("E", "A", "B", "C", "D"):
            acc = get("rn50", arm, head)
            tag = COST_TAG.get((arm, head), f"RN50_{arm}" + ("_B1" if head == "B1" else ""))
            r = cost.get(tag)
            if not acc or acc[1] is None or not r or r["status"] != "ok":
                continue
            x, y, gb = float(r["ms_iter"]), acc[1], float(r["alloc_peak_gb"])
            pts.append((arm, x, y, gb))
            ax.scatter([x], [y], s=70, marker=MARK[arm], color=COLOR[arm], edgecolor="white",
                       linewidth=2, zorder=3, label=NAME[arm])
            ax.annotate(f"{arm}  {y:.2f}%\n{gb:.1f} GB", (x, y), xytext=(8, 6), textcoords="offset points",
                        fontsize=9, color="#333333")
        pts_all += pts
        ax.set_title(f"RN50 · {head} 頭", fontsize=11, loc="left")
        ax.set_xlabel("ms / iter（fp32，越左越快）")
        ax.grid(axis="both", color="#e6e6e6", linewidth=0.8)
        ax.set_axisbelow(True)
        if not pts:
            ax.text(0.5, 0.5, "尚無 50k 結果", transform=ax.transAxes, ha="center", color="#888888")
    axes[0].set_ylabel("50k test accuracy (%)")
    if pts_all:
        xs = [p[1] for p in pts_all]
        for ax in axes:
            ax.set_xlim(min(xs) * 0.85, max(xs) * 1.12)
    h, l = [], []
    for ax in axes:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh)
                l.append(ll)
    if h:
        fig.legend(h, l, loc="upper center", ncol=len(l), frameon=False, fontsize=9)
    fig.text(0.01, 0.01, "E0（全微調、不開 checkpoint）fp32 在 12 GB 上 OOM，無資料點。點旁數字：50k accuracy／峰值 alloc。"
             "單一 seed（42），未重調 lr。", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    os.makedirs(os.path.join(REPO, "figs"), exist_ok=True)
    out = os.path.join(REPO, "figs", "lst_tradeoff_rn50.png")
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main()
