#!/usr/bin/env python3
"""
scripts/compare_ckpt_runs_1102.py — 1102 Phase 2／3 離線比對（不需要 GPU）。

  python scripts/compare_ckpt_runs_1102.py --out logs/1102_p2_resnet18 --backbone resnet18 --prefix 8 \
      --run A=<ckpt_root>:<loss_csv>:<stdout_log> --run Ap=... --run B=... --run C=...

<ckpt_root> 是 run.py 的 -c（底下的時間戳子目錄自動找最新的）。第一個 --run 是參考組（A）。
每個存檔點（checkpoint<N>.pt，N 是 run.py 的 save 編號：存的時候已經做完 N−1 個 iteration；
fully_trained.pt 標 "final"）對參考組算：
  - 權重（所有參數，不含 BN buffer）：最大絕對差、相對 L2 距離 ‖Δw‖/‖w_A‖，另拆 backbone／head
  - BN running_mean／running_var：被重算（chain[:prefix]）與沒被重算的 BN 分開，最大絕對差與相對 L2
  - num_batches_tracked（只列，不判定）；optimizer state（SGD 無 momentum → 應為空）
  - loss 曲線：逐 iteration 差的絕對值；500 iteration 移動平均的差
輸出 <out>_ckpt.csv、<out>_loss.csv、<out>_summary.json、<out>_*.png。
"""
import argparse
import csv
import glob
import json
import math
import os
import re

import numpy as np
import torch
import torchvision

COLORS = {"B": "#2a78d6", "C": "#eb6834", "Ap": "#1baf7a", "D": "#e87ba4"}
LABEL = {"Ap": "A′", "B": "B", "C": "C", "A": "A", "L-B": "L-B", "L-Ap": "L-A′"}
FLOOR = 1e-12   # log 軸上「恰好 0」畫在這條線（圖上註明）


def chain_names(backbone):
    m = getattr(torchvision.models, backbone)(weights=None)
    out = []
    for i, c in enumerate(list(m.children())[:-1]):
        if isinstance(c, torch.nn.Sequential):
            out += [f"resnet.{i}.{j}" for j in range(len(c))]
        else:
            out.append(f"resnet.{i}")
    return out


def latest_subdir(root):
    subs = sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d))
    return subs[-1] if subs else None


def load_sd(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(d, dict) and "model_state_dict" in d:
        return d["model_state_dict"], d.get("optimizer_state_dict"), d.get("iteration")
    return d, None, None


def split_keys(sd, covered_prefixes):
    params, bn_cov, bn_unc, nbt = [], [], [], []
    for k in sd:
        if k.endswith("num_batches_tracked"):
            nbt.append(k)
        elif k.endswith("running_mean") or k.endswith("running_var"):
            mod = k.rsplit(".", 1)[0]
            (bn_cov if any(mod == p or mod.startswith(p + ".") for p in covered_prefixes) else bn_unc).append(k)
        else:
            params.append(k)
    return params, bn_cov, bn_unc, nbt


def dist(sa, sb, keys):
    if not keys:
        return dict(maxabs=float("nan"), rel_l2=float("nan"), equal=None, n=0)
    mx, d2, n2, eq = 0.0, 0.0, 0.0, True
    for k in keys:
        a, b = sa[k].double(), sb[k].double()
        d = b - a
        mx = max(mx, float(d.abs().max()) if d.numel() else 0.0)
        d2 += float((d * d).sum())
        n2 += float((a * a).sum())
        eq = eq and torch.equal(sa[k], sb[k])
    return dict(maxabs=mx, rel_l2=math.sqrt(d2) / math.sqrt(n2) if n2 > 0 else float("nan"), equal=eq, n=len(keys))


def read_loss(path):
    it, lo = [], []
    if not path or not os.path.exists(path):
        return np.array([]), np.array([])
    with open(path) as f:
        for r in csv.DictReader(f):
            it.append(int(r["iteration"]))
            lo.append(float(r["loss"]))
    return np.array(it), np.array(lo)


def movavg(x, w):
    if len(x) < w:
        return np.array([])
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[w:] - c[:-w]) / w


def parse_stdout(path):
    """run.py 的 stdout：accuracy dict、[mem] RUN PEAK、[time] 中位數、memsplit MEDIAN。"""
    out = {}
    if not path or not os.path.exists(path):
        return out
    txt = open(path, errors="replace").read()
    acc = re.findall(r"\{'(\w+)': \{'accuracy': ([0-9.eE+-]+), 'confidence': ([0-9.eE+-]+)\}\}", txt)
    if acc:
        out["accuracy"], out["ci"] = float(acc[-1][1]), float(acc[-1][2])
    m = re.findall(r"\[mem\] RUN PEAK\s+alloc ([0-9.]+) GB\s+reserved ([0-9.]+) GB", txt)
    if m:
        out["peak_alloc_gib"], out["peak_res_gib"] = float(m[-1][0]), float(m[-1][1])
    m = re.findall(r"\[time\] n=(\d+)\(iter21-(\d+)\).*?total\(wall\)=([0-9.]+)ms", txt)
    if m:
        out["ms_per_iter_median"] = float(m[-1][2])
        out["time_n"] = int(m[-1][0])
    m = re.findall(r"backbone\(支\+查\)=([0-9.]+)ms", txt)
    if m:
        out["backbone_ms_median"] = float(m[-1])
    m = re.findall(r"grad_ckpt on \(bn_fix_scope=(\w+)\): adjusted momentum of\s+(.*)", txt)
    if m:
        out["bn_fix"] = f"{m[-1][0]}: {m[-1][1].strip()}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--prefix", type=int, required=True, help="被重算的 chain 前綴長度（B 的 --ckpt_prefix）")
    ap.add_argument("--run", action="append", required=True, help="TAG=ckpt_root:loss_csv[:stdout_log]")
    ap.add_argument("--ma", type=int, default=500)
    a = ap.parse_args()

    runs = []
    for spec in a.run:
        tag, rest = spec.split("=", 1)
        parts = rest.split(":")
        runs.append(dict(tag=tag, root=parts[0], loss=parts[1] if len(parts) > 1 else "",
                         stdout=parts[2] if len(parts) > 2 else ""))
    ref = runs[0]
    covered = chain_names(a.backbone)[:a.prefix]
    rows, J = [], dict(backbone=a.backbone, prefix=a.prefix, covered_chain=covered, runs={}, ckpt={}, loss={})

    def ckpts(r):
        d = latest_subdir(r["root"])
        r["dir"] = d
        if d is None:
            return {}
        out = {}
        for p in glob.glob(os.path.join(d, "checkpoint[0-9]*.pt")):
            out[int(re.search(r"checkpoint(\d+)\.pt", p).group(1))] = p
        if os.path.exists(os.path.join(d, "fully_trained.pt")):
            out["final"] = os.path.join(d, "fully_trained.pt")
        return out

    ref_ck = ckpts(ref)
    for r in runs:
        r["ck"] = ref_ck if r is ref else ckpts(r)
        J["runs"][r["tag"]] = dict(dir=r.get("dir"), n_ckpt=len(r["ck"]), stdout=parse_stdout(r["stdout"]))

    keys_order = sorted([k for k in ref_ck if k != "final"]) + (["final"] if "final" in ref_ck else [])
    for k in keys_order:
        sa, oa, _ = load_sd(ref_ck[k])
        P, BC, BU, NBT = split_keys(sa, covered)
        head = [x for x in P if not x.startswith("resnet.")]
        bbp = [x for x in P if x.startswith("resnet.")]
        for r in runs[1:]:
            if k not in r["ck"]:
                continue
            sb, ob, _ = load_sd(r["ck"][k])
            row = dict(run=r["tag"], ckpt=k)
            for lab, ks in (("w_all", P), ("w_backbone", bbp), ("w_head", head)):
                d = dist(sa, sb, ks)
                row.update({f"{lab}_maxabs": d["maxabs"], f"{lab}_rel_l2": d["rel_l2"], f"{lab}_equal": d["equal"]})
            for lab, ks in (("bn_recomp", BC), ("bn_notrecomp", BU)):
                for stat in ("running_mean", "running_var"):
                    d = dist(sa, sb, [x for x in ks if x.endswith(stat)])
                    s_ = "rm" if stat == "running_mean" else "rv"
                    row.update({f"{lab}_{s_}_maxabs": d["maxabs"], f"{lab}_{s_}_rel_l2": d["rel_l2"],
                                f"{lab}_{s_}_equal": d["equal"]})
                row[f"{lab}_n"] = len(ks) // 2
            nb_a = sorted({int(sa[x]) for x in NBT})
            nb_b = sorted({int(sb[x]) for x in NBT})
            row["nbt_ref"] = " ".join(map(str, nb_a))
            row["nbt_run"] = " ".join(map(str, nb_b))
            if oa is not None and ob is not None:
                row["opt_state_entries_ref"] = len(oa.get("state", {}))
                row["opt_state_entries_run"] = len(ob.get("state", {}))
                row["opt_lr_same"] = [g["lr"] for g in oa["param_groups"]] == [g["lr"] for g in ob["param_groups"]]
            rows.append(row)
            print(f"[ckpt {k}] {r['tag']}: w_rel={row['w_all_rel_l2']:.3e} w_max={row['w_all_maxabs']:.3e} "
                  f"bnR_rm={row['bn_recomp_rm_maxabs']:.3e} bnN_rm={row['bn_notrecomp_rm_maxabs']:.3e}", flush=True)

    fields = sorted({f for r in rows for f in r}, key=lambda f: (f not in ("run", "ckpt"), f))
    with open(a.out + "_ckpt.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    J["ckpt"] = rows

    # ---- loss ----
    ia, la = read_loss(ref["loss"])
    lrows = []
    curves = {}
    for r in runs[1:]:
        ib, lb = read_loss(r["loss"])
        n = min(len(la), len(lb))
        if n == 0:
            continue
        assert (ia[:n] == ib[:n]).all()
        d = np.abs(lb[:n] - la[:n])
        nz = np.nonzero(d)[0]
        ma_a, ma_b = movavg(la[:n], a.ma), movavg(lb[:n], a.ma)
        dma = np.abs(ma_b - ma_a) if len(ma_a) else np.array([])
        curves[r["tag"]] = (ia[:n], d, dma)
        s = dict(run=r["tag"], n=int(n), max_abs=float(d.max()), mean_abs=float(d.mean()),
                 first_nonzero_iter=int(ia[nz[0]]) if len(nz) else None, n_nonzero=int(len(nz)),
                 ma_max_abs=float(dma.max()) if len(dma) else float("nan"),
                 ma_mean_abs=float(dma.mean()) if len(dma) else float("nan"),
                 ref_loss_mean=float(la[:n].mean()))
        lrows.append(s)
        print(f"[loss] {r['tag']}: n={n} max|Δ|={s['max_abs']:.3e} mean|Δ|={s['mean_abs']:.3e} "
              f"first≠0 @ {s['first_nonzero_iter']} MA{a.ma} max|Δ|={s['ma_max_abs']:.3e}", flush=True)
    with open(a.out + "_loss.csv", "w", newline="") as f:
        if lrows:
            w = csv.DictWriter(f, fieldnames=list(lrows[0]))
            w.writeheader()
            w.writerows(lrows)
    J["loss"] = lrows
    with open(a.out + "_summary.json", "w") as f:
        json.dump(J, f, ensure_ascii=False, indent=1, default=str)

    # ---- plots ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for fam in ("DejaVu Sans",):
        plt.rcParams["font.family"] = fam
    try:
        from matplotlib import font_manager
        cjk = [f.name for f in font_manager.fontManager.ttflist if "CJK" in f.name or "Noto Sans TC" in f.name]
        if cjk:
            plt.rcParams["font.family"] = [cjk[0], "DejaVu Sans"]
    except Exception:
        pass

    def it_of(k, total):
        return total if k == "final" else k

    total = int(ia.max()) if len(ia) else 0
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    panels = (("w_all_rel_l2", "weights ‖Δw‖/‖w_A‖"), ("bn_recomp_rm_maxabs", "BN recomputed: running_mean max|Δ|"),
              ("bn_notrecomp_rm_maxabs", "BN not recomputed: running_mean max|Δ|"))
    for ax, (col, title) in zip(axes, panels):
        for r in runs[1:]:
            pts = [(it_of(x["ckpt"], total), x[col]) for x in rows if x["run"] == r["tag"]]
            if not pts:
                continue
            pts.sort()
            xs = [p[0] for p in pts]
            ys = [max(p[1], FLOOR) if not math.isnan(p[1]) else np.nan for p in pts]
            ax.plot(xs, ys, marker="o", ms=5, lw=2, color=COLORS.get(r["tag"].replace("L-", ""), "#555"),
                    label=LABEL.get(r["tag"], r["tag"]) + " vs " + LABEL.get(ref["tag"], ref["tag"]))
        ax.set_yscale("log")
        ax.axhline(FLOOR, color="#999", lw=0.8, ls=":")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("save point (iteration)")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{a.backbone}: differences vs {ref['tag']} (exact 0 drawn at {FLOOR:g})", fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out + "_ckpt.png", dpi=120)
    plt.close(fig)

    if curves:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
        for tag, (xs, d, dma) in curves.items():
            c = COLORS.get(tag.replace("L-", ""), "#555")
            lab = LABEL.get(tag, tag) + " vs " + LABEL.get(ref["tag"], ref["tag"])
            axes[0].plot(xs, np.maximum(d, FLOOR), lw=1, alpha=0.8, color=c, label=lab)
            if len(dma):
                axes[1].plot(xs[a.ma - 1:], np.maximum(dma, FLOOR), lw=2, color=c, label=lab)
        for ax, t in zip(axes, ("|Δ loss| per iteration", f"|Δ| of {a.ma}-iteration moving-average loss")):
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.axhline(FLOOR, color="#999", lw=0.8, ls=":")
            ax.set_title(t, fontsize=10)
            ax.set_xlabel("iteration")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        fig.suptitle(f"{a.backbone}: loss differences (exact 0 drawn at {FLOOR:g})", fontsize=10)
        fig.tight_layout()
        fig.savefig(a.out + "_loss.png", dpi=120)
        plt.close(fig)
        # 疊圖：原始 loss 的移動平均
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ma_ref = movavg(la, a.ma)
        if len(ma_ref):
            ax.plot(ia[a.ma - 1:], ma_ref, lw=2, color="#333", label=LABEL.get(ref["tag"], ref["tag"]))
        for r in runs[1:]:
            ib, lb = read_loss(r["loss"])
            m = movavg(lb, a.ma)
            if len(m):
                ax.plot(ib[a.ma - 1:], m, lw=1.5, color=COLORS.get(r["tag"].replace("L-", ""), "#555"),
                        label=LABEL.get(r["tag"], r["tag"]), alpha=0.85)
        ax.set_title(f"{a.backbone}: training loss, {a.ma}-iteration moving average", fontsize=10)
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(a.out + "_loss_overlay.png", dpi=120)
        plt.close(fig)
    print(f"[compare] wrote {a.out}_*.csv/json/png")


if __name__ == "__main__":
    main()
