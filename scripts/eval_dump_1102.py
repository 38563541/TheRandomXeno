#!/usr/bin/env python3
"""
scripts/eval_dump_1102.py — 1102 Phase 3 的 3-4：逐題預測一致率。

對 L-A／L-B／L-A′ 的 25k checkpoint，各自用 run.py 的 Learner（與 run.py -m 純評估相同的建法：
--seed 42 → 同一組固定的評估 episode）照 Learner.test() 的迴圈跑一次（不改 run.py），
記每個 query 的預測類別與 logits（logsumexp 後，與 aggregate_accuracy 相同）。
自我檢查：每顆 checkpoint 的平均準確率要等於訓練中 25k 評估印出的值（logs/1102_p3_<tag>.log）。
輸出 logs/1102_p3_evaldump_<tag>.pt、logs/1102_p3_evaldump.csv（一致率、logits 平均絕對差、配對差與 CI）。

  python scripts/eval_dump_1102.py [--tags LA LB LAp] [--num_test_tasks N] [--device cuda]
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import argparse  # noqa: E402
import csv       # noqa: E402
import glob      # noqa: E402
import re        # noqa: E402

import numpy as np  # noqa: E402
import torch        # noqa: E402

import run          # noqa: E402

CK = os.path.expanduser("~/trx_data/checkpoints")
SCR = os.path.expanduser("~/Documents/work/trx/trx_data")
LOGS = os.path.join(REPO, "logs")


def find_ckpt(tag, it=25000):
    subs = sorted(glob.glob(os.path.join(CK, f"1102_p3_{tag}", "*")))
    for d in reversed(subs):
        p = os.path.join(d, f"checkpoint{it}.pt")
        if os.path.exists(p):
            return p
    return None


def logged_acc(tag):
    p = os.path.join(LOGS, f"1102_p3_{tag}.log")
    if not os.path.exists(p):
        return None
    m = re.findall(r"\{'hmdb': \{'accuracy': ([0-9.eE+-]+), 'confidence': ([0-9.eE+-]+)\}\}", open(p).read())
    return (float(m[-1][0]), float(m[-1][1])) if m else None


def dump(tag, ckpt, n_tasks, device):
    argv = ["run.py", "--dataset", "hmdb", "--split", "3", "--config", "configs/stage2_true_hyrsm_b4_tuple.yaml",
            "--method", "resnet18", "--seed", "42", "--scratch", SCR, "--amp", "off",
            "-c", os.path.join(CK, f"1102_evaldump_{tag}"), "-m", ckpt]
    if n_tasks:
        argv += ["--num_test_tasks", str(n_tasks)]
    sys.argv = argv
    L = run.Learner()
    if device == "cpu":
        L.device = torch.device("cpu")
        L.model = L.model.cpu()
    sd = torch.load(ckpt, map_location=L.device, weights_only=False)
    L.model.load_state_dict(sd["model_state_dict"])
    # 與 Learner.test() 相同的迴圈，多記逐題結果
    L.model.eval()
    preds, labels, logits_all, accs = [], [], [], []
    with torch.no_grad():
        L.vd.train = False
        it = 0
        for td in L.test_loader:
            if it >= L.args.num_test_tasks:
                break
            it += 1
            ci, ti, cl, tl, _, _ = L.prepare_task(td)
            lg = L.model(ci, cl, ti)["logits"]
            avg = torch.logsumexp(lg, dim=0)
            accs.append(L.accuracy_fn(lg, tl).item())
            preds.append(avg.argmax(-1).cpu())
            labels.append(tl.cpu())
            logits_all.append(avg.cpu())
        L.vd.train = True
    acc = np.array(accs)
    res = dict(tag=tag, ckpt=ckpt, preds=torch.stack(preds), labels=torch.stack(labels),
               logits=torch.stack(logits_all), task_acc=torch.tensor(acc),
               accuracy=float(acc.mean() * 100), confidence=float(196.0 * acc.std() / np.sqrt(len(acc))))
    torch.save(res, os.path.join(LOGS, f"1102_p3_evaldump_{tag}.pt"))
    print(f"[evaldump] {tag}: {res['accuracy']:.4f} +/- {res['confidence']:.4f}  logged={logged_acc(tag)}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["LA", "LB", "LAp"])
    ap.add_argument("--num_test_tasks", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt", nargs="*", default=None, help="（測試用）直接指定 checkpoint，順序同 --tags")
    a = ap.parse_args()
    R = {}
    for i, tag in enumerate(a.tags):
        ck = a.ckpt[i] if a.ckpt else find_ckpt(tag)
        if ck is None:
            print(f"[evaldump] {tag}: 找不到 25k checkpoint，跳過", flush=True)
            continue
        R[tag] = dump(tag, ck, a.num_test_tasks, a.device)
    rows = []
    ref = R.get(a.tags[0])
    for tag, r in R.items():
        lg = logged_acc(tag)
        row = dict(tag=tag, accuracy=r["accuracy"], confidence=r["confidence"],
                   logged_accuracy=lg[0] if lg else "", self_check=(abs(lg[0] - r["accuracy"]) < 1e-9) if lg else "",
                   n_tasks=len(r["task_acc"]), n_queries=r["preds"].numel())
        if ref is not None and tag != a.tags[0]:
            agree = (r["preds"] == ref["preds"]).float().mean().item()
            mad = (r["logits"] - ref["logits"]).abs().mean().item()
            d = (r["task_acc"] - ref["task_acc"]).numpy() * 100
            row.update(agree_vs_ref=agree, logits_mean_absdiff_vs_ref=mad, paired_diff=float(d.mean()),
                       paired_ci95=float(1.96 * d.std(ddof=1) / np.sqrt(len(d))),
                       n_tasks_differ=int((d != 0).sum()))
        rows.append(row)
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "tag", k))
    with open(os.path.join(LOGS, "1102_p3_evaldump.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r, flush=True)


if __name__ == "__main__":
    main()
