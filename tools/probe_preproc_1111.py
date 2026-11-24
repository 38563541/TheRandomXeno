#!/usr/bin/env python3
"""
probe_preproc_1111.py — 1111b 乙：前處理探針（凍結 RN50、最近原型、不訓練）。

量 HMDB 前處理（畫面比例、40 幀截斷、ImageNet normalize）對凍結特徵 few-shot 準確率的影響。
不改 video_reader.py、不改 make_hmdb_trx_zip.py、不動既有 zip。

版本：
  A  既有 zip（trx_data/video_datasets/data/hmdb51_256q5.zip；推得 scale=256:256、fps=10、max_frames=40）
  B  ffmpeg -vf "scale=256:256,fps=10"，不截斷         → ~/trx_data/video_datasets/data/hmdb51_probe_B/<class>/<video>/%08d.jpg
  C  ffmpeg -vf "scale=短邊 256 保持比例,fps=10"，不截斷 → ~/trx_data/video_datasets/data/hmdb51_probe_C/...
  JPEG 品質 -q:v 5，與 make_hmdb_trx_zip.py 的 extract_frames_ffmpeg 相同。

子命令（依序）：
  extract  --version B|C [--fps 10] [--workers 8]   只抽 testlist03 的 1292 支
  feats                                             6 組特徵 → ~/trx_data/probe_1111b/feat_<V>_<norm>.npz
  episodes                                          seed 20001、10000 個 5-way 1-shot、每類 1 query → episodes.json
  eval                                              6 組 → results_<V>_<norm>.npz
  analyze                                           印出乙3／乙4 的全部數字

取幀規則照抄 video_reader.py:296–317（get_seq 的評估分支：start=1、end=n−2；end−start<8 改用 0～n−1；
np.linspace 取 8 個轉 int；n==8 直接 0..7）。get_seq 是綁在 VideoDataset 上的方法、會從 split 隨機抽影片，
不能單獨 import，所以照抄成 eval_frame_indices()。
轉換：videotransforms 的 Resize(256) → CenterCrop(224)（直接 import repo 的 videotransforms，與 video_reader.py:91–99 相同），
再逐幀 torchvision ToTensor（video_reader.py:63、:328）。normalize 開 = 再加 ImageNet mean/std。
影片名稱正規化同 video_reader.py:251–253：空白→底線、轉小寫、去副檔名。少於 8 幀的影片丟掉（video_reader.py:171）。
CI = 1.96 × std(每題 accuracy) / √N，std 為 np.std（ddof=0），與 run.py:1097 相同。
"""
import argparse
import collections
import io
import json
import math
import os
import random
import subprocess
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

ZIP_A = os.path.join(REPO, "trx_data/video_datasets/data/hmdb51_256q5.zip")
TESTLIST = os.path.join(REPO, "trx_data/video_datasets/splits/hmdb_ARN/testlist03.txt")
RAW = os.path.expanduser("~/Documents/mpl-fsar-repro/data/hmdb51/_data/videos")
D = os.path.expanduser("~/trx_data/video_datasets/data")
OUT = os.path.expanduser("~/trx_data/probe_1111b")
VERSIONS = ("A", "B", "C")
NORMS = ("off", "on")
SEQ_LEN = 8
EP_SEED = 20001
N_EP = 10000
WAY = 5
FOCUS = ("talk", "smoke", "pour")


def norm_name(s):
    return os.path.splitext(os.path.split(s.replace(" ", "_").lower())[1])[0]


def testlist():
    out = []
    for l in open(TESTLIST):
        l = l.strip()
        if l:
            c, v = l.split("/", 1)
            out.append((c, v))
    return out


# ── extract ────────────────────────────────────────────────────────────────
def _ffmpeg_one(job):
    src, dst, vf = job
    os.makedirs(dst, exist_ok=True)
    if any(f.endswith(".jpg") for f in os.listdir(dst)):
        return dst, "exists"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", src, "-vf", vf, "-q:v", "5",
           os.path.join(dst, "%08d.jpg")]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    return dst, ("ok" if p.returncode == 0 else "FAIL " + p.stderr[-500:])


def cmd_extract(a):
    if a.version == "B":
        vf = "scale=256:256"
    else:
        vf = "scale='if(gt(iw,ih),-2,256)':'if(gt(iw,ih),256,-2)'"
    if a.fps:
        vf += f",fps={a.fps}"
    root = os.path.join(D, f"hmdb51_probe_{a.version}")
    raw = {}
    for c in os.listdir(RAW):
        for f in os.listdir(os.path.join(RAW, c)):
            if f.endswith(".avi"):
                raw[(c, norm_name(f))] = os.path.join(RAW, c, f)
    jobs, missing = [], []
    for c, v in testlist():
        src = raw.get((c, norm_name(v)))
        if src is None:
            missing.append(f"{c}/{v}")
            continue
        jobs.append((src, os.path.join(root, c, os.path.splitext(os.path.basename(src))[0]), vf))
    print(f"[extract {a.version}] vf={vf}  jobs={len(jobs)}  原始影片找不到={len(missing)}", flush=True)
    st = collections.Counter()
    with ProcessPoolExecutor(a.workers) as ex:
        for dst, s in ex.map(_ffmpeg_one, jobs, chunksize=4):
            st[s.split(" ")[0]] += 1
            if s.startswith("FAIL"):
                print(f"  FAIL {dst}: {s}", flush=True)
    print(f"[extract {a.version}] 結果 {dict(st)}", flush=True)


# ── 讀影片 ─────────────────────────────────────────────────────────────────
class Source:
    """回傳 {(class, norm_name): [frame paths 已排序]}，以及讀單張圖的方法。只收 testlist03 的影片。"""

    def __init__(self, version):
        from PIL import Image
        self.Image = Image
        self.version = version
        want = {(c, norm_name(v)) for c, v in testlist()}
        vids = collections.defaultdict(list)
        if version == "A":
            self.z = zipfile.ZipFile(io.BytesIO(open(ZIP_A, "rb").read()))
            for n in self.z.namelist():
                if n.lower().endswith(".jpg"):
                    c, v, _ = n.split("/")[-3:]
                    k = (c, norm_name(v))
                    if k in want:
                        vids[k].append(n)
        else:
            self.z = None
            root = os.path.join(D, f"hmdb51_probe_{version}")
            for c in sorted(os.listdir(root)):
                for v in sorted(os.listdir(os.path.join(root, c))):
                    k = (c, norm_name(v))
                    if k in want:
                        d = os.path.join(root, c, v)
                        vids[k] = [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(".jpg")]
        self.all = {k: sorted(p) for k, p in vids.items()}
        self.vids = {k: p for k, p in self.all.items() if len(p) >= SEQ_LEN}

    def read(self, p):
        if self.z is not None:
            with self.z.open(p, "r") as f:
                with self.Image.open(f) as i:
                    i.load()
                    return i
        with self.Image.open(p) as i:
            i.load()
            return i


def eval_frame_indices(n_frames, seq_len=SEQ_LEN):
    # 照抄 video_reader.py:296–317（評估分支）
    if n_frames == seq_len:
        return [int(f) for f in range(n_frames)]
    start = 1
    end = n_frames - 2
    if end - start < seq_len:
        end = n_frames - 1
        start = 0
    idx_f = np.linspace(start, end, num=seq_len)
    return [int(f) for f in idx_f]


# ── feats ──────────────────────────────────────────────────────────────────
def cmd_feats(a):
    import torch
    import torchvision
    from torchvision import transforms
    from videotransforms.video_transforms import Compose, Resize, CenterCrop

    os.makedirs(OUT, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = torchvision.models.resnet50(pretrained=True)   # 與 models/stage1_model.py、stage2_model.py 相同的呼叫
    m.fc = torch.nn.Identity()                         # 拿掉 fc → avgpool 後 2048 維
    m = m.to(dev).eval()
    vt = Compose([Resize(256), CenterCrop(224)])
    to_t = transforms.ToTensor()
    nm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    for V in VERSIONS:
        if all(os.path.exists(os.path.join(OUT, f"feat_{V}_{n}.npz")) for n in NORMS) and not a.force:
            print(f"[feats {V}] 已存在，略過", flush=True)
            continue
        src = Source(V)
        keys = sorted(src.vids)
        print(f"[feats {V}] testlist03 找到 {len(src.all)} 支，≥8 幀可用 {len(keys)} 支", flush=True)
        feats = {n: np.zeros((len(keys), 2048), np.float32) for n in NORMS}
        nfr = np.array([len(src.all[k]) for k in keys])
        bs = 32
        with torch.no_grad():
            for i0 in range(0, len(keys), bs):
                batch = keys[i0:i0 + bs]
                clips = []
                for k in batch:
                    paths = src.vids[k]
                    imgs = [src.read(paths[i]) for i in eval_frame_indices(len(paths))]
                    clips.append(torch.stack([to_t(v) for v in vt(imgs)]))
                x = torch.stack(clips).to(dev)               # [b, 8, 3, 224, 224]
                b = x.shape[0]
                for n in NORMS:
                    xi = nm(x.view(-1, 3, 224, 224)) if n == "on" else x.view(-1, 3, 224, 224)
                    f = m(xi).view(b, SEQ_LEN, -1).mean(1)
                    f = torch.nn.functional.normalize(f, dim=1)
                    feats[n][i0:i0 + b] = f.cpu().numpy()
        for n in NORMS:
            np.savez(os.path.join(OUT, f"feat_{V}_{n}.npz"), feat=feats[n],
                     cls=np.array([k[0] for k in keys]), name=np.array([k[1] for k in keys]), n_frames=nfr,
                     n_found=len(src.all), n_usable=len(keys))
        print(f"[feats {V}] done", flush=True)


def load_feat(V, n):
    d = np.load(os.path.join(OUT, f"feat_{V}_{n}.npz"))
    keys = [f"{c}/{v}" for c, v in zip(d["cls"], d["name"])]
    return dict(zip(keys, d["feat"])), dict(zip(keys, d["n_frames"])), d


# ── episodes ───────────────────────────────────────────────────────────────
def cmd_episodes(a):
    sets = [set(load_feat(V, "off")[0]) for V in VERSIONS]
    common = sorted(set.intersection(*sets))
    by_cls = collections.defaultdict(list)
    for k in common:
        by_cls[k.split("/")[0]].append(k)
    classes = sorted(by_cls)
    rng = random.Random(EP_SEED)
    eps = []
    for _ in range(N_EP):
        cl = rng.sample(classes, WAY)
        sup, qry = [], []
        for c in cl:
            s, q = rng.sample(sorted(by_cls[c]), 2)
            sup.append(s)
            qry.append(q)
        eps.append({"classes": cl, "support": sup, "query": qry})
    p = os.path.join(OUT, "episodes.json")
    json.dump({"seed": EP_SEED, "n_episodes": N_EP, "way": WAY, "shot": 1, "query_per_class": 1,
               "n_common_videos": len(common), "per_class": {c: len(by_cls[c]) for c in classes},
               "episodes": eps}, open(p, "w"))
    print(f"[episodes] 三版本交集 {len(common)} 支；各類 {[(c, len(by_cls[c])) for c in classes]}；→ {p}", flush=True)


# ── eval ───────────────────────────────────────────────────────────────────
def cmd_eval(a):
    E = json.load(open(os.path.join(OUT, "episodes.json")))["episodes"]
    for V in VERSIONS:
        for n in NORMS:
            F, _, _ = load_feat(V, n)
            ep_acc = np.zeros(len(E))
            q_ok, q_cls, q_name = [], [], []
            for i, e in enumerate(E):
                S = np.stack([F[k] for k in e["support"]])
                Q = np.stack([F[k] for k in e["query"]])
                pred = (Q @ S.T).argmax(1)              # 特徵已 L2 正規化 → 內積 = cosine
                ok = pred == np.arange(WAY)
                ep_acc[i] = ok.mean()
                q_ok += ok.tolist()
                q_cls += e["classes"]
                q_name += e["query"]
            np.savez(os.path.join(OUT, f"results_{V}_{n}.npz"), episode_acc=ep_acc,
                     query_correct=np.array(q_ok), query_class=np.array(q_cls), query_name=np.array(q_name))
            print(f"[eval {V} {n}] acc {ep_acc.mean() * 100:.2f}", flush=True)


# ── analyze ────────────────────────────────────────────────────────────────
def ci(x):
    x = np.asarray(x, float)
    return x.mean() * 100, 196.0 * x.std() / math.sqrt(len(x))


def cmd_analyze(a):
    R = {(V, n): np.load(os.path.join(OUT, f"results_{V}_{n}.npz")) for V in VERSIONS for n in NORMS}
    meta = json.load(open(os.path.join(OUT, "episodes.json")))
    print(f"episodes：{meta['n_episodes']}，seed {meta['seed']}，三版本交集 {meta['n_common_videos']} 支", flush=True)
    for V in VERSIONS:
        d = load_feat(V, "off")[2]
        print(f"  版本 {V}：testlist03 找到 {int(d['n_found'])} 支，≥8 幀可用 {int(d['n_usable'])} 支")

    print("\n[1] 準確率 ± 95% CI（N=10000 episodes）")
    print("| 版本 | normalize 關 | normalize 開 |\n|---|---|---|")
    acc = {}
    for V in VERSIONS:
        row = []
        for n in NORMS:
            m, c = ci(R[(V, n)]["episode_acc"])
            acc[(V, n)] = m
            row.append(f"{m:.2f} ± {c:.2f}")
        print(f"| {V} | " + " | ".join(row) + " |")

    def pdiff(x, y):
        return ci(R[x]["episode_acc"] - R[y]["episode_acc"])

    print("\n[2] 配對差（逐題相減後平均 ± 95% CI，pp）")
    print("| 比較 | 差 | CI 下限 | CI 上限 | CI 跨 0 |\n|---|---|---|---|---|")
    D_ = {}
    for name, x, y in [("B − A（normalize 關）", ("B", "off"), ("A", "off")),
                       ("B − A（normalize 開）", ("B", "on"), ("A", "on")),
                       ("C − B（normalize 關）", ("C", "off"), ("B", "off")),
                       ("C − B（normalize 開）", ("C", "on"), ("B", "on")),
                       ("開 − 關（A）", ("A", "on"), ("A", "off")),
                       ("開 − 關（B）", ("B", "on"), ("B", "off")),
                       ("開 − 關（C）", ("C", "on"), ("C", "off"))]:
        m, c = pdiff(x, y)
        D_[name] = (m, c)
        print(f"| {name} | {m:+.2f} ± {c:.2f} | {m - c:+.2f} | {m + c:+.2f} | {'是' if m - c <= 0 <= m + c else '否'} |")

    print("\n[3] 各類 query 準確率（%）；* = talk／smoke／pour")
    classes = sorted(set(R[("A", "off")]["query_class"].tolist()))
    hdr = [f"{V}-{n}" for V in VERSIONS for n in NORMS]
    print("| 類別 | n query | " + " | ".join(hdr) + " |\n|---|---|" + "---|" * len(hdr))
    for c in classes:
        mask = R[("A", "off")]["query_class"] == c
        vals = [R[(V, n)]["query_correct"][mask].mean() * 100 for V in VERSIONS for n in NORMS]
        star = "*" if c in FOCUS else ""
        print(f"| {c}{star} | {int(mask.sum())} | " + " | ".join(f"{v:.2f}" for v in vals) + " |")

    print("\n[4] 依 query 在 A 的幀數是否 = 40 分群，B − A 的配對差（逐 query 相減，平均 ± 1.96·std/√n，pp）")
    _, nfA, _ = load_feat("A", "off")
    qn = R[("A", "off")]["query_name"]
    is40 = np.array([nfA[k] == 40 for k in qn])
    n_vid40 = sum(1 for k in set(qn.tolist()) if nfA[k] == 40)
    print(f"  A 中 = 40 幀的影片（交集內，出現在 query 者）：{n_vid40} 支")
    print("| 群 | n query | B − A（關） | B − A（開） |\n|---|---|---|---|")
    for lab, mk in [("A 幀數 = 40（被截斷）", is40), ("A 幀數 < 40", ~is40)]:
        cells = []
        for n in NORMS:
            dlt = R[("B", n)]["query_correct"][mk].astype(float) - R[("A", n)]["query_correct"][mk].astype(float)
            m, c = ci(dlt)
            cells.append(f"{m:+.2f} ± {c:.2f}")
        print(f"| {lab} | {int(mk.sum())} | " + " | ".join(cells) + " |")

    print("\n[乙4] 決策規則（normalize 關）")
    ba, cb = D_["B − A（normalize 關）"], D_["C − B（normalize 關）"]
    cross = lambda mc: mc[0] - mc[1] <= 0 <= mc[0] + mc[1]
    print(f"  B − A = {ba[0]:+.2f} ± {ba[1]:.2f}（|差| {'<' if abs(ba[0]) < 1 else '≥'} 1.0，CI {'跨' if cross(ba) else '不跨'} 0）")
    print(f"  C − B = {cb[0]:+.2f} ± {cb[1]:.2f}（|差| {'<' if abs(cb[0]) < 1 else '≥'} 1.0，CI {'跨' if cross(cb) else '不跨'} 0）")
    if all(abs(x[0]) < 1.0 and cross(x) for x in (ba, cb)):
        verdict = "沿用舊資料"
    elif any(abs(x[0]) >= 1.0 and not cross(x) for x in (ba, cb)):
        best = max(VERSIONS, key=lambda V: acc[(V, "off")])
        verdict = f"建議重做資料（normalize 關下準確率最高的版本：{best}，{acc[(best, 'off')]:.2f}）"
    else:
        verdict = "不確定"
    print(f"  判定：{verdict}")
    print("  normalize（開 − 關）不套用規則：" + "；".join(f"{V} {D_[f'開 − 關（{V}）'][0]:+.2f} ± {D_[f'開 − 關（{V}）'][1]:.2f}" for V in VERSIONS))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract")
    e.add_argument("--version", choices=["B", "C"], required=True)
    e.add_argument("--fps", type=int, default=10)
    e.add_argument("--workers", type=int, default=8)
    f = sp.add_parser("feats")
    f.add_argument("--force", action="store_true")
    sp.add_parser("episodes")
    sp.add_parser("eval")
    sp.add_parser("analyze")
    a = ap.parse_args()
    {"extract": cmd_extract, "feats": cmd_feats, "episodes": cmd_episodes,
     "eval": cmd_eval, "analyze": cmd_analyze}[a.cmd](a)
