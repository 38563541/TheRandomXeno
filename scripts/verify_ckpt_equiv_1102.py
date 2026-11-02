#!/usr/bin/env python3
"""
scripts/verify_ckpt_equiv_1102.py — 1102 Phase 1：GPU 單步 checkpoint 等價比對（確定性模式）。

模式
  episode               用 run.py 的 Learner（--seed 42）取第一個訓練 episode → logs/1102_episode.pt，
                        另存 RN50 縮小版（每類 1 個 query）logs/1102_episode_q1.pt 與 seq_len 4 退路版。
  hashes  --tag T  -- <run.py 參數>
                        用 run.py 的 Learner 建同一個 DataLoader，記第 1、100、1000 個 episode 的雜湊（Phase 3 的 3-2）。
  run --backbone BB --group G
                        一個 (backbone, 組別) 做一次 forward＋loss＋backward（不 step），結果存
                        ~/trx_data/checkpoints/1102_p1/<BB>_<G>.pt；A 另外把涵蓋段的逐 module 輸出寫到
                        ~/trx_data/checkpoints/1102_p1/<BB>_Aref/（之後各組讀這份比對）。
  compare --backbone BB  讀各組結果，輸出 logs/1102_p1_<BB>_{summary,grads,bn}.csv 與 _summary.json。

不改既有程式：模型、BN 修正、參數解析全部呼叫 run.py 本身（Learner.parse_command_line／init_model 以
unbound 方式呼叫，stub 只帶 args 與 device）。只關 dropout（nn.Dropout 設 p=0 並 eval、MultiheadAttention
的 dropout 設 0）；BN momentum 保持 init_model() 設定後的值，不歸零。

注意：non-reentrant checkpoint 的重算不觸發 forward hook，所以 (c)「重算」是用 B 存下的組輸入，在
no_grad＋train mode 下對該組的 deepcopy 手動重跑一次來代表（deepcopy 讓 BN running stats 不被動到）。
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # 必須在 import torch 之前
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import argparse
import copy
import csv
import hashlib
import json
import math
import time
import types
import warnings

import numpy as np
import torch
import torch.nn as nn

WARN = {}


def _showwarning(message, category, filename, lineno, file=None, line=None):
    key = f"{category.__name__}: {str(message).strip()}"
    WARN[key] = WARN.get(key, 0) + 1
    if WARN[key] == 1:
        sys.stderr.write(f"[warn] {os.path.basename(filename)}:{lineno} {key[:400]}\n")


warnings.showwarning = _showwarning
warnings.simplefilter("always")


def set_deterministic():
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_deterministic()

import run          # noqa: E402  （會 import tensorflow，慢但無副作用）
import utils        # noqa: E402

CFG = "configs/stage2_true_hyrsm_b4_tuple.yaml"
SCR = os.path.expanduser("~/Documents/work/trx/trx_data")
CK = os.path.expanduser("~/trx_data/checkpoints")
P1DIR = os.path.join(CK, "1102_p1")
LOGS = os.path.join(REPO, "logs")
EP_FULL = os.path.join(LOGS, "1102_episode.pt")
EP_Q1 = os.path.join(LOGS, "1102_episode_q1.pt")
EP_Q1S4 = os.path.join(LOGS, "1102_episode_q1_s4.pt")
SEQ_MARK = os.path.join(LOGS, "1102_p1_resnet50_seq.txt")

PREFIX = {"resnet18": (8, 4), "resnet34": (14, 7), "resnet50": (14, 7)}


def group_flags(bb, grp):
    n, s = PREFIX[bb]
    b = ["--grad_ckpt", "--ckpt_prefix", str(n), "--ckpt_segments", str(s)]
    return {"A": [], "Ap": [], "B": b, "C": b + ["--bn_fix_scope", "all"], "D": b,
            "B2": ["--grad_ckpt", "--ckpt_segments", "8"]}[grp]


def base_argv(bb, ckdir, extra=()):
    return ["run.py", "--dataset", "hmdb", "--split", "3", "--config", CFG, "--method", bb,
            "--seed", "42", "--scratch", SCR, "-c", ckdir, "--amp", "off"] + list(extra)


def parse_args_like_run(argv):
    old = sys.argv
    sys.argv = argv
    try:
        return run.Learner.parse_command_line(None)
    finally:
        sys.argv = old


def seed_all(s=42):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def thash(t):
    t = t.detach().cpu().contiguous()
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()[:16]


def sd_hash(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def episode_hash(ep):
    h = hashlib.sha256()
    for k in ("support_set", "support_labels", "target_set", "target_labels", "real_target_labels",
              "batch_class_list"):
        h.update(ep[k].contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def env_info():
    return dict(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
                cublas_ws=os.environ.get("CUBLAS_WORKSPACE_CONFIG"))


# ---------------------------------------------------------------------------
# episode / hashes
# ---------------------------------------------------------------------------
def subset_q1(ep, seq_len=8, keep_frames=None):
    tl = ep["target_labels"]
    idx = sorted(int((tl == c).nonzero()[0]) for c in range(int(tl.max()) + 1))
    out = dict(ep)
    out["target_set"] = torch.cat([ep["target_set"][i * seq_len:(i + 1) * seq_len] for i in idx])
    out["target_labels"] = tl[idx].clone()
    out["real_target_labels"] = ep["real_target_labels"][idx].clone()
    out["kept_query_idx"] = torch.tensor(idx)
    if keep_frames is not None:
        def sub(x):
            v = x.reshape(-1, seq_len, *x.shape[1:])[:, keep_frames]
            return v.reshape(-1, *x.shape[1:]).contiguous()
        out["support_set"] = sub(out["support_set"])
        out["target_set"] = sub(out["target_set"])
        out["kept_frames"] = torch.tensor(keep_frames)
    return out


def mode_episode():
    sys.argv = base_argv("resnet18", os.path.join(CK, "1102_p1_episode"))
    L = run.Learner()
    it = iter(L.video_loader)
    td = next(it)
    ep = {k: v[0].clone() for k, v in td.items()}
    del it
    meta = dict(img_size=L.args.img_size, seq_len=L.args.seq_len, way=L.args.way, shot=L.args.shot,
                query_per_class=L.args.query_per_class, seed=L.args.seed, num_workers=L.args.num_workers)
    ep_q1 = subset_q1(ep, L.args.seq_len)
    ep_q1s4 = subset_q1(ep, L.args.seq_len, keep_frames=[0, 2, 4, 6])
    for path, e, extra in ((EP_FULL, ep, {}), (EP_Q1, ep_q1, dict(query_per_class=1)),
                           (EP_Q1S4, ep_q1s4, dict(query_per_class=1, seq_len=4))):
        m = dict(meta, **extra)
        m["hash"] = episode_hash(e)
        e["meta"] = m
        torch.save(e, path)
        print(f"[episode] {path}  support {tuple(e['support_set'].shape)}  target {tuple(e['target_set'].shape)}"
              f"  hash {m['hash']}  labels {e['target_labels'].tolist()}", flush=True)


def mode_hashes(tag, rest):
    sys.argv = ["run.py"] + rest
    L = run.Learner()
    want = {1, 100, 1000}
    rows = []
    for i, td in enumerate(L.video_loader, start=1):
        if i in want:
            ep = {k: v[0] for k, v in td.items()}
            rows.append(dict(tag=tag, episode=i, hash=episode_hash(ep),
                             batch_classes=" ".join(str(int(x)) for x in ep["batch_class_list"])))
            print(f"[hash] {tag} episode {i}: {rows[-1]['hash']}", flush=True)
        if i >= max(want):
            break
    path = os.path.join(LOGS, f"1102_p3_episode_hash_{tag}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tag", "episode", "hash", "batch_classes"])
        w.writeheader()
        w.writerows(rows)
    print(f"[hash] wrote {path}", flush=True)


# ---------------------------------------------------------------------------
# run one (backbone, group)
# ---------------------------------------------------------------------------
def ckpt_groups(model, args):
    """回傳 (branch 預期, 被重算的 chain 區段 list[(s,e)])，與 stage2_model.forward 的切法一致。"""
    n = len(model._ckpt_chain())
    if not getattr(args, "grad_ckpt", False):
        return "none", []
    if getattr(args, "ckpt_prefix", None):
        N = min(args.ckpt_prefix, n)
        segs = max(1, min(args.ckpt_segments or 8, N))
        return "prefix_full", list(model._split_even(N, segs))
    segs = min(args.ckpt_segments or 8, n)
    ss = n // segs
    return "checkpoint_sequential", [(i * ss, (i + 1) * ss) for i in range(segs - 1)]


def disable_dropout(model):
    n_do, n_mha = 0, 0
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
            m.eval()
            n_do += 1
        elif isinstance(m, nn.MultiheadAttention):
            m.dropout = 0.0
            n_mha += 1
    return n_do, n_mha


def bn_modules(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]


def covered_bn_names(model, groups):
    chain = model._ckpt_chain()
    ids = {id(sub) for (s, e) in groups for blk in chain[s:e] for sub in blk.modules()
           if isinstance(sub, nn.modules.batchnorm._BatchNorm)}
    return sorted(n for n, m in bn_modules(model) if id(m) in ids)


def load_episode(bb):
    if bb != "resnet50":
        return torch.load(EP_FULL, weights_only=False), 8
    seq = 8
    if os.path.exists(SEQ_MARK):
        seq = int(open(SEQ_MARK).read().strip())
    return torch.load(EP_Q1 if seq == 8 else EP_Q1S4, weights_only=False), seq


class Ref:
    """A 的逐 module 參考值：A 寫 .npy，其他組讀出來比。"""

    def __init__(self, d, write):
        self.d, self.write = d, write
        os.makedirs(d, exist_ok=True)
        self.cmp = {}

    def handle_any(self, keys, t, table):
        """重算時不知道是哪一個 pass：對每個候選鍵比，取差最小的那個記下來（鍵名記實際配到的）。"""
        best = None
        for key in keys:
            tmp = {}
            self.handle(key, t, tmp)
            v = tmp[key]
            if v["equal"] is None or (isinstance(v["maxabs"], float) and math.isnan(v["maxabs"])):
                continue
            if best is None or v["maxabs"] < best[1]["maxabs"]:
                best = (key, v)
        if best is None:
            table[keys[0] + f"#{len(table)}"] = dict(maxabs=float("nan"), equal=None, refmax=float("nan"),
                                                     note="無可比的參考")
            return None
        k = best[0]
        while k in table:
            k += "+"
        best[1]["note"] = (best[1]["note"] + f" 候選={len(keys)}").strip()
        table[k] = best[1]
        return best[0]

    def handle(self, key, t, table):
        t = t.detach()
        if self.write:
            np.save(os.path.join(self.d, key + ".npy"), t.cpu().numpy())
            return
        p = os.path.join(self.d, key + ".npy")
        if not os.path.exists(p):
            table[key] = dict(maxabs=float("nan"), equal=None, refmax=float("nan"), note="A 無此鍵")
            return
        ref = torch.from_numpy(np.load(p))
        x = t.cpu()
        if ref.shape != x.shape:
            table[key] = dict(maxabs=float("nan"), equal=False, refmax=float(ref.abs().max()),
                              note=f"shape {tuple(ref.shape)} vs {tuple(x.shape)}")
            return
        table[key] = dict(maxabs=float((x - ref).abs().max()), equal=bool(torch.equal(x, ref)),
                          refmax=float(ref.abs().max()), note="")


def mode_run(bb, grp, ckpt_ref_for):
    os.makedirs(P1DIR, exist_ok=True)
    dev = torch.device("cuda")
    ep, seq = load_episode(bb)
    out_path = os.path.join(P1DIR, f"{bb}_{grp}.pt")

    args = parse_args_like_run(base_argv(bb, os.path.join(CK, "1102_p1_dummy"), group_flags(bb, grp)))
    if bb == "resnet50":
        args.query_per_class = 1
    args.seq_len = seq
    assert args.img_size == ep["meta"]["img_size"], (args.img_size, ep["meta"])

    if grp == "D":
        import torch.utils.checkpoint as tuc
        _orig = tuc.checkpoint

        def _reentrant(fn, *a, **kw):
            kw["use_reentrant"] = True
            return _orig(fn, *a, **kw)
        tuc.checkpoint = _reentrant
        print("[D] torch.utils.checkpoint.checkpoint 被 monkeypatch 成 use_reentrant=True", flush=True)

    seed_all(42)
    stub = types.SimpleNamespace(args=args, device=dev)
    model = run.Learner.init_model(stub)          # 同一段 BN 修正邏輯
    init_path = os.path.join(P1DIR, f"{bb}_init.pt")
    if grp == "A" or not os.path.exists(init_path):
        torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, init_path)
    init_sd = torch.load(init_path, weights_only=False)
    model.load_state_dict(init_sd)
    model.train()
    n_do, n_mha = disable_dropout(model)

    branch_exp, my_groups = ckpt_groups(model, args)
    # A/A′ 的「涵蓋段」用 B 的切法（報告拆 BN 用）；A 寫參考值時涵蓋 B 與 B2 的聯集
    b_args = parse_args_like_run(base_argv(bb, os.path.join(CK, "1102_p1_dummy"), group_flags(bb, "B")))
    b2_args = parse_args_like_run(base_argv(bb, os.path.join(CK, "1102_p1_dummy"), group_flags(bb, "B2")))
    _, b_groups = ckpt_groups(model, b_args)
    _, b2_groups = ckpt_groups(model, b2_args)
    rep_groups = my_groups if my_groups else b_groups
    n_ref = max(e for _, e in b_groups + b2_groups)
    starts_all = sorted({s for s, _ in b_groups + b2_groups})
    chain = model._ckpt_chain()
    bns = bn_modules(model)
    momentum = {n: m.momentum for n, m in bns}

    ci = ep["support_set"].to(dev)
    ti = ep["target_set"].to(dev)
    cl = ep["support_labels"].to(dev)
    tl = ep["target_labels"].type(torch.LongTensor).to(dev)

    def step():
        model.zero_grad(set_to_none=True)
        logits = model(ci, cl, ti)["logits"]
        loss = run.loss(logits, tl, dev) / args.tasks_per_batch
        loss.backward()
        return loss, logits

    # ---- (f) 乾淨量測：無 hook，暖機一次＋量測一次，每次都重載初始 state ----
    clean = {}
    try:
        for k in ("warmup", "timed"):
            model.load_state_dict(init_sd)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            step()
            torch.cuda.synchronize()
            clean[k] = dict(ms=(time.perf_counter() - t0) * 1000,
                            peak_alloc_gib=torch.cuda.max_memory_allocated() / 1024 ** 3,
                            peak_res_gib=torch.cuda.max_memory_reserved() / 1024 ** 3)
            print(f"[clean] {k}: {clean[k]}", flush=True)
    except torch.cuda.OutOfMemoryError as e:
        print(f"[OOM] {bb} {grp} seq_len={seq}: {str(e)[:300]}", flush=True)
        if bb == "resnet50" and seq == 8:
            open(SEQ_MARK, "w").write("4\n")
            print(f"[OOM] 寫入 {SEQ_MARK}=4，之後 RN50 各組改用 seq_len 4", flush=True)
            sys.exit(3)
        sys.exit(4)

    # ---- 有 hook 的比對步 ----
    model.load_state_dict(init_sd)
    model.zero_grad(set_to_none=True)
    ref = Ref(os.path.join(P1DIR, f"{bb}_Aref"), write=(grp == "A"))
    st = dict(phase="forward", p=-1, counts={}, bw_calls=0, rerun_calls=0, bw_leaf_calls=0, bw_bn_calls=0,
              bw_order=[])
    rec = dict(stage={}, bn_batch={}, bn_events=[], cmp_fwd={}, cmp_rerun={}, cmp_recompute={}, gin={}, gfeat={},
               versions={})
    n_sup, n_tgt = ep["support_set"].shape[0], ep["target_set"].shape[0]

    def bw_pass_candidates(t):
        if n_sup != n_tgt:
            return [0] if t.shape[0] == n_sup else [1]
        return [0, 1]
    saved_in = {}
    names = {id(m): n for n, m in model.named_modules()}
    stage_idx = {}
    pos = 0
    for i, m in enumerate(model.resnet):
        if isinstance(m, nn.Sequential):
            pos += len(m)
            stage_idx[pos - 1] = f"layer{i - 3}"
        else:
            pos += 1
    stage_idx[len(chain) - 1] = "avgpool"
    hooks = []

    def count(key):
        c = st["counts"].get(key, 0)
        st["counts"][key] = c + 1
        return c

    def table():
        return rec["cmp_fwd"] if st["phase"] == "forward" else rec["cmp_rerun"]

    def chain_pre(k):
        def h(mod, inp):
            if st["phase"] == "backward":
                st["bw_calls"] += 1
                x = inp[0]
                cands = bw_pass_candidates(x)
                if k < n_ref:
                    got = ref.handle_any([f"cin_{p}_{k}" for p in cands], x, rec["cmp_recompute"])
                    st["bw_order"].append((k, got))
                    if got is not None:
                        st["bw_p"] = int(got.split("_")[1])
                else:
                    st["bw_order"].append((k, None))
                return
            if st["phase"] == "rerun":
                st["rerun_calls"] += 1
            if k == 0 and st["phase"] == "forward":
                st["p"] += 1
            p = st["p"]
            x = inp[0]
            if st["phase"] == "forward":
                if k in starts_all:
                    saved_in[(p, k)] = x
                    rec["versions"][(p, k)] = [x._version, None]
                    if x.requires_grad:
                        x.register_hook(lambda g, key=(p, k): rec["gin"].__setitem__(key, g.detach().cpu()))
                if k < n_ref:
                    ref.handle(f"cin_{p}_{k}", x, table())
            elif st["phase"] == "rerun" and k < n_ref:
                ref.handle(f"cin_{p}_{k}", x, table())
        return h

    def chain_fwd(k):
        def h(mod, inp, out):
            if st["phase"] == "backward":
                if k < n_ref:
                    ref.handle_any([f"cout_{p}_{k}" for p in bw_pass_candidates(out)], out, rec["cmp_recompute"])
                return
            p = st["p"]
            if k < n_ref:
                ref.handle(f"cout_{p}_{k}", out, table())
            if st["phase"] == "forward":
                if k in stage_idx:
                    rec["stage"][(p, stage_idx[k])] = out.detach().cpu()
                if k == len(chain) - 1 and out.requires_grad:
                    out.register_hook(lambda g, key=p: rec["gfeat"].__setitem__(key, g.detach().cpu()))
        return h

    def leaf_fwd(name):
        def h(mod, inp, out):
            if st["phase"] == "backward":
                st["bw_leaf_calls"] += 1
                p = st.get("bw_p", 0) if n_sup == n_tgt else bw_pass_candidates(out)[0]
                c = count(("backward", p, name))
                ref.handle_any([f"leaf_{p}_{name}_{c}"], out, rec["cmp_recompute"])
                return
            p = st["p"]
            c = count((st["phase"], p, name))
            ref.handle(f"leaf_{p}_{name}_{c}", out, table())
        return h

    def bn_pre(name):
        def h(mod, inp):
            if st["phase"] == "rerun":
                return
            x = inp[0].detach().double()
            dims = [0] + list(range(2, x.dim()))
            n = x.numel() // x.shape[1]
            var, mean = torch.var_mean(x, dim=dims, unbiased=False)
            d = dict(mean=mean.cpu(), var_unb=(var * n / (n - 1)).cpu(), n=n)
            if st["phase"] == "backward":
                st["bw_bn_calls"] += 1
                p = st.get("bw_p", -1) if n_sup == n_tgt else bw_pass_candidates(x)[0]
            else:
                p = st["p"]
                rec["bn_batch"][(p, name)] = d
            # 每個 BN 實際的 running stats 更新事件（含重算），依發生順序
            rec["bn_events"].append(dict(name=name, phase=st["phase"], p=p, **d))
        return h

    for k, m in enumerate(chain):
        hooks.append(m.register_forward_pre_hook(chain_pre(k)))
        hooks.append(m.register_forward_hook(chain_fwd(k)))
        if k < n_ref:
            for sub in m.modules():
                if len(list(sub.children())) == 0:
                    hooks.append(sub.register_forward_hook(leaf_fwd(names[id(sub)])))
    for n, m in bns:
        hooks.append(m.register_forward_pre_hook(bn_pre(n)))

    torch.cuda.reset_peak_memory_stats()
    logits = model(ci, cl, ti)["logits"]
    loss = run.loss(logits, tl, dev) / args.tasks_per_batch
    branch = getattr(model, "_last_ckpt_branch", None)
    st["phase"] = "backward"
    loss.backward()
    torch.cuda.synchronize()
    hooked_peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    for key, v in rec["versions"].items():
        v[1] = saved_in[key]._version

    grads = {n: (p.grad.detach().cpu().clone() if p.grad is not None else None)
             for n, p in model.named_parameters()}
    bn_state = {n: dict(mean=m.running_mean.detach().cpu().clone(), var=m.running_var.detach().cpu().clone(),
                        nbt=int(m.num_batches_tracked), momentum=m.momentum) for n, m in bns}

    # ---- (c) 手動重跑：B/C/B2 的每個被重算組（deepcopy，不動原本的 BN） ----
    rerun_groups = my_groups if grp in ("B", "C", "B2") else []
    st["phase"] = "rerun"
    for (s, e) in rerun_groups:
        for p in (0, 1):
            st["p"] = p
            g = copy.deepcopy(nn.Sequential(*chain[s:e])).train()
            with torch.no_grad():
                g(saved_in[(p, s)])
            del g
    for h in hooks:
        h.remove()

    res = dict(
        meta=dict(backbone=bb, group=grp, flags=group_flags(bb, grp), branch=branch, branch_expected=branch_exp,
                  groups=my_groups, rep_groups=rep_groups, b_groups=b_groups, b2_groups=b2_groups,
                  covered_bn=covered_bn_names(model, rep_groups), n_bn=len(bns), chain_len=len(chain),
                  episode_hash=ep["meta"]["hash"], episode_tensor_hash=dict(
                      support=thash(ci), target=thash(ti), labels=thash(tl)),
                  init_hash=sd_hash(init_sd), seq_len=seq, n_support_frames=ci.shape[0],
                  n_target_frames=ti.shape[0], dropout_off=(n_do, n_mha), momentum=momentum,
                  clean=clean, hooked_peak_gib=hooked_peak, bw_hook_calls=st["bw_calls"],
                  bw_leaf_calls=st["bw_leaf_calls"], bw_bn_calls=st["bw_bn_calls"], bw_order=st["bw_order"],
                  rerun_hook_calls=st["rerun_calls"], n_passes=st["p"] + 1 if st["p"] >= 0 else 0,
                  env=env_info(), deterministic=torch.are_deterministic_algorithms_enabled(),
                  warnings=dict(WARN)),
        loss=float(loss.detach()), logits=logits.detach().cpu(), grads=grads, bn=bn_state, **rec)
    torch.save(res, out_path)
    print(f"[run] {bb} {grp}: branch={branch} loss={res['loss']:.10f} clean={clean.get('timed')} "
          f"bw_hook_calls={st['bw_calls']} → {out_path}", flush=True)


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------
def _mx(a, b):
    return float((a.double() - b.double()).abs().max()) if a.numel() else 0.0


def emulate(init_sd, bn_batch, name, m, seq):
    """依序套用 running = (1-m)·running + m·stat（float32），stat 取該組自己 forward 時量到的 batch 統計。"""
    r_m = init_sd[name + ".running_mean"].clone().float()
    r_v = init_sd[name + ".running_var"].clone().float()
    for p in seq:
        b = bn_batch[(p, name)]
        r_m = (1 - m) * r_m + m * b["mean"].float()
        r_v = (1 - m) * r_v + m * b["var_unb"].float()
    return r_m, r_v


def mode_compare(bb):
    groups = [g for g in ("A", "Ap", "B", "C", "D", "B2") if os.path.exists(os.path.join(P1DIR, f"{bb}_{g}.pt"))]
    R = {g: torch.load(os.path.join(P1DIR, f"{bb}_{g}.pt"), weights_only=False) for g in groups}
    A = R["A"]
    init_sd = torch.load(os.path.join(P1DIR, f"{bb}_init.pt"), weights_only=False)
    summ, grow, brow = [], [], []
    J = dict(backbone=bb, groups={})

    def add(g, item, maxabs, equal, refmax, note=""):
        summ.append(dict(group=g, item=item, maxabs=maxabs, equal=equal, refmax=refmax, note=note))

    for g in groups:
        X = R[g]
        mx = X["meta"]
        jg = dict(meta={k: mx.get(k) for k in ("branch", "branch_expected", "groups", "episode_hash", "episode_tensor_hash",
                                           "init_hash", "seq_len", "n_support_frames", "n_target_frames", "clean",
                                           "hooked_peak_gib", "bw_hook_calls", "bw_leaf_calls", "bw_bn_calls", "bw_order",
                                           "rerun_hook_calls", "n_passes",
                                           "dropout_off", "deterministic", "env")},
                  warnings=mx["warnings"])
        cov = set(mx["covered_bn"])
        jg["meta"]["n_covered_bn"] = len(cov)
        jg["meta"]["n_bn"] = mx["n_bn"]
        jg["meta"]["momentum_set"] = sorted({f"{v:.6f}" for v in mx["momentum"].values()})
        jg["meta"]["momentum_covered"] = sorted({f"{mx['momentum'][n]:.6f}" for n in cov})
        jg["meta"]["momentum_uncovered"] = sorted({f"{v:.6f}" for n, v in mx["momentum"].items() if n not in cov})
        if g == "A":
            J["groups"][g] = jg
            continue
        # (a) 前提
        add(g, "a_episode_hash_same", 0.0 if mx["episode_hash"] == A["meta"]["episode_hash"] else float("nan"),
            mx["episode_tensor_hash"] == A["meta"]["episode_tensor_hash"], float("nan"), mx["episode_hash"])
        add(g, "a_init_hash_same", float("nan"), mx["init_hash"] == A["meta"]["init_hash"], float("nan"), mx["init_hash"])
        # (b) forward
        for key, t in X["stage"].items():
            add(g, f"b_stage_p{key[0]}_{key[1]}", _mx(t, A["stage"][key]), bool(torch.equal(t, A["stage"][key])),
                float(A["stage"][key].abs().max()))
        add(g, "b_logits", _mx(X["logits"], A["logits"]), bool(torch.equal(X["logits"], A["logits"])),
            float(A["logits"].abs().max()))
        add(g, "b_loss", abs(X["loss"] - A["loss"]), X["loss"] == A["loss"], abs(A["loss"]))
        # (c) 重算與 forward 逐 module
        for tag, tab in (("c_fwd_modules", X["cmp_fwd"]), ("c_rerun_modules", X["cmp_rerun"]),
                         ("c_recompute_modules", X.get("cmp_recompute", {}))):
            if not tab:
                continue
            vals = list(tab.values())
            ma = max((v["maxabs"] for v in vals if not math.isnan(v["maxabs"])), default=float("nan"))
            eq = all(v["equal"] for v in vals)
            rm = max((v["refmax"] for v in vals if not math.isnan(v["refmax"])), default=float("nan"))
            neq = sum(1 for v in vals if not v["equal"])
            add(g, tag, ma, eq, rm, f"n={len(vals)} 不相等={neq}")
            for sub in ("cin", "cout", "leaf"):
                sv = [v for k, v in tab.items() if k.startswith(sub + "_")]
                if sv:
                    add(g, f"{tag}_{sub}", max(v["maxabs"] for v in sv), all(v["equal"] for v in sv),
                        max(v["refmax"] for v in sv), f"n={len(sv)}")
        if X["versions"]:
            bad = {str(k): v for k, v in X["versions"].items() if v[0] != v[1]}
            add(g, "c_input_version_unchanged", float("nan"), not bad, float("nan"),
                f"n={len(X['versions'])} 變動={bad}")
        # (d) backward
        conv1 = "resnet.0.weight"
        gx = X["grads"].get(conv1)
        add(g, "d_stem_conv1_grad_nonzero", float(gx.abs().max()) if gx is not None else float("nan"),
            gx is not None and bool(gx.abs().max() > 0), float(A["grads"][conv1].abs().max()),
            "None" if gx is None else "")
        chain_names = None
        per = []
        none_list = []
        for n, ga in A["grads"].items():
            gb = X["grads"][n]
            if ga is None:
                continue
            region = "head" if not n.startswith("resnet.") else "backbone"
            if gb is None:
                none_list.append(n)
                grow.append(dict(group=g, param=n, region=region, maxabs=float("nan"), rel=float("nan"),
                                 cos=float("nan"), normA=float(ga.norm()), normX=float("nan"), equal=False,
                                 note="X.grad is None"))
                continue
            d = (gb.double() - ga.double())
            na, nb = float(ga.double().norm()), float(gb.double().norm())
            rel = float(d.norm()) / na if na > 0 else float("nan")
            cos = float(torch.nn.functional.cosine_similarity(ga.double().flatten(), gb.double().flatten(), dim=0)) \
                if na > 0 and nb > 0 else float("nan")
            row = dict(group=g, param=n, region=region, maxabs=float(d.abs().max()), rel=rel, cos=cos, normA=na,
                       normX=nb, equal=bool(torch.equal(ga, gb)), note="all-zero" if nb == 0 else "")
            grow.append(row)
            per.append(row)
        if per:
            add(g, "d_param_grads", max(r["maxabs"] for r in per), all(r["equal"] for r in per) and not none_list,
                max(float(A["grads"][r["param"]].abs().max()) for r in per),
                f"n={len(per)} None={len(none_list)} 最大相對誤差={max(r['rel'] for r in per):.3e} "
                f"最小cos={min(r['cos'] for r in per if not math.isnan(r['cos'])) if any(not math.isnan(r['cos']) for r in per) else float('nan'):.10f}")
        for p, t in X["gfeat"].items():
            if p in A["gfeat"]:
                add(g, f"d_feat_grad_p{p}", _mx(t, A["gfeat"][p]), bool(torch.equal(t, A["gfeat"][p])),
                    float(A["gfeat"][p].abs().max()))
        for key, t in X["gin"].items():
            if key in A["gin"]:
                add(g, f"d_group_input_grad_p{key[0]}_chain{key[1]}", _mx(t, A["gin"][key]),
                    bool(torch.equal(t, A["gin"][key])), float(A["gin"][key].abs().max()))
        missing_gin = [k for k in A["gin"] if k not in X["gin"] and k[1] in {s for s, _ in mx["groups"]}]
        if missing_gin:
            add(g, "d_group_input_grad_missing", float("nan"), False, float("nan"), str(missing_gin))
        if none_list:
            add(g, "d_params_grad_None", float(len(none_list)), False, float("nan"),
                f"{len(none_list)} 個參數 .grad 為 None，例：{none_list[:3]}")
        # (e) BN
        m_order = {}
        stats = {}
        for name, bx in X["bn"].items():
            ba = A["bn"][name]
            rec_ = name in cov
            m = bx["momentum"]
            # 預期的更新序列：重算組 (B/C/B2) 的被重算 BN = forward s,q + 重算兩次（順序未知，兩種都算）
            evs = [e for e in X.get("bn_events", []) if e["name"] == name]
            if evs:
                # 實際觀察到的更新事件（forward＋重算），用各事件自己量到的 batch 統計依序套用
                bb_ = {(i, name): e for i, e in enumerate(evs)}
                em, ev = emulate(init_sd, bb_, name, m, list(range(len(evs))))
                lab = ",".join(("f" if e["phase"] == "forward" else "r") + ("s" if e["p"] == 0 else "q" if e["p"] == 1 else "?")
                               for e in evs)
                best = (lab, max(_mx(em, bx["mean"]), _mx(ev, bx["var"])), em, ev)
            else:
                best = None
                cands = {"s,q,q,s": [0, 1, 1, 0], "s,q,s,q": [0, 1, 0, 1]} if (rec_ and g in ("B", "C", "B2")) \
                    else {"s,q": [0, 1]}
                for lab, seq in cands.items():
                    em, ev = emulate(init_sd, X["bn_batch"], name, m, seq)
                    err = max(_mx(em, bx["mean"]), _mx(ev, bx["var"]))
                    if best is None or err < best[1]:
                        best = (lab, err, em, ev)
            eam, eav = emulate(init_sd, A["bn_batch"], name, ba["momentum"], [0, 1])
            row = dict(group=g, bn=name, recomputed=rec_, momentum=m, momentum_A=ba["momentum"],
                       rm_maxabs=_mx(bx["mean"], ba["mean"]), rv_maxabs=_mx(bx["var"], ba["var"]),
                       equal=bool(torch.equal(bx["mean"], ba["mean"]) and torch.equal(bx["var"], ba["var"])),
                       A_shift_from_init=max(_mx(ba["mean"], init_sd[name + ".running_mean"]),
                                             _mx(ba["var"], init_sd[name + ".running_var"])),
                       nbt_A=ba["nbt"], nbt_X=bx["nbt"],
                       emu_order=best[0], X_vs_emuX=best[1],
                       A_vs_emuA=max(_mx(eam, ba["mean"]), _mx(eav, ba["var"])),
                       pred_X_vs_A=max(_mx(best[2], eam), _mx(best[3], eav)))
            brow.append(row)
            stats.setdefault(rec_, []).append(row)
        for rec_, rows in stats.items():
            lab = "recomputed" if rec_ else "not_recomputed"
            add(g, f"e_bn_{lab}_running_stats", max(max(r["rm_maxabs"], r["rv_maxabs"]) for r in rows),
                all(r["equal"] for r in rows), max(r["A_shift_from_init"] for r in rows),
                f"n={len(rows)} 不相等={sum(1 for r in rows if not r['equal'])} "
                f"推算差(emuX−emuA)最大={max(r['pred_X_vs_A'] for r in rows):.3e} "
                f"|X−emuX|最大={max(r['X_vs_emuX'] for r in rows):.3e} |A−emuA|最大={max(r['A_vs_emuA'] for r in rows):.3e} "
                f"重算順序={sorted({r['emu_order'] for r in rows})}")
            nb = sorted({(r["nbt_A"], r["nbt_X"]) for r in rows})
            add(g, f"e_bn_{lab}_num_batches_tracked", float("nan"), None, float("nan"), f"(A,X)={nb}（不納入判定）")
        J["groups"][g] = jg

    for name, rows, fields in (
            ("summary", summ, ["group", "item", "maxabs", "equal", "refmax", "note"]),
            ("grads", grow, ["group", "param", "region", "maxabs", "rel", "cos", "normA", "normX", "equal", "note"]),
            ("bn", brow, ["group", "bn", "recomputed", "momentum", "momentum_A", "rm_maxabs", "rv_maxabs", "equal",
                          "A_shift_from_init", "nbt_A", "nbt_X", "emu_order", "X_vs_emuX", "A_vs_emuA",
                          "pred_X_vs_A"])):
        p = os.path.join(LOGS, f"1102_p1_{bb}_{name}.csv")
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"[compare] wrote {p} ({len(rows)} rows)")
    J["summary"] = summ
    with open(os.path.join(LOGS, f"1102_p1_{bb}_summary.json"), "w") as f:
        json.dump(J, f, ensure_ascii=False, indent=1, default=str)
    for r in summ:
        print(f"  {r['group']:3s} {r['item']:45s} maxabs={r['maxabs']:<12.4g} equal={r['equal']!s:5s} "
              f"ref={r['refmax']:<10.4g} {r['note'][:150]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["episode", "hashes", "run", "compare"])
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--group", default="A")
    ap.add_argument("--tag", default="x")
    a, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    if a.mode == "episode":
        mode_episode()
    elif a.mode == "hashes":
        mode_hashes(a.tag, rest)
    elif a.mode == "run":
        mode_run(a.backbone, a.group, None)
    else:
        mode_compare(a.backbone)


if __name__ == "__main__":
    main()
