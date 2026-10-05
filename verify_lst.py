#!/usr/bin/env python3
"""
verify_lst.py — 1005 階段 3：backbone_mode（full/frozen/partial_l4/lst/top_adapter）的驗證。

做法比照 V4／V4b：CPU、train 模式、BN momentum 0、所有 dropout（含 MultiheadAttention 內部）設 0、
backward 後比 .grad、每項都有控制組、報告梯度量級。

  V1 預設路徑回歸：backbone_mode=full（以及 frozen、E1 的 ckpt_prefix 路徑）的 logits 與 .grad，
     與 stage2-decouple HEAD 的模型程式（另外載入成獨立 package）位元級相同；控制組：擾動一個權重必須抓到差異
  V2 梯度流向：frozen/lst/top_adapter 下 backbone 每個參數 .grad is None；side/adapter 參數有梯度
     （α=0 時只有 α 有梯度，另在 α=0.1 的複本上驗全部 side 參數）；partial_l4 只有 layer4 有梯度；BN 模式
  V3 初始等價：α=0 時 lst／top_adapter 的特徵與 logits 與 frozen 位元級相同；α=0.1 必須改變
  V4 rung 可交換性：pool→conv 與 conv→pool 差 < 1e-5（fp32）
  V5 無 transductive 洩漏：固定一支 query，替換同 batch 其他 query，該 query 的特徵不變；
     BN（train 模式）與 BN(track_running_stats=False，eval 模式) 的 side 是正控制組，必須失敗
  V6 config 落地：新 key 由 run.py argparse 解析、出現在 Options（log）與 CSV 列；改值 → 輸出或參數量跟著變；
     train_aug=none 讓同一支影片兩次讀取完全相同
  V7 舊 checkpoint：strict 載入（full／frozen／partial_l4）；純評估數字見 scripts/v7_eval_1005.py 的 log

用法：python verify_lst.py   （輸出寫到 logs/verify_lst_1005.log 由呼叫端 tee）
"""
import copy
import csv
import importlib
import os
import subprocess
import sys
import tempfile
import types

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

import torch                      # noqa: E402
import torch.nn as nn             # noqa: E402

torch.set_num_threads(8)
REF_COMMIT = "stage2-decouple"   # 1005 改動前的 HEAD（06ddba9）
IMG = 112                        # CPU 可負擔；layer4 為 4×4，side 7×7 時 rung4 會上採樣（形狀無關驗證邏輯）
WAY, SHOT, QPC, SEQ = 5, 1, 2, 8

_pass, _fail = [], []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}", flush=True)
    (_pass if cond else _fail).append(name)


# ----------------------------------------------------------------------------
# 獨立參考：stage2-decouple HEAD 的 models/stage{1,2}_model.py
# ----------------------------------------------------------------------------
def load_reference():
    d = tempfile.mkdtemp(prefix="verify_lst_ref_")
    pkg = os.path.join(d, "ref_models")
    os.makedirs(pkg)
    open(os.path.join(pkg, "__init__.py"), "w").close()
    commit = subprocess.check_output(["git", "rev-parse", "--short", REF_COMMIT], cwd=REPO).decode().strip()
    for name in ("stage1_model", "stage2_model"):
        src = subprocess.check_output(["git", "show", f"{REF_COMMIT}:models/{name}.py"], cwd=REPO).decode()
        assert "backbone_adapt" not in src, "參考版本不應含 1005 改動"
        src = src.replace("from models.stage1_model", "from ref_models.stage1_model")
        open(os.path.join(pkg, f"{name}.py"), "w").write(src)
    sys.path.insert(0, d)
    r1 = importlib.import_module("ref_models.stage1_model")
    r2 = importlib.import_module("ref_models.stage2_model")
    return r1, r2, commit


# ----------------------------------------------------------------------------
def make_args(head="B1", method="resnet18", **kw):
    a = types.SimpleNamespace(
        method=method, way=WAY, shot=SHOT, query_per_class=QPC, seq_len=SEQ, img_size=IMG,
        trans_linear_in_dim=2048 if method == "resnet50" else 512, trans_linear_out_dim=1152,
        trans_dropout=0.0, temp_set=[2], num_gpus=1, matching="bidirectional", set_aggregation="pool",
        tau=0.1, freeze_backbone=False, grad_ckpt=False, ckpt_prefix=None, ckpt_segments=8,
        ckpt_policy="full", profile_time=False, profile_memory=False,
        use_intra_relation=(head == "B4"), use_inter_relation=(head == "B4"),
        relation_level="tuple" if head == "B4" else "none", inter_style="true_hyrsm",
        backbone_mode="full", side_width=-1, side_res=7, side_norm="gn", adapter_hidden=-1,
    )
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def build(module_s1, module_s2, args, seed=0):
    torch.manual_seed(seed)
    if args.use_intra_relation or args.use_inter_relation:
        m = module_s2.CNN_TRXWithRelation(args)
    else:
        m = module_s1.CNN_TRX(args)
    return quiet(m)


def quiet(m):
    """dropout 全設 0（含 MHA 內部），BN momentum 0（重複 forward 時 running stats 不動）。"""
    for mod in m.modules():
        if isinstance(mod, nn.Dropout):
            mod.p = 0.0
        if isinstance(mod, nn.MultiheadAttention):
            mod.dropout = 0.0
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            mod.momentum = 0.0
    return m


def episode(seed=1, method="resnet18"):
    g = torch.Generator().manual_seed(seed)
    sup = torch.rand(WAY * SHOT * SEQ, 3, IMG, IMG, generator=g)
    lab = torch.arange(WAY).repeat_interleave(SHOT).float()
    qry = torch.rand(WAY * QPC * SEQ, 3, IMG, IMG, generator=g)
    tgt = torch.arange(WAY).repeat_interleave(QPC)
    return sup, lab, qry, tgt


def fwd_bwd(m, ep):
    sup, lab, qry, tgt = ep
    m.zero_grad(set_to_none=True)
    out = m(sup, lab, qry)["logits"]
    loss = nn.functional.cross_entropy(out.reshape(-1, WAY), tgt)
    loss.backward()
    grads = {n: (None if p.grad is None else p.grad.detach().clone()) for n, p in m.named_parameters()}
    return out.detach().clone(), grads


def max_grad_diff(g1, g2):
    worst = 0.0
    for n in g1:
        a, b = g1[n], g2.get(n)
        if (a is None) != (b is None):
            return float("inf")
        if a is not None:
            worst = max(worst, (a - b).abs().max().item())
    return worst


def grad_mag(g, prefix=""):
    v = [x.abs().mean().item() for n, x in g.items() if x is not None and n.startswith(prefix)]
    return sum(v) / len(v) if v else 0.0


# ----------------------------------------------------------------------------
def v1(ref1, ref2, new1, new2):
    print("\nV1 預設路徑回歸（新程式 vs stage2-decouple HEAD，位元級）")
    cases = [("B1", "resnet18", {}), ("B4", "resnet18", {}), ("B4", "resnet50", {}),
             ("B1", "resnet18", {"freeze_backbone": True}), ("B4", "resnet18", {"freeze_backbone": True}),
             ("B4", "resnet50", {"grad_ckpt": True, "ckpt_prefix": 14, "ckpt_segments": 7})]
    for head, method, kw in cases:
        a_ref = make_args(head, method, **kw)
        a_new = make_args(head, method, **kw)
        mr = build(ref1, ref2, a_ref)
        mn = build(new1, new2, a_new)
        same_sd = all(torch.equal(v, mn.state_dict()[k]) for k, v in mr.state_dict().items()) \
            and set(mr.state_dict()) == set(mn.state_dict())
        ep = episode(method=method)
        lr, gr = fwd_bwd(mr, ep)
        ln, gn = fwd_bwd(mn, ep)
        dl = (lr - ln).abs().max().item()
        dg = max_grad_diff(gr, gn)
        tag = f"{head}/{method}/{kw or 'default'} (new mode={a_new.backbone_mode})"
        check(f"V1 {tag} state_dict 相同", same_sd)
        check(f"V1 {tag} logits max|Δ|=0", dl == 0.0, f"{dl:.3e}")
        check(f"V1 {tag} .grad max|Δ|=0", dg == 0.0, f"{dg:.3e}；梯度量級 head={grad_mag(gn, 'transformers'):.3e} "
                                                      f"backbone={grad_mag(gn, 'resnet'):.3e}")
    # 控制組：同一個比較在權重被擾動 1e-4 時必須抓到差異
    a = make_args("B4", "resnet18")
    mr = build(ref1, ref2, a)
    mn = build(new1, new2, make_args("B4", "resnet18"))
    with torch.no_grad():
        mn.resnet[7][1].conv2.weight[0, 0, 0, 0] += 1e-4
    ep = episode()
    lr, gr = fwd_bwd(mr, ep)
    ln, gn = fwd_bwd(mn, ep)
    check("V1 控制組：擾動 layer4 一個權重 1e-4 → logits 與 .grad 必須不同",
          (lr - ln).abs().max().item() > 0 and max_grad_diff(gr, gn) > 0,
          f"logits Δ={(lr - ln).abs().max().item():.3e} grad Δ={max_grad_diff(gr, gn):.3e}")


def v2(new1, new2):
    print("\nV2 梯度流向")
    for head in ("B1", "B4"):
        for mode in ("frozen", "lst", "top_adapter", "partial_l4"):
            a = make_args(head, "resnet18", backbone_mode=mode)
            m = build(new1, new2, a)
            m.train()
            _, g = fwd_bwd(m, episode())
            bb = {n: x for n, x in g.items() if n.startswith("resnet.")}
            head_ok = all(x is not None and x.abs().sum() > 0
                          for n, x in g.items() if n.startswith("transformers.") and
                          dict(m.named_parameters())[n].requires_grad and x is not None) and \
                any(x is not None for n, x in g.items() if n.startswith("transformers."))
            tag = f"V2 {head}/{mode}"
            if mode == "partial_l4":
                l4 = {n: x for n, x in bb.items() if n.startswith("resnet.7.")}
                rest = {n: x for n, x in bb.items() if not n.startswith("resnet.7.")}
                check(f"{tag} layer4 全部參數有梯度", all(x is not None and x.abs().sum() > 0 for x in l4.values()),
                      f"{len(l4)} 個；量級 {grad_mag(g, 'resnet.7.'):.3e}")
                check(f"{tag} layer4 以外 backbone .grad is None", all(x is None for x in rest.values()),
                      f"{len(rest)} 個")
                bn_eval = [mm.training for i, s in enumerate(m.resnet) if i != 7 for mm in s.modules()
                           if isinstance(mm, nn.modules.batchnorm._BatchNorm)]
                bn_l4 = [mm.training for mm in m.resnet[7].modules() if isinstance(mm, nn.modules.batchnorm._BatchNorm)]
                check(f"{tag} BN 模式：layer1–3 eval、layer4 train", not any(bn_eval) and all(bn_l4),
                      f"eval={len(bn_eval)} train={len(bn_l4)}")
                check(f"{tag} 不設 freeze_backbone（grad_ckpt 不會被 run.py 關掉）", a.freeze_backbone is False)
            else:
                check(f"{tag} backbone 每個參數 .grad is None", all(x is None for x in bb.values()), f"{len(bb)} 個")
                bn = [mm.training for mm in m.resnet.modules() if isinstance(mm, nn.modules.batchnorm._BatchNorm)]
                check(f"{tag} backbone BN 全部 eval（model.train() 之後）", not any(bn), f"{len(bn)} 個")
            check(f"{tag} 匹配頭有梯度", head_ok, f"量級 {grad_mag(g, 'transformers'):.3e}")
            if mode in ("lst", "top_adapter"):
                name = "side_net" if mode == "lst" else "top_adapter"
                sub = {n: x for n, x in g.items() if n.startswith(name + ".")}
                others = {n: x for n, x in sub.items() if not n.endswith("alpha")}
                check(f"{tag} α=0：α 有梯度", sub[f"{name}.alpha"] is not None and sub[f"{name}.alpha"].abs() > 0,
                      f"|∂L/∂α|={sub[f'{name}.alpha'].abs().item():.3e}")
                check(f"{tag} α=0：其他 side 參數梯度精確為 0（機制如 1-2(e)）",
                      all(x is not None and x.abs().sum() == 0 for x in others.values()), f"{len(others)} 個")
                m2 = copy.deepcopy(m)
                with torch.no_grad():
                    getattr(m2, name).alpha.fill_(0.1)
                _, g2 = fwd_bwd(m2, episode())
                sub2 = {n: x for n, x in g2.items() if n.startswith(name + ".")}
                check(f"{tag} α=0.1 複本：每個 side 參數 .grad.abs().sum()>0",
                      all(x is not None and x.abs().sum() > 0 for x in sub2.values()),
                      f"{len(sub2)} 個；量級 {grad_mag(g2, name):.3e}")
                check(f"{tag} α=0.1 複本：backbone 仍無梯度",
                      all(x is None for n, x in g2.items() if n.startswith("resnet.")))
            # optimizer 會收的參數 = requires_grad 的參數
            n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
            n_head = sum(p.numel() for p in m.transformers.parameters())
            print(f"      可訓練參數 {n_tr/1e6:.3f} M（匹配頭 {n_head/1e6:.3f} M）")
    # 參數量公式核對與 B 對齊 C
    from models.backbone_adapt import side_param_count, _STAGE_CHANNELS
    for method in ("resnet18", "resnet50"):
        ml = build(new1, new2, make_args("B1", method, backbone_mode="lst"))
        mt = build(new1, new2, make_args("B1", method, backbone_mode="top_adapter"))
        n_side = sum(p.numel() for p in ml.side_net.parameters())
        n_ad = sum(p.numel() for p in mt.top_adapter.parameters())
        check(f"V2 {method} side 參數量 = 公式", n_side == side_param_count(_STAGE_CHANNELS[method], ml.args.side_width),
              f"{n_side}")
        check(f"V2 {method} top_adapter 參數量與 lst 相差 ≤10%", abs(n_ad - n_side) / n_side <= 0.10,
              f"adapter {n_ad} vs side {n_side}（{100*(n_ad-n_side)/n_side:+.2f}%）")


def v3(new1, new2):
    print("\nV3 初始等價（α=0 → 與 frozen 位元級相同；α=0.1 → 改變）")
    for head in ("B1", "B4"):
        ep = episode()
        mf = build(new1, new2, make_args(head, "resnet18", backbone_mode="frozen"))
        lf, _ = fwd_bwd(mf, ep)
        mf.eval()
        with torch.no_grad():
            ef = mf(*ep[:3])["logits"]
            ff = mf.resnet(ep[2]).squeeze()
        for mode, name in (("lst", "side_net"), ("top_adapter", "top_adapter")):
            m = build(new1, new2, make_args(head, "resnet18", backbone_mode=mode))
            common = {k: v for k, v in m.state_dict().items() if k in mf.state_dict()}
            check(f"V3 {head}/{mode} 與 frozen 共同參數初始化相同（側支在頭之後建立）",
                  all(torch.equal(v, mf.state_dict()[k]) for k, v in common.items()) and len(common) == len(mf.state_dict()))
            lm, _ = fwd_bwd(m, ep)
            check(f"V3 {head}/{mode} train 模式 logits 與 frozen 位元級相同", torch.equal(lm, lf),
                  f"max|Δ|={(lm - lf).abs().max().item():.3e}")
            m.eval()
            from models.backbone_adapt import backbone_features
            with torch.no_grad():
                em = m(*ep[:3])["logits"]
                fm = backbone_features(m, ep[2])
            check(f"V3 {head}/{mode} eval 模式 logits 與特徵位元級相同",
                  torch.equal(em, ef) and torch.equal(fm, ff))
            with torch.no_grad():
                getattr(m, name).alpha.fill_(0.1)
                em2 = m(*ep[:3])["logits"]
            check(f"V3 {head}/{mode} α=0.1 → logits 改變（控制組）", not torch.equal(em2, ef),
                  f"max|Δ|={(em2 - ef).abs().max().item():.3e}")


def v4(new1, new2):
    print("\nV4 rung 可交換性（pool→conv vs conv→pool）")
    from models.backbone_adapt import SideNet, _STAGE_CHANNELS
    for method, res in (("resnet18", 7), ("resnet50", 7), ("resnet18", 14)):
        chs = _STAGE_CHANNELS[method]
        torch.manual_seed(0)
        s = SideNet(chs, 64, res=res)
        worst = 0.0
        for i, (c, hw) in enumerate(zip(chs, (56, 28, 14, 7))):
            x = torch.randn(6, c, hw, hw)
            a = s.rungs[i](s.pool(x))
            b = s.pool(s.rungs[i](x))
            worst = max(worst, (a - b).abs().max().item())
        check(f"V4 {method} res={res} 四個 rung max|Δ|<1e-5", worst < 1e-5, f"{worst:.3e}")
    # 控制組：在 pool 與 conv 之間插入 ReLU（非線性）就不可交換
    x = torch.randn(6, 64, 56, 56)
    conv = nn.Conv2d(64, 32, 1)
    pool = nn.AdaptiveAvgPool2d(7)
    d = (torch.relu(conv(pool(x))) - pool(torch.relu(conv(x)))).abs().max().item()
    check("V4 控制組：加入 ReLU 後差異必須 ≥1e-3", d >= 1e-3, f"{d:.3e}")


def v5(new1, new2):
    print("\nV5 無 transductive 洩漏（特徵抽取器輸出，不是 logits；B4 頭本身就跨 query）")
    from models.backbone_adapt import backbone_features
    g = torch.Generator().manual_seed(5)
    q = torch.rand(10 * SEQ, 3, IMG, IMG, generator=g)          # 10 支 query 影片
    q2 = q.clone()
    q2[SEQ:] = torch.rand(9 * SEQ, 3, IMG, IMG, generator=g)    # 只保留第 0 支，其餘換掉
    cases = [("gn", "train", True), ("gn", "eval", True),
             ("bn", "train", False), ("bn_notrack", "eval", False)]
    for norm, mode, expect_pass in cases:
        m = build(new1, new2, make_args("B1", "resnet18", backbone_mode="lst", side_norm=norm))
        with torch.no_grad():
            m.side_net.alpha.fill_(1.0)      # 讓 side 真的影響輸出
        m.train() if mode == "train" else m.eval()
        with torch.no_grad():
            f1 = backbone_features(m, q)[:SEQ]
            f2 = backbone_features(m, q2)[:SEQ]
        d = (f1 - f2).abs().max().item()
        same = d == 0.0
        label = "必須不變" if expect_pass else "正控制組，必須改變"
        check(f"V5 side_norm={norm} {mode} 模式：第 0 支 query 的特徵{label}", same == expect_pass, f"max|Δ|={d:.3e}")
    for mode in ("frozen", "top_adapter", "partial_l4", "full"):
        m = build(new1, new2, make_args("B1", "resnet18", backbone_mode=mode))
        if mode == "top_adapter":
            with torch.no_grad():
                m.top_adapter.alpha.fill_(1.0)
        m.eval()
        with torch.no_grad():
            d = (backbone_features(m, q)[:SEQ] - backbone_features(m, q2)[:SEQ]).abs().max().item()
        check(f"V5 {mode} eval 模式：第 0 支 query 的特徵不變", d == 0.0, f"max|Δ|={d:.3e}")


def v6(new1, new2):
    print("\nV6 config 落地")
    import run
    keys = ["backbone_mode", "side_width", "side_res", "side_norm", "adapter_hidden", "train_aug"]
    argv = ["run.py", "--config", "configs/frozen_rn18_b1.yaml", "--dataset", "hmdb", "--split", "3",
            "--scratch", os.path.expanduser("~/Documents/work/trx/trx_data"), "-c", "/nonexistent_v6",
            "--backbone_mode", "lst", "--side_width", "64", "--side_res", "14", "--side_norm", "gn",
            "--train_aug", "none", "--seed", "42"]
    old = sys.argv
    sys.argv = argv
    try:
        lr = run.Learner.__new__(run.Learner)
        args = lr.parse_command_line()
    finally:
        sys.argv = old
    check("V6 argparse 解析新 key", args.backbone_mode == "lst" and args.side_width == 64 and args.side_res == 14
          and args.train_aug == "none" and args.adapter_hidden > 0 and args.freeze_backbone is True,
          f"adapter_hidden 自動={args.adapter_hidden}")
    opt = "Options: %s\n" % args
    check("V6 每個新 key 都出現在 Options（log.txt 第一行）", all(f"{k}=" in opt for k in keys))
    tmp = os.path.join(REPO, "logs", "verify_lst_v6_tmp.csv")
    if os.path.exists(tmp):
        os.remove(tmp)
    saved = run._CSV_PATH
    run._CSV_PATH = tmp
    try:
        run._log_result_csv(args, 25000, 50.0, 0.4)
    finally:
        run._CSV_PATH = saved
    row = list(csv.DictReader(open(tmp)))[0]
    os.remove(tmp)
    check("V6 每個新 key 都寫進 CSV 且值正確",
          all(k in row for k in keys) and row["backbone_mode"] == "lst" and row["side_width"] == "64"
          and row["side_res"] == "14" and row["train_aug"] == "none" and row["seed"] == "42", str({k: row[k] for k in keys}))
    # 改值 → 參數量或輸出跟著變
    ep = episode()

    def lst_out(**kw):
        m = build(new1, new2, make_args("B1", "resnet18", backbone_mode="lst", **kw))
        with torch.no_grad():
            m.side_net.alpha.fill_(0.5)
        m.eval()
        with torch.no_grad():
            return m(*ep[:3])["logits"], sum(p.numel() for p in m.side_net.parameters())
    o_base, n_base = lst_out()
    o_w, n_w = lst_out(side_width=64)
    o_r, n_r = lst_out(side_res=14)
    o_n, n_n = lst_out(side_norm="bn_notrack")
    check("V6 side_width 128→64：參數量改變", n_w != n_base, f"{n_base} → {n_w}")
    check("V6 side_res 7→14：輸出改變（參數量不變）", not torch.equal(o_r, o_base) and n_r == n_base)
    check("V6 side_norm gn→bn_notrack：輸出改變", not torch.equal(o_n, o_base))
    ma = build(new1, new2, make_args("B1", "resnet18", backbone_mode="top_adapter", adapter_hidden=100))
    mb = build(new1, new2, make_args("B1", "resnet18", backbone_mode="top_adapter"))
    check("V6 adapter_hidden 改值：參數量改變",
          sum(p.numel() for p in ma.top_adapter.parameters()) != sum(p.numel() for p in mb.top_adapter.parameters()))
    for mode in ("frozen", "partial_l4", "full"):
        m = build(new1, new2, make_args("B1", "resnet18", backbone_mode=mode))
        print(f"      backbone_mode={mode}：可訓練 {sum(p.numel() for p in m.parameters() if p.requires_grad)/1e6:.3f} M")
    # train_aug：同一支影片讀兩次
    import random as pyrandom
    import video_reader
    for aug, expect_same in (("none", True), ("standard", False)):
        args.train_aug = aug
        args.img_size = 224
        vd = video_reader.VideoDataset(args)
        vd.train = True
        c = vd.get_train_or_test_db()
        label = c.get_unique_classes()[0] if hasattr(c, "get_unique_classes") else 0
        pyrandom.seed(0)
        same_all = True
        for idx in range(3):
            a, _ = vd.get_seq(label, idx)
            b, _ = vd.get_seq(label, idx)
            same_all &= torch.equal(a, b)
        check(f"V6 train_aug={aug}：同一支訓練影片讀兩次{'完全相同' if expect_same else '至少一支不同（有增強）'}",
              same_all == expect_same)
        check(f"V6 train_aug={aug}：仍讀訓練 split（vd.train 未被改動）", vd.train is True and c is vd.train_split)


def v7(new1, new2):
    print("\nV7 舊 checkpoint strict 載入（評估數字見 scripts/v7_eval_1005.py 的 log）")
    import glob
    ck = {
        "frozen B1 s42 (stage1)": (glob.glob(os.path.expanduser(
            "~/trx_data/checkpoints/frozen_rn18_b1_s42/*/checkpoint25000.pt"))[0], "B1"),
        "frozen B4 s42 (stage2)": (glob.glob(os.path.expanduser(
            "~/trx_data/checkpoints/frozen_rn18_b4_s42/*/checkpoint25000.pt"))[0], "B4"),
        "full B1 s43 (stage1)": (os.path.expanduser(
            "~/trx_data/checkpoints/b1_seed43_1shot_hmdb3/hmdb_split3_20260907_123206/checkpoint25000.pt"), "B1"),
    }
    for label, (path, head) in ck.items():
        sd = torch.load(path, map_location="cpu")["model_state_dict"]
        for mode in ("full", "frozen", "partial_l4"):
            a = make_args(head, "resnet18", backbone_mode=mode)
            a.img_size = 224
            m = build(new1, new2, a)
            r = m.load_state_dict(sd, strict=True)
            check(f"V7 {label} → backbone_mode={mode} strict 載入", not r.missing_keys and not r.unexpected_keys)
        m = build(new1, new2, make_args(head, "resnet18", backbone_mode="lst"))
        r = m.load_state_dict(sd, strict=False)
        check(f"V7 {label} → lst（strict=False）只缺 side_net.*、resnet 鍵名不變",
              all(k.startswith("side_net.") for k in r.missing_keys) and not r.unexpected_keys,
              f"缺 {len(r.missing_keys)} 個 side_net 鍵")


def main():
    ref1, ref2, commit = load_reference()
    import models.stage1_model as new1
    import models.stage2_model as new2
    print(f"參考版本：{REF_COMMIT} @ {commit}；新版本：working tree @ "
          f"{subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO).decode().strip()}")
    print(f"CPU, img={IMG}, way={WAY} shot={SHOT} qpc={QPC} seq={SEQ}；dropout=0、BN momentum=0")
    for fn in (v1, v2, v3, v4, v5, v6, v7):
        try:
            fn(ref1, ref2, new1, new2) if fn is v1 else fn(new1, new2)
        except Exception as e:      # 單項例外記為失敗，不中斷其他項
            import traceback
            traceback.print_exc()
            check(f"{fn.__name__} 執行例外", False, repr(e)[:200])
    print(f"\n通過 {len(_pass)} 項，失敗 {len(_fail)} 項")
    for f in _fail:
        print(f"  FAIL: {f}")
    sys.exit(1 if _fail else 0)


if __name__ == "__main__":
    main()
