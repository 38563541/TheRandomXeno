"""
models/backbone_adapt.py — 1005：backbone 端的可調性（arm A–E）。

backbone_mode（config key，預設 full = 現行行為）：
    full         全部可訓練（現行行為，E）
    frozen       只有匹配頭可訓練；backbone requires_grad=False、BN 永遠 eval（= 既有 --freeze_backbone，A）
    top_adapter  凍結 backbone，GAP 後接 D→h→D 的 MLP，f = GAP + α·MLP(GAP)，α 初始 0（B）
    lst          凍結 backbone，4 個 stage 輸出經 rung（AdaptiveAvgPool→1×1 conv）進 side network，
                 f = GAP + α·Linear(GAP(side))，α 初始 0（C）
    partial_l4   只有 layer4 可訓練；conv1/bn1/layer1–3 的 BN 固定 eval、layer4 的 BN 照 train 模式（D）

設計細節與理由見 1005_LST可行性分析.md 1-2。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

BACKBONE_MODES = ("full", "frozen", "partial_l4", "lst", "top_adapter")
FROZEN_MODES = ("frozen", "lst", "top_adapter")
_STAGE_CHANNELS = {
    "resnet18": [64, 128, 256, 512],
    "resnet34": [64, 128, 256, 512],
    "resnet50": [256, 512, 1024, 2048],
}
_DEFAULT_SIDE_WIDTH = {"resnet18": 128, "resnet34": 128, "resnet50": 256}
_L4_INDEX = 7   # self.resnet = [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool]


def side_param_count(chs, width):
    """SideNet 的可訓練參數量（與下方模組逐項對應；verify_lst.py 會核對）。"""
    D = chs[-1]
    rung = sum(c * width + width for c in chs)
    blocks = len(chs) * (9 * width * width + width + 2 * width)
    return rung + blocks + width * D + D + 1


def normalize_backbone_mode(args):
    """解析並補齊 backbone_mode 相關的 args（冪等；run.py 與模型 __init__ 都會呼叫）。

    - freeze_backbone=True 且 backbone_mode=full → 視為 frozen（向後相容）
    - backbone_mode ∈ FROZEN_MODES → freeze_backbone=True（沿用既有凍結／eval／自動關 ckpt 路徑）
    - partial_l4 不准同時 freeze_backbone
    - side_width / adapter_hidden 的 -1 解析成實際值，讓 log 與 CSV 記到真正用的數字
    """
    mode = getattr(args, "backbone_mode", None) or "full"
    if mode not in BACKBONE_MODES:
        raise ValueError(f"backbone_mode={mode!r} 不在 {BACKBONE_MODES}")
    if mode == "full" and getattr(args, "freeze_backbone", False):
        mode = "frozen"
    if mode in FROZEN_MODES:
        args.freeze_backbone = True
    elif mode == "partial_l4" and getattr(args, "freeze_backbone", False):
        raise ValueError("backbone_mode=partial_l4 不能同時 --freeze_backbone")
    args.backbone_mode = mode

    method = getattr(args, "method", "resnet18")
    if (getattr(args, "side_width", -1) or -1) <= 0:
        args.side_width = _DEFAULT_SIDE_WIDTH.get(method, 128)
    if (getattr(args, "adapter_hidden", -1) or -1) <= 0:
        chs = _STAGE_CHANNELS.get(method, _STAGE_CHANNELS["resnet18"])
        D = chs[-1]
        args.adapter_hidden = max(1, round((side_param_count(chs, args.side_width) - D - 1) / (2 * D + 1)))
    if not hasattr(args, "side_res") or args.side_res is None:
        args.side_res = 7
    if not hasattr(args, "side_norm") or args.side_norm is None:
        args.side_norm = "gn"
    return mode


def _make_norm(kind, width):
    if kind == "gn":
        return nn.GroupNorm(8, width)
    if kind == "bn":            # 只給 verify_lst.py V5 當正控制組用
        return nn.BatchNorm2d(width)
    if kind == "bn_notrack":    # 同上：eval 也用 batch 統計量
        return nn.BatchNorm2d(width, track_running_stats=False)
    raise ValueError(f"side_norm={kind!r}")


class SideNet(nn.Module):
    """Ladder side network（1005 提案）。輸入：4 個 stage 的凍結輸出；輸出：α · [N, D]。"""

    def __init__(self, chs, width, res=7, norm="gn"):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(res)
        self.rungs = nn.ModuleList([nn.Conv2d(c, width, 1) for c in chs])
        self.convs = nn.ModuleList([nn.Conv2d(width, width, 3, padding=1) for _ in chs])
        self.norms = nn.ModuleList([_make_norm(norm, width) for _ in chs])
        self.proj = nn.Linear(width, chs[-1])
        self.alpha = nn.Parameter(torch.zeros(()))

    def rung(self, i, f):
        # pool → 1×1 conv：兩步都線性、可交換（V4），rung 輸入可離線快取
        return self.rungs[i](self.pool(f))

    def forward(self, feats):
        h = None
        for i, f in enumerate(feats):
            r = self.rung(i, f)
            z = r if h is None else h + r
            h = z + F.relu(self.norms[i](self.convs[i](z)))
        return self.alpha * self.proj(h.mean(dim=(2, 3)))


class TopAdapter(nn.Module):
    """GAP 後的容量對照（B）：α · Linear(ReLU(Linear(f)))，參數量對齊 SideNet。"""

    def __init__(self, D, hidden):
        super().__init__()
        self.fc1 = nn.Linear(D, hidden)
        self.fc2 = nn.Linear(hidden, D)
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, f):
        return self.alpha * self.fc2(F.relu(self.fc1(f)))


def setup_partial_l4(model):
    """在模型 __init__ 的凍結區塊之後呼叫：partial_l4 時只留 layer4 可訓練、其餘 BN 固定 eval。"""
    if model.args.backbone_mode != "partial_l4":
        return
    for i, m in enumerate(model.resnet):
        if i != _L4_INDEX:
            for p in m.parameters():
                p.requires_grad_(False)
            m.eval()
    n = sum(p.numel() for p in model.resnet.parameters() if p.requires_grad)
    print(f"[INFO] backbone_mode=partial_l4: 只有 layer4 可訓練（{n/1e6:.2f} M），"
          f"layer1–3 BN 固定 eval", flush=True)


def setup_adapters(model):
    """在匹配頭建立「之後」呼叫，讓同 seed 下匹配頭的初始化與 frozen baseline 相同。"""
    args = model.args
    mode = args.backbone_mode
    chs = _STAGE_CHANNELS[args.method]
    if mode == "lst":
        model.side_net = SideNet(chs, args.side_width, args.side_res, args.side_norm)
        n = sum(p.numel() for p in model.side_net.parameters())
        print(f"[INFO] backbone_mode=lst: side width={args.side_width} res={args.side_res} "
              f"norm={args.side_norm}，可訓練 {n/1e6:.3f} M，α 初始 0", flush=True)
    elif mode == "top_adapter":
        model.top_adapter = TopAdapter(chs[-1], args.adapter_hidden)
        n = sum(p.numel() for p in model.top_adapter.parameters())
        print(f"[INFO] backbone_mode=top_adapter: hidden={args.adapter_hidden}，"
              f"可訓練 {n/1e6:.3f} M，α 初始 0", flush=True)


def apply_backbone_train_mode(model):
    """train() override 共用：凍結的部分永遠 eval（BN running stats 不動）。"""
    if getattr(model, "resnet", None) is None:
        return
    args = getattr(model, "args", None)
    if getattr(args, "freeze_backbone", False):
        model.resnet.eval()
    elif getattr(args, "backbone_mode", "full") == "partial_l4":
        for i, m in enumerate(model.resnet):
            if i != _L4_INDEX:
                m.eval()


def backbone_features(model, x):
    """非 checkpoint 路徑的 backbone：回傳 [N, D]。full／frozen／partial_l4 與舊版完全相同。"""
    mode = getattr(model.args, "backbone_mode", "full")
    if mode == "lst":
        feats = []
        h = x
        for i, m in enumerate(model.resnet):     # 與 nn.Sequential.forward 相同的逐層呼叫
            h = m(h)
            if 4 <= i <= _L4_INDEX:
                feats.append(h)
        f = h.squeeze()
        return f + model.side_net(feats)
    f = model.resnet(x).squeeze()
    if mode == "top_adapter":
        return f + model.top_adapter(f)
    return f


def adapter_alpha(model):
    """給 log 用：lst／top_adapter 的 α 目前值；其他模式回傳 None。"""
    for name in ("side_net", "top_adapter"):
        m = getattr(model, name, None)
        if m is not None:
            return float(m.alpha.detach())
    return None
