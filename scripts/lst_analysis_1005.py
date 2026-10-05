#!/usr/bin/env python3
"""
scripts/lst_analysis_1005.py — 1005 階段 1 的靜態分析（CPU，不訓練）。

印出：
  1. RN18／RN50 在 224 輸入下各 stage／各 block 的輸出 shape、forward FLOPs（conv+linear MAC×2）、
     該段內所有 module 輸出的元素數（≈ 訓練時要存的 activation 量的上界）。
  2. side network（提案設計）與 top adapter 的可訓練參數量。
  3. 離線快取的磁碟成本。
"""
import torch
import torch.nn as nn
import torchvision.models as tvm

N_FRAMES = 240  # (5 support + 25 query) × 8


def conv_flops(m, inp, out):
    if isinstance(m, nn.Conv2d):
        k = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
        return 2 * k * out.numel()
    if isinstance(m, nn.Linear):
        return 2 * m.in_features * out.numel()
    return 0


def profile(name):
    net = getattr(tvm, name)(weights=None).eval()
    seq = nn.Sequential(*list(net.children())[:-1])
    stats = {}
    hooks = []
    for top_name, top in seq.named_children():
        for sub in top.modules():
            def h(m, i, o, t=top_name):
                s = stats.setdefault(t, [0, 0])
                s[0] += conv_flops(m, i, o)
                if len(list(m.children())) == 0:
                    s[1] += o.numel()
            hooks.append(sub.register_forward_hook(h))
    x = torch.zeros(1, 3, 224, 224)
    shapes = {}
    with torch.no_grad():
        for n, m in seq.named_children():
            x = m(x)
            shapes[n] = tuple(x.shape[1:])
    for h in hooks:
        h.remove()
    tot = sum(v[0] for v in stats.values())
    print(f"\n== {name}  total fwd {tot/1e9:.3f} GFLOPs/frame")
    print(f"{'module':10s} {'out shape':18s} {'GFLOPs':>8s} {'%':>6s} {'act numel/frame':>16s} {'act GB @240 fp32':>17s}")
    for n in shapes:
        f, a = stats.get(n, [0, 0])
        print(f"{n:10s} {str(shapes[n]):18s} {f/1e9:8.3f} {100*f/tot:6.1f} {a:16,d} {a*4*N_FRAMES/1024**3:17.3f}")
    blocks = []
    for ln in ("layer1", "layer2", "layer3", "layer4"):
        for i, b in enumerate(getattr(seq, ln if False else {"layer1": "4", "layer2": "5", "layer3": "6", "layer4": "7"}[ln])):
            blocks.append((f"{ln}.{i}", b))
    return seq, shapes


def side_params(chs, w, D, res=7):
    rung = sum(c * w + w for c in chs)
    blocks = len(chs) * (9 * w * w + w + 2 * w)        # conv3x3 + bias + GN affine
    head = w * D + D
    alpha = 1
    return rung + blocks + head + alpha


def adapter_params(D, h):
    return D * h + h + h * D + D + 1


if __name__ == "__main__":
    for name in ("resnet18", "resnet50"):
        profile(name)

    print("\n== 參數量")
    cfg = {"resnet18": ([64, 128, 256, 512], 512), "resnet50": ([256, 512, 1024, 2048], 2048)}
    for name, (chs, D) in cfg.items():
        for w in (64, 128, 192, 256):
            p = side_params(chs, w, D)
            # 找 top adapter 隱藏寬度 h 讓參數量 ±10% 對齊
            h = round((p - D - 1) / (2 * D + 1))
            print(f"{name} side w={w:3d}: {p/1e6:.3f} M   top_adapter h={h} → {adapter_params(D, h)/1e6:.3f} M")

    print("\n== 磁碟（fp16）")
    n_train, n_test = 4280, 1292  # splits/hmdb_ARN trainlist03 / testlist03
    fr_tr, fr_te = n_train * 8, n_test * 8
    for name, (chs, D) in cfg.items():
        gap = D * 2
        rung7 = sum(chs) * 49 * 2
        rung14 = sum(chs) * 196 * 2
        for lab, per in (("GAP", gap), ("rung 7x7", rung7), ("rung 14x14", rung14)):
            row = [(fr_te + k * fr_tr) * per / 1e9 for k in (1, 2, 4, 8)]
            print(f"{name} {lab:10s} {per:8d} B/frame  k=1/2/4/8: " + " / ".join(f"{v:.2f}" for v in row) + " GB")
        print(f"{name} rung 7x7 每 episode 讀取量 {rung7*N_FRAMES/1e6:.1f} MB")
