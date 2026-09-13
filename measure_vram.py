"""
measure_vram.py — VRAM and speed benchmark for TRX backbone configurations.

Configurations tested (5-way 1-shot, qpc=5, seq_len=8):
  A: resnet18,  Ω={2},    no grad_ckpt
  B: resnet50,  Ω={2},    no grad_ckpt
  C: resnet50,  Ω={2},    grad_ckpt
  D: resnet50,  Ω={2,3},  grad_ckpt

Reports: peak VRAM (GB), ms/iter, estimated hours for 100k iters.
"""

import os, sys, time, math, argparse
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.checkpoint import checkpoint_sequential
from itertools import combinations

# ── Minimal pair-embedding head ──────────────────────────────────────────────

class PairHead(nn.Module):
    def __init__(self, in_dim, out_dim=1152, temp_set=(2,)):
        super().__init__()
        self.temp_set = list(temp_set)
        self.seq_len  = 8
        # one linear per temporal size
        self.linears = nn.ModuleList([
            nn.Linear(in_dim * t, out_dim) for t in self.temp_set
        ])
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        # x: (N, seq_len, in_dim)
        logits_all = []
        for t, lin in zip(self.temp_set, self.linears):
            idxs = list(combinations(range(self.seq_len), t))
            parts = []
            for combo in idxs:
                idx = torch.tensor(combo, device=x.device)
                xp = x[:, idx, :].reshape(x.shape[0], -1)  # (N, t*in_dim)
                parts.append(xp)
            tuples = torch.stack(parts, dim=1)              # (N, T, t*in_dim)
            z = self.norm(lin(tuples))                      # (N, T, out_dim)
            logits_all.append(z.mean())                     # scalar (just drives grad)
        return sum(logits_all)


class TinyModel(nn.Module):
    def __init__(self, method="resnet18", temp_set=(2,), grad_ckpt=False):
        super().__init__()
        self.grad_ckpt = grad_ckpt
        if method == "resnet18":
            resnet = models.resnet18(pretrained=True)
            in_dim = 512
        elif method == "resnet50":
            resnet = models.resnet50(pretrained=True)
            in_dim = 2048
        else:
            raise ValueError(method)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.head   = PairHead(in_dim, out_dim=1152, temp_set=temp_set)
        self.seq_len = 8

    def forward(self, imgs):
        # imgs: (N*seq_len, 3, H, W)
        if self.training and self.grad_ckpt:
            feats = checkpoint_sequential(
                self.resnet, 4, imgs, use_reentrant=True).squeeze(-1).squeeze(-1)
        else:
            feats = self.resnet(imgs).squeeze(-1).squeeze(-1)
        # reshape to (N, seq_len, dim)
        dim = feats.shape[1]
        feats = feats.reshape(-1, self.seq_len, dim)
        return self.head(feats)


# ── Benchmark function ────────────────────────────────────────────────────────

def run_config(name, method, temp_set, grad_ckpt, device, n_warmup=3, n_bench=10):
    WAY, SHOT, QPC, SEQ = 5, 1, 5, 8
    IMG = 224
    N_sup = WAY * SHOT * SEQ
    N_qry = WAY * QPC  * SEQ

    model = TinyModel(method=method, temp_set=temp_set, grad_ckpt=grad_ckpt)
    model = model.train().to(device)

    # BN momentum correction when grad_ckpt
    if grad_ckpt:
        m = 1 - math.sqrt(1 - 0.1)
        for mod in model.modules():
            if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
                mod.momentum = m

    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    def one_iter():
        opt.zero_grad()
        imgs = torch.randn(N_sup + N_qry, 3, IMG, IMG, device=device)
        out  = model(imgs)
        loss = loss_fn(out, torch.zeros_like(out))
        loss.backward()
        opt.step()
        return loss.item()

    # Warmup
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(n_warmup):
        one_iter()

    # Benchmark
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(n_bench):
        one_iter()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    ms_iter = elapsed / n_bench * 1000
    hrs_100k = (ms_iter / 1000) * 100_000 / 3600

    del model, opt
    torch.cuda.empty_cache()

    return peak_gb, ms_iter, hrs_100k


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_bench", type=int, default=10)
    ap.add_argument("--n_warmup", type=int, default=3)
    args = ap.parse_args()

    device = torch.device(args.device)

    configs = [
        ("A: RN18  Ω={2}       no-ckpt", "resnet18", (2,),    False),
        ("B: RN50  Ω={2}       no-ckpt", "resnet50", (2,),    False),
        ("C: RN50  Ω={2}       +ckpt  ", "resnet50", (2,),    True ),
        ("D: RN50  Ω={2,3}     +ckpt  ", "resnet50", (2, 3),  True ),
    ]

    print(f"\n{'Config':<35} {'Peak VRAM (GB)':>15} {'ms/iter':>10} {'100k hrs':>10}")
    print("-" * 75)
    results = {}
    for (label, method, tset, ckpt) in configs:
        try:
            gb, ms, hrs = run_config(
                label, method, tset, ckpt, device,
                n_warmup=args.n_warmup, n_bench=args.n_bench
            )
            results[label] = (gb, ms, hrs)
            print(f"{label:<35} {gb:>15.3f} {ms:>10.1f} {hrs:>10.2f}")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                results[label] = ("OOM", None, None)
                print(f"{label:<35} {'OOM':>15} {'N/A':>10} {'N/A':>10}")
                torch.cuda.empty_cache()
            else:
                raise

    print()
    # Verdict
    key_c = "C: RN50  Ω={2}       +ckpt  "
    key_d = "D: RN50  Ω={2,3}     +ckpt  "
    for key in (key_c, key_d):
        if key in results and results[key][0] != "OOM":
            gb = results[key][0]
            status = "✓ VIABLE (<10 GB)" if gb < 10.0 else "✗ OVER 10 GB"
            print(f"{key.strip()}: {gb:.3f} GB → {status}")
        else:
            print(f"{key.strip()}: OOM → ✗ NOT VIABLE")


if __name__ == "__main__":
    main()
