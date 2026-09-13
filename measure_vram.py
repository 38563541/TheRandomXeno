"""
measure_vram.py — VRAM and speed benchmark for TRX backbone configurations.

Configurations tested (5-way 1-shot, qpc=5, seq_len=8):
  A: resnet18,  Ω={2},    no grad_ckpt
  B: resnet50,  Ω={2},    no grad_ckpt
  C: resnet50,  Ω={2},    grad_ckpt (children4 = old 4-segment, for baseline comparison)
  D: resnet50,  Ω={2,3},  grad_ckpt

Reports: peak allocated (GB), peak reserved (GB), ms/iter, 學? (grad flows)

NOTE: The Colab sweep in the task prompt reports "peak reserved GB".
      PyTorch's allocator reserves more memory than it actually uses:
        peak_reserved = torch.cuda.max_memory_reserved()
        peak_allocated = torch.cuda.max_memory_allocated()
      Both are reported here to allow direct comparison with Colab numbers.

NOTE: use_reentrant=False (matches training code in models/).
      Old script used use_reentrant=True which caused "None of the inputs have
      requires_grad=True. Gradients will be None" warnings.
"""

import os, sys, time, math, argparse, csv
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
    def __init__(self, method="resnet18", temp_set=(2,), grad_ckpt=False,
                 ckpt_segments=8, frozen=False):
        super().__init__()
        self.grad_ckpt     = grad_ckpt
        self.ckpt_segments = ckpt_segments
        self.frozen        = frozen

        if method == "resnet18":
            resnet = models.resnet18(pretrained=True)
            in_dim = 512
        elif method == "resnet34":
            resnet = models.resnet34(pretrained=True)
            in_dim = 512
        elif method == "resnet50":
            resnet = models.resnet50(pretrained=True)
            in_dim = 2048
        elif method == "resnet101":
            resnet = models.resnet101(pretrained=True)
            in_dim = 2048
        else:
            raise ValueError(method)

        self.resnet = nn.Sequential(*list(resnet.children())[:-1])

        # Disable inplace ReLU for use_reentrant=False compatibility
        for m in self.resnet.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False

        if frozen:
            for p in self.resnet.parameters():
                p.requires_grad_(False)
            self.resnet.eval()

        self.head    = PairHead(in_dim, out_dim=1152, temp_set=temp_set)
        self.seq_len = 8

    def _ckpt_chain(self):
        """攤平 layer1-4 成個別 residual block 視圖，不重新註冊。"""
        out = []
        for m in self.resnet:
            if isinstance(m, nn.Sequential):
                out.extend(list(m))
            else:
                out.append(m)
        return out

    def train(self, mode=True):
        super().train(mode)
        if self.frozen and hasattr(self, "resnet"):
            self.resnet.eval()
        return self

    def forward(self, imgs):
        # imgs: (N*seq_len, 3, H, W)
        if self.training and self.grad_ckpt and not self.frozen:
            chain = self._ckpt_chain()
            segs  = min(self.ckpt_segments or len(chain), len(chain))
            feats = checkpoint_sequential(
                chain, segs, imgs, use_reentrant=False).squeeze(-1).squeeze(-1)
        else:
            feats = self.resnet(imgs).squeeze(-1).squeeze(-1)
        # reshape to (N, seq_len, dim)
        dim = feats.shape[1]
        feats = feats.reshape(-1, self.seq_len, dim)
        return self.head(feats)


# ── Benchmark function ────────────────────────────────────────────────────────

def run_config(method, temp_set, grad_ckpt, device, n_warmup=3, n_bench=10,
               ckpt_segments=8, frozen=False):
    WAY, SHOT, QPC, SEQ = 5, 1, 5, 8
    IMG = 224
    N_sup = WAY * SHOT * SEQ
    N_qry = WAY * QPC  * SEQ

    model = TinyModel(method=method, temp_set=temp_set, grad_ckpt=grad_ckpt,
                      ckpt_segments=ckpt_segments, frozen=frozen)
    model = model.train().to(device)

    # BN momentum correction when grad_ckpt (matches training code)
    if grad_ckpt and not frozen:
        m = 1 - math.sqrt(1 - 0.1)
        for mod in model.modules():
            if isinstance(mod, nn.modules.batchnorm._BatchNorm):
                mod.momentum = m

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(trainable, lr=1e-3)
    loss_fn = nn.MSELoss()

    grad_ok = None

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

    # Benchmark + check grads on final iter
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for i in range(n_bench):
        one_iter()
        if i == n_bench - 1:
            # Check if backbone first-layer weight got a grad
            for p in model.resnet.parameters():
                if p.requires_grad:
                    grad_ok = (p.grad is not None and p.grad.abs().sum().item() > 0)
                    break
            if grad_ok is None and frozen:
                # frozen: no backbone grad expected; check head grad
                head_grad = any(
                    p.grad is not None and p.grad.abs().sum().item() > 0
                    for p in model.head.parameters()
                )
                grad_ok = head_grad  # frozen model "學?" = head is learning
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    peak_alloc_gb    = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    peak_reserved_gb = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
    ms_iter  = elapsed / n_bench * 1000
    hrs_100k = (ms_iter / 1000) * 100_000 / 3600

    del model, opt
    torch.cuda.empty_cache()

    return peak_alloc_gb, peak_reserved_gb, ms_iter, hrs_100k, grad_ok


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device",        default="cuda:0")
    ap.add_argument("--n_bench",       type=int, default=10)
    ap.add_argument("--n_warmup",      type=int, default=3)
    ap.add_argument("--method",        default=None,
                    help="override backbone (resnet18/34/50/101); for single-config run")
    ap.add_argument("--shot",          type=int, default=1)
    ap.add_argument("--qpc",           type=int, default=5)
    ap.add_argument("--ckpt",          action="store_true", default=False,
                    help="enable grad_ckpt for single-config run")
    ap.add_argument("--ckpt_segments", type=int, default=8,
                    help="flattened checkpoint segments (0=max=len(chain))")
    ap.add_argument("--frozen",        action="store_true", default=False)
    ap.add_argument("--sweep",         action="store_true", default=False,
                    help="run full sweep (A/B/C/D configs)")
    ap.add_argument("--csv",           default=None,
                    help="write results to this CSV file")
    args = ap.parse_args()

    # Idle gate
    if torch.cuda.is_available():
        used_mib = torch.cuda.memory_reserved(args.device) / (1024**2)
        # Also check nvidia-smi via subprocess
        import subprocess
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
                text=True).strip()
            smi_mib = int(out.split()[0])
        except Exception:
            smi_mib = 0
        if smi_mib > 500:
            print(f"[WARN] GPU memory.used = {smi_mib} MiB > 500 MiB — not idle!")
            print("       Results will be unreliable. Proceeding anyway (記錄即可).")
        else:
            print(f"[INFO] GPU idle check: memory.used = {smi_mib} MiB ✓")

    device = torch.device(args.device)

    if args.sweep or args.method is None:
        configs = [
            ("A: RN18  Ω={2}  no-ckpt",       "resnet18", (2,),   False, 8,   False),
            ("B: RN50  Ω={2}  no-ckpt",        "resnet50", (2,),   False, 8,   False),
            ("C: RN50  Ω={2}  +ckpt (c4)",     "resnet50", (2,),   True,  4,   False),
            ("D: RN50  Ω={2,3}+ckpt (c4)",     "resnet50", (2,3),  True,  4,   False),
            ("E: RN50  Ω={2}  +ckpt (flat8)",  "resnet50", (2,),   True,  8,   False),
            ("F: RN18  Ω={2}  frozen",         "resnet18", (2,),   False, 8,   True),
            ("G: RN50  Ω={2}  frozen",         "resnet50", (2,),   False, 8,   True),
        ]
    else:
        segs = args.ckpt_segments
        configs = [
            (f"{args.method} ckpt={args.ckpt} segs={segs} frozen={args.frozen}",
             args.method, (2,), args.ckpt, segs, args.frozen)
        ]

    hdr = f"{'Config':<40} {'Alloc(GB)':>10} {'Reserv(GB)':>11} {'ms/iter':>9} {'100k h':>8} {'學?':>4}"
    print(f"\n{hdr}")
    print("-" * 86)

    rows = []
    for (label, method, tset, ckpt, segs, frozen) in configs:
        try:
            alloc, reserv, ms, hrs, grad_ok = run_config(
                method, tset, ckpt, device,
                n_warmup=args.n_warmup, n_bench=args.n_bench,
                ckpt_segments=segs, frozen=frozen,
            )
            g = "✓" if grad_ok else ("✗" if grad_ok is not None else "?")
            print(f"{label:<40} {alloc:>10.3f} {reserv:>11.3f} {ms:>9.1f} {hrs:>8.2f} {g:>4}")
            rows.append({"label": label, "method": method, "grad_ckpt": ckpt,
                         "ckpt_segments": segs, "frozen": frozen,
                         "alloc_gb": round(alloc, 3), "reserved_gb": round(reserv, 3),
                         "ms_iter": round(ms, 1), "hrs_100k": round(hrs, 2),
                         "grad_ok": g})
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{label:<40} {'OOM':>10} {'OOM':>11} {'N/A':>9} {'N/A':>8} {'N/A':>4}")
                rows.append({"label": label, "method": method, "grad_ckpt": ckpt,
                             "ckpt_segments": segs, "frozen": frozen,
                             "alloc_gb": "OOM", "reserved_gb": "OOM",
                             "ms_iter": "N/A", "hrs_100k": "N/A", "grad_ok": "N/A"})
                torch.cuda.empty_cache()
            else:
                raise

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) if os.path.dirname(args.csv) else ".", exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            fieldnames = ["label", "method", "grad_ckpt", "ckpt_segments", "frozen",
                          "alloc_gb", "reserved_gb", "ms_iter", "hrs_100k", "grad_ok"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\n[INFO] written to {args.csv}")


if __name__ == "__main__":
    main()
