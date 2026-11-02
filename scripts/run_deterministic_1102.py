#!/usr/bin/env python3
"""
scripts/run_deterministic_1102.py — 1102 Phase 2 包裝程式：設好確定性模式後呼叫 run.py 的 main()。

  python scripts/run_deterministic_1102.py --warn_log logs/1102_p2_<tag>_warnings.txt -- <run.py 參數原樣>

確定性設定（與 verify_ckpt_equiv_1102.py 相同）：CUBLAS_WORKSPACE_CONFIG=:4096:8（import torch 之前）、
torch.use_deterministic_algorithms(True, warn_only=True)、cudnn.deterministic=True、cudnn.benchmark=False。
seed 由 run.py 的 --seed 處理（Learner.__init__ 四個來源都設）。不改 run.py 任何邏輯；
所有 Python warning 去重後連同次數寫到 --warn_log（non-deterministic 運算的警告會在裡面）。
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import atexit      # noqa: E402
import warnings    # noqa: E402

argv = sys.argv[1:]
warn_log = None
if argv and argv[0] == "--warn_log":
    warn_log = argv[1]
    argv = argv[2:]
if argv and argv[0] == "--":
    argv = argv[1:]

WARN = {}


def _showwarning(message, category, filename, lineno, file=None, line=None):
    key = f"{category.__name__}: {str(message).strip()}"
    WARN[key] = WARN.get(key, 0) + 1
    if WARN[key] == 1:
        sys.stderr.write(f"[warn] {os.path.basename(filename)}:{lineno} {key[:400]}\n")


def _dump():
    if warn_log:
        with open(warn_log, "w") as f:
            f.write(f"# deterministic={torch.are_deterministic_algorithms_enabled()} "
                    f"cudnn.deterministic={torch.backends.cudnn.deterministic} "
                    f"cudnn.benchmark={torch.backends.cudnn.benchmark} "
                    f"CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')}\n")
            for k, v in sorted(WARN.items(), key=lambda kv: -kv[1]):
                f.write(f"{v}\t{k}\n")


warnings.showwarning = _showwarning
warnings.simplefilter("always")

import torch  # noqa: E402

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
atexit.register(_dump)

import run  # noqa: E402

sys.argv = ["run.py"] + argv
print(f"[det] deterministic on; run.py {' '.join(argv)}", flush=True)
run.main()
