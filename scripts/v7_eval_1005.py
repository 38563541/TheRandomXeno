#!/usr/bin/env python3
"""scripts/v7_eval_1005.py — 階段 3 V7：舊 checkpoint 在新程式下純評估，結果要等於原紀錄。

兩個都是「25k checkpoint ＋ 原 run 的 seed」：純評估重播的是第一個評估點的 episode，
所以只能拿 25k（訓練中第一次評估）的數字對。不寫 results.csv（純評估預設不寫）。
"""
import glob
import os
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = os.path.expanduser("~/Documents/work/trx/trx_data")
CK = os.path.expanduser("~/trx_data/checkpoints")

JOBS = [
    ("frozen_b1_s42_25k", "configs/frozen_rn18_b1.yaml", "42",
     glob.glob(f"{CK}/frozen_rn18_b1_s42/*/checkpoint25000.pt")[0], "46.456"),
    ("full_b1_s43_25k", "configs/b1_backbone_1shot.yaml", "43",
     f"{CK}/b1_seed43_1shot_hmdb3/hmdb_split3_20260907_123206/checkpoint25000.pt", "49.502"),
]

for tag, cfg, seed, ckpt, expect in JOBS:
    log = os.path.join(REPO, "logs", f"v7_1005_{tag}.log")
    cmd = ["python", "run.py", "--dataset", "hmdb", "--split", "3", "--scratch", SCRATCH,
           "--config", cfg, "--seed", seed, "-m", ckpt, "-c", f"{CK}/v7_1005_{tag}"]
    print(f"=== {tag} (expect {expect}): {' '.join(cmd)}", flush=True)
    with open(log, "w") as f:
        rc = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT).returncode
    print(f"=== {tag} rc={rc}", flush=True)
print("=== V7 DONE", flush=True)
