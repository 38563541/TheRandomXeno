#!/usr/bin/env python3
"""
scripts/train_queue_1005.py — 1005 階段 5／6 的訓練佇列 driver。

- 每次要開下一個 job 前重新讀 logs/train_queue_1005.json（可以在 driver 跑的時候改順序或刪 job），
  依序取第一個「還沒出現在 logs/train_queue_1005_done.csv」的 job 跑；佇列空了就結束。
- 每個 job 是一個獨立 run.py process；開跑前等 GPU 閒置（nvidia-smi < 200 MB）。
- 結束後記一列：tag、開始／結束時間、時數、rc、狀態（ok／nan／oom／crash）、checkpoint 目錄、
  log 裡所有 'hmdb: x+/-y' 評估結果。單一 job 失敗不會讓 driver 停。
- 檔案 logs/train_queue_1005.STOP 存在時，跑完目前 job 就停。

job 格式（json list）：{"tag": "...", "args": ["--config", "...", ...], "est_hours": 3.4}

§0 預算守門（G4）：開每個 job 前算「已用 ＋ 這個 job 的 est_hours」，超過 48 h 就不跑，
記一列 status=skipped_budget 後看下一個。已用 = logs/gpu_hours_fixed_1005.txt 裡的固定時數
（階段 4 成本量測、純評估等非佇列工作，每行一個數字）＋ done CSV 裡所有 job 的 hours 加總。
固定附加：--dataset hmdb --split 3 --scratch <SCRATCH> -c ~/trx_data/checkpoints/lst1005_<tag>
"""
import csv
import datetime
import json
import os
import re
import subprocess
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = os.path.expanduser("~/Documents/work/trx/trx_data")
CKPT_ROOT = os.path.expanduser("~/trx_data/checkpoints")
QUEUE = os.path.join(REPO, "logs", "train_queue_1005.json")
DONE = os.path.join(REPO, "logs", "train_queue_1005_done.csv")
STOP = os.path.join(REPO, "logs", "train_queue_1005.STOP")
FIXED = os.path.join(REPO, "logs", "gpu_hours_fixed_1005.txt")
BUDGET_H = 48.0
FIELDS = ["tag", "start", "end", "hours", "rc", "status", "ckpt_dir", "evals", "cmd"]


def done_tags():
    if not os.path.exists(DONE):
        return set()
    return {r["tag"] for r in csv.DictReader(open(DONE))}


def used_hours():
    fixed = 0.0
    if os.path.exists(FIXED):
        for line in open(FIXED):
            tok = line.split("#")[0].strip()
            if tok:
                fixed += float(tok.split()[0])
    q = 0.0
    if os.path.exists(DONE):
        for r in csv.DictReader(open(DONE)):
            try:
                q += float(r["hours"])
            except ValueError:
                pass
    return fixed + q


def write_done(row):
    new = not os.path.exists(DONE)
    with open(DONE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def gpu_used_mb():
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
                                   "--format=csv,noheader,nounits"]).decode()
    return int(out.split()[0])


def main():
    while True:
        if os.path.exists(STOP):
            print("=== STOP file found, exiting", flush=True)
            break
        queue = json.load(open(QUEUE))
        dt = done_tags()
        todo = [j for j in queue if j["tag"] not in dt]
        if not todo:
            print("=== queue empty", flush=True)
            break
        job = todo[0]
        used, est = used_hours(), float(job.get("est_hours", 0))
        if used + est > BUDGET_H:
            now = datetime.datetime.now()
            print(f"=== {now:%m-%d %H:%M:%S} SKIP {job['tag']}: 已用 {used:.2f} h ＋ 預估 {est:.2f} h "
                  f"> {BUDGET_H} h", flush=True)
            write_done(dict(tag=job["tag"], start=f"{now:%Y-%m-%d %H:%M:%S}", end="", hours="0",
                            rc="", status="skipped_budget", ckpt_dir="",
                            evals=f"used={used:.2f} est={est:.2f}", cmd=""))
            continue
        print(f"=== budget: 已用 {used:.2f} h ＋ 預估 {est:.2f} h ≤ {BUDGET_H} h", flush=True)
        for _ in range(30):
            if gpu_used_mb() < 200:
                break
            time.sleep(60)
        tag = job["tag"]
        ck = os.path.join(CKPT_ROOT, f"lst1005_{tag}")
        cmd = ["python", "run.py", *job["args"], "--dataset", "hmdb", "--split", "3",
               "--scratch", SCRATCH, "-c", ck]
        log = os.path.join(REPO, "logs", f"lst1005_{tag}.log")
        t0 = datetime.datetime.now()
        print(f"=== {t0:%m-%d %H:%M:%S} START {tag}: {' '.join(cmd)}", flush=True)
        with open(log, "w") as f:
            rc = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT,
                                env={**os.environ, "PYTHONUNBUFFERED": "1"}).returncode
        t1 = datetime.datetime.now()
        text = open(log, errors="replace").read()
        status = "ok" if rc == 0 else ("nan" if rc == 97 else
                                       ("oom" if "out of memory" in text or "OutOfMemoryError" in text else "crash"))
        evals = ";".join(re.findall(r"hmdb: ([\d.]+\+/-[\d.]+)", text))
        write_done(dict(tag=tag, start=f"{t0:%Y-%m-%d %H:%M:%S}", end=f"{t1:%Y-%m-%d %H:%M:%S}",
                            hours=f"{(t1 - t0).total_seconds() / 3600:.3f}", rc=rc, status=status,
                            ckpt_dir=ck, evals=evals, cmd=" ".join(cmd)))
        print(f"=== {t1:%m-%d %H:%M:%S} END {tag}: rc={rc} status={status} evals={evals}", flush=True)
    print("=== driver exit", flush=True)


if __name__ == "__main__":
    main()
