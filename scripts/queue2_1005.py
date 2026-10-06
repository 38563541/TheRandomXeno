#!/usr/bin/env python3
"""
scripts/queue2_1005.py — 1005 第二個佇列 driver（使用者 10-06 指示 3–5）：可並行的 group、
GPU 佔用 wall-clock 預算、開跑前自己做多線吞吐測試並機械判定是否採用並行。用 setsid 啟動，
使用者不在、Claude 也沒在動的時候 GPU 照樣能跑。

流程
  0. 等舊 driver（logs/queue_1005.pid）結束、GPU 閒置。
  1. 多線吞吐測試（logs/parallel_decision_1005.json 不存在時才做）：RN18 A（凍結 B1）300 iter，
     同時跑 1／2／3 個 job，記每個 job 的 --profile_time 中位數、nvidia-smi 峰值（整張卡）、
     系統 RAM 最小 available。採用 n 個並行的條件（全部要成立）：
        總吞吐 Σ(1000/ms_i) ≥ 1.3 × 單 job 吞吐；nvidia-smi 峰值 < 11.0 GiB；RAM 最小 available ≥ 1024 MB。
     max_jobs = 通過條件的最大 n（都不過 = 1，等於序列）。
  2. 依序跑 logs/queue2_1005.json 的 group。每個 group 有多條 lane（lane 內的 job 依序跑）；
     lane 以 min(max_jobs, group.max_parallel) 個槽位並行。group.max_parallel=1 → 永遠單獨跑
     （RN50 E1 全微調）。
  3. 預算（使用者指示 3：改算 GPU 佔用的 wall-clock）：已用 = logs/gpu_hours_fixed_1005.txt
     ＋ 舊佇列 done CSV 的 hours（那時都是單獨跑，等於 wall-clock）＋ 本 driver 每個 group 的 wall-clock。
     group 預估 = Σ lane 預估 ÷ 並行加速比（max_jobs>1 時用測試量到的加速比，否則 1）。
     超過 48 h → 不跑整個 group，改逐 job 檢查（序列），放得下的才跑。
  4. 每個 job 一列（並行時 hours 欄是 wall-clock，標「不可比」），每個 group 一列（wall-clock，計入預算）。
     ms/iter 一律只用階段 4 單獨量的值。
  5. logs/queue2_1005.STOP 存在 → 目前 group 跑完就停。
"""
import csv
import datetime
import json
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(REPO, "logs")
SCRATCH = os.path.expanduser("~/Documents/work/trx/trx_data")
CKPT_ROOT = os.path.expanduser("~/trx_data/checkpoints")
QUEUE = os.path.join(LOG, "queue2_1005.json")
DONE = os.path.join(LOG, "queue2_1005_done.csv")
OLD_DONE = os.path.join(LOG, "train_queue_1005_done.csv")
FIXED = os.path.join(LOG, "gpu_hours_fixed_1005.txt")
DECISION = os.path.join(LOG, "parallel_decision_1005.json")
STOP = os.path.join(LOG, "queue2_1005.STOP")
OLD_PID = os.path.join(LOG, "queue_1005.pid")
BUDGET_H = 48.0
SMI_LIMIT_GIB = 11.0
RAM_FLOOR_MB = 1024
FIELDS = ["kind", "name", "start", "end", "hours", "parallel", "rc", "status", "evals",
          "min_avail_mb", "smi_peak_gib", "note", "cmd"]


def now():
    return datetime.datetime.now()


def say(msg):
    print(f"=== {now():%m-%d %H:%M:%S} {msg}", flush=True)


def gpu_used_mb():
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
                                   "--format=csv,noheader,nounits"]).decode()
    return int(out.split()[0])


def avail_mb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    return -1


class Sampler:
    """背景輪詢 nvidia-smi 與 MemAvailable，記峰值／最小值。"""

    def __init__(self):
        self.smi_peak, self.min_avail, self._stop = 0, 10 ** 9, False
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        while not self._stop:
            try:
                self.smi_peak = max(self.smi_peak, gpu_used_mb())
            except Exception:
                pass
            self.min_avail = min(self.min_avail, avail_mb())
            time.sleep(2)

    def stop(self):
        self._stop = True
        self.t.join(timeout=5)
        return self.smi_peak / 1024.0, self.min_avail


def write_row(row):
    new = not os.path.exists(DONE)
    with open(DONE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def used_hours():
    h = 0.0
    if os.path.exists(FIXED):
        for line in open(FIXED):
            tok = line.split("#")[0].strip()
            if tok:
                h += float(tok.split()[0])
    for path, kind in ((OLD_DONE, None), (DONE, "group")):
        if os.path.exists(path):
            for r in csv.DictReader(open(path)):
                if kind and r.get("kind") != kind:
                    continue
                try:
                    h += float(r["hours"])
                except (ValueError, KeyError):
                    pass
    return h


def done_names(kind):
    if not os.path.exists(DONE):
        return set()
    return {r["name"] for r in csv.DictReader(open(DONE)) if r["kind"] == kind}


def wait_idle(max_min=60):
    for _ in range(max_min):
        if gpu_used_mb() < 200:
            return True
        time.sleep(60)
    return False


def run_job(job, parallel):
    tag = job["tag"]
    ck = os.path.join(CKPT_ROOT, f"lst1005_{tag}")
    cmd = ["python", "run.py", *job["args"], "--dataset", "hmdb", "--split", "3",
           "--scratch", SCRATCH, "-c", ck]
    log = os.path.join(LOG, f"lst1005_{tag}.log")
    t0 = now()
    say(f"START {tag}{'（並行）' if parallel else ''}: {' '.join(cmd)}")
    with open(log, "w") as f:
        rc = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT,
                            env={**os.environ, "PYTHONUNBUFFERED": "1"}).returncode
    t1 = now()
    text = open(log, errors="replace").read()
    status = "ok" if rc == 0 else ("nan" if rc == 97 else
                                   ("oom" if ("out of memory" in text or "OutOfMemoryError" in text) else "crash"))
    evals = ";".join(f"{float(a):.4f}+/-{float(c):.4f}" for a, c in
                     re.findall(r"\{'hmdb': \{'accuracy': ([\d.]+), 'confidence': ([\d.]+)\}\}", text))
    note = "wall-clock 不可比（並行）" if parallel else ""
    if status in ("crash", "oom", "nan"):
        note += " | " + " / ".join(text.splitlines()[-30:])[-600:]
    write_row(dict(kind="job", name=tag, start=f"{t0:%F %T}", end=f"{t1:%F %T}",
                   hours=f"{(t1 - t0).total_seconds() / 3600:.3f}", parallel=int(parallel), rc=rc,
                   status=status, evals=evals, note=note, cmd=" ".join(cmd)))
    say(f"END {tag}: rc={rc} status={status} evals={evals}")
    return status


def run_lane(lane, parallel):
    dn = done_names("job")
    for job in lane:
        if job["tag"] in dn:
            continue
        run_job(job, parallel)


# ----------------------------------------------------------------------------
def parallel_test():
    base = ["python", "run.py", "--config", "configs/frozen_rn18_b1.yaml", "--seed", "42",
            "--dataset", "hmdb", "--split", "3", "--scratch", SCRATCH,
            "--training_iterations", "300", "--print_freq", "100", "--save_freq", "999999",
            "--test_iters", "999999", "--profile_time"]
    res = {}
    for n in (1, 2, 3):
        wait_idle()
        time.sleep(10)
        say(f"吞吐測試 n={n}")
        s = Sampler()
        procs = []
        for i in range(n):
            log = os.path.join(LOG, f"par1005_n{n}_{i}.log")
            procs.append((subprocess.Popen(base + ["-c", os.path.join(CKPT_ROOT, f"par1005_n{n}_{i}")],
                                           cwd=REPO, stdout=open(log, "w"), stderr=subprocess.STDOUT,
                                           env={**os.environ, "PYTHONUNBUFFERED": "1"}), log))
        rcs = [p.wait() for p, _ in procs]
        smi, mavail = s.stop()
        ms = []
        for _, log in procs:
            m = re.search(r"total\(wall\)=([\d.]+)ms", open(log, errors="replace").read())
            ms.append(float(m.group(1)) if m else None)
        ok = all(r == 0 for r in rcs) and None not in ms
        thr = sum(1000.0 / x for x in ms) if ok else 0.0
        res[n] = dict(rc=rcs, ms=ms, it_s=thr, smi_peak_gib=smi, min_avail_mb=mavail, ok=ok)
        say(f"n={n}: rc={rcs} ms={ms} 總 it/s={thr:.3f} smi={smi:.3f} GiB RAM 最小 available={mavail} MB")
    t1 = res[1]["it_s"]
    max_jobs, speed = 1, {1: 1.0}
    for n in (2, 3):
        r = res[n]
        sp = r["it_s"] / t1 if t1 else 0.0
        speed[n] = sp
        r["speedup"] = sp
        r["pass"] = bool(r["ok"] and sp >= 1.3 and r["smi_peak_gib"] < SMI_LIMIT_GIB
                         and r["min_avail_mb"] >= RAM_FLOOR_MB)
        if r["pass"] and n == max_jobs + 1:
            max_jobs = n
    dec = dict(time=f"{now():%F %T}", results=res, max_jobs=max_jobs,
               speedup={str(k): v for k, v in speed.items()},
               rule="總吞吐 ≥ 1.3×單 job、nvidia-smi < 11.0 GiB、RAM 最小 available ≥ 1024 MB；取連續通過的最大 n")
    json.dump(dec, open(DECISION, "w"), indent=1, ensure_ascii=False)
    say(f"吞吐測試結論：max_jobs={max_jobs}，加速比 {speed}")
    return dec


def main():
    say("queue2 driver 啟動")
    try:
        pid = int(open(OLD_PID).read().strip())
        while True:
            os.kill(pid, 0)
            time.sleep(60)
    except (ProcessLookupError, ValueError, FileNotFoundError):
        pass
    say("舊 driver 已結束")
    dec = json.load(open(DECISION)) if os.path.exists(DECISION) else parallel_test()
    max_jobs = int(dec["max_jobs"])
    speed = {int(k): float(v) for k, v in dec["speedup"].items()}

    while True:
        if os.path.exists(STOP):
            say("STOP 檔存在，結束")
            break
        groups = json.load(open(QUEUE))
        gd = done_names("group")
        todo = [g for g in groups if g["name"] not in gd]
        if not todo:
            say("佇列空了")
            break
        g = todo[0]
        dn = done_names("job")
        lanes = [[j for j in lane if j["tag"] not in dn] for lane in g["lanes"]]
        lanes = [l for l in lanes if l]
        slots = max(1, min(max_jobs, int(g.get("max_parallel", 99)), len(lanes)))
        lane_est = [sum(float(j.get("est_hours", 0)) for j in l) for l in lanes]
        sp = speed.get(slots, 1.0) if slots > 1 else 1.0
        est = (sum(lane_est) / sp) if slots > 1 else sum(lane_est)
        used = used_hours()
        t0 = now()
        s = Sampler()
        if used + est <= BUDGET_H:
            say(f"GROUP {g['name']}：{len(lanes)} 條 lane、{slots} 個槽位；已用 {used:.2f} h ＋ 預估 {est:.2f} h ≤ {BUDGET_H}")
            wait_idle()
            with ThreadPoolExecutor(max_workers=slots) as ex:
                list(ex.map(lambda l: run_lane(l, slots > 1), lanes))
            note = f"slots={slots} 預估 {est:.2f} h"
        else:
            say(f"GROUP {g['name']}：已用 {used:.2f} ＋ 預估 {est:.2f} > {BUDGET_H}，改逐 job 序列檢查")
            for l in lanes:
                for j in l:
                    if j["tag"] in done_names("job"):
                        continue
                    u = used_hours() + (now() - t0).total_seconds() / 3600
                    if u + float(j.get("est_hours", 0)) > BUDGET_H:
                        say(f"SKIP {j['tag']}：已用 {u:.2f} ＋ 預估 {j.get('est_hours')} > {BUDGET_H}")
                        write_row(dict(kind="job", name=j["tag"], start=f"{now():%F %T}", hours="0",
                                       status="skipped_budget", note=f"used={u:.2f} est={j.get('est_hours')}"))
                        continue
                    wait_idle()
                    run_job(j, False)
            slots = 1
            note = f"超預算，逐 job 序列；原預估 {est:.2f} h"
        t1 = now()
        smi, mavail = s.stop()
        write_row(dict(kind="group", name=g["name"], start=f"{t0:%F %T}", end=f"{t1:%F %T}",
                       hours=f"{(t1 - t0).total_seconds() / 3600:.3f}", parallel=slots,
                       min_avail_mb=mavail, smi_peak_gib=f"{smi:.3f}", note=note))
        say(f"GROUP END {g['name']}：wall-clock {(t1 - t0).total_seconds() / 3600:.3f} h，"
            f"smi 峰值 {smi:.3f} GiB，RAM 最小 available {mavail} MB")
    say("driver exit")


if __name__ == "__main__":
    main()
