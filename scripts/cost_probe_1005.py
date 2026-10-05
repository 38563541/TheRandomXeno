#!/usr/bin/env python3
"""
scripts/cost_probe_1005.py — 1005 成本量測 driver（改寫自 run_ckpt_methods_0922.py）。

每個設定獨立一個 run.py process，300 iter，--profile_memory --profile_time，
背景 nvidia-smi 輪詢取峰值，結果 append 到 CSV。單一設定失敗不會讓 driver 停。

用法：
  python scripts/cost_probe_1005.py --csv logs/lst_cost_1005.csv --only R0 [--only ...]
  （設定清單寫在 CONFIGS；--only 不給就全跑）
不寫 results.csv：--test_iters 999999 且 --profile_memory（run.py 既有 guard）。
"""
import argparse
import csv
import os
import re
import subprocess
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = os.path.expanduser("~/Documents/work/trx/trx_data")
CKPT_ROOT = os.path.expanduser("~/trx_data/checkpoints")
LOG_DIR = os.path.join(REPO, "logs")
TIMEOUT_S = 25 * 60
GPU_IDLE_THRESHOLD_MB = 200

B4 = "configs/stage2_true_hyrsm_b4_tuple.yaml"
B1 = "configs/b1_backbone_1shot.yaml"
E1 = ["--grad_ckpt", "--ckpt_prefix", "14", "--ckpt_segments", "7"]

CSV_FIELDS = [
    "tag", "backbone", "arm", "head", "amp", "seq_len", "flags",
    "status", "alloc_peak_gb", "reserved_peak_gb", "nvidia_smi_peak_gib",
    "ms_iter", "data_ms", "backbone_fwd_ms", "head_fwd_ms", "backward_ms",
    "trainable_M", "loss_first", "loss_last", "commit", "note",
]

# arm 代號見 1005 任務說明；flags 以外全部沿用 config。
CONFIGS = []
def _add(tag, backbone, arm, head, flags, amp="off", seq_len=8):
    f = list(flags)
    if amp != "off":
        f += ["--amp", amp]
    if seq_len != 8:
        f += ["--seq_len", str(seq_len)]
    CONFIGS.append(dict(tag=tag, backbone=backbone, arm=arm, head=head,
                        config=B4 if head == "B4" else B1, flags=f, amp=amp, seq_len=seq_len))

# 0-4 合併後預設路徑回歸（與 0922 R0 相同指令）
_add("R0_1005", "resnet50", "E1", "B4", E1)

# 階段 4（backbone_mode 的 flag 在實作後才存在）
for amp in ("off", "fp16"):
    s = "" if amp == "off" else "_fp16"
    _add(f"RN50_E0{s}", "resnet50", "E0", "B4", [], amp)
    _add(f"RN50_E1{s}", "resnet50", "E1", "B4", E1, amp)
    _add(f"RN50_D{s}",  "resnet50", "D",  "B4", ["--backbone_mode", "partial_l4"], amp)
    _add(f"RN50_A{s}",  "resnet50", "A",  "B4", ["--backbone_mode", "frozen"], amp)
    _add(f"RN50_B{s}",  "resnet50", "B",  "B4", ["--backbone_mode", "top_adapter"], amp)
    _add(f"RN50_C{s}",  "resnet50", "C",  "B4", ["--backbone_mode", "lst"], amp)
    _add(f"RN18_E0{s}", "resnet18", "E0", "B4", [], amp)
    _add(f"RN18_A{s}",  "resnet18", "A",  "B4", ["--backbone_mode", "frozen"], amp)
    _add(f"RN18_B{s}",  "resnet18", "B",  "B4", ["--backbone_mode", "top_adapter"], amp)
    _add(f"RN18_C{s}",  "resnet18", "C",  "B4", ["--backbone_mode", "lst"], amp)
    _add(f"RN18_D{s}",  "resnet18", "D",  "B4", ["--backbone_mode", "partial_l4"], amp)
# 只在 RN50_D 超過 11.0 GiB 時才跑：layer4 開 checkpoint（chain 共 21 個 module，前 20 個＝到 layer4 結尾）
_add("RN50_D_l4ckpt", "resnet50", "D", "B4", ["--backbone_mode", "partial_l4", "--grad_ckpt",
                                              "--ckpt_prefix", "20", "--ckpt_segments", "3"])
_add("RN50_C_T16", "resnet50", "C", "B4", ["--backbone_mode", "lst"], seq_len=16)
_add("RN18_C_T16", "resnet18", "C", "B4", ["--backbone_mode", "lst"], seq_len=16)


def sh(cmd):
    return subprocess.check_output(cmd, cwd=REPO).decode().strip()


def gpu_used_mb():
    return int(sh(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).splitlines()[0])


def run_one(cfg, commit):
    tag = cfg["tag"]
    log_path = os.path.join(LOG_DIR, f"cost1005_{tag}.log")
    poll_path = os.path.join(LOG_DIR, f"cost1005_{tag}_nvsmi.log")
    cmd = ["python", "run.py", "--dataset", "hmdb", "--split", "3",
           "--config", cfg["config"], "--method", cfg["backbone"], *cfg["flags"],
           "--training_iterations", "300", "--print_freq", "100",
           "--save_freq", "999999", "--test_iters", "999999",
           "--profile_memory", "--profile_time",
           "-c", os.path.join(CKPT_ROOT, f"cost1005_{tag}"), "--scratch", SCRATCH]
    print(f"=== {tag}: {' '.join(cmd)} ===", flush=True)
    poll_f = open(poll_path, "w")
    poll = subprocess.Popen(["nvidia-smi", "--query-gpu=memory.used",
                             "--format=csv,noheader,nounits", "-lms", "500"],
                            stdout=poll_f, stderr=subprocess.DEVNULL)
    status, rc = "ok", None
    try:
        with open(log_path, "w") as lf:
            rc = subprocess.run(cmd, cwd=REPO, stdout=lf, stderr=subprocess.STDOUT,
                                timeout=TIMEOUT_S).returncode
    except subprocess.TimeoutExpired:
        status = "timeout"
    poll.terminate()
    try:
        poll.wait(timeout=5)
    except Exception:
        poll.kill()
    poll_f.close()

    text = open(log_path, errors="replace").read()
    if status != "timeout" and rc != 0:
        status = "nan" if rc == 97 else ("oom" if ("OutOfMemoryError" in text or "out of memory" in text) else "crash")

    row = {k: "未量測" for k in CSV_FIELDS}
    row.update(tag=tag, backbone=cfg["backbone"], arm=cfg["arm"], head=cfg["head"], amp=cfg["amp"],
               seq_len=cfg["seq_len"], flags=" ".join(cfg["flags"]), status=status, commit=commit, note="")
    vals = [int(x) for x in open(poll_path).read().split() if x.strip().isdigit()]
    if vals:
        row["nvidia_smi_peak_gib"] = f"{max(vals) / 1024:.3f}"
    m = re.search(r"\[mem\] RUN PEAK\s+alloc\s+([\d.]+)\s*GB\s+reserved\s+([\d.]+)\s*GB", text)
    if m:
        row["alloc_peak_gb"], row["reserved_peak_gb"] = m.group(1), m.group(2)
    elif status == "oom":
        # OOM 前的峰值：取最後一行 [mem] / [memsplit]
        ms = re.findall(r"\[memsplit\] iter\s+\d+.*?peak ([\d.]+)", text)
        if ms:
            row["note"] = f"OOM 前最後一次 memsplit peak alloc={ms[-1]} GB"
    m = re.search(r"\[time\] n=\d+\(iter21-\d+\)\s+data=([\d.]+)ms\s+H2D=([\d.]+)ms\s+"
                  r"backbone\(支\+查\)=([\d.]+)ms\s+head=([\d.]+)ms\s+backward\+step=([\d.]+)ms"
                  r"\s+\|\s+五段加總=([\d.]+)ms\s+total\(wall\)=([\d.]+)ms", text)
    if m:
        row.update(data_ms=m.group(1), backbone_fwd_ms=m.group(3), head_fwd_ms=m.group(4),
                   backward_ms=m.group(5), ms_iter=m.group(7))
    m = re.search(r"optimizer 收了 ([\d.]+) M", text)
    if m:
        row["trainable_M"] = m.group(1)
    tl = re.findall(r"Train Loss: ([\-\d.eE]+|nan|inf)", text)
    if tl:
        row["loss_first"], row["loss_last"] = tl[0], tl[-1]
    if status == "oom" and not row["note"]:
        m = re.search(r"OutOfMemoryError:.*", text)
        row["note"] = m.group(0)[:300] if m else "OOM"
    elif status in ("crash", "nan"):
        row["note"] = " | ".join(text.splitlines()[-30:])[:800]
    print(f"=== {tag} done: {status} ms/iter={row['ms_iter']} alloc={row['alloc_peak_gb']} "
          f"smi={row['nvidia_smi_peak_gib']} ===", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--only", action="append", default=None)
    a = ap.parse_args()
    todo = [c for c in CONFIGS if a.only is None or c["tag"] in a.only]
    commit = sh(["git", "rev-parse", "--short", "HEAD"])
    os.makedirs(LOG_DIR, exist_ok=True)
    hdr = not os.path.exists(a.csv)
    with open(a.csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if hdr:
            w.writeheader()
        for cfg in todo:
            time.sleep(10)
            used = gpu_used_mb()
            if used >= GPU_IDLE_THRESHOLD_MB:
                time.sleep(60)
                used = gpu_used_mb()
            if used >= GPU_IDLE_THRESHOLD_MB:
                row = {k: "未量測" for k in CSV_FIELDS}
                row.update(tag=cfg["tag"], status="gpu_busy", note=f"nvidia-smi {used} MB")
            else:
                try:
                    row = run_one(cfg, commit)
                except Exception as e:  # driver 不准停
                    row = {k: "未量測" for k in CSV_FIELDS}
                    row.update(tag=cfg["tag"], status="driver_error", note=repr(e)[:300])
            w.writerow(row)
            f.flush()
    print("=== ALL DONE ===", flush=True)


if __name__ == "__main__":
    main()
