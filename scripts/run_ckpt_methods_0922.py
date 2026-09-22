#!/usr/bin/env python3
"""
scripts/run_ckpt_methods_0922.py — 0922 driver：無人值守跑完整組
checkpoint／AMP 方法比較（R0、S2-S4、C1-C3、A1-A6）。

規則（見 0922 任務說明）：
  - 每個設定獨立 process，外面包 25 分鐘 timeout。
  - 同時背景跑 nvidia-smi 輪詢（-lms 500）取最大值。
  - 結束狀態分類：ok / oom / nan / timeout / crash / gpu_busy。
  - 任何單一設定的失敗都不能讓 driver 停下來——逐一 try/except，
    寫一列 CSV 就繼續下一個。
  - 下一個設定開始前等 10 秒，確認 nvidia-smi < 200 MB；不是的話再等
    60 秒，還不是就標 gpu_busy 並跳過該設定（不重試、不遞迴等待）。

用法：python scripts/run_ckpt_methods_0922.py
輸出：logs/ckpt_methods_0922.csv（累加寫入，重跑會 append 不會覆蓋）
      logs/m0922_<tag>.log            每個設定的完整 run.py 輸出
      logs/m0922_<tag>_nvsmi.log      每個設定的 nvidia-smi 輪詢原始輸出
"""
import csv
import os
import re
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = os.path.expanduser("~/Documents/work/trx/trx_data")
CKPT_ROOT = os.path.expanduser("~/trx_data/checkpoints")
LOG_DIR = os.path.join(REPO, "logs")
CSV_PATH = os.path.join(LOG_DIR, "ckpt_methods_0922.csv")

TIMEOUT_S = 25 * 60
GPU_IDLE_THRESHOLD_MB = 200

CSV_FIELDS = [
    "tag", "stage", "backbone", "grad_ckpt", "ckpt_prefix", "ckpt_segments", "ckpt_policy", "amp",
    "status", "alloc_peak_gb", "reserved_peak_gb", "nvidia_smi_peak_gib",
    "persist_gb", "bb_retained_gb", "head_retained_gb", "bwd_extra_gb",
    "ms_iter", "data_ms", "backbone_fwd_ms", "head_fwd_ms", "backward_ms",
    "loss_first", "loss_last", "loss_finite", "commit", "note",
]

# 只跑這裡列出的格子——不准自己加設定。
CONFIGS = [
    dict(tag="R0", stage="1c", backbone="resnet50",
         flags=["--grad_ckpt", "--ckpt_prefix", "14", "--ckpt_segments", "7"]),
    dict(tag="S2", stage="2", backbone="resnet50",
         flags=["--grad_ckpt", "--ckpt_prefix", "11", "--ckpt_segments", "4"]),
    dict(tag="S3", stage="2", backbone="resnet18", flags=[]),
    dict(tag="S4", stage="2", backbone="resnet34",
         flags=["--grad_ckpt", "--ckpt_prefix", "14", "--ckpt_segments", "7"]),
    dict(tag="C1", stage="3d", backbone="resnet50",
         flags=["--grad_ckpt", "--ckpt_prefix", "14", "--ckpt_segments", "7", "--ckpt_policy", "save_conv"]),
    dict(tag="C2", stage="3d", backbone="resnet50",
         flags=["--grad_ckpt", "--ckpt_prefix", "11", "--ckpt_segments", "4", "--ckpt_policy", "save_conv"]),
    dict(tag="C3", stage="3d", backbone="resnet50",
         flags=["--grad_ckpt", "--ckpt_prefix", "21", "--ckpt_segments", "10", "--ckpt_policy", "save_conv"]),
    dict(tag="A1", stage="4b", backbone="resnet50", flags=["--amp", "bf16"]),
    dict(tag="A2", stage="4b", backbone="resnet50",
         flags=["--amp", "bf16", "--grad_ckpt", "--ckpt_prefix", "14", "--ckpt_segments", "7"]),
    dict(tag="A3", stage="4b", backbone="resnet50",
         flags=["--amp", "bf16", "--grad_ckpt", "--ckpt_prefix", "11", "--ckpt_segments", "4"]),
    dict(tag="A4", stage="4b", backbone="resnet50",
         flags=["--amp", "fp16", "--grad_ckpt", "--ckpt_prefix", "14", "--ckpt_segments", "7"]),
    dict(tag="A5", stage="4b", backbone="resnet18", flags=[]),
    dict(tag="A6", stage="4b", backbone="resnet18", flags=["--amp", "bf16"]),
]


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO
        ).decode().strip()
    except Exception:
        return "unknown"


def nvidia_smi_used_mb():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    ).decode().strip()
    return int(out.splitlines()[0])


def _extract_flag_val(flags, name):
    if name in flags:
        idx = flags.index(name)
        if idx + 1 < len(flags):
            return flags[idx + 1]
    return None


def _is_finite_str(s):
    try:
        v = float(s)
        return v == v and v not in (float("inf"), float("-inf"))
    except ValueError:
        return False


def run_one(cfg, commit):
    tag = cfg["tag"]
    log_path = os.path.join(LOG_DIR, f"m0922_{tag}.log")
    poll_path = os.path.join(LOG_DIR, f"m0922_{tag}_nvsmi.log")
    ckpt_dir = os.path.join(CKPT_ROOT, f"m0922_{tag}")

    cmd = [
        "python", "run.py",
        "--dataset", "hmdb", "--split", "3",
        "--config", "configs/stage2_true_hyrsm_b4_tuple.yaml",
        "--method", cfg["backbone"],
        *cfg["flags"],
        "--training_iterations", "300", "--print_freq", "100",
        "--save_freq", "999999", "--test_iters", "999999",
        "--profile_memory", "--profile_time",
        "-c", ckpt_dir,
        "--scratch", SCRATCH,
    ]
    print(f"=== {tag}: {' '.join(cmd)} ===", flush=True)

    poll_f = open(poll_path, "w")
    poll_proc = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-lms", "500"],
        stdout=poll_f, stderr=subprocess.DEVNULL,
    )

    status = "ok"
    rc = None
    t0 = time.time()
    try:
        with open(log_path, "w") as logf:
            proc = subprocess.run(
                cmd, cwd=REPO, stdout=logf, stderr=subprocess.STDOUT, timeout=TIMEOUT_S,
            )
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        status = "timeout"
    wall_s = time.time() - t0

    poll_proc.terminate()
    try:
        poll_proc.wait(timeout=5)
    except Exception:
        poll_proc.kill()
    poll_f.close()

    nvidia_smi_peak_gib = None
    try:
        with open(poll_path) as pf:
            vals = [int(x) for x in pf.read().split() if x.strip().isdigit()]
        if vals:
            nvidia_smi_peak_gib = max(vals) / 1024.0
    except Exception:
        pass

    log_text = ""
    if os.path.exists(log_path):
        with open(log_path, errors="replace") as lf:
            log_text = lf.read()

    if status != "timeout":
        if rc == 97:
            status = "nan"
        elif rc != 0:
            if "OutOfMemoryError" in log_text or "CUDA out of memory" in log_text:
                status = "oom"
            else:
                status = "crash"

    row = {k: "未量測" for k in CSV_FIELDS}
    row["tag"] = tag
    row["stage"] = cfg["stage"]
    row["backbone"] = cfg["backbone"]
    row["grad_ckpt"] = "yes" if "--grad_ckpt" in cfg["flags"] else "no"
    row["ckpt_prefix"] = _extract_flag_val(cfg["flags"], "--ckpt_prefix") or "N/A"
    row["ckpt_segments"] = _extract_flag_val(cfg["flags"], "--ckpt_segments") or "N/A"
    row["ckpt_policy"] = _extract_flag_val(cfg["flags"], "--ckpt_policy") or "full"
    row["amp"] = _extract_flag_val(cfg["flags"], "--amp") or "off"
    row["status"] = status
    row["commit"] = commit
    row["nvidia_smi_peak_gib"] = (
        f"{nvidia_smi_peak_gib:.3f}" if nvidia_smi_peak_gib is not None else "未量測"
    )

    m = re.search(r"\[mem\] RUN PEAK\s+alloc\s+([\d.]+)\s*GB\s+reserved\s+([\d.]+)\s*GB", log_text)
    if m:
        row["alloc_peak_gb"] = m.group(1)
        row["reserved_peak_gb"] = m.group(2)

    m = re.search(
        r"\[memsplit\] MEDIAN.*?persist\s+([\-\d.]+)\s+bb\s+([\-\d.]+)\s+head\s+([\-\d.]+)"
        r"\s+bwd_extra\s+([\-\d.]+)\s+peak\s+([\-\d.]+)", log_text)
    if m:
        row["persist_gb"] = m.group(1)
        row["bb_retained_gb"] = m.group(2)
        row["head_retained_gb"] = m.group(3)
        row["bwd_extra_gb"] = m.group(4)

    m = re.search(
        r"\[time\] n=\d+\(iter21-\d+\)\s+data=([\d.]+)ms\s+H2D=([\d.]+)ms\s+"
        r"backbone\(支\+查\)=([\d.]+)ms\s+head=([\d.]+)ms\s+backward\+step=([\d.]+)ms"
        r"\s+\|\s+五段加總=([\d.]+)ms\s+total\(wall\)=([\d.]+)ms", log_text)
    if m:
        row["data_ms"] = m.group(1)
        row["backbone_fwd_ms"] = m.group(3)
        row["head_fwd_ms"] = m.group(4)
        row["backward_ms"] = m.group(5)
        row["ms_iter"] = m.group(7)  # total(wall) 的中位數

    task_losses = re.findall(r"Task \[(\d+)/\d+\], Train Loss: ([\-\d.eE]+|nan|inf|-inf)", log_text)
    if task_losses:
        row["loss_first"] = task_losses[0][1]
        row["loss_last"] = task_losses[-1][1]
        finite = all(_is_finite_str(v) for _, v in task_losses)
        row["loss_finite"] = "yes" if finite else "no"

    if status == "nan":
        m = re.search(r"\[FATAL\] non-finite loss at iteration (\d+).*", log_text)
        row["note"] = m.group(0) if m else "non-finite loss（詳見 log）"
    elif status == "oom":
        m2 = re.search(r"torch\.OutOfMemoryError:.*", log_text)
        row["note"] = m2.group(0)[:300] if m2 else "OOM（詳見 log）"
    elif status == "crash":
        tail = "\n".join(log_text.splitlines()[-30:])
        row["note"] = tail.replace("\n", " | ")[:500]
    elif status == "timeout":
        row["note"] = f"timeout after {TIMEOUT_S}s"
    else:
        row["note"] = ""

    print(f"=== {tag} done: status={status} wall={wall_s:.1f}s "
          f"nvidia_smi_peak={row['nvidia_smi_peak_gib']}GiB ===", flush=True)
    return row


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    commit = get_git_commit()
    write_header = not os.path.exists(CSV_PATH)

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        for cfg in CONFIGS:
            tag = cfg["tag"]
            time.sleep(10)
            try:
                used = nvidia_smi_used_mb()
            except Exception as e:
                used = -1
                print(f"[WARN] nvidia-smi query failed before {tag}: {e}", flush=True)

            if used >= GPU_IDLE_THRESHOLD_MB:
                time.sleep(60)
                try:
                    used = nvidia_smi_used_mb()
                except Exception:
                    used = -1
                if used >= GPU_IDLE_THRESHOLD_MB:
                    row = {k: "未量測" for k in CSV_FIELDS}
                    row.update(tag=tag, stage=cfg["stage"], backbone=cfg["backbone"],
                               status="gpu_busy", commit=commit,
                               note=f"GPU busy ({used} MB used), skipped without retry")
                    writer.writerow(row)
                    f.flush()
                    print(f"=== {tag}: SKIPPED (gpu_busy, {used} MB) ===", flush=True)
                    continue

            try:
                row = run_one(cfg, commit)
            except Exception as e:
                row = {k: "未量測" for k in CSV_FIELDS}
                row.update(tag=tag, stage=cfg["stage"], backbone=cfg["backbone"],
                           status="crash", commit=commit,
                           note=f"driver-side exception: {e!r}"[:500])
                print(f"=== {tag}: driver-side exception: {e!r} ===", flush=True)

            writer.writerow(row)
            f.flush()

    print("=== ALL CONFIGS DONE ===", flush=True)


if __name__ == "__main__":
    main()
