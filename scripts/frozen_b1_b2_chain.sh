#!/usr/bin/env bash
# 凍結軌道 B1→B2 連續訓練腳本（2026-09-13 夜間）
# B1 完成後自動啟動 B2；B4 明天啟動。
# 本腳本由 launch.sh 以 setsid 啟動，PPID 應追到 1。
set -euo pipefail

SCRATCH=/home/ccwu/Documents/work/trx/trx_data
CKPT_ROOT=/home/ccwu/trx_data/checkpoints
REPO=/home/ccwu/Documents/work/trx
CONDA_ENV=trx

echo "=== frozen_b1_b2_chain.sh 啟動 $(date) ==="

# ── B1 frozen ────────────────────────────────────────────────────────────────
echo "=== [B1 frozen] 開始 $(date) ==="
conda run -n "${CONDA_ENV}" python "${REPO}/run.py" \
    --config "${REPO}/configs/frozen_rn18_b1.yaml" \
    --dataset hmdb --split 3 \
    --scratch "${SCRATCH}" \
    --training_iterations 100020 \
    --test_iters 25000 50000 75000 100000 \
    --seed 42 \
    -c "${CKPT_ROOT}/frozen_rn18_b1_s42"
echo "=== [B1 frozen] 完成 $(date) ==="

# ── B2 frozen ────────────────────────────────────────────────────────────────
echo "=== [B2 frozen] 開始 $(date) ==="
conda run -n "${CONDA_ENV}" python "${REPO}/run.py" \
    --config "${REPO}/configs/frozen_rn18_b2.yaml" \
    --dataset hmdb --split 3 \
    --scratch "${SCRATCH}" \
    --training_iterations 100020 \
    --test_iters 25000 50000 75000 100000 \
    --seed 42 \
    -c "${CKPT_ROOT}/frozen_rn18_b2_s42"
echo "=== [B2 frozen] 完成 $(date) ==="

echo "=== frozen_b1_b2_chain.sh 全部完成 $(date) ==="
