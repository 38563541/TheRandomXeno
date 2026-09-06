#!/usr/bin/env bash
# night_0906b.sh — overnight training, 2026-09-06 夜間
#
# ⚠️ 不要 set -e ——前面失敗後面照樣要跑
#
# 目的：B4 seed=43 已於 15:46-00:45 CST Sep5→6 完成訓練（結果：25k=51.7%）
#       只剩 B2 seed=43 需要跑。
#
# B4 seed=43 跳過原因：
#   checkpoints 已存在 ~/trx_data/checkpoints/b4_seed43_1shot_hmdb3/
#   測試結果：25k=51.7%, 50k=49.3%, 75k=50.6%, 100k=50.1%（已寫進 reports/工作_0906.md）

CKPT_ROOT=~/trx_data/checkpoints
SCRATCH=~/Documents/work/trx/trx_data

echo "=== night_0906b.sh start: $(date) ==="

# ── B2 seed=43 ──────────────────────────────────────────────────────────────
# B2: intra-only relation, bidirectional mean-Hausdorff, 1-shot HMDB split3
# 目的：與 B4 seed=43 對比，評估 inter-relation 的貢獻；提供訓練變異估計
echo ""
echo "=== Launching B2 seed=43 at $(date) ==="

conda run -n trx python run.py \
    --config configs/stage2_b2_intra_only.yaml \
    --dataset hmdb --split 3 \
    --scratch "$SCRATCH" \
    --test_iters 25000 50000 75000 100000 \
    --training_iterations 100020 \
    --seed 43 \
    -c "$CKPT_ROOT/b2_seed43_1shot_hmdb3" \
    > logs/b2_seed43_1shot.log 2>&1

echo "=== B2 seed43 exit=$? at $(date) ==="

echo ""
echo "=== night_0906b.sh complete: $(date) ==="
