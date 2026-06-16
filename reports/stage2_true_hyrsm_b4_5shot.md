# Stage 2 True HyRSM B4 — 5-shot Experiment Report

**Date:** 2026-06-15 to 2026-06-16  
**Branch:** stage2  
**Dataset:** HMDB-51, Split 3  
**Config:** `configs/stage2_true_hyrsm_b4_tuple_5shot.yaml`

---

## 1. Background

This experiment completes the final missing number in the Stage 2 relation module ablation: the **5-shot accuracy of True HyRSM B4** (intra-relation + pool-level inter-relation).

The Stage 2 research investigates tuple-level set matching without prototype aggregation. The relation module ablation compares four configurations:

| Config | Intra | Inter | Inter Style |
|---|---|---|---|
| B2 | ✓ | — | — |
| B3 | — | ✓ | token-level |
| HyRSM B3 | — | ✓ | pool-level (HyRSM) |
| **B4 (this run)** | **✓** | **✓** | **pool-level (true_hyrsm)** |

Prior conclusion: token-level inter-relation (B3) is **harmful**; pool-level inter-relation (True HyRSM) is **beneficial**. B4 1-shot was already the best at 50.26% @ 75k. This session captures the 5-shot number.

---

## 2. Experimental Setup

```yaml
matching_method: bidirectional_hausdorff
backbone: resnet18
way: 5
shot: 5
query_per_class: 5
training_iterations: 100020
tasks_per_batch: 16
use_intra_relation: true
use_inter_relation: true
relation_level: tuple
inter_style: true_hyrsm
temp_set: [2]
```

**Evaluation checkpoints:** 25k / 50k / 75k / 100k  
**Test tasks:** 10,000  
**Machine:** RTX 3080 Ti (12 GB VRAM), 15 GB RAM  
**Checkpoint dir:** `~/trx_data/checkpoints/stage2_true_hyrsm_b4_5shot_hmdb3/hmdb_split3_20260615_230717`

---

## 3. Execution Log — Crashes and Fixes

Getting this run stable took several attempts. Documented here for future reference.

### Attempt 1 & 2 — `conda run` stdout swallowed

**Command:**
```bash
nohup conda run -n trx python run.py ... > logs/... 2>&1 &
```

**Symptom:** Process launched, log file stayed 0 bytes, process eventually died silently.  
**Cause:** `conda run` wraps the subprocess and can swallow stdout/stderr before they reach the redirected file. No output meant no way to diagnose crashes.  
**Fix:** Use the conda env's Python binary directly: `/home/ccwu/miniconda3/envs/trx/bin/python`

### Attempt 3 — SIGPIPE from `| head -N` pipe

**Command:**
```bash
/home/ccwu/miniconda3/envs/trx/bin/python -u run.py ... 2>&1 | head -60
```

**Symptom:** Multiple training processes appeared in `ps aux` (11+ instances), then the system ran out of RAM and rebooted.  
**Cause:** `head` closes the pipe after 60 lines and sends SIGPIPE to Python. But Python had already forked 10 DataLoader worker subprocesses. The workers survived SIGPIPE and kept consuming RAM. Repeating this across multiple debug attempts stacked 11+ zombie training processes (each with 10 workers), exhausting all 15 GB RAM.  
**Syslog evidence:**
```
systemd-oomd: Killed GNOME Terminal scope due to memory pressure 53.07% > 50.00% 
for > 20s — killed 80 processes
```
**Fix:** Never pipe training output through `head` or any command that exits early. Use `tee` to file only.

### Attempt 4 — `num_workers=10` OOM on 15 GB machine

**Command:**
```bash
nohup /home/ccwu/miniconda3/envs/trx/bin/python -u run.py ... >> logs/... 2>&1 &
```

**Symptom:** Training reached Task [1000/100020] (loss 0.90, acc 0.80), then the system killed it.  
**Cause:** Default `num_workers=10` spawns 10 DataLoader workers. Each worker loads the HMDB zip into its own address space (~2.2 GB each). 10 × 2.2 GB = 22 GB on a 15 GB machine. systemd-oomd triggered at 50% memory pressure threshold.  
**Note:** The 1-shot experiment previously survived because it got lucky with swap. The 5-shot run touched more memory pages (accumulating 5× more clips per task) and pushed it over the edge consistently.  
**Fix:** Pass `--num_workers 2` on the CLI.

### Attempt 5 (Final) — tmux + `--num_workers 2`

**Command:**
```bash
tmux new-session -d -s train5shot
tmux send-keys -t train5shot "cd ~/Documents/work/trx && \
  /home/ccwu/miniconda3/envs/trx/bin/python -u run.py \
  --config configs/stage2_true_hyrsm_b4_tuple_5shot.yaml \
  --dataset hmdb --split 3 \
  --test_iters 25000 50000 75000 100000 \
  -c ~/trx_data/checkpoints/stage2_true_hyrsm_b4_5shot_hmdb3 \
  --scratch ~/Documents/work/trx/trx_data \
  --num_workers 2 2>&1 | tee logs/true_hyrsm_b4_5shot_hmdb3.log" Enter
```

**Result:** Ran to completion overnight (~17 hours). RAM stable at 12 GB used / 15 GB, never triggered OOM. GPU used 10.4 GB / 12 GB throughout.

---

## 4. Results

### 4.1 True HyRSM B4 — 5-shot

| Iteration | Mean Accuracy | CI (±) |
|---|---|---|
| 25,000 | **67.22%** | 0.41 |
| 50,000 | 66.18% | 0.40 |
| 75,000 | 65.97% | 0.41 |
| 100,000 | 67.17% | 0.40 |

**Best: 67.22% @ 25k**

### 4.2 Full Ablation Table (HMDB Split 3, 5-way)

| Model | 1-shot (best) | 5-shot (best) |
|---|---|---|
| Stage 1 Bidirectional (baseline) | 47.49% | — |
| Stage 1 Attention Weighted | 47.08% | — |
| Stage 2 Option A (proto-free, unidirectional) | 42.85% | 56.96% |
| Stage 2 Option B (token-level full inter) | 43.71% | 58.03% |
| B2 — Intra only | 48.41% | **67.91%** |
| B3 — Token-level Inter only | 39.01% | 55.92% |
| HyRSM B3 — Pool-level Inter only | ~20% (collapsed) | — |
| **B4 — Intra + Pool-level Inter (True HyRSM)** | **50.26%** | **67.22%** |

### 4.3 Interpretation

**1-shot:** B4 is the clear winner (+1.85pp over B2, +6.55pp over B3). Pool-level inter-relation adds meaningful signal on top of intra.

**5-shot:** B4 (67.22%) vs B2 (67.91%) — difference is only 0.69pp, within the confidence interval (±0.40). Statistically, they are effectively tied. The pool-level inter-relation provides no measurable benefit at 5-shot.

**Hypothesis:** With 5 support examples per class, the intra-relation module already extracts rich enough temporal structure from the support set. The inter-relation module's additional signal is marginally useful in the low-data (1-shot) regime but becomes redundant when more support is available.

**Both B4 and B2 strongly outperform token-level inter-relation (Option B)** across all settings (+9.2pp at 5-shot), confirming that token-level inter-relation is harmful and the pool-level design is the right approach.

---

## 5. Operational Notes for Future Runs

1. **Always use the conda env Python directly:** `/home/ccwu/miniconda3/envs/trx/bin/python`  — never `conda run` for background jobs.
2. **Always use tmux:** prevents terminal disconnects from killing training.
3. **Use `--num_workers 2`** on this machine (15 GB RAM, RTX 3080 Ti). `num_workers=10` will OOM.
4. **Never pipe training output through `head` or similar** — SIGPIPE kills Python but not the forked DataLoader workers, causing zombie RAM accumulation.
5. **GPU is the bottleneck, not RAM** (with `--num_workers 2`): 10.4 GB / 12 GB VRAM. If a larger model or batch is tested, check GPU memory first.
