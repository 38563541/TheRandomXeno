# B2 / B3 Ablation Study — HMDB Split 3

**Goal:** Isolate whether IntraRelation or InterRelation is responsible for Stage 2 underperforming Stage 1.

- **B2** (`stage2_b2_intra_only.yaml`): IntraRelation only, InterRelation disabled  
- **B3** (`stage2_b3_inter_only.yaml`): InterRelation only, IntraRelation disabled  
- **Full** (`stage2_option_b.yaml`): Both IntraRelation + InterRelation (tuple-level)  
- **Baseline**: Stage 1 bidirectional Hausdorff (best Stage 1 config)

All runs: HMDB split 3, 5-way, ResNet18, 100k iterations, tuple-level relation.  
"Best" = highest accuracy across the 4 test checkpoints (25k / 50k / 75k / 100k).

---

## 1-Shot Results

| Model | 25k | 50k | 75k | 100k | **Best** | Δ vs S1 Baseline |
|---|---|---|---|---|---|---|
| Stage 1 Bidir (baseline) | 46.74 | **47.49** | 46.34 | 46.44 | **47.49** | — |
| Stage 2 Full (intra+inter) | 41.34 | 43.04 | **43.71** | 42.73 | **43.71** | −3.78 pp |
| B2 — Intra only | **48.41** | 47.66 | 47.09 | 46.66 | **48.41** | **+0.92 pp** |
| B3 — Inter only | 37.28 | 38.18 | **39.01** | 37.14 | **39.01** | −8.48 pp |

## 5-Shot Results

| Model | 25k | 50k | 75k | 100k | **Best** | Δ vs S2 Full |
|---|---|---|---|---|---|---|
| Stage 2 Full (intra+inter) | 55.80 | 57.61 | **58.03** | 57.82 | **58.03** | — |
| B2 — Intra only | **67.91** | 66.26 | 66.12 | 65.76 | **67.91** | **+9.88 pp** |
| B3 — Inter only | 54.45 | **55.92** | 55.10 | 55.42 | **55.92** | −2.11 pp |

*(Stage 1 5-shot was not run; comparisons are relative to Stage 2 Full.)*

---

## Interpretation

**InterRelation is the culprit.** The ablation clearly shows that disabling InterRelation (B2, intra only) recovers and even surpasses Stage 1 baseline on 1-shot (+0.92 pp), and dramatically outperforms Stage 2 Full on 5-shot (+9.88 pp). By contrast, disabling IntraRelation (B3, inter only) collapses 1-shot accuracy to 39.0%, a −8.48 pp drop below Stage 1 and −4.70 pp below Stage 2 Full.

The cross-attention mechanism in InterRelation—which lets query tokens attend to all support tokens and vice versa—appears to contaminate the set representations used for Hausdorff matching. The model likely shortcircuits the set-matching signal by blending query and support features before distance computation, making the resulting embeddings no longer independently comparable.

IntraRelation alone (B2) is strictly beneficial: it enriches each video's internal temporal structure without mixing query and support information, so the Hausdorff matching step still operates on clean, independent representations.

**Recommendation:** Drop InterRelation from Stage 2. The B2 config (intra only) is the strongest variant found so far and should be the Stage 2 final model.

---

## Anomaly Check

- B2 ≈ B3? No — B2 leads B3 by **9.4 pp** (1-shot) and **12.0 pp** (5-shot). Hypothesis strongly supported.
- Both ≈ baseline? No — B2 beats baseline; B3 is far below. Relations are correctly wired.
- B2 5-shot peaks early (25k, 67.91%) then slowly declines — possible mild overfitting; learning rate schedule may need tuning for 5-shot.
