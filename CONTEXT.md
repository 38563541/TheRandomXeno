# Research Context

Internal notes for this project. Not intended for public release.

---

## Research Goal

Extend TRX (Perrett et al., CVPR 2021) toward **tuple-level set matching without prototype aggregation**.

In the original TRX, query-specific support prototypes are formed by aggregating support tuple embeddings into a single per-class vector before computing distances:

```python
# Original TRX — prototype aggregation bottleneck
query_prototype = torch.matmul(class_scores, class_v)   # weighted sum over support
dist = torch.sum(query_prototype, dim=1)                 # collapse to scalar
```

This destroys within-class diversity. Two support videos showing the same action at different speeds produce a prototype that represents neither accurately.

**This work removes that aggregation entirely.** Each query tuple embedding is compared directly against the full pool of K × T support tuple embeddings using mean-Hausdorff distance:

```
d(Q, S_k) = 0.5 * [ (1/|Q|) Σ_q min_s ||q - s||
                   + (1/|S|) Σ_s min_q ||s - q|| ]
```

where Q is the set of query tuples and S_k is the **union** of all support tuple sets for class k (the pool). This retains diversity across all K support instances.

---

## Implementation Status

### Completed

| Component | File | Status |
|---|---|---|
| Tuple embedding + pool matching | `models/stage1_model.py` | Done |
| Original TRX reference (prototype) | `models/trx_original.py` | Done (annotated) |
| Mean-Hausdorff (pool-based) | `matching/mean_hausdorff.py` | Done |
| Mean-Hausdorff (instance-based) | `matching/mean_hausdorff.py` | Done |
| Attention-weighted Hausdorff | `matching/attention_weighted_hausdorff.py` | Done |
| Bidirectional Hausdorff | `matching/bidirectional_hausdorff.py` | Done |
| YAML config system | `configs/` + `run.py` | Done |
| CSV result logging | `run.py` → `experiments/results/results.csv` | Done |
| Relation module skeletons | `relation/` + `models/stage2_model.py` | Skeleton |

### Pending (Stage 2)

- `IntraRelation`: temporal self-attention within each video
- `InterRelation`: bidirectional cross-attention between query and support
- `TRXSetMatchingWithRelation`: wires backbone + relation + matching together
- Implement only after Stage 1 ablation results are in hand

---

## Option A vs Option B

Both options add relation modules on top of the Stage 1 set-matching backbone.
They differ in **where** the relation modules are applied.

### Option A — Relation at frame level

```
ResNet → [IntraRelation] → [InterRelation] → Tuple Sampling → TRXSetMatching → Logits
```

- Relation enriches per-frame representations before tuples are formed.
- IntraRelation: self-attention across the 8 frames of each video.
- InterRelation: cross-attention between query frames and all support frames.
- Similar in spirit to HyRSM (Wang et al., 2022).
- Risk: frame-level relation may lose temporal tuple structure.

### Option B — Relation at tuple level

```
ResNet → Tuple Sampling → [IntraRelation] → [InterRelation] → TRXSetMatching → Logits
```

- Relation enriches per-tuple representations after tuples are formed.
- IntraRelation: self-attention across the T tuples of each video.
- InterRelation: cross-attention between query tuples and all support tuples.
- This is the **primary contribution** of Stage 2: tuple-level relational reasoning.
- Hypothesis: tuple-level context is richer than frame-level context for action recognition.

### Ablation flags (set in YAML)

```yaml
use_intra_relation: true   # toggle IntraRelation on/off
use_inter_relation: true   # toggle InterRelation on/off
relation_level: tuple      # "frame" (Option A) or "tuple" (Option B)
```

---

## Ablation Plan

### Stage 1 — Matching function ablation (no relation)

Goal: establish which matching function performs best before adding relation modules.

| Experiment | Config | Key variable |
|---|---|---|
| A1 | `stage1_hausdorff.yaml` | Unidirectional Q→S |
| A2 | `stage1_bidirectional.yaml` | Symmetric Q↔S |
| A3 | `stage1_attention_weighted.yaml` | Attention-weighted, τ=0.1 |

Run each on HMDB splits 1, 2, 3. Report mean ± 95% CI across splits.

Secondary sweep for A3: τ ∈ {0.01, 0.05, 0.1, 0.5} to find best temperature.

Expected winner: A2 or A3. If unidirectional is competitive, use it for speed.

### Stage 2 — Relation module ablation

Requires Stage 1 winner as the matching backbone. Four conditions:

| Experiment | `use_intra` | `use_inter` | `relation_level` |
|---|---|---|---|
| B1 | false | false | — | (Stage 1 winner, baseline) |
| B2 | true | false | tuple | intra only |
| B3 | false | true | tuple | inter only |
| B4 | true | true | tuple | Option B full |
| B5 | true | true | frame | Option A full |

Compare B4 vs B5 to test the frame vs tuple hypothesis.
Compare B2/B3 vs B4 to measure contribution of each relation module independently.

---

## Hardware Constraints

**GPU**: NVIDIA RTX 3080 Ti (12 GB VRAM)

Key implications:
- ResNet-50 at 224×224 with `way=5, shot=5, query_per_class=5` exceeds VRAM. Use ResNet-18 or reduce query_per_class.
- `tasks_per_batch=16` gradient accumulation is used instead of a large batch. This is the standard TRX setting.
- Chunked distance computation in `mean_hausdorff_bidir` (chunk_s=64) prevents OOM during the K×T tuple pool scan.
- InterRelation cross-attention across all support tokens will be memory-intensive at Stage 2. Consider chunking or keeping n_support small.
- All Stage 1 configs use `backbone: resnet18` and `shot: 1` to stay within VRAM budget during initial experiments.

**Baseline reference run** (HMDB split 3, ResNet-18, 20000 iterations, shot=1):
- iter 5000:  test accuracy **51.3 ± 3.1%**
- iter 10000: test accuracy **48.5 ± 3.1%** (dropped — likely still converging)
- iter 20000: train accuracy 93.8% (no test evaluation recorded at final checkpoint)

Full runs use `training_iterations: 100020` with `test_iters` milestone at 75000.

---

## Branch Structure

| Branch | Purpose |
|---|---|
| `main` | Original TRX — untouched reference |
| `stage1` | This branch — all work above lives here |
| `stage2` | To be created from `stage1` when relation modules are ready |

Push `stage1` from your local machine:

```bash
git push origin stage1
```
