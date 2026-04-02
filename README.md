# TRX-SetBased: Set Matching for Few-Shot Action Recognition

A research project extending [TRX (CVPR 2021)](https://arxiv.org/abs/2101.06184) by replacing its prototype-aggregation matching with bidirectional set matching over temporal tuple embeddings.

Based on: https://github.com/tobyperrett/few-shot-action-recognition

---

## Motivation

### English

Few-shot action recognition (FSAR) requires classifying videos from only a handful of examples. TRX addresses this with *temporal tuples* — pairs (or triples) of frames whose relative ordering captures motion — and a CrossTransformer that aggregates support tuples into a query-specific prototype before computing distance.

The key limitation: **aggregation discards set structure**. Once you compute a prototype, you lose the individual support tuple embeddings and can no longer reason about which parts of the support set are actually relevant to a given query.

This project asks: *what if we skip the aggregation and match query tuples directly against the full set of support tuples?*

### 動機（繁體中文）

TRX 的流程是：support tuple → 加權聚合 → prototype → 與 query 計算距離。  
這個 prototype 是 query-specific 的，但聚合之後 support set 的結構就消失了。

我們的做法是移除 prototype 聚合，改為直接對 query tuple set 和 support tuple set 做 **set-level matching**（雙向 Mean-Hausdorff），讓每個 query tuple 都能找到最近的 support tuple，反之亦然。這在概念上更接近 HyRSM 的精神，但作用在 tuple embedding 層，而非 frame 層。

---

## Method

### TRX (Baseline)

```
Support frames → tuples → K/V projections → CrossTransformer attention → prototype
Query frames   → tuples → Q projections  →              ↓
                                         distance(query_v, prototype) → logit
```

The CrossTransformer computes a *query-specific prototype* by soft-attending over support tuples. Distance is measured between the query tuple set and this aggregated prototype.

### This Work: Tuple-Level Set Matching

```
Support frames → PE → pair concatenation → embed+norm → support tuple set S
Query frames   → PE → pair concatenation → embed+norm → query tuple set   Q

logit = −Hausdorff_mean_bidir(Q, S)
```

The **bidirectional mean-Hausdorff distance** is:

```
D(Q, S) = 0.5 × (mean_q min_s ||q−s||²  +  mean_s min_q ||s−q||²)
```

- No prototype aggregation — every support tuple participates directly in matching.
- Symmetric: both directions (Q→S and S→Q) contribute.
- VRAM-efficient: support tuples are processed in chunks.

---

## Implementation Status

### Stage 1 — Implemented ✓

| Component | Status | Notes |
|---|---|---|
| `TemporalCrossTransformerHausdorffPairs` | ✓ Done | `model.py` — replaces original CrossTransformer |
| Positional encoding on frame features | ✓ Done | Sinusoidal PE (from TRX) |
| Pair tuple construction | ✓ Done | C(seq\_len, 2) = 28 tuples for seq\_len=8 |
| Bidirectional mean-Hausdorff distance | ✓ Done | Chunked matmul for VRAM efficiency |
| Episodic training loop | ✓ Done | `run.py` — multi-step LR, gradient accumulation |
| HMDB-51 data loading | ✓ Done | Tested on split 3 |
| TensorBoard logging + checkpointing | ✓ Done | Auto-timestamped checkpoint dirs |
| ResNet-18/34/50 backbone | ✓ Done | Default: ResNet-18 for single-GPU use |

The original `TemporalCrossTransformer` (TRX attention) is preserved in `model.py` as commented-out code for reference.

### Stage 2 — Planned

- [ ] Intra/inter relation module (HyRSM-style) operating on tuple embeddings
- [ ] Higher-order tuples (triples, i.e., `temporal_set_size=3`)
- [ ] Multi-cardinality ensemble (pairs + triples, as in original TRX)
- [ ] Evaluation on Kinetics and Something-Something V2

---

## Differences from TRX and HyRSM

| | TRX | HyRSM | This work |
|---|---|---|---|
| Representation | Temporal tuples | Individual frames | Temporal tuples |
| Matching level | Tuple → prototype | Frame → set | Tuple → set |
| Aggregation | Before matching (prototype) | After matching | None (direct set match) |
| Distance metric | L2 to prototype | Mean-Hausdorff on frames | Mean-Hausdorff on tuples |
| Temporal modeling | Pair/triple combos | Intra+inter relation | Pair combos (Stage 1) |

---

## Setup

### Requirements

```bash
pip install torch torchvision tensorflow tensorboard
```

Python 3.6+ recommended. Tested on RTX 3080 Ti (single GPU).

### Data Preparation

Download your dataset and extract frames as:
```
dataset/class/video/00000001.jpg  (8-digit zero-padded)
```

Then zip with no compression (required by the data loader):
```bash
zip -0 -r dataset.zip dataset/
```

Place the zip and split files under `~/trx_data/video_datasets/`:
```
~/trx_data/
  video_datasets/
    data/
      hmdb51_256q5.zip
    splits/
      hmdb_ARN/
        trainlist03.txt
        vallist03.txt
        testlist03.txt
```

Split files for HMDB, UCF, Kinetics, and SSv2 are included in the `splits/` directory.

---

## Running

### Quick model sanity check (no data needed)

```bash
python model.py
```

Runs a forward pass with random tensors (5-way 1-shot, ResNet-18, seq_len=8).

### Training on HMDB-51

```bash
python run.py \
  -c checkpoints_hmdb_hausdorff \
  --dataset hmdb \
  --split 3 \
  --method resnet18 \
  --shot 1 \
  --way 5 \
  --query_per_class 5 \
  --tasks_per_batch 16 \
  --trans_linear_out_dim 128 \
  --img_size 84 \
  --seq_len 8 \
  --learning_rate 0.001
```

Checkpoint and logs are saved to `checkpoints_hmdb_hausdorff/hmdb_split3_<timestamp>/`.

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `ssv2` | `hmdb`, `ucf`, `kinetics`, `ssv2` |
| `--method` | `resnet50` | `resnet18`, `resnet34`, `resnet50` |
| `--shot` | `5` | Support shots per class |
| `--way` | `5` | N-way classification |
| `--seq_len` | `8` | Frames per video |
| `--trans_linear_out_dim` | `1152` | Tuple embedding dimension |
| `--split` | `7` | Dataset split index |
| `--scratch` | `~/trx_data` | Root data directory |

---

## Project Structure

```
.
├── model.py            # TemporalCrossTransformerHausdorffPairs + CNN_TRX
├── run.py              # Episodic training/testing loop
├── video_reader.py     # Dataset loader (zip-based)
├── utils.py            # Loss, accuracy, logging utilities
├── videotransforms/    # Video augmentation (from torch_videovision)
├── splits/             # Train/val/test split files for all datasets
└── make_hmdb_trx_zip.py  # Helper to prepare HMDB zip
```

---

## Citation

If you build on TRX, please cite the original paper:

```bibtex
@inproceedings{perrett2021trx,
  title     = {Temporal Relational CrossTransformers for Few-Shot Action Recognition},
  booktitle = {Computer Vision and Pattern Recognition},
  year      = {2021}
}
```

---

## Acknowledgements

This repo is forked from [tobyperrett/few-shot-action-recognition](https://github.com/tobyperrett/few-shot-action-recognition). Training infrastructure is based on [CNAPs](https://github.com/cambridge-mlg/cnaps). Video transforms from [torch_videovision](https://github.com/hassony2/torch_videovision). The set-matching approach draws conceptual inspiration from [HyRSM (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Hybrid_Relation_Guided_Set_Matching_for_Few-Shot_Action_Recognition_CVPR_2022_paper.pdf).
