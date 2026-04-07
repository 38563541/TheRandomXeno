# Temporal-Relational Cross-Transformers — Tuple-Level Set Matching

This repo extends [TRX (CVPR 2021)](https://arxiv.org/abs/2101.06184) toward **tuple-level set matching without prototype aggregation**.
Query tuple embeddings search directly against the full support tuple pool, bypassing the prototype bottleneck of the original method.

---

## Quick experiment guide

### 1. Prerequisites

```bash
pip install torch torchvision pyyaml tensorflow tensorboard
```

Data must be prepared as described in the [Data setup](#data-setup) section below.
Set your data root once via `--scratch` (or edit the default in `run.py`):

```
~/trx_data/
  video_datasets/
    data/      ← dataset zips (hmdb51_256q5.zip etc.)
    splits/    ← split text files (hmdb_ARN/, etc.)
  checkpoints/ ← output goes here
```

---

### 2. Running an experiment with a config file

All experiments are driven by YAML configs in `configs/`.
Pass `--config` to select a config; any CLI flag overrides the YAML value.

```bash
python run.py \
  --config configs/stage1_hausdorff.yaml \
  --dataset hmdb --split 3 \
  -c ~/trx_data/checkpoints/stage1_hausdorff_hmdb3
```

That is the canonical launch command for Stage 1 experiments.

---

### 3. Available configs

| Config file | Matching method | Direction | Notes |
|---|---|---|---|
| `configs/stage1_hausdorff.yaml` | Mean-Hausdorff | Q→S only | Stage 1 baseline |
| `configs/stage1_bidirectional.yaml` | Mean-Hausdorff | Q↔S (avg) | Symmetric variant |
| `configs/stage1_attention_weighted.yaml` | Attention-weighted Hausdorff | Q→S | Softmax-attends query tuples |
| `configs/stage2_option_a.yaml` | Bidirectional Hausdorff | Q↔S | + Relation at **frame** level |
| `configs/stage2_option_b.yaml` | Bidirectional Hausdorff | Q↔S | + Relation at **tuple** level |

Stage 2 configs require the relation modules in `relation/` to be implemented first.

---

### 4. Overriding config values at the command line

Every YAML key maps directly to a CLI flag. CLI always wins over YAML:

```bash
# Use the bidirectional config but switch to HMDB split 1
python run.py \
  --config configs/stage1_bidirectional.yaml \
  --dataset hmdb --split 1 \
  -c ~/trx_data/checkpoints/bidir_hmdb1

# Quick smoke-test: 1-shot, fewer iterations, small batch
python run.py \
  --config configs/stage1_hausdorff.yaml \
  --shot 1 --training_iterations 5000 --tasks_per_batch 4 \
  --dataset hmdb --split 3 \
  -c ~/trx_data/checkpoints/smoke_test

# Change attention tau without editing the yaml
python run.py \
  --config configs/stage1_attention_weighted.yaml \
  --tau 0.05 \
  --dataset hmdb --split 3 \
  -c ~/trx_data/checkpoints/attn_tau005
```

---

### 5. Ablation matrix (Stage 1)

Run these three commands to produce the core Stage 1 ablation table.
Results are appended automatically to `experiments/results/results.csv`.

```bash
# Unidirectional
python run.py --config configs/stage1_hausdorff.yaml \
  --dataset hmdb --split 3 \
  -c ~/trx_data/checkpoints/abl_uni

# Bidirectional
python run.py --config configs/stage1_bidirectional.yaml \
  --dataset hmdb --split 3 \
  -c ~/trx_data/checkpoints/abl_bidir

# Attention-weighted
python run.py --config configs/stage1_attention_weighted.yaml \
  --dataset hmdb --split 3 \
  -c ~/trx_data/checkpoints/abl_attn
```

To sweep all three HMDB splits:

```bash
for split in 1 2 3; do
  for cfg in stage1_hausdorff stage1_bidirectional stage1_attention_weighted; do
    python run.py --config configs/${cfg}.yaml \
      --dataset hmdb --split ${split} \
      -c ~/trx_data/checkpoints/${cfg}_hmdb${split}
  done
done
```

---

### 6. Resuming from a checkpoint

```bash
python run.py \
  --config configs/stage1_hausdorff.yaml \
  --dataset hmdb --split 3 \
  -c ~/trx_data/checkpoints/stage1_hausdorff_hmdb3 \
  -r
```

The `-r` flag (`--resume_from_checkpoint`) picks up from the latest `checkpoint.pt` in the checkpoint directory.

---

### 7. Checking results

Test accuracy is printed to the log file inside the checkpoint directory and to stdout at every `test_iters` milestone (default: `[75000]`).

After any run, the summary CSV is at:

```
experiments/results/results.csv
```

Columns: `timestamp, config_file, dataset, split, way, shot, iteration, mean_accuracy, confidence_interval`

To view it:

```bash
python -c "import csv, sys; [print(r) for r in csv.DictReader(open('experiments/results/results.csv'))]"
```

---

### 8. Repo structure

```
TheRandomXeno/
├── configs/                   ← YAML experiment configs
│   ├── stage1_hausdorff.yaml
│   ├── stage1_bidirectional.yaml
│   ├── stage1_attention_weighted.yaml
│   ├── stage2_option_a.yaml   (frame-level relation — Stage 2)
│   └── stage2_option_b.yaml   (tuple-level relation — Stage 2)
│
├── models/
│   ├── stage1_model.py        ← CNN_TRX + TRXSetMatching (active)
│   ├── stage2_model.py        ← TRXSetMatchingWithRelation (skeleton)
│   └── trx_original.py        ← original TRX with prototype aggregation (reference)
│
├── matching/
│   ├── mean_hausdorff.py      ← mean_hausdorff_bidir (pool) + mean_hausdorff (instance)
│   ├── attention_weighted_hausdorff.py
│   └── bidirectional_hausdorff.py
│
├── relation/
│   ├── intra_relation.py      ← IntraRelation (skeleton for Stage 2)
│   └── inter_relation.py      ← InterRelation (skeleton for Stage 2)
│
├── experiments/
│   └── results/               ← results.csv written here during training
│
├── model.py                   ← thin wrapper (backward compat with run.py)
├── run.py                     ← training entrypoint
└── utils.py
```

---

### 9. Key hyperparameters reference

| YAML key | CLI flag | Default | Description |
|---|---|---|---|
| `matching_method` | — | — | `mean_hausdorff` / `attention_weighted_hausdorff` / `bidirectional_hausdorff` |
| `bidirectional` | `--bidirectional` | `false` | Enable symmetric Q↔S distance |
| `tau` | `--tau` | `0.1` | Softmax temperature (attention-weighted only) |
| `backbone` | `--method` | `resnet18` | `resnet18` / `resnet34` / `resnet50` |
| `trans_linear_out_dim` | `--trans_linear_out_dim` | `1152` | Tuple embedding output dim |
| `temp_set` | `--temp_set` | `[2]` | Tuple cardinalities (pairs = `[2]`) |
| `way` | `--way` | `5` | N-way classification |
| `shot` | `--shot` | `1` | K-shot support per class |
| `query_per_class` | `--query_per_class` | `5` | Query videos per class (train) |
| `training_iterations` | `-i` | `100020` | Total meta-training steps |
| `tasks_per_batch` | `--tasks_per_batch` | `16` | Gradient accumulation steps |
| `dataset` | `--dataset` | `ssv2` | `ssv2` / `kinetics` / `hmdb` / `ucf` |
| `split` | `--split` | `7` | Dataset split index |

---

## Data setup

Download your chosen dataset and extract frames in the format:

```
dataset/class/video/00000001.jpg   ← 8-digit zero-padded frame numbers
```

Zip the dataset folder with no compression:

```bash
zip -r -0 hmdb51_256q5.zip hmdb51_256q5/
```

Place the zip and split text files under `~/trx_data/` as shown in section 1.

---

## Method overview

**Original TRX** samples temporal tuples from query and support videos, builds cross-attended tuple embeddings, then **aggregates support tuples into per-class prototypes** before computing distances.

**This work** removes prototype aggregation entirely. Query tuple embeddings are matched against the **full pool of K × T support tuples** using mean-Hausdorff distance:

```
d(Q, S) = 0.5 * [ mean_q min_s ||q - s|| + mean_s min_q ||s - q|| ]
```

This preserves within-class tuple diversity that prototype averaging destroys.

---

## Citation

If you build on the original TRX method, please cite:

```bibtex
@inproceedings{perrett2021trx,
  title     = {Temporal Relational CrossTransformers for Few-Shot Action Recognition},
  booktitle = {Computer Vision and Pattern Recognition},
  year      = {2021}
}
```

---

## Acknowledgements

Based on [TRX](https://github.com/tobyperrett/trx) (Perrett et al., CVPR 2021).
Logging and training loop adapted from [CNAPs](https://github.com/cambridge-mlg/cnaps).
Video transforms from [torch_videovision](https://github.com/hassony2/torch_videovision).
