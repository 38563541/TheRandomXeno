# TRX 延遲聚合：tuple 層級的集合匹配

## 1. 這個 repo 是什麼

以 [TRX (CVPR 2021)](https://arxiv.org/abs/2101.06184) 為基礎的小樣本影片動作辨識。差別在於**延遲聚合（set-level matching）**：不先把 support 聚合成 prototype，而是把每支影片的「幀對 tuple」嵌入當成一個集合，query 集合與某類全部 support tuple 之間算 mean-Hausdorff 距離，logit = −距離。

## 2. 目錄結構

| 位置 | 內容 |
|---|---|
| `run.py` | 訓練／評估入口（YAML config + argparse，CLI 覆蓋 YAML） |
| `model.py` | `CNN_TRX(args)` 工廠：有 relation 模組 → Stage 2 模型，否則 Stage 1 |
| `video_reader.py`、`videotransforms/` | 從 zip 讀影格、組 episode、資料增強 |
| `utils.py` | loss、accuracy、log 工具 |
| `visualize.py` | 視覺化與 embedding 抽取（`tools/reselect_val.py` 也用它的 `load_model`） |
| `make_hmdb_trx_zip.py` | 用 ffmpeg 把 HMDB51 影片抽成 JPEG zip |
| `models/` | `stage1_model.py`（tuple 嵌入 + 集合匹配）、`stage2_model.py`（加 intra／inter relation）、`backbone_adapt.py`（frozen／partial_l4／lst／top_adapter）、`recipe.py`（1126：logit_scale、輔助分類頭）、`trx_original.py`（原版 TRX，`head: prototype` 對照組用） |
| `matching/` | mean-Hausdorff（pool／instance、單向／雙向）、attention-weighted Hausdorff |
| `relation/` | `intra_relation.py`、`inter_relation.py`（global，transductive）、`true_hyrsm_inter_relation.py`（逐 query，inductive） |
| `configs/` | 現行實驗 config（見下表） |
| `tests/` | 回歸測試（`git archive` 打包時排除，見 `.gitattributes`） |
| `tools/` | 常用工具：`reselect_val.py`、`probe_preproc_1111.py`、`compare_ckpt_runs_1102.py`、`build_from_config_1111.py`、`measure_vram.py` |
| `splits/` | 各資料集的 train／val／test 清單（HMDB 用 `hmdb_ARN`，split 3） |

## 3. 主要 config 對照

| 名稱 | config | 內容 |
|---|---|---|
| B1 | `configs/stage1_bidirectional.yaml` | 雙向 mean-Hausdorff、pool，沒有 relation |
| B2 | `configs/stage2_b2_intra_only.yaml`（`stage2_b2_intra_x2.yaml`：intra 疊 2 層） | tuple 層 intra |
| B3（true_hyrsm） | `configs/stage2_true_hyrsm_b3_inter_only.yaml` | tuple 層 inter，逐 query（inductive） |
| B3（global，transductive 對照） | `configs/stage2_b3_inter_only.yaml` | tuple 層 inter，所有 query 與 support 互相注意 |
| B4 | `configs/stage2_true_hyrsm_b4_tuple.yaml`（5-shot：`..._5shot.yaml`） | intra + true_hyrsm inter |
| m1–m4 | `configs/m1_uni_pool_1shot.yaml`、`m2_attn_pool_1shot.yaml`、`m3_bidir_pool_5shot.yaml`、`m4_bidir_instance_5shot.yaml` | 距離函數／集合聚合的對照（單向、attention-weighted、5-shot pool vs instance） |
| 凍結版 | `configs/frozen_rn18_b1.yaml`、`frozen_rn18_b2.yaml`、`frozen_rn18_b4.yaml` | 同 B1／B2／B4，`freeze_backbone: true` |
| prototype 對照組 | `configs/proto_1shot.yaml` | 同 B1，`head: prototype`（原版 TRX query-specific prototype，只用幀對） |

## 4. 怎麼跑

資料根目錄用 `--scratch` 指定，底下要有 `video_datasets/data/<zip>` 與 `video_datasets/splits/hmdb_ARN/`（新 clone 時從 `splits/hmdb_ARN/` 複製過去）。所有指令都在 repo 根目錄執行。

```bash
# 訓練（B4、HMDB split 3）
python run.py --config configs/stage2_true_hyrsm_b4_tuple.yaml --dataset hmdb --split 3 \
  --scratch ~/Documents/work/trx/trx_data --seed 42 \
  --test_iters 25000 50000 75000 100000 -c ~/trx_data/checkpoints/b4_s42

# 純評估一個 checkpoint（預設不寫 results.csv；要寫加 --eval_write_csv）
python run.py --config configs/stage1_bidirectional.yaml --dataset hmdb --split 3 \
  --scratch ~/Documents/work/trx/trx_data -c /tmp/eval_b1 \
  -m ~/trx_data/checkpoints/abl_bidir_hmdb3/hmdb_split3_20260407_220041/checkpoint25000.pt

# val 選點重算（checkpoint 路徑寫在腳本裡）
python tools/reselect_val.py

# 回歸測試（每個腳本各自執行，CPU 即可；verify_fixed_test_1111 需要 GPU 與資料）
python tests/verify_matching.py
```

## 5. 評估流程

- 每個 `--test_iters` 檢查點先評 val、再評 test。最後一步只有加 `--eval_at_end` 才會評估。
- val 用 `--val_seed`（預設 10001）抽 `--num_val_tasks`（預設 1000）題；test 用 `--test_seed`（預設 20001）抽 `--num_test_tasks`（預設 10000）題。評估前存下亂數狀態、評估後還原，所以每次評估、每個模型抽到同一批題目，訓練的亂數序列也不受影響。
- val 只記錄，不自動挑 best、不提前停止。
- 每題結果存成 `<checkpoint_dir>/eval_<split>_<iteration>.npz`（加 `--eval_save_preds` 會多存每個 query 的預測）。
- 彙總結果附加到 `experiments/results/results.csv`。這個檔案**不進版控**，只在本機。

## 6. 已移除的東西去哪找

以下都保留在 tag **`archive/pre-cleanup-1125`**（`git show archive/pre-cleanup-1125:<路徑>` 或 `git worktree add <目錄> archive/pre-cleanup-1125`）：

- `inter_style: hyrsm`（舊逐類 HyRSM，`relation/_deprecated/`）、`inter_style: decouple`（`relation/support_decouple_relation.py`）——現在設定這兩個值會直接報錯；
- `configs/stage2_option_a.yaml`、`configs/stage2_option_b.yaml`、`configs/_deprecated/`、`configs/exp_1112/`、`configs/b1_backbone_1shot.yaml`（與 `stage1_bidirectional.yaml` 相同）；
- 一次性腳本（`scripts/` 底下帶日期的 driver／分析、`compute_*.py`、`smoke_b2_b3.py`、`debug_test_crash.py`、`verify_d1.py`、`verify_amp_grad*.py`、`verify_results_csv_1005.py` 等）；
- 實驗紀錄（`logs/`、`reports/`、`figs/`、`checkpoints*/` 的 log、`transfer/`、根目錄帶日期的筆記）。這些目錄現在只留在本機、已加進 `.gitignore`。

實驗筆記與報告在 `~/Documents/work/trx_lab_notes`（`notes/`、`patches/`、`scripts/`），不在 repo 裡。

## 7. 訓練配方選項（1126）

所有選項都可以寫在 YAML config 或用 CLI 覆蓋。**預設值等於 1111 之前的行為**：不寫這些鍵，建出的模型、logits、訓練過程都和以前逐位元相同。

| 鍵 | 可選值 | 預設 | 意義 |
|---|---|---|---|
| `distance` | `sqeuclid`、`sqeuclid_over_d`、`cosine` | `sqeuclid` | tuple 之間的距離：平方 L2；平方 L2 ÷ 特徵維度 d；1 − cos（先 L2 正規化）。pool／instance、單向／雙向／attention_weighted 都適用。attention_weighted 的 `tau` 作用在距離上，換距離後 τ 的意義會變（預設 τ 沒改）。`head: prototype` 時不作用 |
| `logit_scale` | 正浮點數 | 不設定 | logit 溫度：logits = logit_scale × (−距離)，在 CNN 層級套用（set、prototype 都作用）。不設定＝不縮放 |
| `logit_scale_learnable` | `true`／`false` | `false` | 可學的 logit_scale（存 log(scale)，前向取 exp，初始值 = `logit_scale`；必須同時設 `logit_scale`） |
| `momentum` | 浮點數 | `0.0` | SGD momentum。`opt: adam` 時印警告並忽略 |
| `weight_decay` | 浮點數 | `0.0` | weight decay（SGD、Adam 都傳）。沒有做 norm／bias 不加 decay 的參數分組 |
| `lr_schedule` | `step`、`cosine` | `step` | `step`＝MultiStepLR（`--sch`，gamma 0.1）；`cosine`＝`warmup_iters` 內從 0 線性升到 lr，之後 cosine 降到 0，終點是 `training_iterations`。步數單位是 iteration（episode） |
| `warmup_iters` | 整數 | `0` | cosine 的 warmup 長度；`lr_schedule: step` 時不作用 |
| `aux_cls_weight` | 浮點數 | `0.0` | train 類別（HMDB 31 類）輔助分類 loss 的權重。> 0 時才建分類頭：每支影片 8 幀 backbone 特徵取平均 → Linear(D, K)，support 和 query 都用；task_loss = (原 loss + aux_cls_weight × CE) / tasks_per_batch。只在訓練時算；遇到不在 train 類別裡的影片直接報錯 |
| `head` | `set`、`prototype` | `set` | `set`＝本 repo 的集合匹配；`prototype`＝原版 TRX 的 query-specific prototype（`models/trx_original.py`，只用幀對），用它自己的距離（平方距離 ÷ tuple 數）。不能和 relation 模組一起開。對照組 config：`configs/proto_1shot.yaml` |

其他行為：
- config 裡有 run.py 不認得的鍵會**直接報錯**（以前只印警告）。
- 每次訓練開始時，解析完的所有參數會存成 `<checkpoint_dir>/resolved_config.yaml`。
- `results.csv` 最後面多了上表 9 欄，舊的列留空。
