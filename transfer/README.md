# TRX 三日實驗移交材料
## 2026-09-06 ~ 2026-09-08

本資料夾包含三日實驗的全部報告、訓練記錄與關鍵設定，供撰寫論文報告使用。

---

## 資料夾結構

```
transfer/
├── README.md               ← 本文件
├── reports/                ← 每日報告 + 最終總表
│   ├── 整日_0906.md        Phase 0-3 全日實驗（B2-x2 訓練、uni 初嘗試被砍、B1 seed=43 重跑）
│   ├── 晚間_0907.md        B1 config 確認、N=1000 val 重跑方法、uni_m1 夜間訓練啟動
│   ├── 最後一晚_0907.md    nn.Sequential 向後相容修復、B4/B2 val 補跑（GPU 不足 defer）
│   └── 最終總表_0907.md    ★ 主表：所有設定的 1-shot/5-shot 結果（uni_m1 100k 填入後完整）
├── logs/                   ← scripts/ 的輸出 log（stdout，部分因 buffering 為 0 bytes）
│   ├── b1_seed43.log       B1 seed=43 訓練 stdout（已完整）
│   ├── b2_intra_x2_1shot.log  B2-x2 訓練 stdout（已完整）
│   ├── reselect.log        N_VAL=1000/N_TEST=10000 重選 val 腳本輸出
│   ├── phase0eval.log      Phase 0 綜合評估腳本（N_VAL=200, N_TEST=200 舊方法）
│   └── uni_m1.log          uni_m1 stdout（Python block-buffered → 0 bytes；見 checkpoint_logs/）
├── checkpoint_logs/        ← run.py 直接寫入 checkpoint 目錄的 log.txt（無 buffering，可靠）
│   ├── b1_seed43_training.log    B1 seed=43 訓練記錄（含四個檢查點 test acc）
│   ├── b2_intra_x2_training.log  B2-x2 訓練記錄（含四個檢查點 test acc）
│   └── uni_m1_training.log       uni_m1 訓練記錄（100k 完成後需重新複製）
└── configs/                ← 各設定的 YAML 設定檔
    ├── b1_backbone_1shot.yaml      B1：純骨幹，無 relation，雙向 pool Hausdorff
    ├── stage2_b2_intra_x2.yaml     B2-x2：intra depth=2，無 inter
    ├── b4_true_hyrsm_1shot.yaml    B4：intra+inter（TrueHyRSM），雙向
    ├── m1_uni_pool_1shot.yaml      uni_m1：單向 pool Hausdorff，無 relation（= B1 單向版）
    └── stage1_hausdorff.yaml       Stage 1 原始 config（廢棄鍵 warning，已棄用於 uni 實驗）
```

---

## 最終總表快速索引（見 reports/最終總表_0907.md）

### 1-shot 關鍵數字

| 設定 | peak test acc | peak ckpt |
|------|:------------:|:---------:|
| B1 骨幹 seed=43（雙向） | **49.5%** | 25k |
| B2 intra-only seed=43 | 48.8% | 75k |
| B4 intra+inter seed=43 | **51.7%** | 25k |
| B2-x2（33M, intra×2）seed=42 | 45.9% | 25k |
| uni_m1（單向）seed=42 | 48.2% | 75k |

B4 高於 B2 的檢查點數：**8 / 8**（4 ckpt × 2 seed）。

### 距離函數對比（雙向 vs 單向，seed=42, B1 backbone）

| | 25k | 50k | 75k | 100k |
|--|-----|-----|-----|------|
| 雙向（B1 s42） | 46.7 | 47.5 | 46.3 | 46.4 |
| 單向（uni_m1 s42）| 47.9 | 47.2 | **48.2** | 待填（~02:05 完成）|

---

## 方法論注意事項

1. **所有結果來自 run.py training loop**（`num_test_tasks=10000`），不混用其他腳本。
2. **N_VAL=1000 / N_TEST=10000** 為晚間_0907 修正後的方法（見 reselect.log）。
   - B4/B2 的 N=1000 val 補跑因 nn.Sequential key mismatch 未完成（修復後可重跑）。
   - B1 的 N=1000 val 已完成：seed=42 最佳@25k→test=46.71%，seed=43 最佳@50k→test=47.90%。
3. **B1 seed=42（abl_bidir_hmdb3）** 使用 `temp_set=[2,3]`（pairs+triples），非嚴格控制組。
4. **stage1_hausdorff.yaml 廢棄鍵問題**：`matching_method`、`bidirectional` 不被程式碼讀取；
   使用 `m1_uni_pool_1shot.yaml`（新式 `matching: unidirectional`）才能正確設定距離函數。
5. **nn.Sequential 向後相容修復**（最後一晚_0907 階段 1）：
   depth=1 時保留 `self.intra = IntraRelation(...)` 直接指派，保持舊 checkpoint 鍵名相容。

---

## 待補項目（transfer 後手動更新）

- [ ] `reports/最終總表_0907.md` 中 uni_m1 100k 欄位（~02:05 CST 完成，自動更新）
- [ ] `checkpoint_logs/uni_m1_training.log`（100k 完成後重新複製 checkpoint log.txt）
- [ ] B4/B2 N=1000 val 補跑（GPU 釋放後，修復 nn.Sequential 已完成，可直接重跑 reselect_val.py）

---

*最後更新：2026-09-08 01:00 CST*
