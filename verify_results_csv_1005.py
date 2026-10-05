#!/usr/bin/env python3
"""
verify_results_csv_1005.py — 2-1 results.csv 修正的驗收。

  C0 負控制：修正前用 DictReader 讀原檔，已知列的值必須「對不上」（證明這個測試抓得到錯位）
  C1 在複本上跑 _ensure_csv_schema()：重寫後 header == _CSV_COLUMNS，列數不變，
     已知列（各時期各一列）用 DictReader 讀回的 iteration／mean_accuracy／config 精確相等
  C2 每一列 iteration 是正整數、mean_accuracy 在 [0,100]、confidence_interval 在 [0,5]
  C3 冪等：再跑一次不重寫
  C4 寫入時檢查：模擬 _CSV_COLUMNS 之後再加一欄，舊列自動補空欄、已知值不變
  C5 對不上任何 schema 的列 → raise（不默默寫壞）
  C6 正式檔：備份到 logs/results_backup_1005.csv 後重寫，再跑一次 C1/C2
用法：python verify_results_csv_1005.py [--apply]（不給 --apply 只驗複本，不動正式檔）
"""
import csv
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run  # noqa: E402

REPO = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.path.join(REPO, "logs", "verify_csv_1005_tmp")
os.makedirs(SCRATCH, exist_ok=True)

# (timestamp, config 結尾, iteration, mean_accuracy)：19／21／22 欄時期各取幾列，數值來自 checkpoint log.txt
KNOWN = [
    ("2026-08-04 07:18:50", "m3_bidir_pool_5shot.yaml", "75000", "67.724"),      # 19 欄
    ("2026-09-05 18:03:25", "stage2_true_hyrsm_b4_tuple.yaml", "25000", "51.7"),  # 21 欄（B4 s43 25k）
    ("2026-09-06 09:30:16", "stage2_b2_intra_only.yaml", "25000", "47.288"),      # 21 欄
    ("2026-09-07 14:10:16", "b1_backbone_1shot.yaml", "25000", "49.502"),         # 22 欄（B1 s43 25k）
    ("2026-09-13 15:43:42", "frozen_rn18_b1.yaml", "25000", "46.456"),            # 22 欄（凍結 B1 s42 25k）
    ("2026-09-14 12:59:40", "frozen_rn18_b4.yaml", "100000", "44.882"),           # 22 欄
]
fails = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        fails.append(name)


def by_ts(path):
    with open(path, newline="") as f:
        return {r["timestamp"]: r for r in csv.DictReader(f)}


def n_rows(path):
    with open(path, newline="") as f:
        return sum(1 for r in csv.reader(f) if r) - 1


def check_known(path, tag):
    rows = by_ts(path)
    for ts, cfg, it, acc in KNOWN:
        r = rows.get(ts, {})
        ok = (r.get("config_file", "").endswith(cfg) and r.get("iteration") == it
              and r.get("mean_accuracy") == acc)
        check(f"{tag} {ts}", ok, f"iteration={r.get('iteration')} acc={r.get('mean_accuracy')}")


def check_ranges(path, tag):
    bad = []
    for r in csv.DictReader(open(path, newline="")):
        try:
            it, acc, ci = int(r["iteration"]), float(r["mean_accuracy"]), float(r["confidence_interval"])
            if not (it > 0 and 0 <= acc <= 100 and 0 <= ci <= 5):
                bad.append(r["timestamp"])
        except (ValueError, TypeError, KeyError):
            bad.append(r.get("timestamp"))
    check(f"{tag} 每列 iteration/acc/CI 合理", not bad, f"bad={bad}")


def main():
    src = run._CSV_PATH
    n0 = n_rows(src)
    print(f"正式檔 {src}：{n0} 列")

    print("C0 負控制（原檔直接 DictReader）")
    rows = by_ts(src)
    wrong = sum(1 for ts, cfg, it, acc in KNOWN if rows.get(ts, {}).get("iteration") != it)
    already_fixed = open(src, newline="").readline().strip().split(",") == run._CSV_COLUMNS
    if already_fixed:
        print("  （正式檔已是新 header，C0 改用備份檔）")
        bk = os.path.join(REPO, "logs", "results_backup_1005.csv")
        rows = by_ts(bk)
        wrong = sum(1 for ts, cfg, it, acc in KNOWN if rows.get(ts, {}).get("iteration") != it)
    check("C0 原檔讀錯位（21/22 欄列的 iteration 讀錯）", wrong >= 5, f"{wrong}/{len(KNOWN)} 列讀錯")

    print("C1-C3 複本")
    cp = os.path.join(SCRATCH, "results_copy.csv")
    shutil.copy2(bk if already_fixed else src, cp)
    did = run._ensure_csv_schema(cp, backup_dir=SCRATCH)
    check("C1 有重寫", did)
    check("C1 header == _CSV_COLUMNS", open(cp, newline="").readline().strip().split(",") == run._CSV_COLUMNS)
    check("C1 列數不變", n_rows(cp) == n0, f"{n_rows(cp)} vs {n0}")
    check_known(cp, "C1")
    check_ranges(cp, "C2")
    check("C3 冪等（第二次不重寫）", run._ensure_csv_schema(cp, backup_dir=SCRATCH) is False)

    print("C4 模擬之後再加欄位")
    old_cols = list(run._CSV_COLUMNS)
    run._CSV_COLUMNS = old_cols + ["zz_new_col"]
    try:
        check("C4 有重寫", run._ensure_csv_schema(cp, backup_dir=SCRATCH))
        r = list(csv.DictReader(open(cp, newline="")))
        check("C4 新欄補空", all(x["zz_new_col"] == "" for x in r))
        check_known(cp, "C4")
    finally:
        run._CSV_COLUMNS = old_cols

    print("C5 對不上的列要 raise")
    bad = os.path.join(SCRATCH, "results_bad.csv")
    shutil.copy2(src if not already_fixed else bk, bad)
    with open(bad, "a") as f:
        f.write("a,b,c\n")
    try:
        run._ensure_csv_schema(bad, backup_dir=SCRATCH)
        check("C5 raise", False)
    except RuntimeError as e:
        check("C5 raise", True, str(e))

    if "--apply" in sys.argv and not already_fixed:
        print("C6 正式檔")
        bk_path = os.path.join(REPO, "logs", "results_backup_1005.csv")
        check("C6 備份檔名尚未被佔用", not os.path.exists(bk_path))
        run._ensure_csv_schema(src)
        check("C6 備份存在且與原檔列數相同", os.path.exists(bk_path) and n_rows(bk_path) == n0)
        check("C6 列數不變", n_rows(src) == n0)
        check_known(src, "C6")
        check_ranges(src, "C6")
    elif already_fixed:
        print("C6 正式檔（已修過，只重驗）")
        check("C6 列數", n_rows(src) == n0)
        check_known(src, "C6")
        check_ranges(src, "C6")

    shutil.rmtree(SCRATCH)
    print(f"\n失敗 {len(fails)} 項：{fails}" if fails else "\n全部通過")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
