#!/usr/bin/env python3
"""
scripts/summarize_1005.py — 從 logs 產生 1005 結果表與判定（R1–R4、B4−B1、回收比例）。

準確率來源：每個 run 的 logs/lst1005_<tag>.log 裡 run.py 印的
  {'hmdb': {'accuracy': x, 'confidence': y}}（依 test_iters 順序：25k、50k）。
既有對照組（凍結 s42、RN50 E1 B4）直接寫死，來源見 1005_分支盤點.md 0-7。
輸出：把 markdown 寫到 logs/summary_1005.md（結果 md 會貼這一段）。
規則（事先定，不准改）：
  過門檻 = 25k 與 50k 兩點同號，且 |差| 都 ≥ 0.4 pp。
  R1: C B1 − A B1 過門檻且為正
  R2: C B1 − B B1 過門檻且為正
  R3: D B1 ≥ C B1 − 0.4（兩點）且 D ms/iter ≤ 1.5 × C ms/iter
  R4: #9 − A B1（25k）≥ −0.4
  回收比例 = (arm − A)/(E − A)；E − A < 0.4 → 「分母過小，不計算」
"""
import ast
import csv
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(REPO, "logs")
THR = 0.4

# 既有對照組（25k, 50k）
FIXED = {
    ("rn18", "A", "B1"): (46.456, 46.836),   # 凍結 B1 s42（d99ea37）
    ("rn18", "A", "B4"): (45.510, 45.960),   # 凍結 B4 s42（2c15625）
    ("rn50", "E", "B4"): (49.984, 50.362),   # RN50 E1 B4 s42（bf34a2a，0920）
    # 使用者 10-06 指示：RN18 的 E 主列用 §A 那兩次 run（舊程式、無 seed、評估 episode 不配對）
    ("rn18", "E", "B1"): (46.736, 47.488),   # abl_bidir_hmdb3/hmdb_split3_20260407_220041
    ("rn18", "E", "B4"): (49.534, 49.880),   # true_hyrsm_b4_1shot/hmdb_split3_20260526_202630
}
LEGACY = {}
TAGS = {  # (backbone, arm, head) -> queue tag
    ("rn18", "A", "B1_anchor"): "rn18_A_B1_anchor25k",
    ("rn18", "E'", "B1"): "rn18_E_B1_s42",   # E′：1005 seed 42 重跑（G1）
    ("rn18", "E'", "B4"): "rn18_E_B4_s42",
    ("rn18", "C", "B1"): "rn18_C_B1_s42",
    ("rn18", "C", "B4"): "rn18_C_B4_s42",
    ("rn18", "D", "B1"): "rn18_D_B1_s42",
    ("rn18", "D", "B4"): "rn18_D_B4_s42",
    ("rn18", "B", "B1"): "rn18_B_B1_s42",
    ("rn18", "A", "B1_noaug"): "rn18_A_B1_noaug25k",
    ("rn50", "C", "B4"): "rn50_C_B4_s42",
    ("rn50", "A", "B4"): "rn50_A_B4_s42",
    ("rn50", "D", "B4"): "rn50_D_B4_s42",
    ("rn50", "C", "B1"): "rn50_C_B1_s42",
    ("rn50", "A", "B1"): "rn50_A_B1_s42",
    ("rn50", "D", "B1"): "rn50_D_B1_s42",
    ("rn50", "B", "B4"): "rn50_B_B4_s42",
    ("rn50", "E", "B1"): "rn50_E1_B1_s42",
    ("rn18", "C", "B1_s43"): "rn18_C_B1_s43",
    ("rn18", "C", "B4_s43"): "rn18_C_B4_s43",
}


def accs(tag):
    p = os.path.join(LOG, f"lst1005_{tag}.log")
    if not os.path.exists(p):
        return None
    out = []
    for m in re.finditer(r"\{'hmdb': \{'accuracy': ([\d.]+), 'confidence': ([\d.]+)\}\}", open(p, errors="replace").read()):
        out.append((float(m.group(1)), float(m.group(2))))
    return out


def get(bb, arm, head):
    if (bb, arm, head) in FIXED:
        return FIXED[(bb, arm, head)]
    t = TAGS.get((bb, arm, head))
    a = accs(t) if t else None
    if not a:
        return None
    vals = [x for x, _ in a]
    return tuple(vals + [None] * (2 - len(vals)))[:2]


def diff(x, y):
    if x is None or y is None:
        return None
    return tuple(None if (a is None or b is None) else a - b for a, b in zip(x, y))


def fmt(v):
    return "—" if v is None else f"{v:+.2f}"


def fmt_acc(v):
    return "—" if v is None else f"{v:.3f}"


def over(d, positive=True):
    """過門檻＝兩點同號、|差|都 ≥ 0.4；回傳 (成立/不成立/無法判定, 說明)"""
    if d is None or None in d:
        return "無法判定", "缺資料"
    a, b = d
    same = (a > 0 and b > 0) or (a < 0 and b < 0)
    big = abs(a) >= THR and abs(b) >= THR
    ok = same and big and (a > 0 if positive else a < 0)
    near = any(abs(abs(v) - THR) < 0.2 for v in d) or (abs(a) >= THR) != (abs(b) >= THR)
    note = f"25k {a:+.3f}／50k {b:+.3f}" + ("（§1-2(a) 模糊區）" if near else "")
    return ("成立" if ok else "不成立"), note


def main():
    L = []
    L.append("### 逐檢查點準確率（%，seed 42 除非另註）\n")
    L.append("| backbone | arm | head | 25k | 50k | 來源 |")
    L.append("|---|---|---|---|---|---|")
    for key in [("rn18", "A", "B1"), ("rn18", "A", "B1_anchor"), ("rn18", "A", "B1_noaug"), ("rn18", "A", "B4"),
                ("rn18", "B", "B1"), ("rn18", "C", "B1"), ("rn18", "C", "B4"), ("rn18", "D", "B1"),
                ("rn18", "D", "B4"), ("rn18", "E", "B1"), ("rn18", "E", "B4"),
                ("rn18", "E'", "B1"), ("rn18", "E'", "B4"),
                ("rn18", "C", "B1_s43"), ("rn18", "C", "B4_s43"),
                ("rn50", "A", "B1"), ("rn50", "A", "B4"), ("rn50", "B", "B4"), ("rn50", "C", "B1"),
                ("rn50", "C", "B4"), ("rn50", "D", "B1"), ("rn50", "D", "B4"), ("rn50", "E", "B1"),
                ("rn50", "E", "B4")]:
        v = get(*key)
        src = ("§A（舊程式、無 seed、不配對）" if key[:2] == ("rn18", "E") else "既有（0-7）") if key in FIXED else (f"`lst1005_{TAGS[key]}`" if key in TAGS else "")
        if v is None:
            v = (None, None)
            src += "（未跑／未完成）"
        L.append(f"| {key[0].upper()} | {key[1]} | {key[2]} | {fmt_acc(v[0])} | {fmt_acc(v[1])} | {src} |")
    L.append("\n（RN18 的 E ＝ §A 那兩次 run：2026-04-07／05-26 舊程式、沒有 seed、評估 episode 與 seed 42 的 run 不配對；"
             "E′ ＝ 1005 用現行程式 seed 42 重跑，與 A／B／C／D 同 seed、同評估 episode。兩者都列。）\n")

    A1, A4 = get("rn18", "A", "B1"), get("rn18", "A", "B4")
    C1, C4 = get("rn18", "C", "B1"), get("rn18", "C", "B4")
    B1_ = get("rn18", "B", "B1")
    D1, D4 = get("rn18", "D", "B1"), get("rn18", "D", "B4")
    E1, E4 = get("rn18", "E", "B1"), get("rn18", "E", "B4")

    L.append("### 判定（RN18，事先定的規則）\n")
    anc = get("rn18", "A", "B1_anchor")
    if anc and anc[0] is not None:
        d = anc[0] - A1[0]
        L.append(f"- 錨點 #1：{anc[0]:.3f} vs 既有 {A1[0]:.3f}，差 {d:+.3f} pp → "
                 f"{'通過（≤0.4）' if abs(d) <= THR else '**不通過（>0.4）**'}")
    r1 = over(diff(C1, A1))
    r2 = over(diff(C1, B1_))
    L.append(f"- R1 LST 有用（C B1 − A B1 過門檻且為正）：**{r1[0]}**（{r1[1]}）")
    L.append(f"- R2 不只是容量（C B1 − B B1 過門檻且為正）：**{r2[0]}**（{r2[1]}）")
    if D1 and C1 and None not in D1 and None not in C1:
        cond = all(d >= c - THR for d, c in zip(D1, C1))
        ms = {r["tag"]: float(r["ms_iter"]) for r in csv.DictReader(open(os.path.join(LOG, "lst_cost_1005.csv")))
              if r["ms_iter"] not in ("", "未量測")}
        msr = ms["RN18_D_B1"] / ms["RN18_C_B1"]
        L.append(f"- R3 簡單方案優先（D B1 ≥ C B1 − 0.4 兩點，且 D ms ≤ 1.5×C ms）：**{'成立' if cond and msr <= 1.5 else '不成立'}**"
                 f"（D−C：25k {D1[0]-C1[0]:+.3f}／50k {D1[1]-C1[1]:+.3f}；ms 比 {msr:.3f}，B1 頭）")
    else:
        L.append("- R3：無法判定（缺資料）")
    na = get("rn18", "A", "B1_noaug")
    if na and na[0] is not None:
        d = na[0] - A1[0]
        if d >= -THR:
            L.append(f"- R4 快取可行（#9 − A B1 25k ≥ −0.4）：**成立**（{d:+.3f}）")
        else:
            L.append(f"- R4 快取可行（#9 − A B1 25k ≥ −0.4）：**不成立**（{d:+.3f} pp，離門檻 {abs(d)-THR:.2f}，"
                     f"單一 seed、單一檢查點）→ 單份快取有疑慮，需要多份增強或補 seed；不等於「快取不可行」。")
    else:
        L.append("- R4：無法判定（缺資料）")

    L.append("\n### B4 − B1（逐檢查點，pp）\n")
    L.append("| backbone | arm | 25k | 50k |")
    L.append("|---|---|---|---|")
    for bb in ("rn18", "rn50"):
        for arm in ("A", "D", "C", "E") + (("E'",) if bb == "rn18" else ()):
            d = diff(get(bb, arm, "B4"), get(bb, arm, "B1"))
            d = d or (None, None)
            L.append(f"| {bb.upper()} | {arm} | {fmt(d[0])} | {fmt(d[1])} |")
    d = diff(get("rn18", "C", "B4_s43"), get("rn18", "C", "B1_s43"))
    if d:
        L.append(f"| RN18 s43 | C | {fmt(d[0])} | {fmt(d[1])} |")

    L.append("\n### 回收比例 (arm − A)/(E − A)\n")
    L.append("| backbone | E 用哪個 | head | arm | 25k | 50k |")
    L.append("|---|---|---|---|---|---|")
    for bb, ek in (("rn18", "E"), ("rn18", "E'"), ("rn50", "E")):
        for head in ("B1", "B4"):
            A_, E_ = get(bb, "A", head), get(bb, ek, head)
            for arm in ("B", "C", "D"):
                X = get(bb, arm, head)
                cells = []
                for i in range(2):
                    if not (A_ and E_ and X) or None in (A_[i], E_[i], X[i]):
                        cells.append("—")
                    elif E_[i] - A_[i] < THR:
                        cells.append(f"分母過小（E−A={E_[i]-A_[i]:+.2f}），不計算")
                    else:
                        cells.append(f"{(X[i]-A_[i])/(E_[i]-A_[i]):.2f}")
                L.append(f"| {bb.upper()} | {'E（§A）' if ek == 'E' and bb == 'rn18' else ek} | {head} | {arm} | {cells[0]} | {cells[1]} |")
    open(os.path.join(LOG, "summary_1005.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
