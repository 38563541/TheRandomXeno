#!/usr/bin/env python3
"""
scripts/args_diff_1005.py — 印出兩個 run 的完整參數差異（讀 log.txt 第一行的 Options: Namespace(...)）。

用法：python scripts/args_diff_1005.py <reference log.txt 或 lst1005 log> <new log>
忽略純路徑／時間戳欄位以外，全部列出；呼叫端判斷差異是否都是刻意改的欄位。
"""
import ast
import re
import sys


def parse(path):
    text = open(path, errors="replace").read()
    m = re.search(r"Options: Namespace\((.*?)\)\n", text)
    if not m:
        raise SystemExit(f"{path}: 找不到 Options 行")
    node = ast.parse(f"f({m.group(1)})", mode="eval").body
    out = {}
    for kw in node.keywords:
        try:
            out[kw.arg] = ast.literal_eval(kw.value)
        except Exception:
            out[kw.arg] = ast.unparse(kw.value)
    return out


def main():
    a, b = parse(sys.argv[1]), parse(sys.argv[2])
    keys = sorted(set(a) | set(b))
    diff = [(k, a.get(k, "<無>"), b.get(k, "<無>")) for k in keys if a.get(k, "<無>") != b.get(k, "<無>")]
    for k, x, y in diff:
        print(f"{k:28s} {x!r:50s} → {y!r}")
    print(f"共 {len(diff)} 個欄位不同")


if __name__ == "__main__":
    main()
