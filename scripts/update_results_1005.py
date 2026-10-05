#!/usr/bin/env python3
"""scripts/update_results_1005.py — 重跑 summarize_1005.py，把結果貼回 1005_LST對照實驗結果.md 的
SUMMARY 區塊，並更新最上面的進行中狀態行。用法：python scripts/update_results_1005.py "<狀態文字>" """
import os, re, subprocess, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
subprocess.run([sys.executable, os.path.join(REPO, "scripts", "summarize_1005.py")], check=True,
               stdout=subprocess.DEVNULL)
p = os.path.join(REPO, "1005_LST對照實驗結果.md")
s = open(p).read()
summ = open(os.path.join(REPO, "logs", "summary_1005.md")).read()
s = re.sub(r"<!-- SUMMARY-BEGIN -->.*<!-- SUMMARY-END -->",
           lambda m: f"<!-- SUMMARY-BEGIN -->\n{summ}\n<!-- SUMMARY-END -->", s, flags=re.S)
if len(sys.argv) > 1:
    s = re.sub(r"^> \*\*進行中\*\*：.*$", f"> **進行中**：{sys.argv[1]}", s, count=1, flags=re.M)
open(p, "w").write(s)
