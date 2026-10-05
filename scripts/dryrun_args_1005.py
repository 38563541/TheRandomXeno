#!/usr/bin/env python3
"""scripts/dryrun_args_1005.py — 不啟動訓練，用 run.py 的 parser 解析佇列中每個 job 的參數，
與對照組 log.txt 的 Options 逐欄比較（開跑前的完整參數 diff）。
用法：python scripts/dryrun_args_1005.py <queue.json> <ref B1 log.txt> <ref B4 log.txt>"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from args_diff_1005 import parse  # noqa

IGNORE = {"checkpoint_dir", "config_file"}
queue, ref1, ref4 = json.load(open(sys.argv[1])), parse(sys.argv[2]), parse(sys.argv[3])
for job in queue:
    sys.argv = ["run.py", *job["args"], "--dataset", "hmdb", "--split", "3",
                "--scratch", os.path.expanduser("~/Documents/work/trx/trx_data"), "-c", "/dryrun"]
    a = vars(run.Learner.__new__(run.Learner).parse_command_line())
    ref = ref4 if a.get("use_inter_relation") else ref1
    print(f"\n### {job['tag']}（對照：{'凍結 B4 s42' if ref is ref4 else '凍結 B1 s42'}）")
    for k in sorted(set(a) | set(ref)):
        if k in IGNORE:
            continue
        x, y = ref.get(k, "<無>"), a.get(k, "<無>")
        if x != y:
            print(f"  {k:22s} {x!r} → {y!r}")
