"""T003 helper: print the scanned tokens grouped by paper line.

Usage: python code/audit/t003_dump.py summary.tex [sec3_data.tex ...]
Reads data/runs/T003/tokens_raw.csv (written by t003_numbers.py).
"""
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOK = os.path.join(ROOT, "data", "runs", "T003", "tokens_raw.csv")

rows = list(csv.DictReader(open(TOK, encoding="utf-8-sig")))
want = set(sys.argv[1:])
cur = None
for r in rows:
    name = os.path.basename(r["file"])
    if want and name not in want:
        continue
    key = (name, r["line"])
    if key != cur:
        print(f"\n{name}:{r['line']}  | {r['context']}")
        cur = key
    print(f"   [{r['col']:>3}] {r['token']:<12} = {float(r['value']):<14g} {r['kind']}")
