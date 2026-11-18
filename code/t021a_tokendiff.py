"""T021A regression token diff: numeric tokens per paper .tex file at c4be8d1 vs tree.

Writes data/runs/T021A/token_diff.txt.  A token is counted as added when its
multiset count rises and removed when it falls; unchanged repeat occurrences
cancel.  The point is to make every number that moved auditable file by file.
"""
import io
import os
import re
import subprocess
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")
OUT = os.path.join(ROOT, "data", "runs", "T021A", "token_diff.txt")
BASE = "c4be8d1"

NUM = re.compile(r"(?<![A-Za-z0-9])\d[\d,]*(?:\.\d+)?")


def tokens(text):
    return Counter(NUM.findall(text))


def read(path):
    with io.open(path, encoding="utf-8") as fh:
        return fh.read()


def git_show(rel):
    out = subprocess.run(["git", "show", "%s:%s" % (BASE, rel)], cwd=ROOT,
                         capture_output=True)
    if out.returncode != 0:
        return ""
    return out.stdout.decode("utf-8", "replace")


def main():
    files = sorted(f for f in os.listdir(PAPER) if f.endswith(".tex"))
    lines = ["T021A numeric-token diff vs %s (paper/*.tex)" % BASE, ""]
    total_add = total_del = 0
    rows = []
    for name in files:
        rel = "paper/" + name
        a = tokens(git_show(rel))
        b = tokens(read(os.path.join(PAPER, name)))
        added = b - a
        removed = a - b
        na, nr = sum(added.values()), sum(removed.values())
        total_add += na
        total_del += nr
        if not na and not nr:
            rows.append("%-24s unchanged" % name)
            continue
        rows.append("%-24s +%d -%d" % (name, na, nr))
        if na:
            lines.append("%s ADDED: %s" % (rel, ", ".join(
                "%s x%d" % (k, v) for k, v in sorted(added.items()))))
        if nr:
            lines.append("%s REMOVED: %s" % (rel, ", ".join(
                "%s x%d" % (k, v) for k, v in sorted(removed.items()))))
    lines[1:1] = ["per-file counts:", ""] + rows + ["",
                                                    "total added=%d removed=%d" % (total_add, total_del), ""]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(rows))
    print("total added=%d removed=%d -> %s" % (total_add, total_del, OUT))


if __name__ == "__main__":
    main()
