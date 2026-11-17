"""T001 fingerprint snapshot.

Records SHA-256 of data/ (excluding data/runs), paper/, review/, 审查与规划/,
code/lib.py and every file under code/ so that "we only changed the path in
lib.py" can be proven.  Usage:

    python code/audit/t001_fingerprint.py start
    python code/audit/t001_fingerprint.py end

Writes data/runs/T001/fingerprint_<tag>.json.  Read-only elsewhere.
"""
import datetime
import hashlib
import json
import os
import sys

TAG = sys.argv[1] if len(sys.argv) > 1 else "start"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snap(rel_dir, exclude_prefixes=()):
    items = []
    base = os.path.join(ROOT, rel_dir)
    for dp, dn, fn in os.walk(base):
        rel = os.path.relpath(dp, ROOT).replace("\\", "/")
        if any(rel == e or rel.startswith(e + "/") for e in exclude_prefixes):
            dn[:] = []
            continue
        dn.sort()
        for f in sorted(fn):
            p = os.path.join(dp, f)
            items.append({
                "path": os.path.relpath(p, ROOT).replace("\\", "/"),
                "sha256": sha(p),
                "size": os.path.getsize(p),
                "mtime": datetime.datetime.fromtimestamp(
                    os.path.getmtime(p)).isoformat(timespec="seconds"),
            })
    return items


out = {
    "tag": TAG,
    "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "root": ROOT,
    "trees": {
        "data": snap("data", exclude_prefixes=("data/runs",)),
        "paper": snap("paper"),
        "review": snap("review"),
        "审查与规划": snap("审查与规划"),
        "code": snap("code"),
    },
}
outdir = os.path.join(ROOT, "data", "runs", "T001")
os.makedirs(outdir, exist_ok=True)
outpath = os.path.join(outdir, f"fingerprint_{TAG}.json")
with open(outpath, "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1)
print(f"[{TAG}] saved {os.path.relpath(outpath, ROOT)}")
for k, v in out["trees"].items():
    print(f"  {k}: {len(v)} files")
libfile = [i for i in out["trees"]["code"] if i["path"] == "code/lib.py"]
print("  code/lib.py sha256:", libfile[0]["sha256"] if libfile else "MISSING")
