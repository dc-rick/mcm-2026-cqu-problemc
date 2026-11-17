"""T003 step 1: inventory every numeric token in the paper.

Reads paper/*.tex (summary, sec1-sec10, ai_report), extracts numeric tokens
with file:line and the enclosing table (if any), and applies the exclusion
rules stated in the delivery.  Writes only data/runs/T003/.

Outputs
  tokens_raw.csv       one row per in-scope numeric token
  exclusions.csv       one row per excluded token with the rule that removed it
  table_cells.csv      numeric cells inside table environments
  scan_summary.json    counts and the reproducible coverage denominator
"""
import csv
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PAPER = os.path.join(ROOT, "paper")
OUT = os.path.join(ROOT, "data", "runs", "T003")
os.makedirs(OUT, exist_ok=True)

FILES = (["summary.tex"]
         + [f"sec{i}_{n}.tex" for i, n in [
             (1, "intro"), (2, "assumptions"), (3, "data"), (4, "q1"),
             (5, "q2"), (6, "q3"), (7, "q4"), (8, "q5"),
             (9, "sensitivity"), (10, "evaluation")]]
         + ["ai_report.tex"])

SCI = re.compile(r"(\d+(?:\.\d+)?)\s*\$?\\times\s*10\^\{?(-?\d+)\}?\$?")
DIM = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*\\times\s*"
                 r"(\d+(?:\.\d+)?)(?![\d.])(?!\s*\^)")
BRACE = re.compile(r"(?<![\w.])(\d{1,3}(?:\{,\}\d{3})+(?:\.\d+)?)(?![0-9])")
COMMA = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+(?:\.\d+)?)(?![0-9])")
PLAIN = re.compile(r"(?<![\w.\\])(\d+(?:\.\d+)?)(?:([kKmM])(?![A-Za-z]))?")
UNIT = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(pt|cm|mm|em|ex|\\textwidth|"
                  r"\\linewidth|\\columnwidth|\\baselineskip|\\textheight)")

EXCLUDE_CMDS = [
    r"\\label\{[^}]*\}", r"\\ref\{[^}]*\}", r"\\eqref\{[^}]*\}",
    r"\\cite[a-z]*\{[^}]*\}", r"\\input\{[^}]*\}", r"\\includegraphics\[[^]]*\]",
    r"\\vspace\*?\{[^}]*\}", r"\\hspace\*?\{[^}]*\}", r"\\setlength\{[^}]*\}\{[^}]*\}",
    r"\\addtolength\{[^}]*\}\{[^}]*\}", r"\\rule\{[^}]*\}\{[^}]*\}",
    r"\\multicolumn\{[^}]*\}\{[^}]*\}\{[^}]*\}", r"\\cmidrule(?:\([^)]*\))?\{[^}]*\}",
    r"\\cline\{[^}]*\}", r"\\addlinespace(?:\[[^]]*\])?", r"\\\\\s*\[[^]]*\]",
    r"\\captionsetup\{[^}]*\}", r"\\setcounter\{[^}]*\}\{[^}]*\}",
    r"\\begin\{tabular\}\{[^}]*\}", r"\\begin\{tabular\*\}\{[^}]*\}\{[^}]*\}",
    r"\\fontsize\{[^}]*\}\{[^}]*\}", r"\\renewcommand\{[^}]*\}\{[^}]*\}",
    r"\\tag\{[^}]*\}", r"\\hfill", r"\\quad", r"\\,",
    # time-of-day labels are not data numbers
    r"^\s*\d{2}\s*--\s*\d{2}\s*&", r"\b\d{1,2}:\d{2}\b",
    r"(?<=\bin )\d{2}\s*--\s*\d{2}\b", r"\b\d{2}\s*--\s*\d{2}\b(?=\s+or\b)",
    # calendar years are dates, not measurements
    r"\b(?:19|20)\d{2}\b",
    # calendar dates: day before month, date ranges, "between d and d Month"
    r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\b",
    r"\b\d{1,2}\s*--\s*\d{1,2}\s+(?:January|February|March|April|May|June|"
    r"July|August|September|October|November|December)\b",
    r"\bbetween\s+\d{1,2}\s+and\s+\d{1,2}\s+(?:January|February|March|April|"
    r"May|June|July|August|September|October|November|December)\b",
    # index ranges and percentile labels are notation, not data
    r"\d+\s*,\s*\\dots\s*,\s*\d+",
    r"\d+(?:st|nd|rd|th)[- ]percentile",
    r"\[\s*\d{1,3}\s*,\s*\d{1,3}\s*\)",
    # clock labels and code filenames
    r"\bor\s+\d{2}\s*--\s*\d{2}\b",
    r"\d{1,2}\s*(?:a\.m\.|p\.m\.)",
    r"\\texttt\{[^}]*\.py\}",
    # section commands and literal cross-references
    r"\\(?:sub)*section\*?\{[^}]*\}", r"\\paragraph\{[^}]*\}",
    r"§\s*\d+(?:\.\d+)?",
    r"(?:Section|Table|Fig\.?|Figure|Eq\.?|Equation)\s*~?\s*\(?\d+(?:\.\d+)?\)?",
]


def strip_comments(line):
    out, i = [], 0
    while i < len(line):
        if line[i] == "\\" and i + 1 < len(line):
            out.append(line[i:i + 2]); i += 2; continue
        if line[i] == "%":
            break
        out.append(line[i]); i += 1
    return "".join(out)


def extract(line):
    """Return (tokens, excluded).  Tokens carry their column in the raw line."""
    blocked = [False] * len(line)
    excluded = []

    def block(a, b):
        for i in range(max(0, a), min(len(line), b)):
            blocked[i] = True

    for pat in EXCLUDE_CMDS:
        for m in re.finditer(pat, line):
            for tm in re.finditer(r"\d+(?:[.,]\d+)*", m.group(0)):
                excluded.append(dict(token=tm.group(0), rule=pat))
            block(m.start(), m.end())
    for m in UNIT.finditer(line):
        excluded.append(dict(token=m.group(0), rule="length_with_unit"))
        block(m.start(), m.end())
    # subscripts / exponents are indices, not data (but keep 10^{k} for SCI)
    for m in re.finditer(r"([_^])\{([^}]*)\}", line):
        if m.group(1) == "^" and line[max(0, m.start() - 2):m.start()] == "10":
            continue
        for tm in re.finditer(r"\d+(?:[.,]\d+)*", m.group(2)):
            excluded.append(dict(token=tm.group(0), rule="subscript_group"))
        block(m.start(), m.end())

    found = []
    used = [False] * len(line)

    def take(rx, kind, convert):
        for m in rx.finditer(line):
            if any(blocked[m.start():m.end()]) or any(used[m.start():m.end()]):
                continue
            found.append(dict(token=m.group(0), value=convert(m), kind=kind,
                              col=m.start()))
            for i in range(m.start(), m.end()):
                used[i] = True

    take(SCI, "scientific", lambda m: float(m.group(1)) * 10 ** int(m.group(2)))
    # dimension pairs such as 153 x 153 or 555 x 426 (not 10^k)
    for m in DIM.finditer(line):
        if any(blocked[m.start():m.end()]) or any(used[m.start():m.end()]):
            continue
        for g in (1, 2):
            found.append(dict(token=m.group(g), value=float(m.group(g)),
                              kind="dimension", col=m.start(g)))
            for i in range(m.start(g), m.end(g)):
                used[i] = True
    take(BRACE, "brace_comma",
         lambda m: float(m.group(1).replace("{,}", "").replace(",", "")))
    take(COMMA, "comma", lambda m: float(m.group(1).replace(",", "")))
    take(PLAIN, "plain", lambda m: (float(m.group(1)) *
                                    {"k": 1e3, "m": 1e6}[m.group(2).lower()]
                                    if m.group(2) else float(m.group(1))))
    found.sort(key=lambda t: t["col"])
    # drop decorative superscript exponents such as km$^{-1}$ (SCI already consumed 10^k)
    kept = []
    for t in found:
        j = t["col"] - 1
        while j >= 0 and line[j] == " ":
            j -= 1
        if j >= 0 and line[j] in "^{":
            if line[j] == "{" and j - 1 >= 0 and line[j - 1] == "^":
                continue
            if line[j] == "^":
                continue
        kept.append(t)
    return kept, excluded


rows, exc, cells = [], [], []
tid = 0
table_no = 0
in_table = False
cur_table = None
cur_label = None
for fname in FILES:
    path = os.path.join(PAPER, fname)
    lines = open(path, encoding="utf-8").read().splitlines()
    for ln, raw in enumerate(lines, 1):
        line = strip_comments(raw)
        if "\\begin{table}" in line:
            table_no += 1
            in_table = True
            cur_table = f"table{table_no}"
            cur_label = None
        if in_table and "\\label{tab:" in line:
            m = re.search(r"\\label\{(tab:[^}]*)\}", line)
            if m:
                cur_label = m.group(1)
        tokens, excluded = extract(line)
        for t in tokens:
            tid += 1
            rec = dict(id=f"N{tid:04d}", file=f"paper/{fname}", line=ln,
                       col=t["col"], token=t["token"], value=t["value"],
                       kind=t["kind"], context=raw.strip()[:160],
                       table=cur_table if in_table else "")
            rows.append(rec)
            if in_table:
                cells.append(dict(rec, table_label=cur_label or ""))
        for e in excluded:
            exc.append(dict(file=f"paper/{fname}", line=ln, token=e["token"],
                            rule=e["rule"], context=raw.strip()[:120]))
        if "\\end{table}" in line:
            in_table = False
            cur_table = None
            cur_label = None

with open(os.path.join(OUT, "tokens_raw.csv"), "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
with open(os.path.join(OUT, "exclusions.csv"), "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=["file", "line", "token", "rule", "context"])
    w.writeheader(); w.writerows(exc)
with open(os.path.join(OUT, "table_cells.csv"), "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=list(cells[0].keys()))
    w.writeheader(); w.writerows(cells)

per_file = {}
for r in rows:
    per_file.setdefault(r["file"], []).append(r)
per_table = {}
for c in cells:
    per_table.setdefault(c["table"], []).append(c)
summary = {
    "files": {k: len(v) for k, v in per_file.items()},
    "n_tokens": len(rows),
    "n_table_tokens": len(cells),
    "n_excluded": len(exc),
    "exclusion_by_rule": {rule: sum(1 for e in exc if e["rule"] == rule)
                          for rule in sorted({e["rule"] for e in exc})},
    "tables": {k: dict(cells=len(v), records=len({r["id"] for r in v}),
                       labels=sorted({r["table_label"] for r in v}))
               for k, v in per_table.items()},
}
json.dump(summary, open(os.path.join(OUT, "scan_summary.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(json.dumps(summary, ensure_ascii=False, indent=1)[:2500])
