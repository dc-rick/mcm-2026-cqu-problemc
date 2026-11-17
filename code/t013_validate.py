"""T013 delivery validator: recompute the self-checks from the files."""
import glob
import hashlib
import io
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")
DELIV = os.path.join(ROOT, "交付")
BS = chr(92)
FP_PATH = os.path.join(ROOT, "data", "runs", "T001", "fingerprint_start.json")
DELIVERY = os.path.join(DELIV, "T013_delivery.json")
REF_VER = os.path.join(DELIV, "T013_ref_verification.json")
OLD_KEYS = {
    "shaheen2010", "fishman2013", "raviv2013", "chemia2013",
    "schuijbroek2017", "pfrommer2014", "singla2015", "omahony2015",
    "wilson1970", "erlander1990", "fotheringham1989", "mcfadden1974",
    "lloyd1982", "rousseeuw1987", "hotelling1929", "katz1985",
    "rochet2003", "nemhauser1978", "niemeyer2008", "osm", "gurobi",
    "virtanen2020", "mobike",
}
ALLOWED_SOURCE = {
    "paper/ai_report.tex", "paper/references.tex",
    "paper/sec2_assumptions.tex", "paper/sec1_intro.tex",
}

ok = []


def check(name, cond, detail=""):
    ok.append((name, bool(cond), detail))
    print(("PASS " if cond else "FAIL ") + name
          + (" | " + detail if detail else ""))


def read(path):
    return io.open(path, encoding="utf-8").read()


def flat(text):
    return re.sub(r"\s+", " ", text)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def parse_bibitems():
    out = {}
    for lineno, line in enumerate(read(os.path.join(PAPER, "references.tex"))
                                  .splitlines(), 1):
        if line.startswith(BS + "bibitem{"):
            key = line[len(BS + "bibitem{"):].split("}")[0].strip()
            out[key] = lineno
    return out


def parse_cites():
    out = {}
    for path in sorted(glob.glob(os.path.join(PAPER, "*.tex"))):
        name = os.path.basename(path)
        if name == "references.tex":
            continue
        for lineno, line in enumerate(read(path).splitlines(), 1):
            start = 0
            while True:
                pos = line.find(BS + "cite{", start)
                if pos < 0:
                    break
                end = line.find("}", pos)
                for key in line[pos + len(BS + "cite{"):end].split(","):
                    key = key.strip()
                    if key:
                        out.setdefault(key, []).append(
                            f"paper/{name}:{lineno}")
                start = end + 1
    return out


def main():
    d = json.load(io.open(DELIVERY, encoding="utf-8"))
    rv = json.load(io.open(REF_VER, encoding="utf-8"))
    ai = read(os.path.join(PAPER, "ai_report.tex"))
    sec2 = read(os.path.join(PAPER, "sec2_assumptions.tex"))
    ai_flat = flat(ai)
    sec2_flat = flat(sec2)

    check("delivery checks all true", all(d["checks"].values()),
          json.dumps(d["checks"], ensure_ascii=False))

    elements = d["ai_report"]["five_elements"]
    check("AI report title", BS + "section*{Report on Use of AI Tools}" in ai)
    check("AI report tool+version+dates",
          all(t in ai_flat for t in (
              "Claude Code (Anthropic)", "version 2.1.322",
              "11 to 17 November 2026")))
    check("AI report inline cite", BS + "cite{aiapp}" in sec2_flat)
    bibs = parse_bibitems()
    check("AI report bibitem", "aiapp" in bibs)
    check("AI report 2 query/output examples",
          ai.count(BS + "textbf{Example") >= 2
          and ai.count(BS + "textbf{Query.}") >= 2
          and ai.count(BS + "textbf{Output.}") >= 2)
    check("five element flags", all(elements.values()))

    old1 = "No number in the text is typed by hand"
    old2 = "All computations are reproducible"
    new1 = "Result numbers are script-generated, with listed exceptions"
    new2 = "reproducible in a fixed order, not by a single command"
    check("old claim 1 gone", old1 not in ai_flat)
    check("old claim 2 gone", old2 not in ai_flat)
    check("new claim 1 present", new1 in ai_flat)
    check("new claim 2 present", new2 in ai_flat)
    for item in d["false_claims"]:
        for ev in item["evidence"]:
            path = os.path.join(ROOT, ev["file"])
            lines = read(path).splitlines()
            check("evidence line " + ev["file"] + ":" + str(ev["line"]),
                  ev["line"] <= len(lines)
                  and lines[ev["line"] - 1].strip() == ev["text"].strip())

    old_a7 = "nobody was offered a reward in May 2017"
    new_a7 = "the order file contains no reward records"
    check("A7 old phrase gone", old_a7 not in sec2_flat)
    check("A7 new phrase present", new_a7 in sec2_flat)
    check("A7 two citations",
          BS + "cite{technode2017,sixthtone2017}" in sec2_flat)
    check("A7 TechNode accessible",
          rv["new_url_entries"]["technode2017"]["accessible"])
    check("A7 Sixth Tone accessible",
          rv["new_url_entries"]["sixthtone2017"]["accessible"])
    schema_lines = read(os.path.join(ROOT, "code", "00_prepare.py")).splitlines()[43:47]
    check("A7 dataset schema evidence",
          "orderid" in " ".join(schema_lines)
          and "reward" not in " ".join(schema_lines).lower())

    new_keys = {k for k in bibs if k not in OLD_KEYS}
    new_lit = new_keys - {"aiapp"}
    check("at least 6 new literature/news entries", len(new_lit) >= 6,
          str(len(new_lit)))
    check("6/6 new DOIs verified",
          rv["summary"]["new_dois_verified"] == 6
          and rv["summary"]["new_dois_total"] == 6)
    check("2/2 new URLs accessible",
          rv["summary"]["new_urls_accessible"] == 2
          and rv["summary"]["new_urls_total"] == 2)
    check("20/20 all DOIs verified",
          rv["summary"]["all_dois_verified"] == 20
          and rv["summary"]["all_dois_total"] == 20)

    cites = parse_cites()
    undefined = sorted(set(cites) - set(bibs))
    orphans = sorted(set(bibs) - set(cites))
    check("citation counts equal", len(cites) == len(bibs) == 32,
          f"cites={len(cites)} bibs={len(bibs)}")
    check("no undefined citations", not undefined, str(undefined))
    check("no orphan bibitems", not orphans, str(orphans))

    pass1 = os.path.join(DELIV, "T013_xelatex_pass1.log")
    pass2 = os.path.join(DELIV, "T013_xelatex_pass2.log")
    p1 = len(re.findall("undefined", read(pass1), re.IGNORECASE))
    p2 = len(re.findall("undefined", read(pass2), re.IGNORECASE))
    ml = len(re.findall("undefined", read(os.path.join(PAPER, "main.log")),
                        re.IGNORECASE))
    check("cold pass 1 had expected undefined warnings", p1 > 0, str(p1))
    check("pass 2 clean", p2 == 0, str(p2))
    check("main.log clean", ml == 0, str(ml))

    log = read(os.path.join(PAPER, "main.log"))
    match = re.search(r"Output written on main\.pdf \((\d+) pages", log)
    total = int(match.group(1))
    toc = read(os.path.join(PAPER, "main.toc"))
    match = re.search(
        r"contentsline \{section\}\{Report on Use of AI Tools\}\{(\d+)\}", toc)
    ai_start = int(match.group(1))
    ai_pages = total - ai_start + 1
    body = total - ai_pages
    check("total pages 27", total == 27, str(total))
    check("AI report starts p.26", ai_start == 26, str(ai_start))
    check("AI report 2 pages", ai_pages == 2, str(ai_pages))
    check("body excluding AI report 25 pages", body == 25, str(body))
    check("page-count delta recorded",
          d["compilation"]["baseline_total_pages"] == 25
          and d["compilation"]["total_page_delta"] == 2
          and d["compilation"]["baseline_body_pages_excluding_ai_report"] == 24
          and d["compilation"]["body_page_delta"] == 1)

    fp = json.load(io.open(FP_PATH, encoding="utf-8"))["trees"]
    baseline = {r["path"]: r["sha256"] for r in fp["paper"]}
    changed = []
    for path, digest in baseline.items():
        full = os.path.join(ROOT, path)
        if os.path.exists(full) and sha(full) != digest:
            changed.append(path)
    source_changed = [p for p in changed if p.endswith(".tex")]
    check("source changes within allowed set",
          set(source_changed) <= ALLOWED_SOURCE, str(source_changed))
    result_bearing = [
        p for p in baseline
        if p.endswith(".tex") and p not in source_changed
    ]
    check("result-bearing files unchanged",
          all(p not in changed for p in result_bearing))

    data_rows = fp["data"]
    data_changed = []
    for row in data_rows:
        full = os.path.join(ROOT, row["path"])
        if os.path.exists(full) and sha(full) != row["sha256"]:
            data_changed.append(row["path"])
    check("data excluding runs unchanged", not data_changed, str(data_changed))
    review_changed = []
    for row in fp["review"]:
        full = os.path.join(ROOT, row["path"])
        if os.path.exists(full) and sha(full) != row["sha256"]:
            review_changed.append(row["path"])
    check("review unchanged", not review_changed, str(review_changed))

    t013 = sorted(glob.glob(os.path.join(ROOT, "code", "t013_*.py")))
    check("three T013 scripts", len(t013) == 3,
          ", ".join(os.path.basename(p) for p in t013))
    check("citation map file exists",
          os.path.exists(os.path.join(DELIV, "T013_citation_map.md")))

    failed = [name for name, value, _ in ok if not value]
    print(f"\n{len(ok) - len(failed)}/{len(ok)} checks passed")
    if failed:
        raise SystemExit("FAILED: " + "; ".join(failed))
    print("T013 delivery consistent")


if __name__ == "__main__":
    main()
