"""T027 final-state verification.

Checks, with evidence written to 交付/T027/verify_final.json:
  1. Body numeric-token multiset against baseline 8708d37, per file.
     Expected: identical everywhere except sec4_q1.tex, where the only
     difference is 3.65 -> 3.64 (L16).  ai_report.tex is excluded from the
     gate (authorised additions) but its diff is reported for the record.
  2. Forbidden strings: internal task ids (T0xx), revision language,
     wrong figure-count statements.
  3. L16 spot checks in sec4_q1.tex.
  4. PDF gates: page count, '??', U+037E (Greek question mark), and the
     main.log overfull/undefined counts.
  5. Page landmarks: References and AI report start pages; counted pages.

Run:  python -I 交付/T027/verify_final.py
"""
import io
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PAPER = os.path.join(ROOT, "paper")
OUT = os.path.join(ROOT, "交付", "T027", "verify_final.json")
BASE = "8708d37"

TOK = re.compile(r"(?<![A-Za-z0-9])\d[\d,]*(?:\.\d+)?")

# Authorised token changes vs 8708d37 (task sheet sections 3 and 6).
EXPECTED = {
    "sec4_q1.tex": ({"3.65": 1, "0.306": 1}, {"3.64": 1, "0.305": 1}),  # L16 + #7
    "summary.tex": ({"24": 1}, {"19": 1}),            # #2 10--24 -> 10--19 May
    "sec2_assumptions.tex": ({}, {"20": 1}),          # #4 trucks travel at 20 km/h
    "references.tex": ({"17": 1}, {"19": 1}),         # #9 aiapp 11--17 -> 11--19 Nov
}


def read_file(path):
    with io.open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def tokens(text):
    return TOK.findall(text.replace("{,}", ","))


def counter_diff(a, b):
    ca, cb = {}, {}
    for t in a:
        ca[t] = ca.get(t, 0) + 1
    for t in b:
        cb[t] = cb.get(t, 0) + 1
    removed = {t: ca[t] - cb.get(t, 0) for t in ca if ca[t] > cb.get(t, 0)}
    added = {t: cb[t] - ca.get(t, 0) for t in cb if cb[t] > ca.get(t, 0)}
    return removed, added


def git_show(rev, path):
    r = subprocess.run(["git", "show", "%s:%s" % (rev, path)],
                       cwd=ROOT, capture_output=True)
    if r.returncode != 0:
        raise SystemExit("git show failed for %s:%s" % (rev, path))
    return r.stdout.decode("utf-8", "replace")


def main():
    ev = {"task": "T027", "baseline": BASE, "date": "2026-11-19"}
    ok = True

    # ---- 1. token multiset per file -------------------------------------
    tex = sorted(f for f in os.listdir(PAPER) if f.endswith(".tex"))
    gate = {}
    for name in tex:
        path = "paper/" + name
        new = read_file(os.path.join(PAPER, name))
        old = git_show(BASE, path)
        if name == "ai_report.tex":
            removed, added = counter_diff(tokens(old), tokens(new))
            gate[name] = {"excluded_from_gate": True,
                          "old_tokens": len(tokens(old)),
                          "new_tokens": len(tokens(new)),
                          "removed_types": len(removed),
                          "added_types": len(added)}
            continue
        removed, added = counter_diff(tokens(old), tokens(new))
        gate[name] = {"removed": removed, "added": added,
                      "identical": not removed and not added}
        if removed or added:
            exp = EXPECTED.get(name)
            if exp is not None and removed == exp[0] and added == exp[1]:
                gate[name]["expected_authorised"] = True
            else:
                ok = False
                gate[name]["UNEXPECTED"] = True
    ev["token_gate"] = gate
    ev["token_gate_pass"] = ok and all(
        v.get("identical", True) or v.get("expected_authorised") or v.get("excluded_from_gate")
        for v in gate.values())

    # ---- 1b. section-6 correction spot checks --------------------------
    def has(name, s):
        return s in read_file(os.path.join(PAPER, name))

    checks6 = {
        "sec3_old_adjacency_claim_gone": not has("sec3_data.tex", "must start in the same 153 m geohash cell"),
        "sec3_old_correction_claim_gone": not has("sec3_data.tex", "Correcting the exact-match rate"),
        "sec3_lower_bound_wording": has("sec3_data.tex", "a lower bound; 17.72"),
        "sec3_wide_rate_wording": has("sec3_data.tex", "The wide rate, less that false-match share"),
        "summary_10_19": has("summary.tex", "10--19 May window") and not has("summary.tex", "10--24 May"),
        "sec5_vrp_gone": (not has("sec5_q2.tex", "vehicle-routing")
                          and not has("sec5_q2.tex", "integer routing program")),
        "sec5_actual_implementation": (has("sec5_q2.tex", "transportation program")
                                       and has("sec5_q2.tex", "earliest finish")),
        "sec2_overnight_gone": not has("sec2_assumptions.tex", "overnight shift"),
        "sec2_20kmh": has("sec2_assumptions.tex", "travel at 20 km/h"),
        "sec7_measured_inventory_gone": (not has("sec7_q4.tex", "measured inventory")
                                         and not has("sec7_q4.tex", "At the measured")),
        "sec7_measured_516_gone": not has("sec7_q4.tex", "measured 5/16"),
        "sec7_sizing_rows": (has("sec7_q4.tex", "Assigned sizing demand")
                             and has("sec7_q4.tex", "Unassigned sizing demand")
                             and not has("sec7_q4.tex", "& Assigned demand")),
        # #7 ARI minimum 0.306 -> 0.305
        "sec4_ari_0305": has("sec4_q1.tex", "0.305") and not has("sec4_q1.tex", "0.306"),
        # #8 AI report terminology and fleet grid
        "ai_report_proxy": (has("ai_report.tex", "initial-inventory proxy")
                            and not has("ai_report.tex", "measured initial inventory")),
        "ai_report_fleet_grid": has("ai_report.tex", "\\{0,20,40,60,80\\}"),
        # #9 aiapp reference date range
        "references_aiapp_1119": (has("references.tex", "used 11--19 November 2026")
                                  and not has("references.tex", "used 11--17 November 2026")),
    }
    ev["section6_checks"] = checks6
    ok = ok and all(checks6.values())

    # ---- 2. forbidden strings ------------------------------------------
    pats = [r"T0\d\d", r"six figures", r"all six", r"8\s*图", r"六图",
            r"earlier version", r"previous version", r"前版", r"上一版",
            r"T0xx"]
    forbid = {}
    for name in tex:
        text = read_file(os.path.join(PAPER, name))
        hits = {}
        for p in pats:
            m = re.findall(p, text)
            if m:
                hits[p] = len(m)
        if hits:
            forbid[name] = hits
    ev["forbidden_strings"] = forbid
    ev["forbidden_strings_pass"] = not forbid

    # ---- 3. L16 spot checks --------------------------------------------
    sec4 = read_file(os.path.join(PAPER, "sec4_q1.tex"))
    ev["L16"] = {
        "has_3.28--3.64": "3.28--3.64" in sec4,
        "has_3.28--3.65": "3.28--3.65" in sec4,
        "table_3.645_present": "3.645" in sec4,
        "pass": ("3.28--3.64" in sec4 and "3.28--3.65" not in sec4
                 and "3.645" in sec4),
    }
    ok = ok and ev["L16"]["pass"]

    # ---- 4. PDF gates ---------------------------------------------------
    try:
        import fitz
    except ImportError:
        fitz = None
    pdf_path = os.path.join(PAPER, "main.pdf")
    log = read_file(os.path.join(PAPER, "main.log"))
    ev["pdf"] = {
        "overfull_count": len(re.findall(r"^Overfull", log, re.M)),
        "undefined_warnings": len(re.findall(r"There were undefined references", log)),
        "latex_warnings": len(re.findall(r"^LaTeX Warning", log, re.M)),
        "output_line": (re.findall(r"Output written on .*", log) or [""])[-1],
    }
    if fitz is not None:
        doc = fitz.open(pdf_path)
        pages = doc.page_count
        qq, gr = [], []
        landmarks = {}
        for p in doc:
            t = p.get_text()
            if "??" in t:
                qq.append(p.number + 1)
            if ";" in t:
                gr.append(p.number + 1)
            for line in t.splitlines():
                s = line.strip()
                if s in ("References", "Report on Use of AI Tools"):
                    landmarks.setdefault(s, []).append(p.number + 1)
        ev["pdf"].update({"pages": pages, "pages_with_qq": qq,
                          "pages_with_u037e": gr, "landmarks": landmarks,
                          "counted_pages": pages - 2})
    # TOC entries
    toc = read_file(os.path.join(PAPER, "main.toc"))
    ev["pdf"]["toc_references"] = "References" in toc
    ev["pdf"]["toc_ai_report"] = "Report on Use of AI Tools" in toc
    ev["pdf"]["pdf_pass"] = (
        ev["pdf"]["overfull_count"] == 0
        and ev["pdf"]["undefined_warnings"] == 0
        and ev["pdf"].get("pages") == 27
        and ev["pdf"].get("pages_with_qq") == []
        and ev["pdf"].get("pages_with_u037e") == []
        and 25 in ev["pdf"].get("landmarks", {}).get("References", [])
        and 26 in ev["pdf"].get("landmarks", {}).get("Report on Use of AI Tools", [])
        and ev["pdf"].get("counted_pages") == 25
        and ev["pdf"]["toc_references"] and ev["pdf"]["toc_ai_report"]
    )
    ok = ok and ev["pdf"]["pdf_pass"]

    ev["all_pass"] = bool(ok)
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(ev, f, ensure_ascii=False, indent=1)
    print(json.dumps({"all_pass": ev["all_pass"],
                      "token_gate_pass": ev["token_gate_pass"],
                      "forbidden_pass": ev["forbidden_strings_pass"],
                      "pdf_pass": ev["pdf"]["pdf_pass"]}, indent=1))
    return 0 if ev["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
