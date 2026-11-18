"""T027 R1/R1b/R1c/R1d verification.

Checks, with evidence written to 交付/T027/verify_final.json:
  1. Numeric-token multiset per paper/*.tex against baseline 3209636 (R0).
     Authorised: sec5_q2.tex gains token 12 exactly once (R1-5, TRI_NEAREST).
     ai_report.tex is excluded from the gate (R1-13) but its diff is reported.
  2. Forbidden strings: internal task ids (T0xx), revision language.
  3. Section-6 legacy wording checks (carried from R0), the R1-1..R1-13
     item checks from task sheet section 8.1, the R1b-1/R1b-2 text
     checks from section 8.4, the R1c-1..R1c-5 text checks from section 8.5,
     and the R1d-1/R1d-2 text checks from section 8.6.
  4. Regression: 975 / 617,055.7515 / 39,984.2332 (and companions) recomputed
     from data/runs/T024/schedule_check.json and matched to the paper strings.
  5. PDF gates: 27 pages, 25 counted, '??', U+037E, overfull / undefined /
     LaTeX-warning counts, TOC landmarks (References p25, AI report p26).

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
SCHED = os.path.join(ROOT, "data", "runs", "T024", "schedule_check.json")
BASE = "3209636"

TOK = re.compile(r"(?<![A-Za-z0-9])\d[\d,]*(?:\.\d+)?")

# Authorised token changes vs 3209636 (task sheet 8.3).
EXPECTED = {
    "sec5_q2.tex": ({}, {"12": 1}),  # R1-5 TRI_NEAREST=12
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


def norm(text):
    return " ".join(text.split())


def main():
    ev = {"task": "T027", "revision": "R1d", "baseline": BASE, "date": "2026-11-19"}
    ok = True

    tex = sorted(f for f in os.listdir(PAPER) if f.endswith(".tex"))
    cache = {name: read_file(os.path.join(PAPER, name)) for name in tex}
    cnorm = {name: norm(text) for name, text in cache.items()}

    def has(name, s):
        return s in cnorm[name]

    # ---- 1. token multiset per file -------------------------------------
    gate = {}
    for name in tex:
        path = "paper/" + name
        new = cache[name]
        old = git_show(BASE, path)
        if name == "ai_report.tex":
            removed, added = counter_diff(tokens(old), tokens(new))
            gate[name] = {"excluded_from_gate": True,
                          "old_tokens": len(tokens(old)),
                          "new_tokens": len(tokens(new)),
                          "removed": removed, "added": added}
            continue
        removed, added = counter_diff(tokens(old), tokens(new))
        gate[name] = {"removed": removed, "added": added,
                      "identical": not removed and not added}
        if removed or added:
            exp = EXPECTED.get(name)
            if exp is not None and (removed, added) == exp:
                gate[name]["expected_authorised"] = True
            else:
                ok = False
                gate[name]["UNEXPECTED"] = True
    ev["token_gate"] = gate
    ev["token_gate_pass"] = ok and all(
        v.get("identical", True) or v.get("expected_authorised") or v.get("excluded_from_gate")
        for v in gate.values())

    # ---- 2. carried section-6 wording checks ---------------------------
    def has_exact(name, s):
        return s in cache[name]

    checks6 = {
        "sec3_old_adjacency_claim_gone": not has_exact("sec3_data.tex", "must start in the same 153 m geohash cell"),
        "sec3_old_correction_claim_gone": not has_exact("sec3_data.tex", "Correcting the exact-match rate"),
        "sec3_lower_bound_wording": has_exact("sec3_data.tex", "a lower bound; 17.72"),
        "sec3_wide_rate_wording": has("sec3_data.tex", "The wide rate, less the false-match share"),
        "summary_10_19": has_exact("summary.tex", "10--19 May window") and not has_exact("summary.tex", "10--24 May"),
        "sec5_vrp_gone": (not has_exact("sec5_q2.tex", "vehicle-routing")
                          and not has_exact("sec5_q2.tex", "integer routing program")),
        "sec5_actual_implementation": (has("sec5_q2.tex", "time-space network program")
                                       and has("sec5_q2.tex", "earliest finish")),
        "sec2_overnight_gone": not has_exact("sec2_assumptions.tex", "overnight shift"),
        "sec2_20kmh": has_exact("sec2_assumptions.tex", "travel at 20 km/h"),
        "sec7_measured_inventory_gone": (not has("sec7_q4.tex", "measured inventory")
                                         and not has("sec7_q4.tex", "At the measured")),
        "sec7_measured_516_gone": not has("sec7_q4.tex", "measured 5/16"),
        "sec7_sizing_rows": (has("sec7_q4.tex", "Assigned sizing demand")
                             and has("sec7_q4.tex", "Unassigned sizing demand")
                             and not has("sec7_q4.tex", "& Assigned demand")),
        "sec4_ari_0305": has_exact("sec4_q1.tex", "0.305") and not has_exact("sec4_q1.tex", "0.306"),
        "ai_report_proxy": (has("ai_report.tex", "initial-inventory proxy")
                            and not has("ai_report.tex", "measured initial inventory")),
        "ai_report_fleet_grid": has_exact("ai_report.tex", "\\{0,20,40,60,80\\}"),
        "references_aiapp_1119": (has("references.tex", "used 11--19 November 2026")
                                  and not has("references.tex", "used 11--17 November 2026")),
    }
    ev["section6_checks"] = checks6

    # ---- 3. R1 item checks (task sheet 8.1) ----------------------------
    sec7_cites = cache["sec7_q4.tex"].count("\\cite{gurobi}")
    r1 = {
        "R1-1_A11_scope": (has("sec2_assumptions.tex", "not restricted to the source zone")
                           and has("sec2_assumptions.tex", "No cross-hour rearrangement is modelled")
                           and not has("sec2_assumptions.tex", "in the same zone")),
        "R1-1b_A12_consistency_fix": has("sec2_assumptions.tex",
                                         "priced in the truck-caliber test of Section~\\ref{sec:q3}"),
        "R1-2_sec6_limits": (has("sec6_q3.tex", "reroutes are cell-level, not zone-level")
                             and has("sec6_q3.tex",
                                     "the only channel through which the incentive can reach the truck layer")
                             and not has("sec6_q3.tex", "restricted to the \\emph{same zone}")),
        "R1-3_sec10_saving": (has("sec10_evaluation.tex",
                                  "neither credits nor charges the induced truck-layer change")
                              and has("sec10_evaluation.tex",
                                      "which the truck-caliber test prices separately")
                              and not has("sec10_evaluation.tex", "Reroutes stay within a zone")),
        "R1-4_truck_fleet": (has("sec5_q2.tex", "conserve the truck count, with $K$ at the")
                             and has("sec5_q2.tex", "no per-zone requirement")
                             and has("sec5_q2.tex", "deadhead relocation reported below, outside the")
                             and not has("sec5_q2.tex", "return them to their starting zones")),
        "R1-5_tri_block": (has("sec5_q2.tex", "the 12 nearest")
                           and has("sec5_q2.tex", "\\label{eq:q2tri}")
                           and has("sec5_q2.tex", "earliest finish")
                           and has("sec5_q2.tex", "smallest integer fleet")),
        "R1-6_siting_block": (has("sec7_q4.tex", "\\label{eq:q4cap}")
                              and has("sec7_q4.tex", "\\emph{uncapped} reference only")
                              and has("sec7_q4.tex", "the per-site cap set above")),
        "R1-7_closed_fleet_shape": has("sec7_q4.tex", "flow-implied one, not the proxy's"),
        "R1-8_inventory_lower_bound": has("sec3_data.tex", "a lower bound on the true stock"),
        "R1-9_coincidence": (has("sec4_q1.tex", "numerical coincidence")
                             and has("sec4_q1.tex", "the two quantities are unrelated")),
        "R1-10_gurobi_cite_once": sec7_cites == 1,
        "R1-11_typo": (has("sec3_data.tex", "less the false-match share")
                       and not has("sec3_data.tex", "less that false-match")),
        "R1-12_precision": has("sec10_evaluation.tex",
                               "printed at the precision of the archived result files"),
        "R1-13_ai_report": has("ai_report.tex",
                               "12 nearest stocked donor cells per deficit cell in the tricycle stage"),
        "no_7_46_anywhere": all("7.46" not in text for text in cache.values()),
        "no_same_zone_claim": (not has("sec2_assumptions.tex", "in the same zone")
                               and not has("sec6_q3.tex", "same zone, which keeps")
                               and not has("sec10_evaluation.tex", "stay within a zone, which")),
    }
    ev["r1_checks"] = r1

    # ---- 3b. R1b item checks (task sheet 8.4) --------------------------
    r1b = {
        "R1b-1_sizing_demand_units": (
            has("sec7_q4.tex", "With $f_{ij}$ the sizing demand at $i$")
            and has("sec7_q4.tex", "$u_i$ the sizing demand left unassigned")
            and has("sec7_q4.tex", "$a_i$ the sizing demand at $i$")
            and not has("sec7_q4.tex", "$f_{ij}$ arrivals at $i$")),
        "R1b-2_caption_row_ref": (
            has("sec7_q4.tex", "in the row labelled Arrival coverage")
            and not has("sec7_q4.tex", "in the last row")),
    }
    ev["r1b_checks"] = r1b

    # ---- 3c. R1c item checks (task sheet 8.5) --------------------------
    r1c = {
        "R1c-1_summary_replay_scope": has(
            "summary.tex", "in this replay reroutes stay within a zone"),
        "R1c-2_A12_tricycle_replay": (
            has("sec2_assumptions.tex",
                "The tricycle-caliber replay of Section~\\ref{sec:q3} keeps the target")
            and has("sec2_assumptions.tex",
                    "inside the source zone, so the truck layer stays fixed there by construction")),
        "R1c-3_sec6_two_calibers": (
            has("sec6_q3.tex", "The replay keeps every reroute inside the source zone")
            and has("sec6_q3.tex",
                    "the unrestricted placement rule of Section~\\ref{sec:assume}")
            and has("sec6_q3.tex", "fixed by construction; the unrestricted")),
        "R1c-4_sec6_truck_caliber_first": (
            has("sec6_q3.tex",
                "First, in the truck-caliber test reroutes are cell-level, not zone-level")
            and not has("sec6_q3.tex", "First, reroutes are cell-level")),
        "R1c-5_summary_4_33_upper": (
            has("summary.tex", "an upper estimate of 4.33 rides each per day")
            and not has("summary.tex", "serving \\textbf{4.33 rides")),
    }
    ev["r1c_checks"] = r1c

    # ---- 3d. R1d item checks (task sheet 8.6) --------------------------
    r1d = {
        "R1d-1_sigma_defined": has(
            "sec8_q5.tex",
            "is $\\sigma(a+2m\\tilde x_i)$, where $\\sigma$ is the logistic function"),
        "R1d-2_deadhead_shared": (
            has("sec5_q2.tex",
                "The truck cost is common to both totals in the comparison")
            and has("sec5_q2.tex",
                    "this deadhead shifts the two sides equally and leaves the percentage saving unchanged")),
    }
    ev["r1d_checks"] = r1d
    ok = (ok and all(checks6.values()) and all(r1.values()) and all(r1b.values())
          and all(r1c.values()) and all(r1d.values()))

    # ---- 4. forbidden strings ------------------------------------------
    pats = [r"T0\d\d", r"six figures", r"all six", r"8\s*图", r"六图",
            r"earlier version", r"previous version", r"前版", r"上一版",
            r"T0xx"]
    forbid = {}
    for name in tex:
        hits = {}
        for p in pats:
            m = re.findall(p, cache[name])
            if m:
                hits[p] = len(m)
        if hits:
            forbid[name] = hits
    ev["forbidden_strings"] = forbid
    ev["forbidden_strings_pass"] = not forbid
    ok = ok and not forbid

    # ---- 5. regression against the archived result file ----------------
    sched = json.load(io.open(SCHED, encoding="utf-8"))
    t = sched["exec_totals"]
    reg = {
        "n_exec": t["n_exec"],
        "unmet_printed": "{:,.4f}".format(t["unmet"]),
        "layer_printed": "{:,.2f}".format(t["layer_cost_fullload"]),
        "total_printed": "{:,.4f}".format(t["total_fullload"]),
        "saving_printed": "{:,.2f}".format(t["saving_fullload"]),
        "share_printed": "{:.3f}%".format(100.0 * t["saving_fullload_share"]),
    }
    joined = norm(" ".join(cache[f] for f in ("sec1_intro.tex", "sec5_q2.tex", "sec10_evaluation.tex")))
    joined = joined.replace("\\", "")
    reg["pass"] = (
        reg["n_exec"] == 975
        and reg["unmet_printed"] == "39,984.2332" and reg["unmet_printed"] in joined
        and reg["layer_printed"] == "533,248.01" and reg["layer_printed"] in joined
        and reg["total_printed"] == "617,055.7515" and reg["total_printed"] in joined
        and reg["saving_printed"] == "902,223.20" and reg["saving_printed"] in joined
        and reg["share_printed"] == "59.385%" and reg["share_printed"] in joined
    )
    ev["regression"] = reg
    ok = ok and reg["pass"]

    # ---- 6. L16 spot checks (carried from R0) --------------------------
    sec4 = cache["sec4_q1.tex"]
    ev["L16"] = {
        "has_3.28--3.64": "3.28--3.64" in sec4,
        "has_3.28--3.65": "3.28--3.65" in sec4,
        "table_3.645_present": "3.645" in sec4,
        "pass": ("3.28--3.64" in sec4 and "3.28--3.65" not in sec4
                 and "3.645" in sec4),
    }
    ok = ok and ev["L16"]["pass"]

    # ---- 7. PDF gates ---------------------------------------------------
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
            tt = p.get_text()
            if "??" in tt:
                qq.append(p.number + 1)
            if ";" in tt:
                gr.append(p.number + 1)
            for line in tt.splitlines():
                s = line.strip()
                if s in ("References", "Report on Use of AI Tools"):
                    landmarks.setdefault(s, []).append(p.number + 1)
        ev["pdf"].update({"pages": pages, "pages_with_qq": qq,
                          "pages_with_u037e": gr, "landmarks": landmarks,
                          "counted_pages": pages - 2})
    toc = read_file(os.path.join(PAPER, "main.toc")) if os.path.exists(os.path.join(PAPER, "main.toc")) else ""
    ev["pdf"]["toc_references"] = "References" in toc
    ev["pdf"]["toc_ai_report"] = "Report on Use of AI Tools" in toc
    ev["pdf"]["pdf_pass"] = (
        ev["pdf"]["overfull_count"] == 0
        and ev["pdf"]["undefined_warnings"] == 0
        and ev["pdf"]["latex_warnings"] == 0
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
                      "r1_checks_pass": all(r1.values()),
                      "r1b_checks_pass": all(r1b.values()),
                      "r1c_checks_pass": all(r1c.values()),
                      "r1d_checks_pass": all(r1d.values()),
                      "section6_pass": all(checks6.values()),
                      "regression_pass": reg["pass"],
                      "forbidden_pass": ev["forbidden_strings_pass"],
                      "pdf_pass": ev["pdf"]["pdf_pass"]}, indent=1))
    return 0 if ev["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
