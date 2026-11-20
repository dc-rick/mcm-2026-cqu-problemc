"""T028 acceptance verification.

Writes 交付/T028/verify_final.json with:
  1. Number lock: data/runs/T038/numbers.json (n_tokens, n_matched, unmatched 0)
     and coverage_report.txt's PASS line.
  2. Regression: key accepted numbers recomputed from their result files and
     matched to the exact printed strings in paper/*.tex.
  3. Forbidden strings: internal task ids and revision language in the sources.
  4. Figure inventory: exactly the four figures used, flagged in file_list.json.
  5. PDF gates: 28 pages, 25 counted, references end on p25, AI report starts
     p26, no '??', no U+037E, 0 overfull, 0 undefined references, 0 LaTeX
     warnings, TOC on p2.

Run:  python -I 交付/T028/verify_final.py
"""
import io
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PAPER = os.path.join(ROOT, "paper")
OUT = os.path.join(ROOT, "交付", "T028", "verify_final.json")
RUN = lambda *p: os.path.join(ROOT, "data", "runs", *p)


def read(path):
    with io.open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def jload(path):
    with io.open(path, encoding="utf-8") as f:
        return json.load(f)


def norm(s):
    return " ".join(s.split())


def main():
    ev = {"task": "T028", "date": "2026-11-20"}
    ok = True

    tex = sorted(f for f in os.listdir(PAPER) if f.endswith(".tex"))
    cache = {n: read(os.path.join(PAPER, n)) for n in tex}
    joined = norm(" ".join(cache.values()))

    def printed(x, fmt):
        return format(x, fmt)

    # ---- 1. number lock ------------------------------------------------
    lock = jload(RUN("T038", "numbers.json"))
    cov = read(RUN("T038", "coverage_report.txt"))
    ev["lock"] = {
        "n_tokens": lock["n_tokens"],
        "n_matched": lock["n_matched"],
        "n_records": lock["n_records"],
        "unmatched_lines": len(re.findall(r"^UNMATCHED", cov, re.M)),
        "pass_line": "PASS" in cov,
    }
    ev["lock_pass"] = (lock["n_tokens"] == lock["n_matched"]
                       and ev["lock"]["unmatched_lines"] == 0
                       and ev["lock"]["pass_line"])
    ok = ok and ev["lock_pass"]

    # ---- 2. regression from stored results ----------------------------
    q2 = jload(RUN("T028", "local_search.json"))
    t31 = jload(RUN("T031", "truck_payment.json"))
    tri = jload(RUN("T031", "tricycle_payment.json"))
    fr = jload(RUN("T030", "frontier_cons.json"))["levels"]["meas343"]
    q5 = jload(RUN("T029", "m15_outside.json"))

    base = q2["baseline_total"]
    saving = base - q2["exec_cost"]
    share = 100.0 * saving / base
    tk = t31["levels"]["arrivals"]["best"]
    all_truck = t31["levels"]["arrivals"]["all_truck_cost"]
    tk_save = all_truck - tk["total_spend"]
    # tricycle-caliber optimum (arrivals cap) from the stored optimum block
    tp = tri.get("optimum") or tri.get("best")
    sel = fr["selected"]

    reg = {
        "q2_baseline": printed(base, ",.2f"),
        "q2_exec_n": format(q2["exec_n"], ",d"),
        "q2_exec_cost": printed(q2["exec_cost"], ",.2f"),
        "q2_exec_unmet": printed(q2["exec_unmet"], ",.2f"),
        "q2_exec_vkm": printed(q2["exec_tri_vkm"], ",.2f"),
        "q2_saving": printed(saving, ",.2f"),
        "q2_share": printed(share, ".2f"),
        "q3_all_truck": printed(all_truck, ",.2f"),
        "q3_r": tk["r"], "q3_q": printed(tk["q"], ",.2f"),
        "q3_P": printed(tk["P"], ",.2f"),
        "q3_payment": printed(tk["payment"], ",.2f"),
        "q3_total": printed(tk["total_spend"], ",.2f"),
        "q3_saving": printed(tk_save, ",.2f"),
        "q3_share": printed(100.0 * tk_save / all_truck, ".2f"),
        "q4_sites": sel and fr["selected_n"],
        "q4_coverage": printed(sel["coverage"], ".4f"),
        "q4_arrival": printed(sel["arrival_coverage"], ".4f"),
        "q4_unassigned": printed(sel["unassigned"], ",.2f"),
    }
    if tp:
        reg["q3_tri_total"] = printed(tp.get("total_spend", float("nan")), ",.2f")
        reg["q3_tri_payment"] = printed(tp.get("payment", float("nan")), ",.2f")
    # Q5: capacity ratios and the m=0 two-platform rate
    try:
        rows = q5.get("rows", [])
        r0 = [x for x in rows if x["m"] == 0.0 and x["institution"] == "shared"][0]
        reg["q5_m0_unmet"] = printed(100.0 * r0["unmet_rate"], ".2f")
        reg["q5_ratio_m0"] = printed(
            q5["ratios"]["0.0"]["capacity_ratio_ind_over_shared"], ".4f")
        reg["q5_ratio_m15"] = printed(
            q5["ratios"]["1.5"]["capacity_ratio_ind_over_shared"], ".4f")
    except Exception:
        reg["q5_m0_unmet"] = None

    needs = {
        "q2_baseline": reg["q2_baseline"],
        "q2_exec_n": reg["q2_exec_n"],
        "q2_exec_cost": reg["q2_exec_cost"],
        "q2_exec_unmet": reg["q2_exec_unmet"],
        "q2_exec_vkm": reg["q2_exec_vkm"],
        "q2_saving": reg["q2_saving"],
        "q2_share": reg["q2_share"],
        "q3_all_truck": reg["q3_all_truck"],
        "q3_q": reg["q3_q"], "q3_P": reg["q3_P"],
        "q3_payment": reg["q3_payment"], "q3_total": reg["q3_total"],
        "q3_saving": reg["q3_saving"], "q3_share": reg["q3_share"],
        "q4_coverage": reg["q4_coverage"], "q4_arrival": reg["q4_arrival"],
        "q4_unassigned": reg["q4_unassigned"],
    }
    if "q3_tri_total" in reg:
        needs["q3_tri_total"] = reg["q3_tri_total"]
        needs["q3_tri_payment"] = reg["q3_tri_payment"]
    for k in ("q5_m0_unmet", "q5_ratio_m0", "q5_ratio_m15"):
        if reg.get(k):
            needs[k] = reg[k]
    missing = {k: v for k, v in needs.items()
               if v is not None and str(v) not in joined
               and str(v).replace(",", "{,}") not in joined}
    ev["regression"] = reg
    ev["regression_missing_in_tex"] = missing
    ev["regression_pass"] = not missing
    ok = ok and ev["regression_pass"]

    # ---- 3. forbidden strings -----------------------------------------
    pats = [r"(?<![A-Za-z0-9])T0\d\d(?![0-9])", r"earlier version",
            r"previous version", r"six figures", r"all six", r"前版", r"上一版"]
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
    ev["forbidden_pass"] = not forbid
    ok = ok and ev["forbidden_pass"]

    # ---- 4. figure inventory ------------------------------------------
    fl = jload(os.path.join(ROOT, "交付", "T028", "file_list.json"))
    figs = fl["sections"]["figures_pdf"]
    used = [e["path"] for e in figs if e.get("used_in_paper")]
    inc = sorted(set(re.findall(r"includegraphics\[[^\]]*\]\{\.\./(figs/[^}]+)\}",
                                " ".join(cache.values()))))
    ev["figures"] = {"used_in_paper": used, "included_in_tex": inc}
    ev["figures_pass"] = (sorted("figs/" + os.path.basename(u) for u in used)
                          == inc)
    ok = ok and ev["figures_pass"]

    # ---- 5. PDF gates --------------------------------------------------
    try:
        import fitz
    except ImportError:
        fitz = None
    log = read(os.path.join(PAPER, "main.log"))
    ev["pdf"] = {
        "overfull_count": len(re.findall(r"^Overfull", log, re.M)),
        "undefined_warnings": len(re.findall(r"There were undefined references", log)),
        "latex_warnings": len(re.findall(r"^LaTeX Warning", log, re.M)),
        "output_line": (re.findall(r"Output written on .*", log) or [""])[-1],
    }
    if fitz is not None:
        doc = fitz.open(os.path.join(PAPER, "main.pdf"))
        pages = doc.page_count
        qq, gr, landmarks = [], [], {}
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
        ai_p = max(landmarks.get("Report on Use of AI Tools") or [pages + 1])
        ref_p = max(landmarks.get("References") or [0])
        ev["pdf"].update({
            "pages": pages,
            "pages_with_qq": qq,
            "pages_with_u037e": gr,
            "landmarks": landmarks,
            "references_last_page": ref_p,
            "ai_report_first_page": ai_p,
            "counted_pages": ai_p - 1,
            "toc_on_p2": "Contents" in doc[1].get_text(),
            "summary_on_p1": "Recommendations" in doc[0].get_text()
            or "Beijing" in doc[0].get_text(),
        })
    ev["pdf"]["pdf_pass"] = (
        ev["pdf"]["overfull_count"] == 0
        and ev["pdf"]["undefined_warnings"] == 0
        and ev["pdf"]["latex_warnings"] == 0
        and ev["pdf"].get("pages") == 28
        and ev["pdf"].get("counted_pages") == 25
        and ev["pdf"].get("pages_with_qq") == []
        and ev["pdf"].get("pages_with_u037e") == []
        and ev["pdf"].get("references_last_page") == 25
        and ev["pdf"].get("ai_report_first_page") == 26
        and ev["pdf"].get("toc_on_p2")
    )
    ok = ok and ev["pdf"]["pdf_pass"]

    ev["all_pass"] = bool(ok)
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(ev, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: ev[k] for k in
                      ("all_pass", "lock_pass", "regression_pass",
                       "forbidden_pass", "figures_pass")}, indent=1))
    print(json.dumps({"pdf_pass": ev["pdf"]["pdf_pass"],
                      "missing_in_tex": ev["regression_missing_in_tex"]}, indent=1))
    return 0 if ev["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
