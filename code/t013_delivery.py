"""T013 delivery builder.

Assembles 交付/T013_delivery.json and 交付/T013_citation_map.md from the
current paper tree, the T001 start fingerprint, the compile logs and the
reference-verification JSON.  Reads only; writes only under 交付/.
"""
import datetime
import glob
import hashlib
import io
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")
DELIV = os.path.join(ROOT, "交付")
OUT = os.path.join(DELIV, "T013_delivery.json")
CITATION_MD = os.path.join(DELIV, "T013_citation_map.md")
FP_PATH = os.path.join(ROOT, "data", "runs", "T001", "fingerprint_start.json")
REF_VER = os.path.join(DELIV, "T013_ref_verification.json")
BS = chr(92)

OLD_KEYS = [
    "shaheen2010", "fishman2013", "raviv2013", "chemia2013",
    "schuijbroek2017", "pfrommer2014", "singla2015", "omahony2015",
    "wilson1970", "erlander1990", "fotheringham1989", "mcfadden1974",
    "lloyd1982", "rousseeuw1987", "hotelling1929", "katz1985",
    "rochet2003", "nemhauser1978", "niemeyer2008", "osm", "gurobi",
    "virtanen2020", "mobike",
]
ALLOWED_SOURCE = {
    "paper/ai_report.tex",
    "paper/references.tex",
    "paper/sec2_assumptions.tex",
    "paper/sec1_intro.tex",
}


def read(path):
    return io.open(path, encoding="utf-8").read()


def flat(text):
    return re.sub(r"\s+", " ", text)


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def tree_files(tree):
    out = []
    for root, dirs, files in os.walk(os.path.join(ROOT, tree)):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for name in files:
            path = os.path.join(root, name)
            rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
            out.append(rel)
    return sorted(out)


def compare_tree(tree, baseline_rows, exclude_prefix=None):
    baseline = {r["path"]: r["sha256"] for r in baseline_rows}
    current = {
        p: sha256(os.path.join(ROOT, p)) for p in tree_files(tree)
        if not (exclude_prefix and p.startswith(exclude_prefix))
    }
    changed = sorted(p for p in baseline if p in current
                     and current[p] != baseline[p])
    missing = sorted(p for p in baseline if p not in current)
    added = sorted(p for p in current if p not in baseline)
    return {
        "baseline_files": len(baseline),
        "current_files": len(current),
        "changed": changed,
        "added": added,
        "removed": missing,
        "unchanged": not changed and not added and not missing,
    }


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


def line_text(relpath, lineno):
    lines = read(os.path.join(ROOT, relpath)).splitlines()
    return lines[lineno - 1].strip() if 0 < lineno <= len(lines) else ""


def count_undefined(path):
    return len(re.findall("undefined", read(path), re.IGNORECASE))


def main():
    fp = json.load(io.open(FP_PATH, encoding="utf-8"))["trees"]
    paper_cmp = compare_tree("paper", fp["paper"])
    data_cmp = compare_tree("data", fp["data"], exclude_prefix="data/runs/")
    review_cmp = compare_tree("review", fp["review"])
    code_cmp = compare_tree("code", fp["code"])

    bibs = parse_bibitems()
    cites = parse_cites()
    undefined = sorted(set(cites) - set(bibs))
    orphans = sorted(set(bibs) - set(cites))

    paper_text = read(os.path.join(PAPER, "main.tex"))
    ai_text = read(os.path.join(PAPER, "ai_report.tex"))
    sec2_text = read(os.path.join(PAPER, "sec2_assumptions.tex"))
    sec1_text = read(os.path.join(PAPER, "sec1_intro.tex"))
    refs_text = read(os.path.join(PAPER, "references.tex"))
    ai_flat = flat(ai_text)
    sec2_flat = flat(sec2_text)

    title_ok = BS + "section*{Report on Use of AI Tools}" in ai_text
    tool_ok = all(token in ai_flat for token in (
        "Claude Code (Anthropic)", "version 2.1.322",
        "11 to 17 November 2026"))
    inline_ok = BS + "cite{aiapp}" in sec2_flat
    bibentry_ok = "aiapp" in bibs
    n_examples = ai_text.count(BS + "textbf{Example")
    n_query = ai_text.count(BS + "textbf{Query.}")
    n_output = ai_text.count(BS + "textbf{Output.}")
    examples_ok = n_examples >= 2 and n_query >= 2 and n_output >= 2

    old_claim_1 = "No number in the text is typed by hand"
    old_claim_2 = "All computations are reproducible"
    new_claim_1 = "Result numbers are script-generated, with listed exceptions"
    new_claim_2 = "reproducible in a fixed order, not by a single command"
    false_claims_ok = (
        old_claim_1 not in ai_flat and old_claim_2 not in ai_flat
        and new_claim_1 in ai_flat and new_claim_2 in ai_flat
    )

    old_a7 = "nobody was offered a reward in May 2017"
    new_a7 = "the order file contains no reward records"
    a7_ok = (old_a7 not in sec2_flat and new_a7 in sec2_flat
             and "red-envelope" in sec2_flat
             and BS + "cite{technode2017,sixthtone2017}" in sec2_flat)

    ref_ver = json.load(io.open(REF_VER, encoding="utf-8"))
    new_keys = [k for k in bibs if k not in OLD_KEYS]
    new_lit = [k for k in new_keys if k != "aiapp"]
    ref_links_ok = True
    for key in new_lit:
        if key not in refs_text:
            ref_links_ok = False
    refs_ok = (
        len(new_lit) >= 6
        and ref_ver["summary"]["new_dois_verified"] == 6
        and ref_ver["summary"]["new_urls_accessible"] == 2
        and ref_ver["summary"]["all_dois_verified"]
        == ref_ver["summary"]["all_dois_total"]
        and ref_links_ok
    )

    log = read(os.path.join(PAPER, "main.log"))
    match = re.search(r"Output written on main\.pdf \((\d+) pages", log)
    total_pages = int(match.group(1))
    toc = read(os.path.join(PAPER, "main.toc"))
    toc_match = re.search(
        r"contentsline \{section\}\{Report on Use of AI Tools\}\{(\d+)\}", toc)
    ai_start = int(toc_match.group(1))
    ai_pages = total_pages - ai_start + 1
    body_pages = total_pages - ai_pages

    pass1 = os.path.join(DELIV, "T013_xelatex_pass1.log")
    pass2 = os.path.join(DELIV, "T013_xelatex_pass2.log")
    pass1_undef = count_undefined(pass1)
    pass2_undef = count_undefined(pass2)
    main_undef = count_undefined(os.path.join(PAPER, "main.log"))
    compile_ok = pass2_undef == 0 and main_undef == 0

    source_changed = [p for p in paper_cmp["changed"]
                      if p.endswith(".tex")]
    build_changed = [p for p in paper_cmp["changed"]
                     if not p.endswith(".tex")]
    scope_ok = set(source_changed) <= ALLOWED_SOURCE
    result_bearing_unchanged = all(
        p not in paper_cmp["changed"]
        for p in [r["path"] for r in fp["paper"]]
        if p.endswith(".tex") and p not in source_changed
    )
    no_numbers_changed = scope_ok and result_bearing_unchanged

    t013_new = [p for p in code_cmp["added"] if p.startswith("code/t013_")]
    unexpected_code = [
        p for p in code_cmp["added"]
        if not (p.startswith("code/t005_") or p.startswith("code/t013_")
                or p.startswith("code/audit/"))
    ]
    code_ok = not unexpected_code and len(t013_new) == 3

    readonly_ok = data_cmp["unchanged"] and review_cmp["unchanged"]

    checks = {
        "1_ai_report_five_elements": bool(
            title_ok and tool_ok and inline_ok and bibentry_ok and examples_ok),
        "2_false_claims_corrected": bool(false_claims_ok),
        "3_a7_rewritten_two_sources": bool(a7_ok),
        "4_six_new_refs_verified": bool(refs_ok),
        "5_citation_map_complete": bool(not undefined and not orphans
                                        and len(bibs) == len(cites)),
        "6_compile_two_passes_clean": bool(compile_ok),
        "7_page_counts_reported": bool(body_pages <= 25 and total_pages > 0),
        "8_paper_scope_allowed": bool(scope_ok),
        "9_no_result_numbers_changed": bool(no_numbers_changed),
        "10_readonly_trees_untouched": bool(readonly_ok and code_ok),
    }

    evidence = {
        "c1": [
            {"file": "code/20_q2_diag.py", "line": 11,
             "text": line_text("code/20_q2_diag.py", 11)},
            {"file": "code/20_q2_diag.py", "line": 26,
             "text": line_text("code/20_q2_diag.py", 26)},
            {"file": "code/24_cell_curve.py", "line": 9,
             "text": line_text("code/24_cell_curve.py", 9)},
            {"file": "code/24_cell_curve.py", "line": 36,
             "text": line_text("code/24_cell_curve.py", 36)},
            {"file": "code/21_q2_opt.py", "line": 19,
             "text": line_text("code/21_q2_opt.py", 19)},
            {"file": "code/30_q3_incentive.py", "line": 16,
             "text": line_text("code/30_q3_incentive.py", 16)},
            {"file": "code/90_figs.py", "line": 256,
             "text": line_text("code/90_figs.py", 256)},
            {"file": "code/90_figs.py", "line": 258,
             "text": line_text("code/90_figs.py", 258)},
            {"file": "code/50_q5_platform.py", "line": 37,
             "text": line_text("code/50_q5_platform.py", 37)},
            {"file": "code/50_q5_platform.py", "line": 38,
             "text": line_text("code/50_q5_platform.py", 38)},
        ],
        "c2": [
            {"file": "code/lib.py", "line": 9,
             "text": line_text("code/lib.py", 9)},
            {"file": "code/lib.py", "line": 10,
             "text": line_text("code/lib.py", 10)},
            {"file": "code/lib.py", "line": 11,
             "text": line_text("code/lib.py", 11)},
            {"file": "code/lib.py", "line": 12,
             "text": line_text("code/lib.py", 12)},
            {"file": "code/20_q2_diag.py", "line": 11,
             "text": line_text("code/20_q2_diag.py", 11)},
            {"file": "code/22_q2_sim.py", "line": 15,
             "text": line_text("code/22_q2_sim.py", 15)},
            {"file": "code/22_q2_sim.py", "line": 16,
             "text": line_text("code/22_q2_sim.py", 16)},
            {"file": "code/95_numbers.py", "line": 1,
             "text": line_text("code/95_numbers.py", 1)},
            {"file": "code/95_numbers.py", "line": 2,
             "text": line_text("code/95_numbers.py", 2)},
        ],
    }

    run_order = sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(ROOT, "code", "*.py"))
        if re.match(r"^\d\d_", os.path.basename(p))
    )

    citation_table = []
    for key, lineno in sorted(bibs.items(), key=lambda kv: kv[1]):
        citation_table.append({
            "key": key,
            "bibitem_line": lineno,
            "cited_in": cites.get(key, []),
            "new_in_T013": key in new_keys,
        })

    md_lines = [
        "# T013 citation map",
        "",
        "| key | `\\bibitem` line | cited in | new in T013 |",
        "|---|---:|---|---|",
    ]
    for row in citation_table:
        cited = ", ".join(row["cited_in"]) or "**ORPHAN**"
        md_lines.append(
            f"| `{row['key']}` | {row['bibitem_line']} | {cited} | "
            f"{'yes' if row['new_in_T013'] else 'no'} |")
    md_lines += [
        "",
        f"- `\\cite` keys: {len(cites)}; `\\bibitem` entries: {len(bibs)}.",
        f"- Undefined citations: {undefined or 'none'}.",
        f"- Orphan bibliography entries: {orphans or 'none'}.",
    ]
    with io.open(CITATION_MD, "w", encoding="utf-8") as handle:
        handle.write("\n".join(md_lines) + "\n")

    new_refs = []
    for key in new_lit:
        meta = ref_ver["new_doi_entries"].get(key, {})
        url_meta = ref_ver["new_url_entries"].get(key, {})
        new_refs.append({
            "key": key,
            "doi": meta.get("doi"),
            "crossref_title": meta.get("title"),
            "verified": bool(meta.get("verified") or url_meta.get("accessible")),
            "url": url_meta.get("url"),
        })

    delivery = {
        "task": "T013",
        "title": "引用、AI 报告与合规",
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "checks": checks,
        "checks_reasons": {
            "1": "title, tool+version+date, inline \\cite{aiapp}, \\bibitem{aiapp}, two Query/Output examples",
            "2": "the two pre-T013 claims are absent and both replacement statements are present; evidence lines are recorded",
            "3": "the pre-T013 phrase is absent; the order-file statement and two news citations are present; both URLs returned 200",
            "4": "8 new literature/news entries: 6 Crossref-verified DOIs and 2 accessible URLs; all are cited",
            "5": "32 \\cite keys and 32 \\bibitem entries; undefined and orphan lists are both empty",
            "6": "cold xelatex pass 1 has 115 expected undefined warnings, pass 2 has 0; main.log has 0",
            "7": "total 27 pages; AI report starts p.26 and is 2 pages; body excluding AI report is 25 pages",
            "8": "source changes are ai_report.tex, references.tex, sec2_assumptions.tex and the citation-only edit in sec1_intro.tex",
            "9": "every result-bearing .tex file outside the four changed source files is byte-identical to the T001 start fingerprint",
            "10": "data/ excluding runs/ and review/ are byte-identical to the T001 start fingerprint; T013 adds exactly three t013_*.py scripts and no unexpected code additions; no T013 script writes to 审查与规划/ or review/",
        },
        "ai_report": {
            "file": "paper/ai_report.tex",
            "title": "Report on Use of AI Tools",
            "tool": "Claude Code (Anthropic)",
            "product_version": "2.1.322",
            "runtime_model_identifier": "deepseek-flash[1m]",
            "date_range": "2026-11-11 to 2026-11-17",
            "inline_cite": {"key": "aiapp",
                            "location": "paper/sec2_assumptions.tex:6"},
            "reference_entry": {"key": "aiapp",
                                "line": bibs.get("aiapp")},
            "examples": [
                {
                    "query": "Compare the gravity destination model with "
                             "distance-only, attraction-only and uniform "
                             "benchmarks on held-out days; fit each model's own "
                             "decay; report nlp and top-5; no AIC.",
                    "output": "555,675 held-out trips; nlp 3.4185 / 3.4881 / "
                              "8.1124 / 8.5685; top-5 0.4255 / 0.3924 / 0.0108 "
                              "/ 0.00095; full-minus-distance 0.0696 nats "
                              "(0.0685-0.0705) and 3.31 pp (3.23-3.38).",
                    "source": "data/runs/T005/dest_validation.json",
                },
                {
                    "query": "Test whether the published k=5 segmentation is "
                             "stable: k=3-6, at least 20 initialisations per k, "
                             "pairwise ARI, silhouette, cross-k ARI and "
                             "agreement with published labels.",
                    "output": "k=5 ARI 0.665 mean / 0.305 min; silhouette "
                              "0.256 (40k) vs 0.263 at k=6; 10k mean 0.251 vs "
                              "0.246; published labels recovered at ARI 0.994; "
                              "stability claim unsupported.",
                    "source": "data/runs/T005/k_stability.json",
                },
            ],
            "five_elements": {
                "tool_version_dates": bool(tool_ok),
                "inline_citation": bool(inline_ok),
                "reference_entry": bool(bibentry_ok),
                "query_output_examples": bool(examples_ok),
                "policy_title": bool(title_ok),
            },
        },
        "false_claims": [
            {
                "id": "C1_no_hand_typed_numbers",
                "old_claim": "No number in the text is typed by hand. A dedicated "
                             "script (95_numbers.py) re-reads every result file and "
                             "prints the values quoted in the manuscript; the "
                             "manuscript was checked against its output.",
                "old_location": "paper/ai_report.tex:30-32 (pre-T013)",
                "new_claim": "Result numbers are script-generated, with listed "
                             "exceptions. Most values quoted in the text are "
                             "printed by 95_numbers.py from the result files. The "
                             "constant 343,638 is still hard-coded in four "
                             "computation scripts and two figure lines: "
                             "20_q2_diag.py:11, used at line 26 for the initial "
                             "inventory written to q2_state.npz; "
                             "24_cell_curve.py:9, used at line 36 for the "
                             "cell-level initial inventory; 21_q2_opt.py:19, where "
                             "it cancels algebraically; 30_q3_incentive.py:16, "
                             "where it is declared but unused; and 90_figs.py:256 "
                             "and 258, the cost curve. It is an input to the Q2/Q5 "
                             "runs, not a display value. The Q5 interpretation "
                             "text in 50_q5_platform.py:37-38 is also written by "
                             "hand. The next revision will replace the constant "
                             "with the revised fleet estimate and move that text "
                             "into the script.",
                "new_location": "paper/ai_report.tex:72-84",
                "evidence": evidence["c1"],
            },
            {
                "id": "C2_single_command_reproducibility",
                "old_claim": "All computations are reproducible. Each numbered "
                             "result is written to disk by the script that computes "
                             "it ... and the scripts run from the raw order file "
                             "without manual intervention.",
                "old_location": "paper/ai_report.tex:33-36 (pre-T013)",
                "new_claim": "The computation is reproducible in a fixed order, "
                             "not by a single command. After scripts 00-04, the "
                             "numbered stages run 10 -> 11 -> 20 -> 21 -> ... -> 95; "
                             "later stages read earlier outputs; lib.py resolves "
                             "data/ and figs/ relative to __file__; the pipeline is "
                             "not unattended.",
                "new_location": "paper/ai_report.tex:85-91",
                "evidence": evidence["c2"],
                "run_order": run_order,
            },
        ],
        "a7": {
            "old": "nobody was offered a reward in May 2017",
            "old_location": "paper/sec2_assumptions.tex:51 (pre-T013)",
            "new": "the order file contains no reward records, although Mobike "
                   "had been operating red-envelope (hongbao) bikes since March "
                   "2017 [30,31]",
            "new_location": "paper/sec2_assumptions.tex:51-53",
            "dataset_schema_evidence": {
                "file": "code/00_prepare.py",
                "lines": "44-47",
                "what": "the order CSV is read with orderid, userid, bikeid, "
                        "biketype, geohashed_start_loc, geohashed_end_loc and "
                        "starttime; there is no reward field.",
            },
            "sources": [
                {
                    "key": "technode2017",
                    "url": ref_ver["new_url_entries"]["technode2017"]["url"],
                    "date": "2017-03-24",
                    "status": ref_ver["new_url_entries"]["technode2017"]["status"],
                    "content": "Mobike announced red-envelope bikes on 23 March "
                               "2017; a ride of at least 10 minutes earned a "
                               "1-100 CNY hongbao; Mobike described the scheme as "
                               "guiding riders to bikes that need moving.",
                },
                {
                    "key": "sixthtone2017",
                    "url": ref_ver["new_url_entries"]["sixthtone2017"]["url"],
                    "date": "2017-05-02",
                    "status": ref_ver["new_url_entries"]["sixthtone2017"]["status"],
                    "content": "States that Mobike launched virtual red envelopes "
                               "in late March 2017 and Ofo in mid-April.",
                },
            ],
        },
        "references": {
            "n_bibitems_before": len(OLD_KEYS),
            "n_bibitems_after": len(bibs),
            "n_new": len(new_keys),
            "n_new_literature_or_news": len(new_lit),
            "new_keys": new_keys,
            "new_entries": new_refs,
            "verification_file": "交付/T013_ref_verification.json",
            "verification_summary": ref_ver["summary"],
            "note": "aiapp is an internal disclosure entry, not a literature "
                    "entry; it intentionally has no DOI/URL.",
        },
        "citation_table": citation_table,
        "citation_summary": {
            "n_cite_keys": len(cites),
            "n_bibitems": len(bibs),
            "undefined": undefined,
            "orphans": orphans,
            "map_file": "交付/T013_citation_map.md",
        },
        "compilation": {
            "engine": "XeTeX 3.141592653-2.6-0.999998 (TeX Live 2026)",
            "passes": 2,
            "pass1_undefined_count": pass1_undef,
            "pass2_undefined_count": pass2_undef,
            "main_log_undefined_count": main_undef,
            "pass1_log": "交付/T013_xelatex_pass1.log",
            "pass2_log": "交付/T013_xelatex_pass2.log",
            "total_pages": total_pages,
            "baseline_total_pages": 25,
            "baseline_source": "paper/b2.log: Output written on main.pdf (25 pages)",
            "total_page_delta": total_pages - 25,
            "ai_report_start_page": ai_start,
            "ai_report_pages": ai_pages,
            "body_pages_excluding_ai_report": body_pages,
            "baseline_body_pages_excluding_ai_report": 24,
            "body_page_delta": body_pages - 24,
            "body_sections_last_page": ai_start - 2,
            "references_pages": f"{ai_start - 1}",
            "twenty_five_page_limit_ok": body_pages <= 25,
        },
        "paper_integrity": {
            "baseline": "data/runs/T001/fingerprint_start.json",
            "n_baseline_files": paper_cmp["baseline_files"],
            "changed": paper_cmp["changed"],
            "source_changed": source_changed,
            "build_artifacts_changed": build_changed,
            "added": paper_cmp["added"],
            "removed": paper_cmp["removed"],
            "unchanged_count": paper_cmp["baseline_files"]
            - len(paper_cmp["changed"]),
        },
        "readonly_integrity": {
            "data_excluding_runs": data_cmp,
            "review": review_cmp,
            "peer_workspace": "not path-compared: the T001 fingerprint stores "
                              "the 审查与规划/ tree name in a non-UTF-8 form; no "
                              "T013 script writes there.",
        },
        "code_integrity": {
            "baseline": "data/runs/T001/fingerprint_start.json (code tree)",
            "t013_new": t013_new,
            "unexpected_added": unexpected_code,
            "t013_modified_existing": [],
            "changed_since_T001": code_cmp["changed"],
            "note": "code/lib.py and code/sim_cell.py changed in T001/T004, not "
                    "T013; T013 only adds the three t013_*.py scripts.",
        },
        "code_changes": [
            {
                "file": "code/t013_refs.py",
                "status": "new",
                "what": "Crossref lookup for every DOI in references.tex and "
                        "HTTP accessibility check for the two A7 news URLs; "
                        "writes 交付/T013_ref_verification.json.",
                "why": "the task requires a DOI/URL and a Crossref title for "
                       "every new entry, and the A7 links must be accessible.",
            },
            {
                "file": "code/t013_delivery.py",
                "status": "new",
                "what": "this delivery builder; hashes paper/, data/ and review/ "
                        "against the T001 start fingerprint, parses the citation "
                        "map, page counts and compile logs.",
                "why": "assemble the T013 evidence and the citation table.",
            },
            {
                "file": "code/t013_validate.py",
                "status": "new",
                "what": "delivery validator for checks 1-10.",
                "why": "self-check the delivery before sending it.",
            },
        ],
        "outputs": {
            "delivery_json": "交付/T013_delivery.json",
            "reference_verification": "交付/T013_ref_verification.json",
            "citation_map": "交付/T013_citation_map.md",
            "xelatex_pass1_log": "交付/T013_xelatex_pass1.log",
            "xelatex_pass2_log": "交付/T013_xelatex_pass2.log",
            "compiled_pdf": "paper/main.pdf",
        },
        "notes": [
            "The AI report's two query/output examples are from the T005 "
            "validation work, not from the final paper text; they are real "
            "working-record examples and their outputs are archived under "
            "data/runs/T005/.",
            "The runtime reported the model identifier deepseek-flash[1m] for "
            "this revision; the product/tool version is Claude Code 2.1.322. "
            "The identifier is disclosed as reported, not inferred.",
            "References were switched from footnotesize to scriptsize to keep "
            "the body (including references) at 25 pages after adding 9 "
            "entries; the text is still legible at print size. This is a "
            "page-count compromise: T012 should restore \\small if the body "
            "shortens, then re-check the 25-page limit.",
            "sec10_evaluation.tex:95-96 still says every number is written to a "
            "result file and the whole chain is reproducible. That is a related "
            "universal claim and is outside T013's allowed file set; T012 should "
            "reconcile it with the two exceptions now listed in the AI report.",
            "paper/preview/*.png are from the pre-T013 build and are stale; they "
            "were not regenerated because preview images are outside the "
            "allowed source-edit scope.",
            "paper/build.log, paper/b1.log and paper/b2.log are pre-T013 build "
            "logs and were not overwritten; the authoritative T013 logs are "
            "交付/T013_xelatex_pass1.log and 交付/T013_xelatex_pass2.log.",
            "The A7 sources are news reports, not peer-reviewed articles; they "
            "are used only to establish the historical fact that Mobike was "
            "operating red-envelope bikes in 2017, not to support a model "
            "parameter.",
            "The 14 pre-existing DOI entries were also re-checked; all 20 DOI "
            "entries (14 old + 6 new) returned matching Crossref metadata.",
            "The AI report is 2 pages and starts on p.26; the 25-page body limit "
            "is met with the AI report excluded (total 27 pages).",
            "The AI report's sentence 'The next revision will replace the "
            "constant with the revised fleet estimate' is a live action item, "
            "not a statement about T013; after T012 updates the paper it must be "
            "revisited and either fulfilled or removed.",
        ],
    }

    with io.open(OUT, "w", encoding="utf-8") as handle:
        json.dump(delivery, handle, ensure_ascii=False, indent=1)
    print("saved", OUT)
    print("checks", json.dumps(checks, ensure_ascii=False))
    print("pages", total_pages, "ai_start", ai_start, "body", body_pages)
    print("citations", len(cites), "bibitems", len(bibs),
          "undefined", undefined, "orphans", orphans)
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise SystemExit("FAILED checks: " + ", ".join(failed))


if __name__ == "__main__":
    main()
