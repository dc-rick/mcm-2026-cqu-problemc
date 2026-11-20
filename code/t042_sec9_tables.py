"""T042: emit the sec9 LaTeX tables from the T032 result files.

Reads data/runs/T032/intervals.json and variance_shares.json and prints
(a) the P5/P50/P95 interval table rows and (b) the variance-share rows at
the printed precision.  Read-only; nothing is written to the paper.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib

RUN = os.path.join(lib.DATA, "runs", "T032")

ORDER = [
    ("q2_total", "Q2 total", "CNY/day", 0),
    ("q2_unmet", "Q2 unmet", "bikes/day", 0),
    ("q2_saving_share", "Q2 saving share", "share", 4),
    ("q2_kstar", "Q2 truck $K^*$", "trucks", 0),
    ("q3_all_truck", "Q3 all-truck pool", "CNY/day", 0),
    ("q3_total", "Q3 optimum total", "CNY/day", 0),
    ("q3_payment", "Q3 payment $rP$", "CNY/day", 0),
    ("q3_saving", "Q3 saving share", "share", 4),
    ("q4_coverage", "Q4 occupancy coverage", "share", 4),
    ("q4_arrival", "Q4 arrival coverage", "share", 4),
    ("q4_unassigned", "Q4 unassigned", "bike-slots", 0),
]


def fmt(v, d):
    if d == 0:
        return "{:,.0f}".format(v)
    return ("{:.%df}" % d).format(v)


def main():
    iv = json.load(open(os.path.join(RUN, "intervals.json"),
                        encoding="utf-8"))["outputs"]
    print("% ---- interval rows")
    for key, label, unit, d in ORDER:
        e = iv[key]
        ref = fmt(e["reference"], d)
        sl = e["slices"]
        f = sl.get("F")
        cells = []
        for g in ("A", "B", "C"):
            s = sl.get(g)
            cells.append("%s--%s" % (fmt(s["p5"], d), fmt(s["p95"], d))
                         if s else "--")
        if f:
            joint = "%s / %s / %s" % (fmt(f["p5"], d), fmt(f["p50"], d),
                                      fmt(f["p95"], d))
        else:
            dist = e["slices"]["F"].get("distribution", {})
            joint = " / ".join("%s: %d" % (k, v) for k, v in dist.items())
        print("%s & %s & %s & %s & %s & %s \\\\" %
              (label, ref, joint, cells[0], cells[1], cells[2]))
        print("%%     unit=%s n_F=%s n_A=%s n_B=%s n_C=%s" %
              (unit, f and f["n"], sl.get("A") and sl["A"]["n"],
               sl.get("B") and sl["B"]["n"], sl.get("C") and sl["C"]["n"]))
    sh = json.load(open(os.path.join(RUN, "variance_shares.json"),
                        encoding="utf-8"))["outputs"]
    print("% ---- variance rows")
    for key, label, unit, d in ORDER:
        if key == "q2_kstar" or key not in sh:
            continue
        e = sh[key]
        s = e["share_first_order"]
        print("%s & %.3f & %.3f & %.3f & %.3f \\\\  %% residual %.3f" %
              (label, s["A"], s["B"], s["C"],
               s["A"] + s["B"] + s["C"], e["residual_1_minus_sum"]))


if __name__ == "__main__":
    main()
