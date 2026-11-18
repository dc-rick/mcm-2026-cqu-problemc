# T021B numeric-token gate: per-file multiset diff of sec3..sec10 against 7105b07.
import io, re, subprocess, json, sys
from collections import Counter

TOK = re.compile(r"(?<![A-Za-z0-9])\d[\d,]*(?:\.\d+)?")
FILES = ["sec3_data.tex", "sec4_q1.tex", "sec5_q2.tex", "sec6_q3.tex",
         "sec7_q4.tex", "sec8_q5.tex", "sec9_sensitivity.tex", "sec10_evaluation.tex"]
BASE = "7105b07"


def git_show(path):
    return subprocess.run(["git", "show", "%s:%s" % (BASE, path)],
                          capture_output=True).stdout.decode("utf-8")


def main():
    report = {}
    for f in FILES:
        old = git_show("paper/" + f)
        new = io.open("paper/" + f, encoding="utf-8").read()
        co, cn = Counter(TOK.findall(old)), Counter(TOK.findall(new))
        removed = co - cn
        added = cn - co
        report[f] = {"removed": dict(removed), "added": dict(added)}
        print("%-22s removed=%s  added=%s" %
              (f, dict(removed) if removed else "{}", dict(added) if added else "{}"))
    io.open("交付/T021B/token_gate.json", "w", encoding="utf-8", newline="\n").write(
        json.dumps(report, indent=2, ensure_ascii=False))
    print("written 交付/T021B/token_gate.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
