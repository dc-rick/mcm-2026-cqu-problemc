# T021B: locate every removed numeric-token occurrence in the baseline text with context.
import io, re, subprocess, json, sys
from collections import Counter

TOK = re.compile(r"(?<![A-Za-z0-9])\d[\d,]*(?:\.\d+)?")
FILES = ["sec4_q1.tex", "sec6_q3.tex", "sec8_q5.tex", "sec10_evaluation.tex"]
BASE = "7105b07"


def git_show(path):
    return subprocess.run(["git", "show", "%s:%s" % (BASE, path)],
                          capture_output=True).stdout.decode("utf-8")


def main():
    out = []
    for f in FILES:
        old = git_show("paper/" + f)
        new = io.open("paper/" + f, encoding="utf-8").read()
        removed = Counter(TOK.findall(old)) - Counter(TOK.findall(new))
        # group old occurrences by token
        occ = {}
        for m in TOK.finditer(old):
            occ.setdefault(m.group(), []).append(m.start())
        print("=" * 78)
        print(f)
        for tok, n in sorted(removed.items()):
            starts = occ[tok]
            print("-- %-12s removed=%d  old_total=%d" % (tok, n, len(starts)))
            for s in starts:
                lo, hi = max(0, s - 70), min(len(old), s + 70)
                ctx = old[lo:hi].replace("\n", " ")
                print("     ...%s..." % ctx)
                out.append({"file": f, "token": tok, "old_context": ctx})
    io.open("交付/T021B/removed_tokens.json", "w", encoding="utf-8",
            newline="\n").write(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
