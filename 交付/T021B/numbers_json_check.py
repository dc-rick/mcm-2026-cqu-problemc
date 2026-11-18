# T021B: every result number in the abstract and in §1.5 must be findable in
# data/runs/T021A/numbers.json. Findability = exact token match (modulo % and commas)
# OR rounding of an entry value/token to the printed precision.
# Structural exemptions (calendar years, LaTeX layout lengths) are reported separately.
import io, json, re, sys
from collections import Counter

TOK = re.compile(r"(?<![A-Za-z0-9])\d[\d,]*(?:\.\d+)?")
YEAR = re.compile(r"^(19|20)\d\d$")


def norm(t):
    return t.rstrip("%").replace(",", "")


def main():
    d = json.load(io.open("data/runs/T021A/numbers.json", encoding="utf-8"))
    have = set()
    vals = []
    for e in d["entries"]:
        t = norm(e["token"])
        have.add(t)
        try:
            vals.append(float(t))
        except ValueError:
            pass
    for path in sys.argv[1:]:
        txt = io.open(path, encoding="utf-8").read()
        txt = txt.replace("{,}", ",")          # LaTeX thousands separator
        toks = TOK.findall(txt)
        missing, exempt = [], []
        for t in toks:
            tn = norm(t)
            if tn in have:
                continue
            if YEAR.match(tn):
                exempt.append((t, "calendar year"))
                continue
            try:
                x = float(tn)
                dec = len(tn.split(".")[1]) if "." in tn else 0
                tol = 0.5 * 10 ** -dec
                if any(abs(v - x) <= tol + 1e-12 for v in vals):
                    continue
            except ValueError:
                pass
            missing.append(t)
        mc = Counter(missing)
        print(path, "tokens:", len(toks), "missing:", len(missing),
              "exempt:", len(exempt))
        for t, n in sorted(mc.items(), key=lambda kv: (-kv[1], kv[0])):
            print("   MISSING %-14s x%d" % (t, n))
        ec = Counter(t for t, _ in exempt)
        for t, n in sorted(ec.items()):
            print("   exempt(year) %-6s x%d" % (t, n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
