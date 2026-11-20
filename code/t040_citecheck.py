"""T040: citation/bibitem consistency check (read-only)."""
import glob
import re

BS = chr(92)
cite = re.compile(BS + BS + r"cite\{([^}]*)\}")
bibitem = re.compile(BS + BS + r"bibitem\{([^}]*)\}")

keys = set()
for p in glob.glob("paper/sec*.tex") + ["paper/summary.tex",
                                        "paper/ai_report.tex"]:
    txt = open(p, encoding="utf-8").read()
    for m in cite.finditer(txt):
        for k in m.group(1).split(","):
            keys.add(k.strip())
ref = open("paper/references.tex", encoding="utf-8").read()
bib = {m.group(1).strip() for m in bibitem.finditer(ref)}
print("cited", len(keys), "bibitems", len(bib))
print("missing", sorted(keys - bib))
print("uncited", sorted(bib - keys))
