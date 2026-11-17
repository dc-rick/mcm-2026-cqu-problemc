"""T013 reference verification: Crossref metadata and URL accessibility.

Reads paper/references.tex, resolves every DOI through the Crossref API, and
checks the two news URLs used for assumption A7.  Writes
交付/T013_ref_verification.json.  No data/ or review/ files are touched.
"""
import datetime
import io
import json
import os
import re
import sys
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCES = os.path.join(ROOT, "paper", "references.tex")
OUT = os.path.join(ROOT, "交付", "T013_ref_verification.json")
UA = "MCM-T013-refcheck/1.0 (mailto:example@example.com)"

NEW_DOIS = {
    "pal2017": (
        "10.1016/j.trc.2017.03.016",
        "Free-floating bike sharing: Solving real-life large-scale static "
        "rebalancing problems",
    ),
    "shen2018": (
        "10.1080/15568318.2018.1429696",
        "Understanding the usage of dockless bike sharing in Singapore",
    ),
    "du2018": (
        "10.3390/su10041244",
        "Better understanding the characteristics and influential factors of "
        "different travel patterns in free-floating bike sharing: Evidence "
        "from Nanjing, China",
    ),
    "faghih2014": (
        "10.1016/j.jtrangeo.2014.01.013",
        "How land-use and urban form impact bicycle flows: evidence from the "
        "bicycle-sharing system (BIXI) in Montreal",
    ),
    "bachand2012": (
        "10.3141/2314-09",
        "Better understanding of factors influencing likelihood of using "
        "shared bicycle systems and frequency of use",
    ),
    "simini2012": (
        "10.1038/nature10856",
        "A universal model for mobility and migration patterns",
    ),
}

NEW_URLS = {
    "technode2017": (
        "https://technode.com/2017/03/24/mobike-jumps-on-hongbao-bandwagon-"
        "with-mobike-mo-money/",
        ["mobike", "hongbao", "red envelope"],
    ),
    "sixthtone2017": (
        "https://www.sixthtone.com/news/1000130",
        ["mobike", "hongbao", "red envelope"],
    ),
}


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=45) as response:
        return json.load(response)


def crossref(doi):
    message = get_json(
        "https://api.crossref.org/works/" + urllib.parse.quote(doi)
    )["message"]
    authors = []
    for author in message.get("author", []):
        family = author.get("family", "")
        given = author.get("given", "")
        authors.append((family + ", " + given).strip(", "))
    year = (message.get("issued", {}).get("date-parts") or [[None]])[0][0]
    return {
        "doi": doi,
        "title": (message.get("title") or [""])[0],
        "authors": authors,
        "year": year,
        "container": (message.get("container-title") or [""])[0],
        "volume": message.get("volume"),
        "issue": message.get("issue"),
        "page": message.get("page"),
        "publisher": message.get("publisher"),
    }


def check_url(url, terms):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/120 Safari/537.36"
            )
        },
    )
    with urllib.request.urlopen(req, timeout=45) as response:
        body = response.read(400000).decode("utf-8", "ignore").lower()
        return {
            "url": url,
            "status": response.status,
            "bytes_sampled": len(body),
            "terms_found": {term: term in body for term in terms},
            "accessible": response.status == 200 and all(
                term in body for term in terms
            ),
        }


def parse_reference_keys():
    """Return key -> (doi or None, url or None) from references.tex."""
    text = io.open(REFERENCES, encoding="utf-8").read()
    parts = re.split(r"\\bibitem\{([^}]+)\}", text)[1:]
    out = {}
    for i in range(0, len(parts), 2):
        key, body = parts[i], parts[i + 1]
        doi_match = re.search(r"\\url\{https://doi\.org/([^}]+)\}", body)
        url_match = re.search(r"\\url\{(https?://[^}]+)\}", body)
        out[key] = {
            "doi": doi_match.group(1) if doi_match else None,
            "url": url_match.group(1) if url_match else None,
        }
    return out


def main():
    refs = parse_reference_keys()
    result = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "method": "Crossref REST API /works/{doi}; direct HTTP GET for URLs",
        "new_doi_entries": {},
        "new_url_entries": {},
        "all_doi_entries": {},
        "summary": {},
    }

    for key, (doi, expected) in NEW_DOIS.items():
        try:
            meta = crossref(doi)
            meta["expected_title"] = expected
            meta["title_match"] = norm(meta["title"]) == norm(expected)
            meta["verified"] = bool(meta["title_match"] and meta["authors"])
        except Exception as exc:  # network failure is reported, not hidden
            meta = {"doi": doi, "expected_title": expected,
                    "verified": False, "error": repr(exc)}
        result["new_doi_entries"][key] = meta
        print(key, "OK" if meta.get("verified") else "FAIL", meta.get("title"))

    for key, (url, terms) in NEW_URLS.items():
        try:
            meta = check_url(url, terms)
        except Exception as exc:
            meta = {"url": url, "accessible": False, "error": repr(exc)}
        result["new_url_entries"][key] = meta
        print(key, "OK" if meta.get("accessible") else "FAIL", url)

    for key, ref in refs.items():
        if not ref["doi"]:
            continue
        try:
            meta = crossref(ref["doi"])
            meta["verified"] = bool(meta["title"] and meta["authors"])
        except Exception as exc:
            meta = {"doi": ref["doi"], "verified": False, "error": repr(exc)}
        result["all_doi_entries"][key] = meta

    new_ok = sum(1 for v in result["new_doi_entries"].values()
                 if v.get("verified"))
    url_ok = sum(1 for v in result["new_url_entries"].values()
                 if v.get("accessible"))
    all_ok = sum(1 for v in result["all_doi_entries"].values()
                 if v.get("verified"))
    result["summary"] = {
        "new_dois_verified": new_ok,
        "new_dois_total": len(NEW_DOIS),
        "new_urls_accessible": url_ok,
        "new_urls_total": len(NEW_URLS),
        "all_dois_verified": all_ok,
        "all_dois_total": len(result["all_doi_entries"]),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with io.open(OUT, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=1)
    print("saved", OUT)
    print(json.dumps(result["summary"], ensure_ascii=False))
    if new_ok != len(NEW_DOIS) or url_ok != len(NEW_URLS):
        raise SystemExit("reference verification failed")


if __name__ == "__main__":
    main()
