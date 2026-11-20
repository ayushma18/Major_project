"""Verify every paper in literature/catalog.json against its source, mechanically.

    python scripts/verify_catalog.py            # check, update verified.*, print a report
    python scripts/verify_catalog.py --dry-run  # report only

arXiv papers: the arXiv API record must match the catalog title (normalised similarity
>= 0.85) and the first author's surname; the catalog date is checked against the v1 date.
Other papers: the URL must answer 200 and the page must contain the title words.
Anything that fails is marked `unverified` with the reason; nothing is ever marked
`verified` by hand.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import time
import unicodedata
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "literature" / "catalog.json"
UA = {"User-Agent": "affect-as-attack-catalog-check/1.0 (university research project)"}
# Some publishers (PubMed Central, Frontiers) serve a bot page to unknown agents.
BROWSER_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", s).split())


def sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def surname(name: str) -> str:
    name = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode().strip()
    return name.split()[-1].lower() if name else ""


def arxiv_records(ids: list[str]) -> dict[str, dict]:
    out = {}
    for i in range(0, len(ids), 40):
        chunk = ids[i : i + 40]
        url = "http://export.arxiv.org/api/query?max_results=100&id_list=" + ",".join(chunk)
        for attempt in range(4):
            try:
                xml = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90).read().decode()
                break
            except Exception:  # noqa: BLE001
                time.sleep(5 * (attempt + 1))
        else:
            continue
        for entry in re.findall(r"<entry>(.*?)</entry>", xml, re.S):
            m = re.search(r"<id>https?://arxiv.org/abs/([^<v]+)(v\d+)?</id>", entry)
            if not m:
                continue
            aid = m.group(1)
            title = " ".join(re.search(r"<title>(.*?)</title>", entry, re.S).group(1).split())
            pub = re.search(r"<published>(.*?)</published>", entry).group(1)[:10]
            authors = re.findall(r"<name>(.*?)</name>", entry)
            out[aid] = {"title": title, "published": pub, "authors": authors}
        time.sleep(3)
    return out


def check_url(url: str, title: str) -> tuple[bool, str]:
    try:
        req = urllib.request.Request(url, headers=BROWSER_UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            body = r.read(400_000).decode("utf-8", "ignore")
            code = r.status
    except Exception as exc:  # noqa: BLE001
        return False, f"url error: {exc}"
    if code != 200:
        return False, f"HTTP {code}"
    words = [w for w in re.findall(r"[a-z]{4,}", title.lower())][:8]
    low = body.lower()
    hit = sum(w in low for w in words)
    return (hit >= max(2, int(0.7 * len(words)))), f"HTTP 200, {hit}/{len(words)} title words on page"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cat = json.loads(CATALOG.read_text())
    papers = cat["papers"]
    ids = sorted({p["arxiv"].split("v")[0] for p in papers if p.get("arxiv")})
    recs = arxiv_records(ids)
    today = date.today().isoformat()
    bad = 0
    for p in papers:
        if p.get("arxiv"):
            aid = p["arxiv"].split("v")[0]
            r = recs.get(aid)
            if not r:
                status, how = "unverified", f"arXiv id {aid} not returned by the API"
            else:
                s = sim(p["title"], r["title"])
                fa_ok = bool(p.get("authors")) and surname(p["authors"][0]) == surname(r["authors"][0] if r["authors"] else "")
                if s >= 0.85 and fa_ok:
                    status, how = "verified", f"arXiv API: title match {s:.2f}, first author {r['authors'][0]}, v1 {r['published']}"
                    if p.get("date") and p["date"] != r["published"]:
                        how += f" (catalog date {p['date']} corrected)"
                        p["date"] = r["published"]
                        p["year"] = int(r["published"][:4])
                else:
                    status, how = "unverified", f"arXiv API mismatch: title {s:.2f} ('{r['title'][:80]}'), first author {r['authors'][:1]}"
        elif p.get("url"):
            ok, how = check_url(p["url"], p["title"])
            status = "verified" if ok else "unverified"
            how = "page check: " + how
        else:
            status, how = "unverified", "no arXiv id and no url"
        if status != "verified":
            bad += 1
            print(f"UNVERIFIED {p['id']}: {how}")
        p["verified"] = {"status": status, "how": how, "at": today}
    print(f"{len(papers) - bad}/{len(papers)} verified")
    if not args.dry_run:
        cat["updated"] = today
        CATALOG.write_text(json.dumps(cat, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
