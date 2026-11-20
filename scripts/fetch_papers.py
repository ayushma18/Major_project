"""Download the paper PDFs listed in literature/catalog.json into literature/pdfs/.

    python scripts/fetch_papers.py           # only missing files
    python scripts/fetch_papers.py --force   # re-download everything

PDFs are not committed (copyright stays with the authors and publishers), so every
teammate runs this once. Each file is checked to start with %PDF; failures are listed.
Sets `pdf_local` in the catalog for every file that is present.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "literature" / "catalog.json"
PDFS = ROOT / "literature" / "pdfs"
UA = {"User-Agent": "Mozilla/5.0 (affect-as-attack literature fetch; university research)"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cat = json.loads(CATALOG.read_text())
    PDFS.mkdir(parents=True, exist_ok=True)
    failed = []
    for p in cat["papers"]:
        dest = PDFS / f"{p['id']}.pdf"
        url = p.get("pdf_url") or (f"https://arxiv.org/pdf/{p['arxiv']}" if p.get("arxiv") else None)
        if dest.exists() and dest.stat().st_size > 10_000 and dest.read_bytes()[:4] == b"%PDF" and not args.force:
            p["pdf_local"] = f"literature/pdfs/{dest.name}"
            continue
        if not url:
            p["pdf_local"] = None
            continue
        try:
            data = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120).read()
            if data[:4] != b"%PDF":
                raise ValueError("not a PDF")
            dest.write_bytes(data)
            p["pdf_local"] = f"literature/pdfs/{dest.name}"
            print(f"ok   {p['id']} ({len(data) / 1e6:.1f} MB)")
        except Exception as exc:  # noqa: BLE001
            p["pdf_local"] = None
            failed.append((p["id"], str(exc)))
            print(f"FAIL {p['id']}: {exc}")
        time.sleep(1.5 if "arxiv.org" in (url or "") else 0.5)
    CATALOG.write_text(json.dumps(cat, indent=2, ensure_ascii=False))
    print(f"{sum(1 for p in cat['papers'] if p.get('pdf_local'))}/{len(cat['papers'])} PDFs present; {len(failed)} failed")


if __name__ == "__main__":
    main()
