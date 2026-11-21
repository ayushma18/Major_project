"""Render paper/paper.md to paper/paper.pdf with headless Chrome (no LaTeX needed).

    .venv/bin/python paper/make_figures.py   # figures first
    .venv/bin/python paper/build_pdf.py

Needs the `markdown` package and Google Chrome. The page is A4, STIX Two Text (a journal face that ships
with macOS), figures from paper/figures/. The HTML goes to a temp folder; only the PDF is kept.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: A4; margin: 20mm 20mm 22mm 20mm;
  @bottom-center { content: counter(page); font: 9pt 'STIX Two Text', serif; color: #444; } }
html { font-family: 'STIX Two Text', 'Times New Roman', serif; font-size: 10.5pt; line-height: 1.38; color: #111; }
body { margin: 0; }
p, li { text-align: justify; hyphens: auto; }
h1 { font-size: 17pt; line-height: 1.2; text-align: center; margin: 0 0 6pt; font-weight: 700; }
h1 + p { text-align: center; font-size: 11pt; margin: 0 0 10pt; }
.authors { text-align: center; margin: 0 0 2pt; }
.affil { text-align: center; font-size: 9.5pt; color: #333; margin: 0 0 4pt; }
.draft { text-align: center; font-size: 8.5pt; color: #555; margin: 0 0 14pt; }
h2 { font-size: 12pt; margin: 14pt 0 4pt; break-after: avoid; }
h3 { font-size: 10.5pt; margin: 10pt 0 3pt; break-after: avoid; }
.abstract { margin: 0 8mm 6pt; font-size: 9.8pt; }
.abstract h2 { text-align: center; font-size: 10.5pt; margin-top: 0; }
code { font-family: 'IBM Plex Mono', Menlo, monospace; font-size: 8.6pt; }
img { display: block; max-width: 100%; margin: 6pt auto 2pt; break-inside: avoid; }
p.caption { font-size: 9pt; margin-top: 2pt; break-before: avoid; }
table { border-collapse: collapse; width: 100%; font-size: 8.4pt; line-height: 1.25; margin: 4pt 0 10pt;
  border-top: 1.2pt solid #111; border-bottom: 1.2pt solid #111; break-inside: avoid; }
thead th { border-bottom: 0.7pt solid #111; text-align: left; font-weight: 700; padding: 3pt 4pt; vertical-align: bottom; }
td { padding: 2.2pt 4pt; vertical-align: top; text-align: left; }
.tbl { break-inside: avoid; }
.refs li { font-size: 9pt; text-align: left; margin-bottom: 2pt; }
ul, ol { padding-left: 16pt; }
"""


def build(md_path: Path = HERE / "paper.md", out: Path = HERE / "paper.pdf") -> Path:
    text = md_path.read_text()
    html = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists", "attr_list"])
    # figures: absolute file paths so the temp HTML can find them
    html = re.sub(r'src="(figures/[^"]+)"', lambda m: f'src="file://{HERE / m.group(1)}"', html)
    # captions are the paragraphs that open with a bold "Figure n." or "Table n."
    html = re.sub(r"<p><strong>((?:Figure|Table) \d+\.)</strong>", r'<p class="caption"><strong>\1</strong>', html)
    # keep each table with its caption (captions sit above tables)
    html = re.sub(r'(<p class="caption"><strong>Table \d+\.</strong>.*?</p>\s*<table>.*?</table>)', r'<div class="tbl">\1</div>', html, flags=re.S)
    # title block: the three plain paragraphs after the subtitle are authors, affiliation, draft note
    parts = html.split("<h2>Abstract</h2>", 1)
    if len(parts) == 2:
        head, rest = parts
        paras = re.findall(r"<p>(.*?)</p>", head, flags=re.S)
        if len(paras) >= 3:
            authors_aff = paras[1].split("\n", 1)
            block = f'<p class="authors">{authors_aff[0]}</p>'
            if len(authors_aff) > 1:
                block += f'<p class="affil">{authors_aff[1]}</p>'
            block += f'<p class="draft">{paras[2]}</p>'
            head = head.replace(f"<p>{paras[1]}</p>", block).replace(f"<p>{paras[2]}</p>", "")
        abstract, _, after = rest.partition("<h2>")
        html = f'{head}<div class="abstract"><h2>Abstract</h2>{abstract}</div><h2>{after}'
    html = re.sub(r"(<h2>References</h2>\s*)<ul>", r'\1<ul class="refs">', html)
    page = f'<!doctype html><html lang="en"><meta charset="utf-8"><title>Affect as Attack?</title><style>{CSS}</style><body>{html}</body></html>'
    tmp = Path(os.environ.get("AFFECT_TMP", "/Users/anuj/Agents/.tmp"))
    with tempfile.TemporaryDirectory(dir=tmp if tmp.exists() else None) as d:
        src = Path(d) / "paper.html"
        src.write_text(page)
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--user-data-dir={d}/chrome", "--allow-file-access-from-files",
                        f"--print-to-pdf={out}", src.as_uri()], check=True, capture_output=True, timeout=180)
    print("wrote", out, f"{out.stat().st_size // 1024} KB")
    return out


if __name__ == "__main__":
    build()
