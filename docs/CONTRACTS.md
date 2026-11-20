# Data contracts

Every agent and script in this project reads and writes these files in these shapes. The web
interface (`web/`) only reads them. Change a shape here first, then in the code.

All dates are ISO 8601 (`2024-06-17`). All files are UTF-8 JSON. No em or en dashes in any text.

## 1. `literature/catalog.json`, the paper library

```json
{
  "version": 1,
  "updated": "2026-11-20",
  "categories": [
    {"id": "refusal-geometry", "label": "Refusal geometry", "blurb": "How refusal is stored inside the model"}
  ],
  "eras": [
    {"id": "foundations", "label": "Foundations", "start": "2017-01-01", "end": "2022-12-31", "summary": "plain words"}
  ],
  "papers": [
    {
      "id": "arditi2024refusal",
      "title": "Refusal in Language Models Is Mediated by a Single Direction",
      "authors": ["Andy Arditi", "Oscar Obeso", "..."],
      "date": "2024-06-17",
      "year": 2024,
      "venue": "NeurIPS 2024",
      "arxiv": "2406.11717",
      "doi": null,
      "url": "https://arxiv.org/abs/2406.11717",
      "pdf_url": "https://arxiv.org/pdf/2406.11717",
      "pdf_local": "literature/pdfs/arditi2024refusal.pdf",
      "code_url": "https://github.com/andyrdt/refusal_direction",
      "categories": ["refusal-geometry"],
      "priority": "core",
      "reading_order": 3,
      "cited_in_proposal": "[1]",
      "verified": {"status": "verified", "how": "arXiv API: title and first author match", "at": "2026-11-20"},
      "summary_simple": "2 to 4 plain sentences a second-year student understands.",
      "key_findings": ["one plain sentence per finding"],
      "key_numbers": ["13 open chat models tested", "..."],
      "method": "one or two plain sentences",
      "relevance": "how this paper shapes OUR project, concretely",
      "limitations": "one or two sentences",
      "tags": ["llama-3", "directional-ablation"]
    }
  ],
  "proposal_reference_audit": [
    {
      "ref": "[3]",
      "as_cited": "M. Jeong, S. Kim, J. Park, Shared geometry of emotion representations..., arXiv:2603.XXXXX",
      "status": "real-correct | real-wrong-metadata | wrong-paper | not-found",
      "match_id": "paper id in this catalog, or null",
      "note": "what is wrong and what to cite instead"
    }
  ]
}
```

- `priority`: `core` (read first, the project rests on it), `important`, `background`.
- `verified.status`: `verified` (title and first author checked against arXiv, Semantic Scholar,
  OpenReview, ACL Anthology or the publisher page), `unverified`, `not-found`.
  Only `verified` papers appear in the timeline by default.
- `pdf_local` is relative to the repo root. PDFs are not committed (copyright); run
  `scripts/fetch_papers.py` to download them.
- Fields a writer has not filled yet are `null` or `[]`, never invented.

## 2. `results/index.json` and `results/<experiment-id>.json`, experiment results

`results/index.json`:

```json
{"updated": "2026-11-20T12:00:00", "experiments": [
  {"id": "E01-baseline-refusal", "title": "Baseline refusal on HarmBench", "module": 2,
   "status": "planned | running | done | failed", "file": "results/E01-baseline-refusal.json",
   "headline": "one plain sentence with the main number"}
]}
```

`results/<id>.json`:

```json
{
  "id": "E01-baseline-refusal",
  "title": "...",
  "module": 2,
  "status": "done",
  "question": "the question this experiment answers, plain words",
  "started": "ISO", "finished": "ISO",
  "model": "unsloth/Llama-3.1-8B-Instruct",
  "config": {"n_prompts": 159, "max_new_tokens": 256, "layer": 12, "alpha": 8.0},
  "headline": "plain sentence with the main number",
  "metrics": [{"name": "refusal_rate", "label": "Refusal rate", "value": 0.97, "unit": "fraction", "ci95": [0.93, 0.99]}],
  "tables": [{"id": "per_condition", "title": "...", "columns": ["condition", "asr", "refusal_rate", "n"], "rows": [["baseline", 0.03, 0.97, 159]]}],
  "charts": [
    {"id": "asr_by_condition", "type": "bar | line | scatter | heatmap", "title": "...",
     "x": {"label": "...", "values": ["..."]}, "y": {"label": "...", "unit": "fraction"},
     "series": [{"name": "ASR", "values": [0.03, 0.81]}]}
  ],
  "interpretation": ["plain sentence", "..."],
  "caveats": ["plain sentence", "..."],
  "artifacts": ["path/relative/to/repo"]
}
```

Scatter charts use `"points": [{"x": 0.12, "y": 0.30, "label": "desperate"}]` instead of `series`.
Heatmaps use `"x": {"values": [...]}, "y": {"values": [...]}, "z": [[...], ...]`.

**Never put model completions to harmful prompts in any results file.** Raw generations stay in
`outputs/generations/` (git-ignored, local only). Results carry labels and rates only. Benign
examples (for instance an emotion-steered answer to "describe your morning") may be quoted.

## 3. `project/status.json`, the project dashboard

```json
{
  "updated": "ISO",
  "title": "Affect as Attack",
  "one_line": "plain sentence",
  "team": ["Anuj Paudel", "Ashish Pandey", "Ayushma Pudasaini"],
  "model": {"id": "unsloth/Llama-3.1-8B-Instruct", "why": "plain sentence"},
  "phases": [
    {"id": "P1", "label": "Literature and setup", "planned": ["2026-07-01", "2026-08-31"],
     "status": "done | in_progress | not_started | blocked", "progress": 0.0,
     "items": [{"label": "...", "status": "done", "note": "..."}]}
  ],
  "findings": [{"date": "ISO", "text": "plain sentence", "experiment": "E01-baseline-refusal"}],
  "blockers": [{"text": "...", "owner": "Anuj"}],
  "next": ["plain sentence"]
}
```

## 4. Story corpus

See `data/stories/BRIEF.md`. Raw files: `data/stories/raw/<emotion>.jsonl`.
