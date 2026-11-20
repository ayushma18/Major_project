# Project interface: build brief

A local web interface for the whole "Affect as Attack" research project. Three students and
their supervisor use it to understand the project, study the papers, follow the field's
timeline, read papers, and watch experiment results come in. It runs on a laptop from
`python scripts/serve.py` at http://127.0.0.1:8765/web/ (repo root is served, read-only).

## Hard technical rules

- Plain HTML, CSS and vanilla JavaScript (ES modules). No build step, no npm, no framework,
  no CDN at runtime: the site must work offline. Vendor anything you need into `web/vendor/`.
- Everything lives in `web/` (index.html, app.js plus modules, styles.css, fonts/, vendor/).
  Do not edit any file outside `web/`.
- Data comes ONLY from these JSON files via `fetch()`, shapes defined in `docs/CONTRACTS.md`:
  `literature/catalog.json`, `results/index.json` + `results/<id>.json`, `project/status.json`.
  The files fill up over the coming days: every view must handle empty arrays, `null` fields,
  missing result files and missing PDFs gracefully (show "not yet" states, never crash).
  Re-fetch on page load; no caching.
- Hash routing (`#/timeline`, `#/paper/<id>`, ...) so links can be shared and the back button works.
- PDFs: `literature/pdfs/<id>.pdf` when `pdf_local` exists (check with a HEAD request), shown in
  an `<iframe>` reader view; otherwise show "PDF not downloaded yet" plus the `url` link.
  The proposal and slides are at `private/proposal.pdf` and `private/proposal-presentation.pdf`
  (local only; hide the page section if they 404).
- Charts: draw them yourself as inline SVG from the results JSON (`bar`, `line`, `scatter`,
  `heatmap` as described in CONTRACTS.md). Axis labels, units, a legend when there is more
  than one series, value labels on hover. No chart library needed.
- Accessible: semantic HTML, keyboard reachable, visible focus, WCAG AA contrast, `prefers-reduced-motion` respected.
- Works from 390 px phones to 1840 px wide screens. Check both, plus 1440.

## Views

1. **Overview** (`#/`): the research question in one big statement, the four-module pipeline
   (1 emotion vectors, 2 safety evaluation, 3 mechanism, 4 multi-turn persistence) as a clear
   diagram you build in HTML/SVG, the current phase and its progress from `project/status.json`,
   the latest findings, blockers, next steps.
2. **Timeline** (`#/timeline`): every `verified` paper on a vertical time axis by `date`,
   grouped by year, with the `eras` from the catalog as labelled bands and their summaries.
   Colour or mark by category, filter by category and priority, toggle "show unverified".
   Each node: title, date, venue, the first key finding. Click opens the paper page. This is
   how the team sees how the field evolved towards our project.
3. **Library** (`#/library`): searchable, filterable list (category, priority, cited in
   proposal, verified status, has PDF). Sort by date, by reading order, by priority.
4. **Paper page** (`#/paper/<id>`): title, authors, date, venue, links (arXiv, code), the
   plain summary, key findings, key numbers, method, why it matters for us (`relevance`),
   limitations, and a "Read the paper" reader (the iframe) on the same page or a full-width
   reader route `#/read/<id>`.
5. **Reading path** (`#/reading`): papers ordered by `reading_order`, with a checkbox per paper
   stored in `localStorage` so a student can tick off what they have read.
6. **Reference audit** (`#/audit`): the `proposal_reference_audit` table: each proposal
   reference, its status (colour-coded: real-correct, real-wrong-metadata, wrong-paper,
   not-found), the note, and a link to the matching paper.
7. **Experiments** (`#/experiments` and `#/experiment/<id>`): cards from `results/index.json`
   grouped by module, status badge, headline. The detail page renders metrics (value plus CI),
   tables, charts, interpretation and caveats from the result file.
8. **Methods** (`#/methods`): a static explainer you write in plain words of how the project
   works: residual stream, emotion vectors by CAA (x' = x + alpha * v), the refusal direction
   (difference of means), directional ablation (x' = x - (x . r_hat) r_hat), cosine geometry,
   ASR, the sleeper protocol. Use the proposal text (`private/proposal.txt`, if present, else
   `docs/CONTRACTS.md` and the repo README) as the source. Formulas in plain HTML with
   Unicode and `<sub>`/`<sup>`, no math library.
9. **Proposal** (`#/proposal`): the two private PDFs in the reader, if present.

## Design rules (the owner has rejected these before, so they are hard rules)

- **Fonts**: Schibsted Grotesk for everything (self-host the woff2 files from Google Fonts in
  `web/fonts/`, SIL Open Font License, include the licence file). A mono face only for code
  and numbers if needed (JetBrains Mono or IBM Plex Mono, self-hosted). **Never** Fraunces,
  Inter, Geist, Space Grotesk, Plus Jakarta Sans, system-ui as the main face, or any italic
  serif accent words.
- **Colour**: neutral grey base `#F2F2F0`, near-black ink `#0E0E0E`, white and `#E9E9E6`
  cards, ONE accent used sparingly (pick a deep red such as `#B3261E` for "attack/guardrail"
  meaning). Category colours for the timeline may use a small muted palette. Offer a dark mode
  via `prefers-color-scheme` that is a real dark system, not inverted. No purple to blue
  gradients, no glow shadows, no gradient text.
- **Type scale**: one harmonic scale in CSS custom properties (ratio about 1.25), decisive
  jumps between tiers. The big statement on Overview is the largest text on every screen.
- **No** "01 —" or "— 01" eyebrows above sections, no decorative numbering, no emoji icons,
  no stock illustration, no fake product shots, no lorem ipsum, no placeholder copy that
  pretends to be data. Numbers only inside real sequences (reading order, pipeline modules).
- Plain uppercase small labels are fine. Rounded cards, black pill buttons. Generous
  whitespace, one left edge per section.
- Copy: direct, plain words, no filler, no marketing tone. No em or en dashes anywhere.

## Done means

- `python scripts/serve.py` then every route renders with the current (sparse) data and with
  a fuller test fixture. Make a fixture set in `web/fixtures/` (a catalog with 6 papers across
  3 eras, a status file, two result files with one chart of each type) and a `?fixtures=1`
  URL switch that loads them instead of the real files, so the design can be judged before
  real data lands. Fixture papers must be clearly marked as fixtures in the UI.
- You opened it in a real browser at 390, 1440 and 1840 px wide, in light and dark mode, and
  fixed what looked wrong. No console errors.
- Report back the list of files, the screenshots you took (save them under
  `/Users/anuj/Agents/.tmp/2026-11-20_affect-web/`), and anything you could not do.
