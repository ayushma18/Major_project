#!/bin/bash
# Deploy the project interface to Cloudflare Pages (project "affect-as-attack"), password gated.
#
#   bash scripts/deploy_pages.sh
#
# Builds a clean folder with web/, literature/catalog.json and the paper PDFs, results/, paper/paper.pdf,
# project/status.json and the private proposal files, adds the Basic Auth middleware from
# deploy/functions/, and deploys it with wrangler. Raw outputs/ (model completions) are never
# included. The password lives only in the Pages secret SITE_PASSWORD; the local copy is
# ~/Agents/.secrets/affect-as-attack/site-password.txt.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
ACCOUNT_ID="${CLOUDFLARE_ACCOUNT_ID:-7b10d7d0f4d06f76bb0e7ddbd59e2405}"
PROJECT="affect-as-attack"
BUILD="$(mktemp -d "${AFFECT_TMP:-/Users/anuj/Agents/.tmp}/$(date +%F)_affect-deploy.XXXX")"
trap 'rm -rf "$BUILD"' EXIT
SITE="$BUILD/site"
mkdir -p "$SITE/literature/pdfs" "$SITE/project" "$SITE/private"
cp -R "$REPO/web" "$SITE/web"
rm -f "$SITE/web/BRIEF.md"
cp "$REPO/literature/catalog.json" "$SITE/literature/"
find "$REPO/literature/pdfs" -name '*.pdf' -size -25M -exec cp {} "$SITE/literature/pdfs/" \;
rsync -a --exclude 'raw/' "$REPO/results/" "$SITE/results/"
cp "$REPO/project/status.json" "$SITE/project/"
if [ -f "$REPO/paper/paper.pdf" ]; then mkdir -p "$SITE/paper"; cp "$REPO/paper/paper.pdf" "$SITE/paper/"; fi
if [ -e "$REPO/private" ]; then cp -L "$REPO/private/"*.pdf "$SITE/private/" 2>/dev/null || true; cp -L "$REPO/private/"*.txt "$SITE/private/" 2>/dev/null || true; fi
cat > "$SITE/index.html" <<'HTML'
<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0; url=/web/"><title>Affect as Attack</title><a href="/web/">Open the project interface</a>
HTML
cat > "$SITE/404.html" <<'HTML'
<!doctype html><meta charset="utf-8"><title>Not found</title><p>Not found. <a href="/web/">Back to the project interface</a></p>
HTML
printf '/*\n  X-Robots-Tag: noindex, nofollow\n' > "$SITE/_headers"
printf 'User-agent: *\nDisallow: /\n' > "$SITE/robots.txt"
cp -R "$REPO/deploy/functions" "$BUILD/functions"
cd "$BUILD"
echo "files: $(find site -type f | wc -l | tr -d ' ')  size: $(du -sh site | cut -f1)"
CLOUDFLARE_ACCOUNT_ID="$ACCOUNT_ID" wrangler pages deploy ./site --project-name="$PROJECT" --branch=main --commit-dirty=true 2>&1 | tail -4
