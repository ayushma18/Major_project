"""Validate data/stories/ood/validation.jsonl (see data/stories/BRIEF_ood.md)."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_stories import EMOTIONS, FAMILY, HARM_TERMS, has_any, words  # noqa: E402

PATH = Path(__file__).resolve().parent.parent / "data" / "stories" / "ood" / "validation.jsonl"


def main() -> int:
    rows = [json.loads(l) for l in PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    fails = []
    count = Counter(r.get("emotion") for r in rows)
    for e in EMOTIONS:
        if count.get(e, 0) != 12:
            fails.append(f"{e}: {count.get(e, 0)} items, want 12")
    ids = [r.get("id") for r in rows]
    if len(set(ids)) != len(ids):
        fails.append("duplicate ids")
    for r in rows:
        e, t = r.get("emotion"), r.get("text", "")
        n = len(words(t))
        if not 22 <= n <= 75:
            fails.append(f"{r.get('id')}: {n} words (want 25-70)")
        if e in FAMILY and has_any(t, FAMILY[e]):
            fails.append(f"{r.get('id')}: names its own emotion {has_any(t, FAMILY[e])}")
        if has_any(t, HARM_TERMS):
            fails.append(f"{r.get('id')}: harm terms {has_any(t, HARM_TERMS)}")
        if "—" in t or "–" in t:
            fails.append(f"{r.get('id')}: em or en dash")
        if r.get("form") not in ("monologue", "spoken", "message", "diary"):
            fails.append(f"{r.get('id')}: bad form {r.get('form')!r}")
    for f in fails:
        print("FAIL", f)
    print(f"INFO rows={len(rows)}")
    print("RESULT:", "PASS" if not fails else "FAIL")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
