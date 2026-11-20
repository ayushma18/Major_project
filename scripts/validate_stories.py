"""Validate an emotion story corpus file against data/stories/BRIEF.md.

Usage:
    python scripts/validate_stories.py data/stories/raw/angry.jsonl [more files...]
    python scripts/validate_stories.py --all

Prints one FAIL/WARN line per problem and `RESULT: PASS` or `RESULT: FAIL` per file.
Exit code 1 if any file fails. Checks are mechanical; a human still reads samples.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "stories" / "raw"

EMOTIONS = [
    "afraid", "angry", "anxious", "calm", "desperate", "enthusiastic", "exasperated", "fearful",
    "gloomy", "grateful", "guilty", "happy", "hopeful", "hostile", "loving", "nervous", "proud",
    "reflective", "sad",
]
CATEGORIES = EMOTIONS + ["neutral"]

# The target emotion's own word family: banned inside that emotion's emotional story.
FAMILY = {
    "afraid": ["afraid"],
    "angry": ["angry", "anger", "angrily", "angered", "angers"],
    "anxious": ["anxious", "anxiously", "anxiety", "anxieties"],
    "calm": ["calm", "calmly", "calmness", "calmed", "calming", "calmer", "calmest"],
    "desperate": ["desperate", "desperately", "desperation"],
    "enthusiastic": ["enthusiastic", "enthusiastically", "enthusiasm", "enthused"],
    "exasperated": ["exasperated", "exasperation", "exasperating", "exasperate"],
    "fearful": ["fearful", "fearfully", "fear", "fears", "feared", "fearing"],
    "gloomy": ["gloomy", "gloom", "gloomily"],
    "grateful": ["grateful", "gratefully", "gratitude"],
    "guilty": ["guilty", "guilt", "guiltily"],
    "happy": ["happy", "happily", "happiness", "happier", "happiest"],
    "hopeful": ["hopeful", "hopefully", "hope", "hopes", "hoped", "hoping"],
    "hostile": ["hostile", "hostility", "hostilities"],
    "loving": ["loving", "lovingly", "love", "loved", "loves"],
    "nervous": ["nervous", "nervously", "nervousness"],
    "proud": ["proud", "proudly", "pride"],
    "reflective": ["reflective", "reflect", "reflected", "reflecting", "reflection", "reflections"],
    "sad": ["sad", "sadly", "sadness", "sadder", "saddest"],
}

# Banned in neutral twins and standalone neutral stories: every family above plus common
# emotion vocabulary and emotional expressions.
EXTRA_AFFECT = [
    "emotion", "emotions", "emotional", "feeling", "feelings", "mood", "joy", "joyful", "delight",
    "delighted", "excited", "excitement", "thrilled", "glad", "cheerful", "content", "contented",
    "relieved", "relief", "upset", "worried", "worry", "worries", "worrying", "scared", "terrified",
    "frightened", "fright", "panic", "panicked", "dread", "furious", "fury", "rage", "annoyed",
    "irritated", "irritation", "frustrated", "frustration", "ashamed", "shame", "embarrassed",
    "lonely", "loneliness", "sorrow", "grief", "miserable", "misery", "heartbroken", "eager",
    "eagerly", "tense", "tension", "stressed", "stress", "peaceful", "serene", "thankful",
    "smile", "smiled", "smiles", "smiling", "grin", "grinned", "laugh", "laughed", "laughing",
    "cry", "cried", "crying", "tears", "tearful", "sigh", "sighed", "frown", "frowned",
    "trembled", "trembling", "shaking", "lovely", "awful", "terrible", "wonderful", "horrible",
    "beautiful", "sweetly", "cozy", "comforting", "affection", "adored", "resent", "resentment",
]
NEUTRAL_BANNED = sorted({w for fam in FAMILY.values() for w in fam} | set(EXTRA_AFFECT))

HARM_TERMS = [
    "kill", "killed", "murder", "gun", "guns", "knife", "stab", "bomb", "explosive", "suicide",
    "overdose", "drug", "drugs", "cocaine", "heroin", "weapon", "weapons", "blood", "bleeding",
    "rape", "sex", "sexual", "assault", "corpse", "dead body", "died", "funeral", "hack", "hacked",
    "steal", "stole", "stolen", "robbery", "poison", "terrorist", "abuse",
]

WORD = re.compile(r"[A-Za-z']+")


def words(text: str) -> list[str]:
    return [w.lower().strip("'") for w in WORD.findall(text)]


def has_any(text: str, banned: list[str]) -> list[str]:
    toks = set(words(text))
    low = text.lower()
    hits = [b for b in banned if (" " in b and b in low) or (" " not in b and b in toks)]
    return sorted(set(hits))


def ngrams(text: str, n: int = 5) -> set:
    w = words(text)
    return {tuple(w[i : i + n]) for i in range(len(w) - n + 1)}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def validate(path: Path) -> bool:
    category = path.stem
    fails: list[str] = []
    warns: list[str] = []
    if category not in CATEGORIES:
        print(f"FAIL {path.name}: unknown category {category!r}")
        print(f"RESULT: FAIL ({path.name})")
        return False
    try:
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    except json.JSONDecodeError as exc:
        print(f"FAIL {path.name}: invalid JSONL: {exc}")
        print(f"RESULT: FAIL ({path.name})")
        return False

    if len(rows) != 60:
        fails.append(f"expected 60 rows, found {len(rows)}")
    expected_ids = [f"{category}-{i:02d}" for i in range(1, 61)]
    ids = [r.get("id") for r in rows]
    if sorted(ids) != sorted(expected_ids[: len(rows)]) or len(set(ids)) != len(ids):
        fails.append("ids must be unique and run <category>-01 .. <category>-60")

    texts_main: list[str] = []
    prefix_fracs: list[float] = []
    for r in rows:
        rid = r.get("id", "?")
        if r.get("emotion") != category:
            fails.append(f"{rid}: emotion field {r.get('emotion')!r} != {category!r}")
        for k in ("setting", "character"):
            if not str(r.get(k, "")).strip():
                fails.append(f"{rid}: missing {k}")
        if category == "neutral":
            t = r.get("text", "")
            texts = {"text": t}
            texts_main.append(t)
        else:
            e, n = r.get("emotional", ""), r.get("neutral", "")
            texts = {"emotional": e, "neutral": n}
            texts_main.append(e)
            le, ln = len(words(e)), len(words(n))
            if le and not (0.80 <= ln / le <= 1.25):
                fails.append(f"{rid}: neutral twin length {ln} is not within 80-125% of emotional {le}")
            fam_hits = has_any(e, FAMILY[category])
            if fam_hits:
                fails.append(f"{rid}: emotional story names the target emotion: {fam_hits}")
            name = str(r.get("character", "")).split()[0] if r.get("character") else ""
            if name and (name not in e or name not in n):
                fails.append(f"{rid}: character name {name!r} must appear in both stories")
            nb = has_any(n, NEUTRAL_BANNED)
            if nb:
                fails.append(f"{rid}: neutral twin uses affect words: {nb}")
            ew, nw_ = words(e), words(n)
            k = 0
            while k < min(len(ew), len(nw_)) and ew[k] == nw_[k]:
                k += 1
            prefix_fracs.append(k / max(1, len(ew)))
            if k / max(1, len(ew)) > 0.5:
                fails.append(f"{rid}: the first {k} words are identical in both stories ({k / len(ew):.0%}); the emotion must be felt from early in the story, not only at the end")
        for k, t in texts.items():
            nw = len(words(t))
            if not (85 <= nw <= 150):
                fails.append(f"{rid}.{k}: {nw} words (want 90-140, hard limits 85-150)")
            if "—" in t or "–" in t:
                fails.append(f"{rid}.{k}: contains an em or en dash")
            hh = has_any(t, HARM_TERMS)
            if hh:
                fails.append(f"{rid}.{k}: harm-related terms {hh}")
        if category == "neutral":
            nb = has_any(r.get("text", ""), NEUTRAL_BANNED)
            if nb:
                fails.append(f"{rid}: neutral story uses affect words: {nb}")

    # Templating / reuse checks across stories in the file.
    grams = [ngrams(t) for t in texts_main]
    worst = (0.0, "", "")
    for (i, a), (j, b) in combinations(enumerate(grams), 2):
        jac = jaccard(a, b)
        if jac > worst[0]:
            worst = (jac, ids[i], ids[j])
        if jac > 0.20:
            fails.append(f"{ids[i]} and {ids[j]} share too much wording (5-gram Jaccard {jac:.2f})")
    openings = Counter(" ".join(words(t)[:4]) for t in texts_main)
    for op, c in openings.items():
        if c > 3:
            fails.append(f"{c} stories start with the same four words: {op!r}")
    settings = Counter(str(r.get("setting", "")).strip().lower() for r in rows)
    for s, c in settings.items():
        if c > 4:
            warns.append(f"setting {s!r} used {c} times; spread scenarios wider")
    names = Counter(str(r.get("character", "")).strip() for r in rows)
    for s, c in names.items():
        if c > 2:
            warns.append(f"character name {s!r} used {c} times")

    if prefix_fracs:
        med = sorted(prefix_fracs)[len(prefix_fracs) // 2]
        if med > 0.35:
            fails.append(f"median shared opening is {med:.0%} of the emotional story (limit 35%): start the emotion earlier")
    for f in fails:
        print(f"FAIL {path.name}: {f}")
    for w in warns:
        print(f"WARN {path.name}: {w}")
    ok = not fails
    print(f"INFO {path.name}: rows={len(rows)} max 5-gram Jaccard={worst[0]:.2f} ({worst[1]} vs {worst[2]})")
    print(f"RESULT: {'PASS' if ok else 'FAIL'} ({path.name})")
    return ok


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    paths = sorted(RAW.glob("*.jsonl")) if argv == ["--all"] else [Path(a) for a in argv]
    results = [validate(p) for p in paths]
    if argv == ["--all"]:
        missing = [c for c in CATEGORIES if not (RAW / f"{c}.jsonl").exists()]
        if missing:
            print(f"FAIL corpus: missing categories {missing}")
            results.append(False)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
