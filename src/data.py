"""Dataset loaders. Every split is deterministic (fixed seeds) and documented.

HarmBench (Mazeika et al. 2024), text behaviours, `data/benchmarks/harmbench/`:
    standard test (159) is the evaluation set for every ASR number in this project.
    standard val (41) is used only for quick pilots, never for selecting a direction.
Refusal-direction splits (Arditi et al. 2024, `data/benchmarks/refusal_direction_splits/`):
    harmful_train (260) / harmless_train (18,793) to build candidate directions,
    harmful_val (39) / harmless_val (6,264) to score and select them.
    None of these instructions appear verbatim in the HarmBench test set (checked).
Emotion stories: `data/stories/raw/<emotion>.jsonl`, pairs 01 to 50 train, 51 to 60 held out.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "data" / "benchmarks"
STORIES = ROOT / "data" / "stories" / "raw"
PROMPTS = ROOT / "data" / "prompts"

EMOTIONS = [
    "afraid", "angry", "anxious", "calm", "desperate", "enthusiastic", "exasperated", "fearful",
    "gloomy", "grateful", "guilty", "happy", "hopeful", "hostile", "loving", "nervous", "proud",
    "reflective", "sad",
]
N_TRAIN_PAIRS = 50


@dataclass
class Behavior:
    id: str
    behavior: str
    category: str
    context: str | None = None


def harmbench(split: str = "test", functional: str = "standard") -> list[Behavior]:
    path = BENCH / "harmbench" / f"harmbench_behaviors_text_{split}.csv"
    rows = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["FunctionalCategory"] != functional:
            continue
        rows.append(Behavior(r["BehaviorID"], r["Behavior"], r["SemanticCategory"], r["ContextString"] or None))
    return rows


def harmbench_prompt(b: Behavior) -> str:
    """The user message for a behaviour (HarmBench DirectRequest format)."""
    return f"{b.context}\n\n---\n\n{b.behavior}" if b.context else b.behavior


def _split(name: str) -> list[str]:
    return [x["instruction"] for x in json.load(open(BENCH / "refusal_direction_splits" / f"{name}.json", encoding="utf-8"))]


def refusal_sets(n_train: int = 128, n_val: int = 32, seed: int = 0) -> dict[str, list[str]]:
    rng = random.Random(seed)
    harmful_train = rng.sample(_split("harmful_train"), n_train)
    harmless_train = rng.sample(_split("harmless_train"), n_train)
    harmful_val = _split("harmful_val")[: max(n_val, 32)]
    harmless_val = rng.sample(_split("harmless_val"), n_val)
    return {"harmful_train": harmful_train, "harmless_train": harmless_train, "harmful_val": harmful_val, "harmless_val": harmless_val}


def harmless_eval(n: int = 100, seed: int = 1) -> list[str]:
    """Harmless instructions for over-refusal / coherence checks (Alpaca, from Arditi's test split)."""
    return random.Random(seed).sample(_split("harmless_test"), n)


def load_stories(emotion: str) -> list[dict]:
    path = STORIES / f"{emotion}.jsonl"
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    rows.sort(key=lambda r: r["id"])
    for i, r in enumerate(rows):
        r["split"] = "train" if i < N_TRAIN_PAIRS else "heldout"
    return rows


def available_emotions() -> list[str]:
    return [e for e in EMOTIONS if (STORIES / f"{e}.jsonl").exists()]


def priming() -> dict[str, list[str]]:
    return json.load(open(PROMPTS / "priming.json", encoding="utf-8"))


def multiturn_benign() -> list[dict]:
    return json.load(open(PROMPTS / "multiturn_benign.json", encoding="utf-8"))
