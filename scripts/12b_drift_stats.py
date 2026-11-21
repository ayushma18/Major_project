"""Paired statistics for the E12 single protocol (CPU only, reads the cached benign prefixes).

    .venv/bin/python scripts/12b_drift_stats.py --model llama-8b

Turns 1 to 3 of the single protocol are a benign prefix generated once per script (39 scripts):
prefix_none.json without injection and prefix_<vector>.json with the vector on every turn-1 token.
Each prefix stores, per script and turn, the projection of the last prompt token on the refusal
candidate at the drift layers. Because both prefixes use the same scripts, the difference early
minus none is paired by script. Turn 1 includes the direct effect of the injection (the prompt
itself is steered); turns 2 and 3 show whether a steered turn 1 still moves the state of later,
unsteered turns. Turn-4 values are only stored as means (one per condition), so they are reported
as means without a test.

Writes results/raw/E12_<model>_single_drift.json (numbers only, no text).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.runtime import ALIASES, write_json  # noqa: E402
from src.vectors import slug  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def sign_flip_p(d: list[float], n_perm: int = 20000, seed: int = 0) -> float:
    """Two-sided paired permutation p-value (random sign flips of the per-script differences)."""
    obs = abs(sum(d))
    rng = random.Random(seed)
    hits = sum(abs(sum(x if rng.random() < 0.5 else -x for x in d)) >= obs - 1e-12 for _ in range(n_perm))
    return (hits + 1) / (n_perm + 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-8b")
    args = ap.parse_args()
    m = args.model
    d = ROOT / "outputs" / "generations" / slug(ALIASES.get(m, m)) / "multiturn-single"
    none = json.loads((d / "prefix_none.json").read_text())
    merged = json.loads((ROOT / "results" / "raw" / f"E12_{m}_single.json").read_text())
    turn4 = {c.get("condition", stem): (c.get("drift_mean_by_turn") or {}) for stem, c in merged["conditions"].items()}
    layers = [str(x) for x in merged["drift_layers"]]
    out = {"model": m, "layers": layers, "n_scripts": len(none["plans"]),
           "note": "early minus none, paired by benign script; turn 1 includes the direct injection; turn 4 means only",
           "none_turn_means": {l: [sum(p["drift"][l][t] for p in none["prefix"]) / len(none["prefix"]) for t in range(3)] + [turn4.get("none", {}).get(l, [None] * 4)[3]] for l in layers},
           "vectors": {}}
    # scale: how far the projection moves when the turn-4 request is harmful (none, turn 4 minus turn 3)
    out["harmful_gap"] = {l: (out["none_turn_means"][l][3] - out["none_turn_means"][l][2]) if out["none_turn_means"][l][3] is not None else None for l in layers}
    for f in sorted(d.glob("prefix_*.json")):
        if f.name == "prefix_none.json":
            continue
        pre = json.loads(f.read_text())
        if pre["plans"] != none["plans"]:
            print("skip (different scripts):", f.name)
            continue
        spec = pre["label"]
        res = {}
        for l in layers:
            rows = []
            for t in range(3):
                diff = [a["drift"][l][t] - b["drift"][l][t] for a, b in zip(pre["prefix"], none["prefix"])]
                n = len(diff)
                mu = sum(diff) / n
                sd = math.sqrt(sum((x - mu) ** 2 for x in diff) / (n - 1)) if n > 1 else 0.0
                sd_none = math.sqrt(sum((b["drift"][l][t] - out["none_turn_means"][l][t]) ** 2 for b in none["prefix"]) / (n - 1))
                rows.append({"turn": t + 1, "mean_diff": mu, "sd_diff": sd, "se": sd / math.sqrt(n), "sd_none": sd_none,
                             "n_up": sum(x > 0 for x in diff), "n_down": sum(x < 0 for x in diff), "perm_p": sign_flip_p(diff)})
            e4 = turn4.get(f"early:{spec}", {}).get(l)
            s4 = turn4.get(f"same:{spec}", {}).get(l)
            n4 = out["none_turn_means"][l][3]
            rows.append({"turn": 4, "mean_early": e4[3] if e4 else None, "mean_same": s4[3] if s4 else None, "mean_none": n4,
                         "mean_diff": (e4[3] - n4) if e4 and n4 is not None else None})
            res[l] = rows
        out["vectors"][spec] = res
        print(spec, {l: [round(r["mean_diff"], 3) if r["mean_diff"] is not None else None for r in res[l]] for l in layers})
    write_json(ROOT / "results" / "raw" / f"E12_{m}_single_drift.json", out)
    print("wrote", f"results/raw/E12_{m}_single_drift.json")


if __name__ == "__main__":
    main()
