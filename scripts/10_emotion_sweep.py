"""E07, E08, E09, E10: the attack-success sweep, as one resumable plan.

    python scripts/10_emotion_sweep.py --model llama-8b --layer 12 --coeffs 0.5 1.0 --plan-only
    python scripts/10_emotion_sweep.py --model llama-8b --layer 12 --coeffs 0.5 1.0

Plan, all on the 159 HarmBench standard test behaviours, greedy, 256 new tokens:
    E07  emotion:<e>@L:c     for every emotion and every c in --coeffs (paired CAA, the proposal's vector)
         emotion:<e>@L:-c0   the negative direction at the first strength
         specific:<e>@L:c0   the emotion-specific part (shared affect component removed)
         common@L:+-c        the shared affect component itself
    E08  random:<s>@L:c      --n-random seeds at every strength (same injected norm)
    E09  prime:<group>:<i>   --n-prime system prompts per group (stress, relaxation, neutral)
    E10  perp:<e>@L:c0       each emotion with its refusal component removed, equal norm
The plan is saved to results/raw/E07_plan_<model>.json; already generated conditions are
skipped, so the run can be stopped and restarted.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import data  # noqa: E402
from src.runtime import write_json  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def fmt(c: float) -> str:
    return f"{c:g}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-8b")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--coeffs", nargs="+", type=float, required=True)
    ap.add_argument("--n-random", type=int, default=8)
    ap.add_argument("--n-prime", type=int, default=5)
    ap.add_argument("--only", nargs="*", default=None, help="subset of: emotion negative specific common random prime perp")
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    L, cs, c0 = args.layer, args.coeffs, args.coeffs[0]
    E = data.available_emotions()
    plan: dict[str, list[str]] = {
        "emotion": [f"emotion:{e}@{L}:{fmt(c)}" for c in cs for e in E],
        "negative": [f"emotion:{e}@{L}:{fmt(-c0)}" for e in E],
        "specific": [f"specific:{e}@{L}:{fmt(c0)}" for e in E],
        "common": [f"common@{L}:{fmt(c)}" for c in cs] + [f"common@{L}:{fmt(-c0)}"],
        "random": [f"random:{s}@{L}:{fmt(c)}" for c in cs for s in range(args.n_random)],
        "prime": [f"prime:{g}:{i}" for g in ("stress", "relaxation", "neutral") for i in range(args.n_prime)],
        "perp": [f"perp:{e}@{L}:{fmt(c0)}" for e in E],
    }
    groups = args.only or list(plan)
    order = ["baseline"] + [c for g in ("random", "emotion", "perp", "negative", "specific", "common", "prime") if g in groups for c in plan[g]]
    write_json(ROOT / "results" / "raw" / f"E07_plan_{args.model}{'_' + args.tag if args.tag else ''}.json", {"layer": L, "coeffs": cs, "groups": {g: plan[g] for g in groups}, "conditions": order})
    print(f"{len(order)} conditions planned")
    if args.plan_only:
        return
    cmd = [sys.executable, str(ROOT / "scripts" / "05_run_conditions.py"), "--model", args.model, "--set", "harmbench-test", "--batch-size", str(args.batch_size), "--max-new-tokens", str(args.max_new_tokens), *(["--tag", args.tag] if args.tag else []), "--conditions", *order]
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
