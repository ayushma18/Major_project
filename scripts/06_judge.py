"""Label generations with the HarmBench classifier and summarise ASR per condition.

    python scripts/06_judge.py --model llama-8b --set harmbench-test

Reads outputs/generations/<model>/<set>/*.jsonl, writes <condition>.labels.jsonl next to
each (label only: behaviour id, harmful, p_yes), skips files already labelled, then writes
results/raw/judged_<model>_<set>.json with ASR and a Wilson 95% interval per condition.
Run it when no target model is loaded: the 13B judge needs about 26 GB.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.judge import HarmBenchJudge  # noqa: E402
from src.runtime import ALIASES, read_jsonl, write_json, write_jsonl  # noqa: E402
from src.vectors import slug  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-8b")
    ap.add_argument("--set", default="harmbench-test")
    ap.add_argument("--judge", default="harmbench-cls")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--only", nargs="*", default=None, help="condition names to (re)label")
    ap.add_argument("--max-files", type=int, default=None, help="label at most this many files, then exit (restart the process to free MPS memory)")
    args = ap.parse_args()

    gdir = ROOT / "outputs" / "generations" / slug(ALIASES.get(args.model, args.model)) / args.set
    files = sorted(p for p in gdir.glob("*.jsonl") if not p.name.endswith(".labels.jsonl"))
    todo = [p for p in files if not p.with_suffix(".labels.jsonl").exists() and (args.only is None or p.stem in args.only)]
    if args.max_files:
        todo = todo[: args.max_files]
    judge = HarmBenchJudge(args.judge) if todo else None
    for p in todo:
        rows = read_jsonl(p)
        print(f"judging {p.stem} ({len(rows)})")
        labels = judge.classify([r["behavior"] for r in rows], [r["completion"] for r in rows], [r.get("context") for r in rows], batch_size=args.batch_size)
        write_jsonl(p.with_suffix(".labels.jsonl"), [{"behavior_id": r["behavior_id"], "category": r.get("category"), "turn": r.get("turn"), **l, "refusal_substring": r["refusal_substring"]} for r, l in zip(rows, labels)])

    summary = {"model": ALIASES.get(args.model, args.model), "set": args.set, "judge": ALIASES.get(args.judge, args.judge), "conditions": {}}
    for p in files:
        lp = p.with_suffix(".labels.jsonl")
        if not lp.exists():
            continue
        labels = read_jsonl(lp)
        n = len(labels)
        k = sum(l["harmful"] for l in labels)
        cats: dict[str, list[int]] = {}
        for l in labels:
            cats.setdefault(l.get("category") or "?", []).append(int(l["harmful"]))
        summary["conditions"][p.stem] = {
            "n": n, "asr": k / max(1, n), "ci95": wilson(k, n),
            "refusal_substring_rate": sum(l["refusal_substring"] for l in labels) / max(1, n),
            "by_category": {c: sum(v) / len(v) for c, v in cats.items()},
            "harmful_ids": sorted(l["behavior_id"] for l in labels if l["harmful"]),
        }
        turns = sorted({l.get("turn") for l in labels if l.get("turn")})
        if turns:
            summary["conditions"][p.stem]["by_turn"] = {
                str(t): {"asr": sum(l["harmful"] for l in labels if l.get("turn") == t) / max(1, sum(1 for l in labels if l.get("turn") == t)),
                         "n": sum(1 for l in labels if l.get("turn") == t),
                         "refusal": sum(l["refusal_substring"] for l in labels if l.get("turn") == t) / max(1, sum(1 for l in labels if l.get("turn") == t))}
                for t in turns}
    write_json(ROOT / "results" / "raw" / f"judged_{args.model}_{args.set}.json", summary)
    for c, s in sorted(summary["conditions"].items()):
        print(f"{c:40s} ASR {s['asr']:.3f}  [{s['ci95'][0]:.2f}, {s['ci95'][1]:.2f}]  refusal {s['refusal_substring_rate']:.3f}")


if __name__ == "__main__":
    main()
