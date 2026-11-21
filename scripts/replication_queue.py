"""Queue the second-model replication (E13) once its refusal direction exists.

    python scripts/replication_queue.py --model qwen2.5-7b            # print the jobs
    python scripts/replication_queue.py --model qwen2.5-7b --insert   # put them at the front of logs/queue.txt

Stage 1 (queued by hand): 04_refusal_direction, 07_emotion_vectors, 07b_ood_eval, then this script
with --insert. It reads the selected refusal layer and applies the rule used on Llama 3.1 8B,
fixed before any replication result was seen: inject one layer after the refusal layer
(Llama: refusal 11, injection 12), strength c = 0.5 of the mean residual norm.

Deviation (2026-11-21): on Qwen2.5-7B the mean norm is dominated by massive activations (one
newline token per prompt at about 13,700 against a median of 100; scripts/15_norm_profile.py),
so c = 0.5 of the raw mean pushed 2.2 times a typical token's norm and broke 45 to 100% of the
answers (kept as harmbench-test-c0.5). The rule is restated as 0.5 of the mean norm over
non-outlier tokens (<= 5 x median), which is the same push on Llama (no outliers) and
c = 0.115 of the raw mean on Qwen: pass it as --c 0.115.

Stage 2 (what this script queues), the conditions behind the main claims, 256-token answers:
    steering check at that layer (E05), geometry (E06), harmless over-refusal (E01/E03),
    baseline, ablate-refusal, 10 random seeds, the 19 paired emotion vectors, the 19 specific
    vectors and the 19 paired vectors subtracted; then the HarmBench judge, stats, correlation,
    results and the site.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import data  # noqa: E402
from src.runtime import ALIASES  # noqa: E402
from src.vectors import slug  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ENV = ("AFFECT_MODELS_DIR=/Users/anuj/Agents/tools/models/hf PYTHONUNBUFFERED=1 "
       "PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.8")
PY = ".venv/bin/python"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-7b")
    ap.add_argument("--c", default="0.5")
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--sanity-coeffs", nargs="+", default=None, help="strengths for the E05 steering check (default: --c)")
    ap.add_argument("--insert", action="store_true")
    args = ap.parse_args()
    m = args.model
    meta = json.loads((ROOT / "artifacts" / "vectors" / slug(ALIASES.get(m, m)) / "refusal.json").read_text())
    L, c = meta["layer"] + 1, args.c
    emos = data.EMOTIONS

    def run(conds, set_="harmbench-test", bs=args.batch_size):
        base = f"{ENV} {PY} scripts/05_run_conditions.py --model {m} --set {set_} --conditions {' '.join(conds)}"
        return f"{base} --batch-size {bs} || {base} --batch-size {max(8, bs // 3)}"

    gdir = f"outputs/generations/{slug(ALIASES.get(m, m))}/harmbench-test"
    jobs = [
        f"{ENV} {PY} scripts/08_steering_sanity.py --model {m} --layers {L} --coeffs {' '.join(args.sanity_coeffs or [c])}",
        f"{ENV} {PY} scripts/09_geometry.py --model {m}",
        run(["baseline", "ablate-refusal"], set_="harmless", bs=32),
        run(["baseline", "ablate-refusal"] + [f"random:{s}@{L}:{c}" for s in range(10)]),
        run([f"emotion:{e}@{L}:{c}" for e in emos]),
        run([f"specific:{e}@{L}:{c}" for e in emos]),
        run([f"emotion:{e}@{L}:-{c}" for e in emos]),
        (f"for i in $(seq 1 25); do {ENV} {PY} scripts/06_judge.py --model {m} --set harmbench-test --batch-size 4 --max-files 4 || true; "
         f"n=$(ls {gdir}/*.jsonl | grep -v labels | wc -l); k=$(ls {gdir}/*.labels.jsonl | wc -l); [ $n -le $k ] && break; done"),
        f"{ENV} {PY} scripts/13_coherence.py --model {m} --set harmbench-test --batch-size 32",
        (f"AFFECT_MODELS_DIR=/Users/anuj/Agents/tools/models/hf {PY} scripts/14_stats.py --model {m} --layer {L} --c {c}; "
         f"{PY} scripts/11_correlation.py --model {m} --layer {L} --c {c} --kind emotion --geom caa; "
         f"{PY} scripts/11_correlation.py --model {m} --layer {L} --c {c} --kind specific --geom specific; "
         f"{PY} scripts/make_results.py --model {m}; {PY} scripts/make_results.py --model llama-8b && bash scripts/deploy_pages.sh"),
    ]
    print(f"# {m}: refusal layer {meta['layer']}, injection layer {L}, c = {c}")
    if not args.insert:
        print("\n".join(jobs))
        return
    q = ROOT / "logs" / "queue.txt"
    rest = q.read_text().splitlines() if q.exists() else []
    q.write_text("\n".join(jobs + rest) + "\n")
    print(f"inserted {len(jobs)} jobs at the front of {q.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
