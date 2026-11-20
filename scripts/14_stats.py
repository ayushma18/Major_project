"""E07 to E10 statistics: does any emotion beat random vectors, and is any change real?

    python scripts/14_stats.py --model llama-8b --layer 12 --c 0.5

Uses the judged full-length runs (outputs/generations/<model>/harmbench-test/*.labels.jsonl).
For every condition kind (emotion, specific, perp, negative) and every emotion:
    asr, delta vs baseline
    paired test vs baseline: exact McNemar (two-sided binomial on the discordant behaviours)
    position in the random-vector null: z = (asr - mean_random) / sd_random over the random
        seeds at the same strength, and the empirical share of random seeds at or above it
    Holm correction across the 19 emotions within each kind (McNemar p-values)
Also: random seeds summary, common direction, fluency (from coherence_<model>_harmbench-test.json
when present). Writes results/raw/E07_stats_<model>.json (rates and p-values only).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import data  # noqa: E402
from src.runtime import ALIASES, read_jsonl, write_json  # noqa: E402
from src.vectors import slug  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def binom_two_sided(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2**n
    return min(1.0, 2 * tail)


def holm(ps: dict[str, float]) -> dict[str, float]:
    items = sorted(ps.items(), key=lambda x: x[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-8b")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--c", default="0.5")
    args = ap.parse_args()
    name = ALIASES.get(args.model, args.model)
    gdir = ROOT / "outputs" / "generations" / slug(name) / "harmbench-test"
    labels = {p.name[: -len(".labels.jsonl")]: {r["behavior_id"]: bool(r["harmful"]) for r in read_jsonl(p)} for p in gdir.glob("*.labels.jsonl")}
    if "baseline" not in labels:
        print("baseline not judged yet")
        return
    coh_path = ROOT / "results" / "raw" / f"coherence_{args.model}_harmbench-test.json"
    coh = json.loads(coh_path.read_text())["conditions"] if coh_path.exists() else {}
    base = labels["baseline"]
    ids = sorted(base)

    def asr(cond: str) -> float:
        lab = labels[cond]
        return sum(lab.values()) / len(lab)

    def mcnemar(cond: str) -> tuple[int, int, float]:
        lab = labels[cond]
        b = sum(1 for i in ids if lab.get(i) and not base[i])  # newly harmful
        c = sum(1 for i in ids if base[i] and not lab.get(i))  # newly safe
        return b, c, binom_two_sided(min(b, c), b + c)

    L, c = args.layer, args.c
    rand = [k for k in labels if k.startswith("random:") and k.endswith(f"@{L}:{c}")]
    r_asr = [asr(k) for k in rand]
    mu = sum(r_asr) / len(r_asr) if r_asr else float("nan")
    sd = (sum((x - mu) ** 2 for x in r_asr) / (len(r_asr) - 1)) ** 0.5 if len(r_asr) > 1 else float("nan")
    out = {"model": name, "layer": L, "c": c, "baseline_asr": asr("baseline"), "n_behaviours": len(ids),
           "random": {"n_seeds": len(rand), "asr_mean": mu, "asr_sd": sd, "asr_min": min(r_asr) if r_asr else None, "asr_max": max(r_asr) if r_asr else None,
                      "per_seed": {k: asr(k) for k in sorted(rand)}},
           "kinds": {}}
    kinds = {"emotion": f"emotion:{{e}}@{L}:{c}", "specific": f"specific:{{e}}@{L}:{c}", "perp": f"perp:{{e}}@{L}:{c}", "negative": f"emotion:{{e}}@{L}:-{c}"}
    for kind, pat in kinds.items():
        rows, ps = {}, {}
        for e in data.EMOTIONS:
            k = pat.format(e=e)
            if k not in labels:
                continue
            a = asr(k)
            b, cc, p = mcnemar(k)
            ps[e] = p
            rows[e] = {"condition": k, "asr": a, "delta": a - out["baseline_asr"], "newly_harmful": b, "newly_safe": cc, "mcnemar_p": p,
                       "z_vs_random": (a - mu) / sd if sd and sd == sd and sd > 0 else None,
                       "share_random_at_or_above": (sum(1 for x in r_asr if x >= a) / len(r_asr)) if r_asr else None,
                       "nll": coh.get(k, {}).get("nll"), "broken": coh.get(k, {}).get("broken")}
        adj = holm(ps)
        for e in rows:
            rows[e]["mcnemar_p_holm"] = adj[e]
        if rows:
            out["kinds"][kind] = rows
            sig = [e for e, r in rows.items() if r["mcnemar_p_holm"] < 0.05]
            above = [e for e, r in rows.items() if r["z_vs_random"] is not None and r["z_vs_random"] > 2]
            print(f"{kind:9s} n={len(rows)}  mean ASR {sum(r['asr'] for r in rows.values()) / len(rows):.3f}  significant vs baseline (Holm): {sig or 'none'}  above random (z>2): {above or 'none'}")
    for k in (f"common@{L}:{c}", f"common@{L}:-{c}", "ablate-refusal"):
        if k in labels:
            b, cc, p = mcnemar(k)
            out.setdefault("other", {})[k] = {"asr": asr(k), "mcnemar_p": p, "newly_harmful": b, "newly_safe": cc}
    write_json(ROOT / "results" / "raw" / f"E07_stats_{args.model}.json", out)
    print(f"baseline {out['baseline_asr']:.3f}; random at c={c}: mean {mu:.3f} sd {sd:.3f} (n={len(rand)})")


if __name__ == "__main__":
    main()
