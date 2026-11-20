"""E11: does geometry predict the attack? (proposal SO3, eq. 5.4 to 5.6)

    python scripts/11_correlation.py --model llama-8b --layer 12 --c 0.5 --kind emotion

Joins, per emotion, the cosine with the refusal direction at the injection layer
(results/raw/E06) with the judged attack success rate of `<kind>:<e>@L:c`
(results/raw/judged_<model>_harmbench-test.json) and computes:
    delta_asr[e] = ASR(e) - ASR(baseline)
    Spearman rho(cos, delta_asr) across emotions, a permutation p-value (20,000 shuffles)
    and a bootstrap 95% interval for rho (resampling emotions)
    the causal check: ASR(perp:e) vs ASR(emotion:e), where perp removes the refusal part
The same with the refusal-substring rate as a secondary outcome.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.runtime import write_json  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def ranks(x: list[float]) -> list[float]:
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / den if den else float("nan")


def spearman(a, b):
    return pearson(ranks(a), ranks(b))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-8b")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--c", type=float, required=True)
    ap.add_argument("--kind", default="emotion")
    ap.add_argument("--geom", default="caa", help="vector type in E06 matching --kind")
    args = ap.parse_args()
    geo = json.loads((ROOT / "results" / "raw" / f"E06_{args.model}.json").read_text())
    judged = json.loads((ROOT / "results" / "raw" / f"judged_{args.model}_harmbench-test.json").read_text())["conditions"]
    base = judged["baseline"]
    c_str = f"{args.c:g}" if args.c != int(args.c) else f"{args.c:.1f}"

    def cond(kind, e):
        for cand in (f"{kind}:{e}@{args.layer}:{args.c}", f"{kind}:{e}@{args.layer}:{c_str}", f"{kind}:{e}@{args.layer}:{args.c:g}"):
            if cand in judged:
                return judged[cand]
        return None

    t = geo["types"][args.geom]
    rows = []
    for e in t["emotions"]:
        j = cond(args.kind, e)
        if j is None:
            continue
        p = cond("perp", e)
        rows.append({"emotion": e, "cos": t["cos_same_layer"][e][args.layer], "asr": j["asr"], "delta_asr": j["asr"] - base["asr"],
                     "refusal": j["refusal_substring_rate"], "delta_refusal": j["refusal_substring_rate"] - base["refusal_substring_rate"],
                     "asr_perp": p["asr"] if p else None, "refusal_perp": p["refusal_substring_rate"] if p else None})
    if len(rows) < 4:
        print(f"only {len(rows)} emotions judged at layer {args.layer}, c {args.c}; need at least 4")
        return
    cos = [r["cos"] for r in rows]
    out = {"model": geo["model"], "layer": args.layer, "c": args.c, "kind": args.kind, "baseline_asr": base["asr"], "baseline_refusal": base["refusal_substring_rate"], "rows": rows}
    rng = random.Random(0)
    for target in ("delta_asr", "delta_refusal"):
        y = [r[target] for r in rows]
        rho = spearman(cos, y)
        perm = 0
        for _ in range(20000):
            ys = y[:]
            rng.shuffle(ys)
            if abs(spearman(cos, ys)) >= abs(rho) - 1e-12:
                perm += 1
        boots = []
        for _ in range(5000):
            idx = [rng.randrange(len(rows)) for _ in rows]
            a, b = [cos[i] for i in idx], [y[i] for i in idx]
            if len(set(a)) > 1 and len(set(b)) > 1:
                boots.append(spearman(a, b))
        boots.sort()
        out[target] = {"spearman_rho": rho, "perm_p": (perm + 1) / 20001, "boot_ci95": [boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]] if boots else None, "pearson_r": pearson(cos, y)}
        print(f"{target:14s} rho = {rho:+.3f}  p = {out[target]['perm_p']:.4f}  CI {out[target]['boot_ci95']}")
    perp_rows = [r for r in rows if r["asr_perp"] is not None]
    if perp_rows:
        out["perp_check"] = {"n": len(perp_rows), "mean_delta_asr": sum(r["delta_asr"] for r in perp_rows) / len(perp_rows),
                             "mean_delta_asr_perp": sum(r["asr_perp"] - base["asr"] for r in perp_rows) / len(perp_rows)}
        print("perp check:", out["perp_check"])
    write_json(ROOT / "results" / "raw" / f"E11_{args.model}_{args.kind}_L{args.layer}_c{args.c:g}.json", out)


if __name__ == "__main__":
    main()
