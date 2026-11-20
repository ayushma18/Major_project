"""E06: how do the emotion vectors line up with the refusal (guardrail) direction?

    python scripts/09_geometry.py --model llama-8b

For every vector type (caa, specific, centered) and layer l:
    cos_same[e][l]  = cos(v_e[l], r[l])    with r[l] the refusal candidate at layer l and the
                                           selected post-instruction position
    cos_star[e][l]  = cos(v_e[l], r*)      with r* the selected refusal direction
Also the cosine of the shared affect direction (emotion_common) with r, and the null
distribution for random directions (|cos| of 10,000 random unit vectors with r*), so every
number can be read against chance (about 1/sqrt(hidden)).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import data  # noqa: E402
from src.runtime import ALIASES, write_json  # noqa: E402
from src.vectors import VectorLibrary  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-8b")
    args = ap.parse_args()
    name = ALIASES.get(args.model, args.model)
    lib = VectorLibrary(name)
    cands, _ = lib.load("refusal_candidates")
    r_star, meta = lib.load("refusal")
    pos = meta["pos_index"]
    n_layers, _, hidden = cands.shape
    R = unit(cands[:, pos])  # [L, H]
    rs = unit(r_star)
    out = {"model": name, "refusal": meta, "n_layers": n_layers, "hidden": hidden, "types": {}}
    for kind in ("caa", "specific", "centered"):
        emos = [e for e in data.EMOTIONS if lib.exists(f"emotion_{kind}_{e}")]
        if not emos:
            continue
        same, star = {}, {}
        for e in emos:
            V = unit(lib.load(f"emotion_{kind}_{e}")[0])  # [L, H]
            same[e] = (V * R).sum(-1).tolist()
            star[e] = (V @ rs).tolist()
        out["types"][kind] = {"emotions": emos, "cos_same_layer": same, "cos_with_r_star": star}
    if lib.exists("emotion_common"):
        C = unit(lib.load("emotion_common")[0])
        out["common"] = {"cos_same_layer": (C * R).sum(-1).tolist(), "cos_with_r_star": (C @ rs).tolist()}
    g = torch.Generator().manual_seed(0)
    rnd = unit(torch.randn(10000, hidden, generator=g))
    null = (rnd @ rs).abs()
    out["null_abs_cos"] = {"mean": float(null.mean()), "p95": float(null.quantile(0.95)), "p99": float(null.quantile(0.99)), "max": float(null.max())}
    # refusal direction stability across layers: cos(r[l], r*)
    out["refusal_cos_across_layers"] = (R @ rs).tolist()
    write_json(ROOT / "results" / "raw" / f"E06_{args.model}.json", out)
    L = meta["layer"]
    print(f"r* at layer {L}, pos {meta['pos']}; random |cos| p99 = {out['null_abs_cos']['p99']:.3f}")
    for kind, t in out["types"].items():
        row = sorted(((e, t["cos_same_layer"][e][L]) for e in t["emotions"]), key=lambda x: -x[1])
        print(f"{kind:9s} at layer {L}: " + " ".join(f"{e}:{c:+.2f}" for e, c in row))


if __name__ == "__main__":
    main()
