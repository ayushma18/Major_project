"""Per-token residual norm profile at chosen layers (E13 strength calibration).

    python scripts/15_norm_profile.py --model qwen2.5-7b --layers 18 19
    python scripts/15_norm_profile.py --model llama-8b --layers 11 12

Same prompts and masking as the steering norm reference (scripts/04_refusal_direction.py,
src/vectors.py::layer_norm_stats): the first 64 harmless training instructions in the chat
template, padding and each prompt's first real token left out. Reports the mean (the reference
the steering strength c multiplies), the median and percentiles, and which tokens carry outlier
norms. Qwen2.5 has massive activations on a few tokens, so its mean is far above a typical token's
norm; this file is what the replication's strength rule uses instead.

Writes results/raw/E13_norms_<model>.json (numbers and token strings only).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import data  # noqa: E402
from src.activations import get_layers  # noqa: E402
from src.generate import build_chat_prompt  # noqa: E402
from src.runtime import load, write_json  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-7b")
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    model, tok, prov = load(args.model)
    device = next(model.parameters()).device
    instr = data.refusal_sets()["harmless_train"][:64]
    layers = get_layers(model)
    store: dict[int, list[torch.Tensor]] = {l: [] for l in args.layers}
    toks: list[str] = []
    cur: dict[str, torch.Tensor] = {}

    def make(l):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, (tuple, list)) else out
            store[l].append(h.float().norm(dim=-1)[cur["m"]].cpu())
        return hook

    handles = [layers[l].register_forward_hook(make(l)) for l in args.layers]
    try:
        prompts = [build_chat_prompt(tok, x) for x in instr]
        for b in range(0, len(prompts), args.batch_size):
            enc = tok(prompts[b : b + args.batch_size], return_tensors="pt", padding=True, add_special_tokens=False)
            m = enc["attention_mask"].bool().clone()
            m[:, :1] = False
            first = enc["attention_mask"].float().argmax(dim=1)
            m[torch.arange(m.shape[0]), first] = False
            cur["m"] = m.to(device)
            toks += [tok.convert_ids_to_tokens(int(i)) for i in enc["input_ids"][m]]
            with torch.no_grad():
                model(**{k: v.to(device) for k, v in enc.items()}, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    out = {"model": prov["model_name"], "n_prompts": len(instr), "layers": {}}
    for l in args.layers:
        x = torch.cat(store[l])
        med = float(x.median())
        q = lambda p: float(torch.quantile(x, p))
        big = (x > 5 * med).nonzero().flatten().tolist()
        out["layers"][str(l)] = {
            "n_tokens": int(x.numel()), "mean": float(x.mean()), "median": med, "p10": q(0.10), "p90": q(0.90), "p99": q(0.99), "max": float(x.max()),
            "mean_without_outliers": float(x[x <= 5 * med].mean()), "outlier_share": len(big) / x.numel(),
            "outlier_tokens": Counter(toks[i] for i in big).most_common(8),
        }
        print(l, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out["layers"][str(l)].items()})
    write_json(ROOT / "results" / "raw" / f"E13_norms_{args.model}.json", out)


if __name__ == "__main__":
    main()
