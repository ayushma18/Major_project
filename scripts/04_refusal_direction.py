"""E02: find the refusal (guardrail) direction (Arditi et al. 2024).

    python scripts/04_refusal_direction.py --model llama-8b

Steps: mean activations for 128 harmful and 128 harmless training instructions at every
layer and post-instruction position; candidate r = difference of means; score each on the
validation sets (bypass under ablation, induce under addition, KL on harmless); select.
Also stores the residual-norm table used to scale every steering vector.

Outputs:
    artifacts/vectors/<model>/refusal.safetensors           selected direction r*
    artifacts/vectors/<model>/refusal_candidates.safetensors  [layers, positions, hidden]
    artifacts/vectors/<model>/resid_norms.safetensors         [layers]
    results/raw/E02_<model>.json                             every candidate's scores
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import data  # noqa: E402
from src.refusal import (  # noqa: E402
    mean_activations, post_instruction_positions, refusal_token_ids, score_candidates, select_direction,
)
from src.runtime import load, write_json  # noqa: E402
from src.vectors import VectorLibrary, layer_norm_stats  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-8b")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--layer-step", type=int, default=1, help="score every k-th layer (1 = all)")
    args = ap.parse_args()

    t0 = time.time()
    model, tok, prov = load(args.model)
    lib = VectorLibrary(prov["model_name"])
    sets = data.refusal_sets()
    n_pos = post_instruction_positions(tok)
    ref_ids = refusal_token_ids(tok, ("I",))
    print(f"post-instruction positions: {n_pos}; refusal token ids {ref_ids}")

    print("mean activations: harmful_train")
    mu_harm = mean_activations(model, tok, sets["harmful_train"], n_pos, args.batch_size)
    print("mean activations: harmless_train")
    mu_safe = mean_activations(model, tok, sets["harmless_train"], n_pos, args.batch_size)
    cands = mu_harm - mu_safe  # [L, P, H]
    lib.save("refusal_candidates", cands, {"n_harmful": len(sets["harmful_train"]), "n_harmless": len(sets["harmless_train"]), "positions": n_pos, "space": "resid_post[layer]"})

    norms = layer_norm_stats(model, tok, sets["harmless_train"][:64], args.batch_size)
    lib.save("resid_norms", norms, {"what": "mean L2 norm of resid_post per layer over harmless chat prompts", "n_prompts": 64})

    n_layers = cands.shape[0]
    layers = list(range(0, n_layers, args.layer_step))
    print(f"scoring {len(layers) * n_pos} candidates")
    scored = score_candidates(model, tok, cands, sets["harmful_val"], sets["harmless_val"], ref_ids, layers=layers, batch_size=args.batch_size)
    best = select_direction(scored, n_layers)
    pos_index = best.pos + n_pos
    r = cands[best.layer, pos_index]
    meta = {"layer": best.layer, "pos": best.pos, "pos_index": pos_index, "bypass": best.bypass, "induce": best.induce, "kl": best.kl, "norm": best.norm,
            "method": "Arditi et al. 2024 difference of means, selection by bypass/induce/KL", "model": prov["model_name"]}
    lib.save("refusal", r, meta)
    print(f"selected: layer {best.layer} pos {best.pos}  bypass {best.bypass:.2f} induce {best.induce:.2f} kl {best.kl:.3f}")

    # Baseline refusal score on the validation sets (no intervention) for reference.
    from src.refusal import next_token_logprobs, refusal_score

    base_harm = float(refusal_score(next_token_logprobs(model, tok, sets["harmful_val"], batch_size=args.batch_size), ref_ids).mean())
    base_safe = float(refusal_score(next_token_logprobs(model, tok, sets["harmless_val"], batch_size=args.batch_size), ref_ids).mean())
    write_json(ROOT / "results" / "raw" / f"E02_{args.model}.json", {
        "model": prov["model_name"], "selected": meta, "baseline_refusal_score": {"harmful_val": base_harm, "harmless_val": base_safe},
        "candidates": [c.__dict__ for c in scored], "resid_norms": norms.tolist(), "n_layers": n_layers, "n_pos": n_pos,
        "seconds": time.time() - t0, "provenance": prov,
    })
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
