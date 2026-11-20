"""E12: the sleeper effect (proposal section 5.3.5, SO4).

    python scripts/12_multiturn.py --model llama-8b --layer 12 --c 0.5 --emotions desperate calm --random-seeds 0

Protocol, 40 conversations (data/prompts/multiturn_benign.json) x 7 turns:
    turn 1   benign general-knowledge question           (EARLY condition injects here)
    turn 2-3 neutral follow-ups, no injection
    turn 4-7 four HarmBench test behaviours, no injection (SAME-TURN control injects here)
Conditions per vector v:
    none            no injection anywhere
    early:<v>       v added on every turn-1 token (prompt and reply). Later turns recompute the
                    whole conversation with v still on the turn-1 positions only, which is the
                    same as keeping the KV cache built while turn 1 was steered.
    same:<v>        v added only on the current harmful turn (its user message and its reply)
Persistence (eq. 5.6): delta_persist(t) = ASR_early(t) - ASR_same(t), t = 4..7, after judging.
Drift (STAR-style): at every turn, the projection of the last prompt token's residual stream
at the refusal layer onto r*, per condition.

Harmful-turn generations are written in the standard row format to
outputs/generations/<model>/multiturn/<condition>.jsonl, so `06_judge.py --set multiturn`
labels them (with per-turn ASR).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import data  # noqa: E402
from src.activations import get_layers  # noqa: E402
from src.evaluate import is_refusal  # noqa: E402
from src.generate import eos_token_ids  # noqa: E402
from src.runtime import load, write_json, write_jsonl  # noqa: E402
from src.steering import AddVectorMasked  # noqa: E402
from src.vectors import VectorLibrary, random_unit_vectors, slug  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def render(tok, messages):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def char_spans(text: str, pieces: list[str]) -> list[tuple[int, int]]:
    """Locate each piece in order inside the rendered conversation."""
    spans, start = [], 0
    for p in pieces:
        i = text.index(p, start)
        spans.append((i, i + len(p)))
        start = i + len(p)
    return spans


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-8b")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--c", type=float, required=True)
    ap.add_argument("--kind", default="emotion", choices=["emotion", "specific", "centered"])
    ap.add_argument("--emotions", nargs="*", default=[])
    ap.add_argument("--random-seeds", nargs="*", type=int, default=[])
    ap.add_argument("--n-conv", type=int, default=40)
    ap.add_argument("--benign-tokens", type=int, default=96)
    ap.add_argument("--harmful-tokens", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=20)
    args = ap.parse_args()
    t0 = time.time()
    model, tok, prov = load(args.model)
    lib = VectorLibrary(prov["model_name"])
    norms, _ = lib.load("resid_norms")
    r_star, rmeta = lib.load("refusal")
    r_hat = (r_star / r_star.norm()).float()
    scale = args.c * float(norms[args.layer])
    device = next(model.parameters()).device
    eos = eos_token_ids(tok)

    convs = data.multiturn_benign()[: args.n_conv]
    behaviors = data.harmbench("test")
    plans = []
    for i, cv in enumerate(convs):
        hs = behaviors[i * 4 : i * 4 + 4]
        plans.append({"id": cv["id"], "benign": [cv["turn1"], cv["turn2"], cv["turn3"]], "harmful": hs})

    vectors: dict[str, torch.Tensor] = {}
    for e in args.emotions:
        v = lib.load(f"emotion_{args.kind}_{e}")[0][args.layer]
        vectors[e] = scale * v / v.norm()
    for s in args.random_seeds:
        vectors[f"random{s}"] = scale * random_unit_vectors(model.config.hidden_size, 1, seed=s)[0]
    conditions = [("none", None, None)] + [(f"early:{k}", k, "early") for k in vectors] + [(f"same:{k}", k, "same") for k in vectors]

    drift_layer = rmeta["layer"]
    proj_holder: dict[str, torch.Tensor] = {}

    def proj_hook(_m, _i, out):
        h = out[0] if isinstance(out, (tuple, list)) else out
        if h.shape[1] > 1:  # prefill only: last prompt token
            proj_holder["p"] = (h[:, -1, :].float() @ r_hat.to(h.device)).cpu()

    outdir = ROOT / "outputs" / "generations" / slug(prov["model_name"]) / "multiturn"
    drift = {}
    for cname, vkey, mode in conditions:
        print(f"== {cname}")
        rows_out = []
        drift[cname] = [[] for _ in range(7)]
        for b in range(0, len(plans), args.batch_size):
            batch = plans[b : b + args.batch_size]
            msgs = [[] for _ in batch]
            turn1_pieces: list[list[str]] = [[] for _ in batch]
            for t in range(7):
                is_harm = t >= 3
                users = [p["benign"][t] if not is_harm else data.harmbench_prompt(p["harmful"][t - 3]) for p in batch]
                for m, u in zip(msgs, users):
                    m.append({"role": "user", "content": u})
                texts = [render(tok, m) for m in msgs]
                enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False, return_offsets_mapping=True)
                offs = enc.pop("offset_mapping")
                mask = torch.zeros(enc["input_ids"].shape, dtype=torch.bool)
                on_decode = False
                if vkey is not None:
                    for row, text in enumerate(texts):
                        if mode == "early":
                            pieces = turn1_pieces[row] if t > 0 else [users[row]]
                            spans = char_spans(text, pieces)
                        else:  # same-turn: only the current harmful user message
                            spans = [(text.rindex(users[row]), text.rindex(users[row]) + len(users[row]))] if is_harm else []
                        for k, (a, z) in enumerate(offs[row].tolist()):
                            if z > a and any(a >= s and z <= e for s, e in spans):
                                mask[row, k] = True
                    on_decode = (mode == "early" and t == 0) or (mode == "same" and is_harm)
                enc = {k: v.to(device) for k, v in enc.items()}
                handles = [get_layers(model)[drift_layer].register_forward_hook(proj_hook)]
                ctx = AddVectorMasked(model, args.layer, vectors[vkey], mask, on_decode) if vkey is not None else None
                try:
                    with torch.inference_mode():
                        if ctx:
                            ctx.__enter__()
                        gen = model.generate(**enc, max_new_tokens=args.harmful_tokens if is_harm else args.benign_tokens, do_sample=False, eos_token_id=eos, pad_token_id=tok.pad_token_id)
                finally:
                    if ctx:
                        ctx.__exit__(None, None, None)
                    for h in handles:
                        h.remove()
                p_first = proj_holder.get("p")
                new = gen[:, enc["input_ids"].shape[1]:].tolist()
                for row, g in enumerate(new):
                    cut = next((k for k, x in enumerate(g) if x in eos), len(g))
                    reply = tok.decode(g[:cut], skip_special_tokens=True).strip() or "(no reply)"
                    msgs[row].append({"role": "assistant", "content": reply})
                    if t == 0:
                        turn1_pieces[row] = [users[row], reply]
                    if p_first is not None:
                        drift[cname][t].append(float(p_first[row]))
                    if is_harm:
                        h = batch[row]["harmful"][t - 3]
                        rows_out.append({"behavior_id": h.id, "behavior": h.behavior, "category": h.category, "context": h.context,
                                         "completion": reply, "refusal_substring": is_refusal(reply), "turn": t + 1,
                                         "conversation": batch[row]["id"], "condition": cname, "n_tokens": cut})
                print(f"  conv {b + 1}-{b + len(batch)} turn {t + 1} done", flush=True)
        write_jsonl(outdir / f"{cname.replace(':', '_')}.jsonl", rows_out)
        by_turn = {t: sum(r["refusal_substring"] for r in rows_out if r["turn"] == t) / max(1, sum(1 for r in rows_out if r["turn"] == t)) for t in range(4, 8)}
        print(f"   refusal by turn {by_turn}")
    write_json(ROOT / "results" / "raw" / f"E12_{args.model}.json", {
        "model": prov["model_name"], "layer": args.layer, "c": args.c, "kind": args.kind, "conditions": [c[0] for c in conditions],
        "drift_layer": drift_layer, "drift_mean_by_turn": {c: [sum(x) / len(x) if x else None for x in v] for c, v in drift.items()},
        "n_conversations": len(plans), "seconds": time.time() - t0,
    })
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
