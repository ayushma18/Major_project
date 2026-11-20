"""E05: do the emotion vectors actually steer emotion? (causal check, benign prompts only)

    python scripts/08_steering_sanity.py --model llama-8b --layers 8 12 16 --coeffs 0.25 0.5 1.0

For every emotion, layer and strength c, the vector (c * mean residual norm * unit v_e) is
added while the model answers 12 open benign prompts ("Describe your morning so far.").
Random unit vectors at the same norm are the control. Each output is then read by the
UNSTEERED model, which is asked which of 20 emotion words (19 emotions plus neutral) best
describes its tone (20-way choice, scored from next-token probabilities over the option
numbers), and whether the tone is positive, negative or neutral. Coherence = mean token
negative log-likelihood of the output under the unsteered model (higher = less fluent).

Writes results/raw/E05_<model>.json and benign example outputs for the interface.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import data  # noqa: E402
from src.generate import build_chat_prompt, eos_token_ids  # noqa: E402
from src.runtime import load, write_json  # noqa: E402
from src.steering import AddVectorRows  # noqa: E402
from src.vectors import VectorLibrary, random_unit_vectors  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
BENIGN = __import__("importlib").import_module("05_run_conditions").BENIGN_FEEL
OPTIONS = data.EMOTIONS + ["neutral"]
VALENCE_WORDS = ["positive", "negative", "neutral"]


def judge_prompt(text: str) -> str:
    opts = "\n".join(f"{i + 1}. {w}" for i, w in enumerate(OPTIONS))
    return (f"Here is a short text.\n\n<text>\n{text}\n</text>\n\nWhich ONE of these words best describes the emotional tone "
            f"of the text?\n{opts}\n\nAnswer with the number only.")


def valence_prompt(text: str) -> str:
    return f"Here is a short text.\n\n<text>\n{text}\n</text>\n\nIs the emotional tone of the text positive, negative or neutral? Answer with one word."


@torch.inference_mode()
def first_token_probs(model, tok, users: list[str], option_ids: list[int], bs: int = 16) -> torch.Tensor:
    device = next(model.parameters()).device
    out = []
    for b in range(0, len(users), bs):
        enc = tok([build_chat_prompt(tok, u) for u in users[b : b + bs]], return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc, use_cache=False).logits[:, -1, :].float()
        out.append(logits.softmax(-1)[:, option_ids].cpu())
    return torch.cat(out)


@torch.inference_mode()
def mean_nll(model, tok, prompts: list[str], texts: list[str], bs: int = 16) -> list[float]:
    """Mean NLL of each text as the assistant reply to its prompt, unsteered."""
    device = next(model.parameters()).device
    res = []
    for b in range(0, len(texts), bs):
        ps = [build_chat_prompt(tok, p) for p in prompts[b : b + bs]]
        full = [p + t for p, t in zip(ps, texts[b : b + bs])]
        enc = tok(full, return_tensors="pt", padding=True, add_special_tokens=False)
        plen = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in ps]
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc, use_cache=False).logits.float()
        logp = logits[:, :-1].log_softmax(-1)
        tgt = enc["input_ids"][:, 1:]
        lp = logp.gather(-1, tgt[..., None])[..., 0]
        seq = enc["attention_mask"].shape[1]
        for row in range(len(full)):
            pad = int((enc["attention_mask"][row] == 0).sum())
            start = pad + plen[row] - 1
            vals = lp[row, start:]
            res.append(float(-vals.mean()) if vals.numel() else float("nan"))
    return res


def repetition(text: str) -> float:
    w = text.split()
    bi = list(zip(w, w[1:]))
    return 1 - len(set(bi)) / max(1, len(bi))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-8b")
    ap.add_argument("--layers", nargs="+", type=int, default=[8, 12, 16])
    ap.add_argument("--coeffs", nargs="+", type=float, default=[0.25, 0.5, 1.0])
    ap.add_argument("--vector", default="caa", choices=["caa", "centered"])
    ap.add_argument("--n-random", type=int, default=3)
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    t0 = time.time()
    model, tok, prov = load(args.model)
    lib = VectorLibrary(prov["model_name"])
    norms, _ = lib.load("resid_norms")
    emotions = [e for e in data.EMOTIONS if lib.exists(f"emotion_{args.vector}_{e}")]
    rand = random_unit_vectors(model.config.hidden_size, args.n_random, seed=1234)
    device = next(model.parameters()).device
    eos = eos_token_ids(tok)

    rows = []  # one per generation
    for L in args.layers:
        specs = []
        for c in args.coeffs:
            scale = c * float(norms[L])
            for e in emotions:
                v = lib.load(f"emotion_{args.vector}_{e}")[0][L]
                specs += [(L, c, e, scale * v / v.norm(), p) for p in BENIGN]
            for k in range(args.n_random):
                specs += [(L, c, f"random{k}", scale * rand[k], p) for p in BENIGN]
        specs += [(L, 0.0, "none", torch.zeros(model.config.hidden_size), p) for p in BENIGN]
        for b in range(0, len(specs), args.batch_size):
            chunk = specs[b : b + args.batch_size]
            enc = tok([build_chat_prompt(tok, s[4]) for s in chunk], return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.inference_mode(), AddVectorRows(model, L, torch.stack([s[3] for s in chunk])):
                gen = model.generate(**enc, max_new_tokens=args.new_tokens, do_sample=False, eos_token_id=eos, pad_token_id=tok.pad_token_id)
            for s, g in zip(chunk, gen[:, enc["input_ids"].shape[1]:].tolist()):
                cut = next((k for k, t in enumerate(g) if t in eos), len(g))
                rows.append({"layer": s[0], "c": s[1], "vector": s[2], "prompt": s[4], "text": tok.decode(g[:cut], skip_special_tokens=True).strip()})
            print(f"  layer {L}: {min(b + args.batch_size, len(specs))}/{len(specs)}", end="\r", flush=True)
        print()

    # Read every output with the unsteered model.
    opt_ids = [tok(str(i + 1), add_special_tokens=False)["input_ids"][0] for i in range(len(OPTIONS))]
    assert len(set(opt_ids)) == len(opt_ids), "option numbers must be single distinct tokens"
    val_ids = [tok(w.capitalize(), add_special_tokens=False)["input_ids"][0] for w in VALENCE_WORDS]
    texts = [r["text"] or "(empty)" for r in rows]
    P = first_token_probs(model, tok, [judge_prompt(t) for t in texts], opt_ids)
    V = first_token_probs(model, tok, [valence_prompt(t) for t in texts], val_ids)
    nll = mean_nll(model, tok, [r["prompt"] for r in rows], texts)
    for r, p, v, n in zip(rows, P, V, nll):
        p = p / p.sum()
        v = v / v.sum()
        r["probs"] = p.tolist()
        r["pred"] = OPTIONS[int(p.argmax())]
        r["p_target"] = float(p[OPTIONS.index(r["vector"])]) if r["vector"] in OPTIONS else None
        r["valence"] = {w: float(x) for w, x in zip(VALENCE_WORDS, v)}
        r["nll"] = n
        r["repetition"] = repetition(r["text"])

    # Aggregate.
    agg = {}
    for r in rows:
        key = f"{r['vector']}|{r['layer']}|{r['c']}"
        a = agg.setdefault(key, {"vector": r["vector"], "layer": r["layer"], "c": r["c"], "n": 0, "top1": 0, "p_target": 0.0, "nll": 0.0, "rep": 0.0, "pos": 0.0, "neg": 0.0, "pred_counts": {}})
        a["n"] += 1
        a["top1"] += int(r["pred"] == r["vector"])
        a["p_target"] += r["p_target"] or 0.0
        a["nll"] += r["nll"]
        a["rep"] += r["repetition"]
        a["pos"] += r["valence"]["positive"]
        a["neg"] += r["valence"]["negative"]
        a["pred_counts"][r["pred"]] = a["pred_counts"].get(r["pred"], 0) + 1
    for a in agg.values():
        for k in ("top1", "p_target", "nll", "rep", "pos", "neg"):
            a[k] /= a["n"]
    # Baseline p(target) for each emotion = how often the UNSTEERED outputs already read as that emotion.
    base = [r for r in rows if r["vector"] == "none"]
    base_p = {e: sum(r["probs"][OPTIONS.index(e)] for r in base) / len(base) for e in emotions}
    examples = {}
    for r in rows:
        if r["vector"] in emotions and r["prompt"] == BENIGN[0]:
            examples.setdefault(r["vector"], []).append({"layer": r["layer"], "c": r["c"], "text": r["text"]})
    examples["none"] = [{"layer": r["layer"], "c": 0.0, "text": r["text"]} for r in base if r["prompt"] == BENIGN[0]][:1]
    write_json(ROOT / "results" / "raw" / f"E05_{args.model}.json", {
        "model": prov["model_name"], "vector": args.vector, "layers": args.layers, "coeffs": args.coeffs, "emotions": emotions, "options": OPTIONS,
        "aggregate": list(agg.values()), "baseline_p_target": base_p, "examples": examples, "seconds": time.time() - t0,
    })
    write_json(ROOT / "outputs" / "E05_rows.json", rows)
    for L in args.layers:
        for c in args.coeffs:
            em = [a for a in agg.values() if a["layer"] == L and a["c"] == c and a["vector"] in emotions]
            rn = [a for a in agg.values() if a["layer"] == L and a["c"] == c and a["vector"].startswith("random")]
            print(f"layer {L:2d} c {c:.2f}: emotion top1 {sum(a['top1'] for a in em) / len(em):.2f}  p_target {sum(a['p_target'] for a in em) / len(em):.2f}  nll {sum(a['nll'] for a in em) / len(em):.2f}  | random nll {sum(a['nll'] for a in rn) / max(1, len(rn)):.2f}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
