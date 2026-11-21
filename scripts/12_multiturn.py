"""E12: the sleeper effect (proposal section 5.3.5, SO4).

Both protocols use the benign 3-turn scripts in data/prompts/multiturn_benign.json and the 159
HarmBench test behaviours: 39 scripts x 4 behaviours = 156 harmful requests (the last 3 are unused).

  stacked   The proposal's protocol. One conversation per script, 7 turns: turns 1-3 benign,
            turns 4-7 four different HarmBench behaviours. Found on Llama 3.1 8B: once the model
            refuses at turn 4 it refuses every later harmful turn (refusal at turns 5-7 is 100%
            with no injection), so turns 5-7 sit at a floor.
  single    One conversation per behaviour: the script's 3 benign turns, then exactly one
            harmful request at turn 4. No earlier refusal in the context, and all 156 requests
            are the same distance from the injection. The benign prefix is generated once per
            script and shared by its 4 behaviours (exact: nothing differs before turn 4).

Conditions for each vector v (a condition-grammar spec, e.g. specific:happy@12:0.5):
    none        no injection anywhere
    early:v     v added on every turn-1 token (prompt and reply). Later turns recompute the whole
                conversation with v still on the turn-1 positions only, which equals keeping the
                KV cache built while turn 1 was steered.
    same:v      v added only on the harmful turn (its user message and its reply)
Persistence (eq. 5.6): delta_persist(t) = ASR_early(t) - ASR_same(t), after judging.
Drift (STAR-style): at every turn, the projection of the last prompt token's residual stream on
the refusal direction at layer+2 and layer+6 (a layer cannot see an injection made above it).

    python scripts/12_multiturn.py --protocol single --layer 12 \\
        --vectors specific:happy@12:0.5 random:2@12:1 --only none early:random:2@12:1
    python scripts/12_multiturn.py --protocol stacked --layer 12 --c 0.5 --emotions happy   # legacy flags

Each condition writes outputs/generations/<model>/<multiturn|multiturn-single>/<condition>.jsonl
(the standard row format, so `06_judge.py --set <dir>` labels it with per-turn ASR) and a
<condition>.drift.json. Conditions already on disk are skipped, so the queue runs one condition
per process (`--only`): MPS keeps cached blocks between generate calls, and a 7-turn condition
left about 16 GB of them behind. Every run rewrites results/raw/E12_<model>_<protocol>.json from
all conditions on disk.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import data  # noqa: E402
from src.activations import get_layers  # noqa: E402
from src.conditions import parse as parse_condition, steering_delta  # noqa: E402
from src.evaluate import is_refusal  # noqa: E402
from src.generate import eos_token_ids  # noqa: E402
from src.runtime import load, read_jsonl, write_json, write_jsonl  # noqa: E402
from src.steering import AddVectorMasked  # noqa: E402
from src.vectors import VectorLibrary, slug  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
N_TURNS = {"stacked": 7, "single": 4}


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


def last_span(text: str, piece: str) -> list[tuple[int, int]]:
    i = text.rindex(piece)
    return [(i, i + len(piece))]


def free_memory(device) -> None:
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()


class Turn:
    """One batched chat turn: optional masked injection, greedy reply, drift projections."""

    def __init__(self, model, tok, drift_dirs: dict[int, torch.Tensor]):
        self.model, self.tok, self.drift_dirs = model, tok, drift_dirs
        self.device = next(model.parameters()).device
        self.eos = eos_token_ids(tok)

    def __call__(self, texts, spans, layer, delta, on_decode, max_new):
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False, return_offsets_mapping=True)
        offs = enc.pop("offset_mapping")
        ctx = None
        if delta is not None and (on_decode or any(spans)):
            mask = torch.zeros(enc["input_ids"].shape, dtype=torch.bool)
            for row, sp in enumerate(spans):
                if sp:
                    for k, (a, z) in enumerate(offs[row].tolist()):
                        if z > a and any(a >= s and z <= e for s, e in sp):
                            mask[row, k] = True
            ctx = AddVectorMasked(self.model, layer, delta, mask, on_decode)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        proj: dict[int, torch.Tensor] = {}

        def make_hook(l):
            def hook(_m, _i, out):
                h = out[0] if isinstance(out, (tuple, list)) else out
                if h.shape[1] > 1:  # prefill only: the last prompt token
                    proj[l] = (h[:, -1, :].float() @ self.drift_dirs[l].to(h.device)).cpu()
            return hook

        handles = [get_layers(self.model)[l].register_forward_hook(make_hook(l)) for l in self.drift_dirs]
        try:
            with torch.inference_mode():
                if ctx:
                    ctx.__enter__()
                gen = self.model.generate(**enc, max_new_tokens=max_new, do_sample=False, eos_token_id=self.eos, pad_token_id=self.tok.pad_token_id)
        finally:
            if ctx:
                ctx.__exit__(None, None, None)
            for h in handles:
                h.remove()
        new = gen[:, enc["input_ids"].shape[1]:].tolist()
        replies = []
        for g in new:
            cut = next((k for k, x in enumerate(g) if x in self.eos), len(g))
            replies.append((self.tok.decode(g[:cut], skip_special_tokens=True).strip() or "(no reply)", cut))
        drift = {l: [float(x) for x in proj[l]] if l in proj else [None] * len(texts) for l in self.drift_dirs}
        del gen, enc, new, proj
        free_memory(self.device)
        return replies, drift


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-8b")
    ap.add_argument("--protocol", default="stacked", choices=list(N_TURNS))
    ap.add_argument("--layer", type=int, required=True, help="injection layer; drift is read at layer+2 and layer+6")
    ap.add_argument("--vectors", nargs="*", default=[], help="condition-grammar specs, e.g. specific:happy@12:0.5 random:2@12:1")
    ap.add_argument("--only", nargs="*", default=None, help="run only these conditions (none, early:<v>, same:<v>; single protocol also early-full:<v>, same-full:<v>)")
    ap.add_argument("--c", type=float, default=None, help="legacy: strength for --emotions and --random-seeds")
    ap.add_argument("--kind", default="emotion", choices=["emotion", "specific", "centered"], help="legacy: vector kind for --emotions")
    ap.add_argument("--emotions", nargs="*", default=[], help="legacy: condition labels are the bare emotion names")
    ap.add_argument("--random-seeds", nargs="*", type=int, default=[], help="legacy: labels random<seed>")
    ap.add_argument("--n-conv", type=int, default=40)
    ap.add_argument("--benign-tokens", type=int, default=96)
    ap.add_argument("--harmful-tokens", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8, help="conversations per generate call (harmful turns)")
    ap.add_argument("--prefix-batch-size", type=int, default=13, help="single protocol: scripts per generate call for the benign prefix")
    args = ap.parse_args()
    t0 = time.time()

    # label -> spec. Legacy labels keep the file names of the first stacked runs (early_happy.jsonl).
    vectors: dict[str, str] = {}
    if (args.emotions or args.random_seeds) and args.c is None:
        ap.error("--emotions and --random-seeds need --c")
    for e in args.emotions:
        vectors[e] = f"{args.kind}:{e}@{args.layer}:{args.c:g}"
    for s in args.random_seeds:
        vectors[f"random{s}"] = f"random:{s}@{args.layer}:{args.c:g}"
    for spec in args.vectors:
        parse_condition(spec)
        vectors[spec] = spec
    conditions = [("none", None, None)] + [(f"early:{k}", k, "early") for k in vectors] + [(f"same:{k}", k, "same") for k in vectors]
    if args.protocol == "single":  # full-turn modes: the template tokens of the turn are steered too
        conditions += [(f"early-full:{k}", k, "early-full") for k in vectors] + [(f"same-full:{k}", k, "same-full") for k in vectors]
    if args.only is not None:
        unknown = set(args.only) - {c[0] for c in conditions}
        if unknown:
            ap.error(f"--only names unknown conditions: {sorted(unknown)}")
        conditions = [c for c in conditions if c[0] in args.only]

    model, tok, prov = load(args.model)
    device = next(model.parameters()).device
    lib = VectorLibrary(prov["model_name"])
    deltas = {k: steering_delta(spec, model, lib) for k, spec in vectors.items()}

    _, rmeta = lib.load("refusal")
    cands, _ = lib.load("refusal_candidates")
    n_layers = model.config.num_hidden_layers
    drift_layers = sorted({min(n_layers - 1, args.layer + 2), min(n_layers - 1, args.layer + 6)})
    drift_dirs = {l: (cands[l, rmeta["pos_index"]] / cands[l, rmeta["pos_index"]].norm()).float() for l in drift_layers}
    turn = Turn(model, tok, drift_dirs)

    convs = data.multiturn_benign()[: args.n_conv]
    behaviors = data.harmbench("test")
    plans = []
    for i, cv in enumerate(convs):
        hs = behaviors[i * 4 : i * 4 + 4]
        if len(hs) == 4:  # 159 behaviours fill 39 conversations of 4; the leftover 3 are not used
            plans.append({"id": cv["id"], "benign": [cv["turn1"], cv["turn2"], cv["turn3"]], "harmful": hs})
    n_rows = 4 * len(plans)

    outdir = ROOT / "outputs" / "generations" / slug(prov["model_name"]) / ("multiturn" if args.protocol == "stacked" else "multiturn-single")
    outdir.mkdir(parents=True, exist_ok=True)
    n_turns = N_TURNS[args.protocol]

    def row(h, reply, cut, t, conv_id, cname):
        return {"behavior_id": h.id, "behavior": h.behavior, "category": h.category, "context": h.context,
                "completion": reply, "refusal_substring": is_refusal(reply), "turn": t, "conversation": conv_id,
                "condition": cname, "n_tokens": cut}

    def run_stacked(cname, label, mode):
        layer, delta = deltas[label] if label else (None, None)
        rows_out, drift = [], {l: [[] for _ in range(n_turns)] for l in drift_layers}
        for b in range(0, len(plans), args.batch_size):
            batch = plans[b : b + args.batch_size]
            msgs = [[] for _ in batch]
            turn1: list[list[str]] = [[] for _ in batch]
            for t in range(n_turns):
                is_harm = t >= 3
                # The chat template trims message text, so trim here too: spans are found by text match.
                users = [(p["benign"][t] if not is_harm else data.harmbench_prompt(p["harmful"][t - 3])).strip() for p in batch]
                for m, u in zip(msgs, users):
                    m.append({"role": "user", "content": u})
                texts = [render(tok, m) for m in msgs]
                if mode == "early":
                    spans = [char_spans(x, turn1[r] if t > 0 else [users[r]]) for r, x in enumerate(texts)]
                    on_decode = t == 0
                elif mode == "same":
                    spans = [last_span(x, users[r]) if is_harm else [] for r, x in enumerate(texts)]
                    on_decode = is_harm
                else:
                    spans, on_decode = [[] for _ in texts], False
                replies, dr = turn(texts, spans, layer, delta, on_decode, args.harmful_tokens if is_harm else args.benign_tokens)
                for r, (reply, cut) in enumerate(replies):
                    msgs[r].append({"role": "assistant", "content": reply})
                    if t == 0:
                        turn1[r] = [users[r], reply]
                    for l in drift_layers:
                        drift[l][t].append(dr[l][r])
                    if is_harm:
                        rows_out.append(row(batch[r]["harmful"][t - 3], reply, cut, t + 1, batch[r]["id"], cname))
                print(f"  conv {b + 1}-{b + len(batch)} turn {t + 1} done", flush=True)
        return rows_out, {str(l): [mean(x) for x in drift[l]] for l in drift_layers}

    def turn1_block(text: str, t1: list[str]) -> list[tuple[int, int]]:
        """One span from the turn-1 user text to the end of the turn-1 reply, template tokens included."""
        sp = char_spans(text, t1)
        return [(sp[0][0], sp[-1][1])]

    def benign_prefix(label, full=False):
        """Turns 1-3 for every script, with v on turn 1 (early) or no injection (label None).

        full=False steers the turn-1 user text and reply only; full=True steers every turn-1 position
        from the user text to the end of the reply, including the chat-template tokens in between
        (during turn-1 generation: the user text through the end of the prompt, then every reply token)."""
        name = "none" if label is None else ("full_" if full else "") + label.replace(":", "_")
        path = outdir / f"prefix_{name}.json"
        if path.exists():
            d = json.loads(path.read_text())
            if d["plans"] == [p["id"] for p in plans]:
                return d["prefix"]
        layer, delta = deltas[label] if label else (None, None)
        prefix = []
        for b in range(0, len(plans), args.prefix_batch_size):
            batch = plans[b : b + args.prefix_batch_size]
            msgs = [[] for _ in batch]
            turn1: list[list[str]] = [[] for _ in batch]
            drift = [{str(l): [] for l in drift_layers} for _ in batch]
            for t in range(3):
                users = [p["benign"][t].strip() for p in batch]
                for m, u in zip(msgs, users):
                    m.append({"role": "user", "content": u})
                texts = [render(tok, m) for m in msgs]
                if not label:
                    spans = [[] for _ in texts]
                elif full:
                    spans = [turn1_block(x, turn1[r]) if t > 0 else [(x.rindex(users[r]), len(x))] for r, x in enumerate(texts)]
                else:
                    spans = [char_spans(x, turn1[r] if t > 0 else [users[r]]) for r, x in enumerate(texts)]
                replies, dr = turn(texts, spans, layer, delta, label is not None and t == 0, args.benign_tokens)
                for r, (reply, _) in enumerate(replies):
                    msgs[r].append({"role": "assistant", "content": reply})
                    if t == 0:
                        turn1[r] = [users[r], reply]
                    for l in drift_layers:
                        drift[r][str(l)].append(dr[l][r])
                print(f"  prefix {name}: scripts {b + 1}-{b + len(batch)} turn {t + 1} done", flush=True)
            prefix += [{"msgs": m, "turn1": t1, "drift": d} for m, t1, d in zip(msgs, turn1, drift)]
        write_json(path, {"label": label, "full": full, "plans": [p["id"] for p in plans], "prefix": prefix})
        return prefix

    def run_single(cname, label, mode):
        prefix = benign_prefix(label if mode in ("early", "early-full") else None, full=mode == "early-full")
        layer, delta = deltas[label] if label else (None, None)
        items = [(i, j) for i in range(len(plans)) for j in range(4)]
        rows_out, drift4 = [], {l: [] for l in drift_layers}
        for b in range(0, len(items), args.batch_size):
            chunk = items[b : b + args.batch_size]
            users = [data.harmbench_prompt(plans[i]["harmful"][j]).strip() for i, j in chunk]
            texts = [render(tok, prefix[i]["msgs"] + [{"role": "user", "content": u}]) for (i, _), u in zip(chunk, users)]
            if mode == "early":
                spans, on_decode = [char_spans(x, prefix[i]["turn1"]) for (i, _), x in zip(chunk, texts)], False
            elif mode == "early-full":
                spans, on_decode = [turn1_block(x, prefix[i]["turn1"]) for (i, _), x in zip(chunk, texts)], False
            elif mode == "same":
                spans, on_decode = [last_span(x, u) for x, u in zip(texts, users)], True
            elif mode == "same-full":  # the request and every template token after it, as in single-turn steering
                spans, on_decode = [[(x.rindex(u), len(x))] for x, u in zip(texts, users)], True
            else:
                spans, on_decode = [[] for _ in texts], False
            replies, dr = turn(texts, spans, layer, delta, on_decode, args.harmful_tokens)
            for r, ((i, j), (reply, cut)) in enumerate(zip(chunk, replies)):
                rows_out.append(row(plans[i]["harmful"][j], reply, cut, 4, f"{plans[i]['id']}-{j + 1}", cname))
                for l in drift_layers:
                    drift4[l].append(dr[l][r])
            print(f"  harmful turn: {b + len(chunk)}/{len(items)} done", flush=True)
        drift = {str(l): [mean([p["drift"][str(l)][t] for p in prefix]) for t in range(3)] + [mean(drift4[l])] for l in drift_layers}
        return rows_out, drift

    for cname, label, mode in conditions:
        stem = cname.replace(":", "_")
        target, meta_path = outdir / f"{stem}.jsonl", outdir / f"{stem}.drift.json"
        meta = {"condition": cname, "mode": mode, "vector": vectors.get(label), "protocol": args.protocol, "layer": args.layer,
                "c": parse_condition(vectors[label])["c"] if label else None}
        if target.exists() and len(read_jsonl(target)) >= n_rows - 1:
            print(f"skip {cname} (exists)")
            if not meta_path.exists():  # first stacked runs: no drift saved, keep the metadata
                write_json(meta_path, {**meta, "drift_mean_by_turn": None})
            continue
        print(f"== {cname}", flush=True)
        tc = time.time()
        rows_out, drift = (run_stacked if args.protocol == "stacked" else run_single)(cname, label, mode)
        write_jsonl(target, rows_out)
        write_json(meta_path, {**meta, "drift_layers": drift_layers, "drift_mean_by_turn": drift, "seconds": time.time() - tc})
        by_turn = {t: mean([r["refusal_substring"] for r in rows_out if r["turn"] == t]) for t in sorted({r["turn"] for r in rows_out})}
        print(f"   refusal phrase rate by turn {by_turn}  ({time.time() - tc:.0f}s)", flush=True)
        free_memory(device)

    # Merged results file from everything on disk for this protocol.
    merged = {}
    for f in sorted(outdir.glob("*.jsonl")):
        if f.name.endswith(".labels.jsonl"):
            continue
        rows = read_jsonl(f)
        mp = outdir / f"{f.stem}.drift.json"
        meta = json.loads(mp.read_text()) if mp.exists() else {"condition": rows[0]["condition"] if rows else f.stem}
        turns = sorted({r["turn"] for r in rows})
        meta["n_rows"] = len(rows)
        meta["refusal_substring_by_turn"] = {str(t): mean([r["refusal_substring"] for r in rows if r["turn"] == t]) for t in turns}
        merged[f.stem] = meta
    write_json(ROOT / "results" / "raw" / f"E12_{args.model}_{args.protocol}.json", {
        "model": prov["model_name"], "protocol": args.protocol, "layer": args.layer, "drift_layers": drift_layers,
        "n_turns": n_turns, "n_scripts": len(plans), "n_harmful": n_rows, "conditions": merged,
    })
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
