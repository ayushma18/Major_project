"""Experimental conditions as short strings, turned into (intervention, system prompt).

Grammar (one condition per string):
    baseline                         no intervention
    ablate-refusal                   project the selected refusal direction out everywhere
    add-refusal@L:c                  add c * norm[L] * unit(r[L]) at layer L (induce refusal)
    emotion:<e>@L:c                  add c * norm[L] * unit(v_e[L])         (CAA paired vector)
    centered:<e>@L:c                 same with the one-vs-mean vector of emotion e
    specific:<e>@L:c                 paired vector with the shared "affect" component removed
    common@L:c                       the shared affect component itself (mean of the 19 unit vectors)
    (c may be negative: subtract the vector)
    perp:<e>@L:c                     v_e[L] with its r[L] component removed, rescaled to the same norm
    random:<seed>@L:c                a random unit direction (seeded), same injected norm
    prime:<group>:<i>                system prompt priming.json[group][i]   (no activation change)

L is a decoder-layer index, c the strength as a fraction of the layer's mean residual norm.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .data import priming
from .steering import AblateDirection, AddVector, orthogonalize
from .vectors import VectorLibrary, random_unit_vectors

_PAT = re.compile(r"^(?P<kind>[a-z\-]+)(?::(?P<arg>[^@]+))?(?:@(?P<layer>\d+):(?P<c>-?[0-9.]+))?$")


@dataclass
class Condition:
    name: str
    intervention: Any
    system: str | None
    info: dict


def parse(name: str) -> dict:
    m = _PAT.match(name)
    if not m:
        raise ValueError(f"Bad condition {name!r}")
    d = m.groupdict()
    return {"kind": d["kind"], "arg": d["arg"], "layer": int(d["layer"]) if d["layer"] else None, "c": float(d["c"]) if d["c"] else None}


def build(name: str, model: Any, lib: VectorLibrary) -> Condition:
    p = parse(name)
    kind, arg, layer, c = p["kind"], p["arg"], p["layer"], p["c"]
    if kind == "baseline":
        return Condition(name, None, None, p)
    if kind == "prime":
        group, idx = arg.split(":") if ":" in arg else (arg, "0")
        return Condition(name, None, priming()[group][int(idx)], p)
    if kind == "ablate-refusal":
        r, meta = lib.load("refusal")
        return Condition(name, AblateDirection(model, r), None, {**p, "refusal": meta})

    norms, _ = lib.load("resid_norms")
    scale = c * float(norms[layer])
    v = _direction(kind, arg, layer, model, lib)
    return Condition(name, AddVector(model, layer, v, scale, normalize=True), None, {**p, "injected_norm": scale})


def _direction(kind: str, arg: str | None, layer: int, model: Any, lib: VectorLibrary):
    """The raw (unnormalised) direction behind an additive condition at `layer`."""
    if kind == "add-refusal":
        cand, _ = lib.load("refusal_candidates")
        _, meta = lib.load("refusal")
        v = cand[layer, meta["pos_index"]]
    elif kind == "emotion":
        v = lib.load(f"emotion_caa_{arg}")[0][layer]
    elif kind == "centered":
        v = lib.load(f"emotion_centered_{arg}")[0][layer]
    elif kind == "specific":
        v = lib.load(f"emotion_specific_{arg}")[0][layer]
    elif kind == "common":
        v = lib.load("emotion_common")[0][layer]
    elif kind == "perp":
        cand, _ = lib.load("refusal_candidates")
        _, meta = lib.load("refusal")
        v = orthogonalize(lib.load(f"emotion_caa_{arg}")[0][layer], cand[layer, meta["pos_index"]])
    elif kind == "random":
        v = random_unit_vectors(model.config.hidden_size, 1, seed=int(arg))[0]
    else:
        raise ValueError(f"Unknown condition kind {kind!r}")
    return v


def steering_delta(name: str, model: Any, lib: VectorLibrary):
    """(layer, delta) for an additive condition, delta = c * norm[L] * unit(v).

    For hooks that take a ready vector (the multi-turn protocol); `build` adds the same delta.
    """
    p = parse(name)
    if p["layer"] is None or p["c"] is None or p["kind"] in ("baseline", "prime", "ablate-refusal"):
        raise ValueError(f"{name!r} is not an additive condition")
    norms, _ = lib.load("resid_norms")
    scale = p["c"] * float(norms[p["layer"]])
    v = _direction(p["kind"], p["arg"], p["layer"], model, lib)
    return p["layer"], scale * v / v.norm()
