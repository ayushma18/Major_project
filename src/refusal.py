"""The refusal ("guardrail") direction, following Arditi et al. 2024.

Pipeline:
  1. Mean residual-stream activation over harmful and over harmless instructions, at
     every layer and at each post-instruction token position (the chat-template tokens
     between the end of the user message and the assistant's first token).
  2. Candidate direction r[l, p] = mean_harmful - mean_harmless.
  3. Score every candidate on held-out validation prompts:
       bypass  : refusal score on harmful prompts with r ablated everywhere (lower is better)
       induce  : refusal score on harmless prompts with r added at layer l (should be > 0)
       kl      : KL(base || ablated) of the next-token distribution on harmless prompts
                 (ablation should not wreck ordinary behaviour; Arditi keep kl < 0.1)
     and drop candidates in the last 20 percent of layers.
  4. Pick the remaining candidate with the lowest bypass score.

Refusal score (Arditi et al.): log-odds of the probability mass on refusal-start tokens
(for Llama 3, the token "I", as in "I can't") at the first generated position.

Conventions: candidate r[l] lives in the OUTPUT space of decoder layer l (resid_post[l]),
the same space where `steering.AddVector(layer=l)` adds vectors, so "add r at layer l" and
"r extracted at layer l" refer to the same residual-stream location.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from .activations import get_layers
from .generate import build_chat_prompt
from .steering import AblateDirection, AddVector


def post_instruction_positions(tokenizer: Any) -> int:
    """Number of template tokens after the user message (Llama 3: 5)."""
    with_gen = build_chat_prompt(tokenizer, "X")
    ids = tokenizer(with_gen, add_special_tokens=False)["input_ids"]
    x_id = tokenizer("X", add_special_tokens=False)["input_ids"][-1]
    last_x = max(i for i, t in enumerate(ids) if t == x_id)
    return len(ids) - 1 - last_x


def refusal_token_ids(tokenizer: Any, words: Sequence[str] = ("I",)) -> list[int]:
    ids = []
    for w in words:
        t = tokenizer(w, add_special_tokens=False)["input_ids"]
        if t:
            ids.append(t[0])
    return sorted(set(ids))


@torch.inference_mode()
def mean_activations(model: Any, tokenizer: Any, instructions: Sequence[str], n_pos: int, batch_size: int = 16) -> torch.Tensor:
    """Mean resid_post over instructions -> [n_layers, n_pos, hidden] (fp32)."""
    layers = get_layers(model)
    device = next(model.parameters()).device
    total = None
    store: dict[int, torch.Tensor] = {}

    def make(i):
        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, (tuple, list)) else out
            store[i] = h[:, -n_pos:, :].float().sum(dim=0)
        return hook

    handles = [layer.register_forward_hook(make(i)) for i, layer in enumerate(layers)]
    try:
        prompts = [build_chat_prompt(tokenizer, x) for x in instructions]
        for b in range(0, len(prompts), batch_size):
            enc = tokenizer(prompts[b : b + batch_size], return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.to(device) for k, v in enc.items()}
            model(**enc, use_cache=False)
            batch = torch.stack([store[i] for i in range(len(layers))]).cpu()
            total = batch if total is None else total + batch
    finally:
        for h in handles:
            h.remove()
    return total / len(instructions)


@torch.inference_mode()
def next_token_logprobs(model: Any, tokenizer: Any, instructions: Sequence[str], intervention: Any = None, batch_size: int = 16) -> torch.Tensor:
    """Log-probs of the first assistant token -> [n, vocab] (fp32, cpu)."""
    device = next(model.parameters()).device
    prompts = [build_chat_prompt(tokenizer, x) for x in instructions]
    outs = []
    ctx = intervention
    if ctx is not None:
        ctx.__enter__()
    try:
        for b in range(0, len(prompts), batch_size):
            enc = tokenizer(prompts[b : b + batch_size], return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc, use_cache=False).logits[:, -1, :].float()
            outs.append(F.log_softmax(logits, dim=-1).cpu())
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)
    return torch.cat(outs)


def refusal_score(logprobs: torch.Tensor, refusal_ids: Sequence[int]) -> torch.Tensor:
    """Per-prompt log-odds of starting with a refusal token."""
    p = logprobs[:, list(refusal_ids)].exp().sum(dim=-1).clamp(1e-8, 1 - 1e-8)
    return torch.log(p) - torch.log1p(-p)


def kl_div(base: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """Per-prompt KL(base || other) over next-token distributions."""
    return (base.exp() * (base - other)).sum(dim=-1)


@dataclass
class Candidate:
    layer: int
    pos: int  # negative index into the post-instruction tokens, -n_pos .. -1
    bypass: float
    induce: float
    kl: float
    norm: float


def score_candidates(
    model: Any,
    tokenizer: Any,
    directions: torch.Tensor,  # [n_layers, n_pos, hidden]
    harmful_val: Sequence[str],
    harmless_val: Sequence[str],
    refusal_ids: Sequence[int],
    *,
    layers: Sequence[int] | None = None,
    positions: Sequence[int] | None = None,
    batch_size: int = 16,
    progress: bool = True,
) -> list[Candidate]:
    n_layers, n_pos, _ = directions.shape
    layers = list(layers) if layers is not None else list(range(n_layers))
    positions = list(positions) if positions is not None else list(range(n_pos))
    base_harmless = next_token_logprobs(model, tokenizer, harmless_val, batch_size=batch_size)
    out: list[Candidate] = []
    total = len(layers) * len(positions)
    k = 0
    for l in layers:
        for p in positions:
            r = directions[l, p]
            abl_harmful = next_token_logprobs(model, tokenizer, harmful_val, AblateDirection(model, r), batch_size)
            abl_harmless = next_token_logprobs(model, tokenizer, harmless_val, AblateDirection(model, r), batch_size)
            add_harmless = next_token_logprobs(model, tokenizer, harmless_val, AddVector(model, l, r, 1.0), batch_size)
            c = Candidate(
                layer=l,
                pos=p - n_pos,
                bypass=float(refusal_score(abl_harmful, refusal_ids).mean()),
                induce=float(refusal_score(add_harmless, refusal_ids).mean()),
                kl=float(kl_div(base_harmless, abl_harmless).mean()),
                norm=float(r.norm()),
            )
            out.append(c)
            k += 1
            if progress:
                print(f"  candidate {k}/{total}: layer {l:2d} pos {c.pos}  bypass {c.bypass:7.2f}  induce {c.induce:6.2f}  kl {c.kl:.3f}", flush=True)
    return out


def select_direction(cands: Sequence[Candidate], n_layers: int, *, kl_max: float = 0.1, max_layer_frac: float = 0.8) -> Candidate:
    """Arditi et al.'s rule: lowest bypass among candidates that induce refusal,
    keep KL small, and sit in the first 80 percent of layers."""
    ok = [c for c in cands if c.layer < max_layer_frac * n_layers and c.kl < kl_max and c.induce > 0 and math.isfinite(c.bypass)]
    if not ok:
        ok = [c for c in cands if c.layer < max_layer_frac * n_layers]
    return min(ok, key=lambda c: c.bypass)
