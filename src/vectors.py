"""The vector library: refusal directions, emotion vectors, random controls.

Every vector the experiments use is stored in `artifacts/vectors/<model-slug>/` as
safetensors plus a JSON card, so any result can be traced back to the exact vector.

    lib = VectorLibrary("llama-8b")
    lib.save("refusal", r, {"layer": 12, "pos": -1, ...})
    r, meta = lib.load("refusal")

Steering strength convention (used by every experiment, so emotions, random vectors and
orthogonalised vectors are compared at EQUAL injected norm):
    injected = c * mean_resid_norm[layer] * unit(v)
where mean_resid_norm[layer] is the average L2 norm of the residual stream at that layer
over harmless chat prompts (`layer_norm_stats`). c = 0.5 means "half a typical activation".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch

from .activations import get_layers
from .generate import build_chat_prompt

ROOT = Path(__file__).resolve().parent.parent


def slug(model_name: str) -> str:
    return model_name.replace("/", "--")


class VectorLibrary:
    def __init__(self, model_name: str):
        self.dir = ROOT / "artifacts" / "vectors" / slug(model_name)
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, tensor: torch.Tensor, meta: dict | None = None) -> None:
        from safetensors.torch import save_file

        save_file({"v": tensor.detach().float().contiguous().cpu()}, str(self.dir / f"{name}.safetensors"))
        (self.dir / f"{name}.json").write_text(json.dumps(meta or {}, indent=2))

    def load(self, name: str) -> tuple[torch.Tensor, dict]:
        from safetensors.torch import load_file

        v = load_file(str(self.dir / f"{name}.safetensors"))["v"]
        meta_path = self.dir / f"{name}.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return v, meta

    def exists(self, name: str) -> bool:
        return (self.dir / f"{name}.safetensors").exists()


@torch.inference_mode()
def layer_norm_stats(model: Any, tokenizer: Any, instructions: Sequence[str], batch_size: int = 16) -> torch.Tensor:
    """Mean L2 norm of resid_post per layer over all real tokens -> [n_layers]."""
    layers = get_layers(model)
    device = next(model.parameters()).device
    sums = torch.zeros(len(layers))
    count = 0
    mask_holder: dict[str, torch.Tensor] = {}

    def make(i):
        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, (tuple, list)) else out
            m = mask_holder["m"]
            sums[i] += float((h.float().norm(dim=-1) * m).sum())
        return hook

    handles = [layer.register_forward_hook(make(i)) for i, layer in enumerate(layers)]
    try:
        prompts = [build_chat_prompt(tokenizer, x) for x in instructions]
        for b in range(0, len(prompts), batch_size):
            enc = tokenizer(prompts[b : b + batch_size], return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.to(device) for k, v in enc.items()}
            m = enc["attention_mask"].float()
            m[:, :1] = 0  # skip BOS: its norm is an outlier in Llama models
            # also skip the first real token of each left-padded row
            first = enc["attention_mask"].float().argmax(dim=1)
            m[torch.arange(m.shape[0]), first] = 0
            mask_holder["m"] = m
            model(**enc, use_cache=False)
            count += int(m.sum())
    finally:
        for h in handles:
            h.remove()
    return sums / max(1, count)


@torch.inference_mode()
def span_means(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    *,
    wrapper: str = "Read this short story.\n\n{text}",
    batch_size: int = 8,
    skip_first: int = 3,
) -> torch.Tensor:
    """Mean resid_post over the tokens of each text inside a chat user turn.

    Each text is wrapped (`wrapper`) and rendered as a user message. Only the tokens of
    the text itself are averaged (the fixed wrapper and template tokens would cancel in a
    pairwise difference anyway, but averaging only the story keeps vectors comparable
    across corpora). The first `skip_first` story tokens are skipped because the very first
    tokens of a span carry little content.

    Returns [n_texts, n_layers, hidden] in fp32.
    """
    layers = get_layers(model)
    device = next(model.parameters()).device
    results: list[torch.Tensor] = []
    holder: dict[str, torch.Tensor] = {}
    store: dict[int, torch.Tensor] = {}

    def make(i):
        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, (tuple, list)) else out
            w = holder["w"]  # [b, seq] weights summing to 1 per row
            store[i] = torch.einsum("bs,bsh->bh", w, h.float())
        return hook

    handles = [layer.register_forward_hook(make(i)) for i, layer in enumerate(layers)]
    try:
        for b in range(0, len(texts), batch_size):
            chunk = texts[b : b + batch_size]
            prompts, spans = [], []
            for t in chunk:
                user = wrapper.format(text=t)
                p = build_chat_prompt(tokenizer, user)
                start_char = p.index(t)
                end_char = start_char + len(t)
                prompts.append(p)
                spans.append((start_char, end_char))
            enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False, return_offsets_mapping=True)
            offsets = enc.pop("offset_mapping")
            w = torch.zeros(enc["input_ids"].shape, dtype=torch.float32)
            for row, (s, e) in enumerate(spans):
                idx = [k for k, (a, z) in enumerate(offsets[row].tolist()) if a >= s and z <= e and z > a and enc["attention_mask"][row, k] == 1]
                idx = idx[skip_first:] if len(idx) > skip_first + 5 else idx
                w[row, idx] = 1.0 / max(1, len(idx))
            holder["w"] = w.to(device)
            enc = {k: v.to(device) for k, v in enc.items()}
            model(**enc, use_cache=False)
            results.append(torch.stack([store[i] for i in range(len(layers))], dim=1).cpu())
            print(f"  read {min(b + batch_size, len(texts))}/{len(texts)} texts", end="\r", flush=True)
    finally:
        for h in handles:
            h.remove()
    print()
    return torch.cat(results)


def random_unit_vectors(hidden: int, n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, hidden, generator=g)
    return v / v.norm(dim=-1, keepdim=True)
