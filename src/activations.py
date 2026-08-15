"""Capture and save hidden activations from a causal LM.

Why forward hooks rather than `output_hidden_states=True`:

`output_hidden_states` only gives the residual stream *between* decoder layers.
The later stages of this project (emotion vectors, refusal directions, SAE
training) need to be able to read the attention and MLP contributions
separately, and to write *into* the stream at a chosen site. Hooks give all of
that from one mechanism; `output_hidden_states` gives one site read-only.

Capture happens on a single forward pass over the prompt -- the prefill -- not
during generation. The activation at the final prompt token is the standard
readout point for steering-vector and refusal-direction work: it is the position
whose residual stream determines the first generated token.

Sites available per layer:
    resid_post   output of the whole decoder layer (the residual stream)
    attn_out     output of self-attention, before it is added to the stream
    mlp_out      output of the MLP, before it is added to the stream
    resid_mid    the stream after attention but before the MLP block

Per layer the identity  resid_post == resid_mid + mlp_out  holds exactly, and
tests/test_activations.py asserts it.

IMPORTANT -- the final layer. `resid_post` for the last layer is the *raw*
residual stream. It deliberately differs from `output_hidden_states[-1]`, which
transformers passes through the model's final RMSNorm first. Raw is the correct
choice here: steering vectors are added to the residual stream, so directions
must be computed in that same un-normalised space. For every other layer i,
`resid_post[i]` equals `output_hidden_states[i + 1]` exactly.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

log = logging.getLogger(__name__)

SITES = ("resid_post", "attn_out", "mlp_out", "resid_mid")
TOKEN_POSITIONS = ("last", "all", "mean")


# --------------------------------------------------------------------------- #
# Locating the layer stack
# --------------------------------------------------------------------------- #


def get_layers(model: Any) -> torch.nn.ModuleList:
    """Find the decoder layer stack.

    Llama exposes it at `model.model.layers`. We probe a few known paths rather
    than hard-coding one, so a config change surfaces as a clear error here
    instead of an AttributeError deep inside a hook.
    """
    for path in ("model.layers", "model.model.layers", "transformer.h", "model.decoder.layers"):
        obj: Any = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if isinstance(obj, torch.nn.ModuleList) and len(obj) > 0:
            return obj
    raise AttributeError(
        f"Could not locate the decoder layer stack on {type(model).__name__}. "
        "Add its path to get_layers()."
    )


def _as_tensor(output: Any) -> torch.Tensor:
    """Normalise a module output to a tensor.

    Decoder layers and attention modules return a bare tensor in some
    transformers versions and a tuple in others; the hidden states are always
    element 0 when it is a tuple.
    """
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Cannot extract a tensor from module output of type {type(output)}")


# --------------------------------------------------------------------------- #
# Capture
# --------------------------------------------------------------------------- #


@dataclass
class CaptureConfig:
    """What to record.

    `store_dtype` is float32 on purpose. Activations are captured in the model's
    compute dtype (fp16/bf16) but every downstream operation -- mean differences,
    SVD, SAE training -- is numerically happier in fp32, and the storage cost is
    negligible next to the weights.
    """

    sites: tuple[str, ...] = ("resid_post",)
    layers: tuple[int, ...] | None = None  # None = every layer
    token_position: str = "last"
    store_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        bad = set(self.sites) - set(SITES)
        if bad:
            raise ValueError(f"Unknown site(s) {sorted(bad)}; valid sites are {SITES}")
        if self.token_position not in TOKEN_POSITIONS:
            raise ValueError(
                f"Unknown token_position {self.token_position!r}; valid: {TOKEN_POSITIONS}"
            )


class ActivationCapture:
    """Context manager that records activations during a forward pass.

        with ActivationCapture(model, CaptureConfig(sites=("resid_post",))) as cap:
            model(**inputs)
        acts = cap.stacked()          # {"resid_post": [n_layers, batch, hidden]}

    Hooks are always removed on exit, including on exception -- a leaked hook
    would silently corrupt every subsequent forward pass in the process.
    """

    def __init__(self, model: Any, config: CaptureConfig | None = None):
        self.model = model
        self.config = config or CaptureConfig()
        self.layers = get_layers(model)
        self.n_layers = len(self.layers)

        selected = self.config.layers
        if selected is None:
            self.selected_layers: tuple[int, ...] = tuple(range(self.n_layers))
        else:
            for i in selected:
                if not -self.n_layers <= i < self.n_layers:
                    raise IndexError(f"Layer {i} out of range for a {self.n_layers}-layer model")
            self.selected_layers = tuple(i % self.n_layers for i in selected)

        self._handles: list[Any] = []
        # {site: {layer_index: tensor}}
        self._buffer: dict[str, dict[int, torch.Tensor]] = {s: {} for s in self.config.sites}
        self._attention_mask: torch.Tensor | None = None

    # -- position reduction ------------------------------------------------- #

    def set_attention_mask(self, mask: torch.Tensor | None) -> None:
        """Give the capture the attention mask so it can ignore padding.

        Required for token_position 'all' and 'mean'. Not required for 'last':
        the tokenizer is configured for left padding, so index -1 is the final
        real token of every sequence in the batch.
        """
        self._attention_mask = mask

    def _reduce(self, hidden: torch.Tensor) -> torch.Tensor:
        """[batch, seq, hidden] -> the positions we want to keep."""
        pos = self.config.token_position
        if pos == "last":
            return hidden[:, -1, :]  # valid because padding_side='left'
        if pos == "all":
            return hidden
        # mean over real tokens only
        if self._attention_mask is None:
            return hidden.mean(dim=1)
        mask = self._attention_mask.to(hidden.dtype).unsqueeze(-1)  # [batch, seq, 1]
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    # -- hooks -------------------------------------------------------------- #

    def _make_hook(self, site: str, layer_idx: int):
        def hook(_module, inputs, output):
            if site == "resid_mid":
                # Read the *input* to post_attention_layernorm. That is the raw
                # residual stream after attention has been added and before the
                # MLP block. Note this is NOT the MLP's own input, which is the
                # normalised version -- hooking the MLP here would silently give
                # a post-layernorm tensor of identical shape.
                tensor = inputs[0] if isinstance(inputs, tuple) else inputs
            else:
                tensor = _as_tensor(output)
            reduced = self._reduce(tensor.detach())
            self._buffer[site][layer_idx] = reduced.to(self.config.store_dtype).cpu()

        return hook

    def _register(self) -> None:
        for site in self.config.sites:
            for idx in self.selected_layers:
                layer = self.layers[idx]
                if site == "resid_post":
                    target = layer
                elif site == "attn_out":
                    target = layer.self_attn
                elif site == "mlp_out":
                    target = layer.mlp
                elif site == "resid_mid":
                    target = layer.post_attention_layernorm
                else:  # pragma: no cover - guarded in CaptureConfig
                    raise ValueError(site)
                self._handles.append(target.register_forward_hook(self._make_hook(site, idx)))

    def clear(self) -> None:
        self._buffer = {s: {} for s in self.config.sites}

    def __enter__(self) -> ActivationCapture:
        self.clear()
        self._register()
        return self

    def __exit__(self, *exc: object) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # -- results ------------------------------------------------------------ #

    def stacked(self) -> dict[str, torch.Tensor]:
        """{site: tensor}, layer-major.

        Shapes: [n_selected_layers, batch, hidden] for 'last'/'mean',
                [n_selected_layers, batch, seq, hidden] for 'all'.
        Layer-major ordering means a single layer's activations across the whole
        dataset are one contiguous slice, which is what the analysis stages read.
        """
        out: dict[str, torch.Tensor] = {}
        for site, per_layer in self._buffer.items():
            if not per_layer:
                continue
            out[site] = torch.stack([per_layer[i] for i in self.selected_layers], dim=0)
        return out


# --------------------------------------------------------------------------- #
# Batched collection over a prompt set
# --------------------------------------------------------------------------- #


@dataclass
class ActivationSet:
    """Captured activations plus everything needed to interpret them."""

    activations: dict[str, torch.Tensor]  # site -> [n_layers, n_prompts, hidden]
    layer_indices: tuple[int, ...]
    prompts: list[dict]
    config: CaptureConfig
    provenance: dict = field(default_factory=dict)

    def site(self, name: str) -> torch.Tensor:
        if name not in self.activations:
            raise KeyError(f"Site {name!r} was not captured; have {sorted(self.activations)}")
        return self.activations[name]

    def layer(self, name: str, layer_index: int) -> torch.Tensor:
        """Activations for one absolute layer index -> [n_prompts, hidden]."""
        if layer_index not in self.layer_indices:
            raise KeyError(f"Layer {layer_index} was not captured; have {self.layer_indices}")
        return self.site(name)[self.layer_indices.index(layer_index)]

    def summary(self) -> str:
        lines = [f"ActivationSet: {len(self.prompts)} prompts, layers {self.layer_indices[0]}..{self.layer_indices[-1]}"]
        for site, tensor in sorted(self.activations.items()):
            lines.append(f"  {site:<12} {tuple(tensor.shape)}  {tensor.dtype}")
        return "\n".join(lines)


@torch.inference_mode()
def collect_activations(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    config: CaptureConfig | None = None,
    batch_size: int = 8,
    provenance: dict | None = None,
    prompt_metadata: Sequence[dict] | None = None,
    progress: bool = True,
) -> ActivationSet:
    """Run prefill over `prompts` and collect activations.

    `prompts` must already be chat-formatted -- use `generate.build_chat_prompt`.
    Passing raw user text here would capture off-distribution activations, which
    is the single easiest way to invalidate everything built on top.
    """
    config = config or CaptureConfig()
    if config.token_position == "all" and batch_size > 1:
        # Different lengths would need ragged storage; force the simple path.
        log.warning("token_position='all' with batch_size>1 pads to the longest prompt in each batch.")

    device = next(model.parameters()).device
    per_site_batches: dict[str, list[torch.Tensor]] = {s: [] for s in config.sites}
    layer_indices: tuple[int, ...] = ()

    n_batches = (len(prompts) + batch_size - 1) // batch_size
    for bi in range(n_batches):
        chunk = list(prompts[bi * batch_size : (bi + 1) * batch_size])
        # add_special_tokens=False: the chat template already emitted BOS.
        enc = tokenizer(chunk, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(device) for k, v in enc.items()}

        with ActivationCapture(model, config) as cap:
            cap.set_attention_mask(enc.get("attention_mask"))
            model(**enc, use_cache=False)
            batch_acts = cap.stacked()
            layer_indices = cap.selected_layers

        for site, tensor in batch_acts.items():
            per_site_batches[site].append(tensor)

        if progress:
            done = min((bi + 1) * batch_size, len(prompts))
            print(f"  captured {done}/{len(prompts)} prompts", end="\r", flush=True)

    if progress:
        print()

    # Concatenate along the prompt axis (dim=1, since storage is layer-major).
    activations = {
        site: torch.cat(batches, dim=1) for site, batches in per_site_batches.items() if batches
    }

    meta = list(prompt_metadata) if prompt_metadata else [{} for _ in prompts]
    prompt_records = [{"index": i, "prompt_text": p, **m} for i, (p, m) in enumerate(zip(prompts, meta))]

    return ActivationSet(
        activations=activations,
        layer_indices=layer_indices,
        prompts=prompt_records,
        config=config,
        provenance=provenance or {},
    )


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def save_activation_set(acts: ActivationSet, out_dir: Path | str) -> Path:
    """Write activations to safetensors plus a JSON sidecar.

    safetensors because it memory-maps on load: the analysis stages can read one
    layer without paging the whole tensor in, which matters once this is a few GB.
    The sidecar carries the provenance so a tensor on disk is never anonymous.
    """
    from safetensors.torch import save_file

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tensors = {site: t.contiguous() for site, t in acts.activations.items()}
    save_file(tensors, str(out_dir / "activations.safetensors"))

    metadata = {
        "capture_config": {
            "sites": list(acts.config.sites),
            "layers": list(acts.layer_indices),
            "token_position": acts.config.token_position,
            "store_dtype": str(acts.config.store_dtype),
        },
        "shapes": {site: list(t.shape) for site, t in acts.activations.items()},
        "axes": {
            site: (
                ["layer", "prompt", "seq", "hidden"]
                if t.ndim == 4
                else ["layer", "prompt", "hidden"]
            )
            for site, t in acts.activations.items()
        },
        "n_prompts": len(acts.prompts),
        "provenance": acts.provenance,
        "prompts": acts.prompts,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    total = sum(t.numel() * t.element_size() for t in acts.activations.values())
    log.info("Saved %.1f MiB of activations to %s", total / 1024**2, out_dir)
    return out_dir


def load_activation_set(in_dir: Path | str) -> ActivationSet:
    """Inverse of `save_activation_set`."""
    from safetensors.torch import load_file

    in_dir = Path(in_dir)
    metadata = json.loads((in_dir / "metadata.json").read_text())
    tensors = load_file(str(in_dir / "activations.safetensors"))

    cc = metadata["capture_config"]
    config = CaptureConfig(
        sites=tuple(cc["sites"]),
        layers=tuple(cc["layers"]),
        token_position=cc["token_position"],
        store_dtype=getattr(torch, cc["store_dtype"].replace("torch.", "")),
    )
    return ActivationSet(
        activations=tensors,
        layer_indices=tuple(cc["layers"]),
        prompts=metadata["prompts"],
        config=config,
        provenance=metadata.get("provenance", {}),
    )
