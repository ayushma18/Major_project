"""Writing INTO the residual stream: activation addition and directional ablation.

Two interventions, both as context managers that always remove their hooks:

    with AddVector(model, layer=12, vector=v, coeff=4.0):
        model.generate(...)          # x_l <- x_l + coeff * v at every position

    with AblateDirection(model, direction=r):
        model.generate(...)          # x <- x - (x . r_hat) r_hat at every residual write

AddVector is Contrastive Activation Addition / ActAdd style steering (Rimsky et al. 2023,
Turner et al. 2023): the vector is added to the OUTPUT of decoder layer `layer`, which is
the residual stream entering layer `layer + 1`. It applies to every token position of every
forward call, so it covers the prompt (prefill) and every generated token (decode steps
with the KV cache). `positions="generated"` restricts it to decode steps only.

AblateDirection is the directional ablation of Arditi et al. 2024. The direction is
projected out of every write to the residual stream: the input of each decoder layer
(which covers the embedding output), the attention output and the MLP output of every
layer. Because every write is projected, no component along r_hat ever exists in the
stream, which is mathematically the same as their weight orthogonalisation. Projecting
only one layer's output would let later layers write r back in.

Both accept a `scale_like` option: the vector is first normalised to unit length and then
multiplied by `coeff`. Experiments in this project express steering strength as a multiple
of the typical residual-stream norm at the injection layer (see `vectors.layer_norm_stats`),
so emotion vectors, random vectors and orthogonalised vectors are compared at equal norm.
"""

from __future__ import annotations

from typing import Any

import torch

from .activations import get_layers


def _replace_output(output: Any, new_hidden: torch.Tensor) -> Any:
    """Put a modified hidden-state tensor back into a module output of either shape."""
    if isinstance(output, torch.Tensor):
        return new_hidden
    if isinstance(output, tuple):
        return (new_hidden, *output[1:])
    if isinstance(output, list):
        return [new_hidden, *output[1:]]
    raise TypeError(f"Unsupported module output type {type(output)}")


def _hidden_of(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Cannot find hidden states in output of type {type(output)}")


class _HookSet:
    """Base: registers hooks on enter, always removes them on exit."""

    def __init__(self) -> None:
        self._handles: list[Any] = []

    def _register(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def __enter__(self):
        self._register()
        return self

    def __exit__(self, *exc: object) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


class AddVector(_HookSet):
    """Add `coeff * vector` to the output of decoder layer(s).

    Args:
        model: HF causal LM.
        layer: one layer index or a list of indices.
        vector: [hidden] tensor (any dtype/device; moved to the model's).
        coeff: multiplier applied to the vector as given, or to its unit vector if
            `normalize=True`.
        normalize: scale the vector to unit norm before multiplying.
        positions: "all" (prefill and decode) or "generated" (decode steps only,
            i.e. forward calls whose sequence length is 1).
    """

    def __init__(
        self,
        model: Any,
        layer: int | list[int],
        vector: torch.Tensor,
        coeff: float = 1.0,
        *,
        normalize: bool = False,
        positions: str = "all",
    ) -> None:
        super().__init__()
        if positions not in ("all", "generated"):
            raise ValueError("positions must be 'all' or 'generated'")
        self.model = model
        self.layers_idx = [layer] if isinstance(layer, int) else list(layer)
        v = vector.detach().float().flatten()
        if normalize:
            v = v / v.norm().clamp(min=1e-12)
        self.delta = coeff * v
        self.positions = positions
        self.enabled = True

    def _hook(self, _module, _inputs, output):
        if not self.enabled:
            return output
        hidden = _hidden_of(output)
        if self.positions == "generated" and hidden.shape[1] != 1:
            return output
        delta = self.delta.to(device=hidden.device, dtype=hidden.dtype)
        return _replace_output(output, hidden + delta)

    def _register(self) -> None:
        layers = get_layers(self.model)
        for i in self.layers_idx:
            self._handles.append(layers[i].register_forward_hook(self._hook))


class AblateDirection(_HookSet):
    """Project `direction` out of every residual-stream write (Arditi et al. 2024).

    Hooks: a pre-hook on every decoder layer (its input stream, which includes the
    embedding output for layer 0), and forward hooks on every self-attention and MLP
    output. After this, the component of the residual stream along r_hat is zero at
    every layer and position.
    """

    def __init__(self, model: Any, direction: torch.Tensor) -> None:
        super().__init__()
        self.model = model
        d = direction.detach().float().flatten()
        self.r_hat = d / d.norm().clamp(min=1e-12)
        self.enabled = True

    def _project(self, hidden: torch.Tensor) -> torch.Tensor:
        r = self.r_hat.to(device=hidden.device, dtype=hidden.dtype)
        # Compute the coefficient in fp32 for stability, then subtract in the model dtype.
        coef = (hidden.float() @ self.r_hat.to(hidden.device)).unsqueeze(-1)
        return hidden - coef.to(hidden.dtype) * r

    def _pre_hook(self, _module, args, kwargs):
        if not self.enabled:
            return None
        if args:
            hidden = args[0]
            return (self._project(hidden), *args[1:]), kwargs
        if "hidden_states" in kwargs:
            kwargs = dict(kwargs)
            kwargs["hidden_states"] = self._project(kwargs["hidden_states"])
            return args, kwargs
        return None

    def _out_hook(self, _module, _inputs, output):
        if not self.enabled:
            return output
        return _replace_output(output, self._project(_hidden_of(output)))

    def _register(self) -> None:
        for layer in get_layers(self.model):
            self._handles.append(layer.register_forward_pre_hook(self._pre_hook, with_kwargs=True))
            self._handles.append(layer.self_attn.register_forward_hook(self._out_hook))
            self._handles.append(layer.mlp.register_forward_hook(self._out_hook))


class Compose(_HookSet):
    """Apply several interventions together (for example ablation plus addition)."""

    def __init__(self, *interventions: _HookSet) -> None:
        super().__init__()
        self.items = [i for i in interventions if i is not None]

    def __enter__(self):
        for i in self.items:
            i.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        for i in reversed(self.items):
            i.__exit__(*exc)


def orthogonalize(vector: torch.Tensor, direction: torch.Tensor, *, keep_norm: bool = True) -> torch.Tensor:
    """Remove the component of `vector` along `direction` (proposal eq. 5.5).

    keep_norm=True rescales the result back to the original norm, so the causal test
    compares perturbations of equal size (otherwise a smaller ASR effect could just
    mean a smaller push).
    """
    v = vector.detach().float().flatten()
    d = direction.detach().float().flatten()
    d_hat = d / d.norm().clamp(min=1e-12)
    out = v - (v @ d_hat) * d_hat
    if keep_norm:
        out = out * (v.norm() / out.norm().clamp(min=1e-12))
    return out


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    return float((a @ b) / (a.norm() * b.norm()).clamp(min=1e-12))


class AddVectorRows(_HookSet):
    """Add a DIFFERENT vector to each batch row at one layer: deltas [batch, hidden].

    Lets one generate() call steer 32 rows with 32 different emotions/strengths. Rows
    must keep their order for the whole call (true for HF generate without beam search).
    """

    def __init__(self, model: Any, layer: int, deltas: torch.Tensor) -> None:
        super().__init__()
        self.model = model
        self.layer = layer
        self.deltas = deltas.detach().float()

    def _hook(self, _module, _inputs, output):
        hidden = _hidden_of(output)
        d = self.deltas.to(device=hidden.device, dtype=hidden.dtype)
        if d.shape[0] != hidden.shape[0]:
            raise ValueError(f"AddVectorRows: {d.shape[0]} deltas for a batch of {hidden.shape[0]}")
        return _replace_output(output, hidden + d[:, None, :])

    def _register(self) -> None:
        self._handles.append(get_layers(self.model)[self.layer].register_forward_hook(self._hook))


class AddVectorMasked(_HookSet):
    """Add `delta` at one layer only at selected positions (multi-turn sleeper protocol).

    `prefill_mask` [batch, seq] (bool) marks which prompt positions get the vector in the
    prefill pass; `on_decode` says whether newly generated tokens (decode steps) get it.
    Recomputing a whole conversation with the vector only on turn-1 positions is the same
    as keeping a KV cache built while turn 1 was steered.
    """

    def __init__(self, model: Any, layer: int, delta: torch.Tensor, prefill_mask: torch.Tensor, on_decode: bool) -> None:
        super().__init__()
        self.model = model
        self.layer = layer
        self.delta = delta.detach().float().flatten()
        self.mask = prefill_mask.bool()
        self.on_decode = on_decode

    def _hook(self, _module, _inputs, output):
        hidden = _hidden_of(output)
        d = self.delta.to(device=hidden.device, dtype=hidden.dtype)
        if hidden.shape[1] == 1:  # decode step
            return _replace_output(output, hidden + d) if self.on_decode else output
        m = self.mask.to(hidden.device)
        if m.shape != hidden.shape[:2]:
            raise ValueError(f"AddVectorMasked: mask {tuple(m.shape)} vs hidden {tuple(hidden.shape[:2])}")
        return _replace_output(output, hidden + m[..., None].to(hidden.dtype) * d)

    def _register(self) -> None:
        self._handles.append(get_layers(self.model)[self.layer].register_forward_hook(self._hook))
