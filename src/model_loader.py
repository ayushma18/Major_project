"""Loading Llama 3.1 8B Instruct (or any HF causal LM) reproducibly.

Design goals:
  * One code path that runs unchanged on CUDA, Apple MPS, or CPU.
  * A memory preflight that fails loudly *before* the OS OOM-killer gets
    involved -- on a small machine a naive `from_pretrained` will lock the box.
  * The raw `model` object is returned untouched, so later stages can attach
    forward hooks to the residual stream without fighting a wrapper.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)

BYTES_PER_GIB = 1024**3


# --------------------------------------------------------------------------- #
# Hardware detection
# --------------------------------------------------------------------------- #


@dataclass
class DeviceInfo:
    kind: str  # "cuda" | "mps" | "cpu"
    name: str
    total_memory_gib: float
    supports_bf16: bool

    def __str__(self) -> str:
        return f"{self.kind} ({self.name}, {self.total_memory_gib:.1f} GiB, bf16={self.supports_bf16})"


def _host_ram_gib() -> float:
    """Total system RAM, without depending on psutil."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / BYTES_PER_GIB
    except (ValueError, OSError, AttributeError):
        return float("nan")


def _available_ram_gib() -> float:
    """RAM actually available right now (Linux MemAvailable), else total."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024**2)
    except OSError:
        pass
    return _host_ram_gib()


def detect_device(prefer: str | None = None) -> DeviceInfo:
    """Pick the best available backend.

    `prefer` ("cuda"/"mps"/"cpu") forces a choice; we still verify it exists so
    that a typo or an absent driver surfaces here instead of mid-load.
    """
    if prefer == "cpu":
        return DeviceInfo("cpu", "cpu", _host_ram_gib(), supports_bf16=True)

    if (prefer in (None, "cuda")) and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return DeviceInfo(
            kind="cuda",
            name=props.name,
            total_memory_gib=props.total_memory / BYTES_PER_GIB,
            supports_bf16=torch.cuda.is_bf16_supported(),
        )
    if prefer == "cuda":
        raise RuntimeError("device='cuda' requested but torch.cuda.is_available() is False.")

    if (prefer in (None, "mps")) and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        # MPS shares system RAM; bf16 support on MPS is patchy, so we say no.
        return DeviceInfo("mps", "apple-mps", _host_ram_gib(), supports_bf16=False)
    if prefer == "mps":
        raise RuntimeError("device='mps' requested but MPS is not available.")

    # CPU bf16 is supported by torch on any modern x86; it is emulated without
    # AVX512-BF16 but numerically fine, which is what matters for us.
    return DeviceInfo("cpu", "cpu", _host_ram_gib(), supports_bf16=True)


def pick_dtype(device: DeviceInfo, override: str | None = None) -> torch.dtype:
    """Choose a compute dtype.

    bf16 is the right default for Llama 3.1: it is the dtype the weights were
    released in, so it avoids a conversion pass and the fp16 overflow risk.
    """
    if override:
        mapping = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        if override not in mapping:
            raise ValueError(f"Unknown dtype {override!r}; expected one of {sorted(mapping)}")
        return mapping[override]

    if device.kind == "cuda":
        return torch.bfloat16 if device.supports_bf16 else torch.float16
    if device.kind == "mps":
        return torch.float16
    return torch.float32  # CPU: many kernels are fp32-only or slower in bf16


# --------------------------------------------------------------------------- #
# Memory preflight
# --------------------------------------------------------------------------- #


def estimate_num_parameters(model_id: str, token: str | None = None) -> tuple[int, int] | None:
    """Estimate (total_params, embedding_params) from the config.

    Returns None if the config cannot be fetched (offline / no access), in which
    case the caller should warn rather than block.
    """
    try:
        cfg = AutoConfig.from_pretrained(model_id, token=token)
    except Exception as exc:  # noqa: BLE001 - network/auth/anything: we degrade
        log.debug("Could not fetch config for %s: %s", model_id, exc)
        return None

    h = getattr(cfg, "hidden_size", None)
    n_layers = getattr(cfg, "num_hidden_layers", None)
    vocab = getattr(cfg, "vocab_size", None)
    inter = getattr(cfg, "intermediate_size", None)
    n_heads = getattr(cfg, "num_attention_heads", None)
    n_kv_heads = getattr(cfg, "num_key_value_heads", n_heads)
    if not all(isinstance(v, int) for v in (h, n_layers, vocab, inter, n_heads, n_kv_heads)):
        return None

    head_dim = getattr(cfg, "head_dim", None) or h // n_heads
    # Per layer: q/k/v/o projections + gated MLP (gate, up, down).
    attn = h * (n_heads * head_dim) + 2 * h * (n_kv_heads * head_dim) + (n_heads * head_dim) * h
    mlp = 3 * h * inter
    embeddings = vocab * h
    if not getattr(cfg, "tie_word_embeddings", False):
        embeddings += vocab * h  # separate lm_head

    return n_layers * (attn + mlp) + embeddings, embeddings


def estimate_weight_gib(model_id: str, dtype: torch.dtype, token: str | None = None) -> float | None:
    """Resident weight size for a given dtype, in GiB."""
    counts = estimate_num_parameters(model_id, token=token)
    if counts is None:
        return None
    total, _embeddings = counts
    return total * dtype.itemsize / BYTES_PER_GIB


def estimate_4bit_gib(model_id: str, token: str | None = None) -> float | None:
    """Resident size under bitsandbytes NF4 with double quantisation.

    Only the linear layers are quantised (~4.5 bits/param including the quant
    constants); embeddings and the lm_head stay in fp16. Ignoring that split
    underestimates the real footprint by a couple of GiB on a 128k-vocab model
    like Llama 3.1, which is exactly the margin that decides whether it fits.
    """
    counts = estimate_num_parameters(model_id, token=token)
    if counts is None:
        return None
    total, embeddings = counts
    quantised = total - embeddings
    return (quantised * 4.5 / 8 + embeddings * 2) / BYTES_PER_GIB


def preflight_memory(
    model_id: str,
    device: DeviceInfo,
    dtype: torch.dtype,
    token: str | None = None,
    headroom_gib: float | None = None,
    allow_insufficient: bool = False,
) -> float | None:
    """Refuse to start a load that cannot possibly fit.

    On CPU we check MemAvailable, not total: an OOM here takes the whole desktop
    down with it, which is a much worse failure than an early exception.
    """
    need = estimate_weight_gib(model_id, dtype, token=token)
    if need is None:
        log.warning("Could not estimate weight size for %s; skipping preflight.", model_id)
        return None

    if headroom_gib is None:
        # Scale with model size: activations and the KV cache grow with the model,
        # so a flat allowance is either wasteful for small models or too thin for
        # large ones. ~20% of weight size, with a 0.5 GiB floor.
        headroom_gib = max(0.5, 0.20 * need)

    if device.kind == "cuda":
        free_b, _total_b = torch.cuda.mem_get_info()
        have = free_b / BYTES_PER_GIB
        where = "free VRAM"
    else:
        have = _available_ram_gib()
        where = "available RAM"

    log.info("Preflight: %s needs ~%.1f GiB in %s; %.1f GiB %s.", model_id, need, str(dtype).replace("torch.", ""), have, where)

    if need + headroom_gib > have:
        four_bit = estimate_4bit_gib(model_id, token=token)
        four_bit_hint = f"~{four_bit:.1f} GiB" if four_bit else "much less"
        msg = (
            f"Refusing to load {model_id}: needs ~{need:.1f} GiB of weights plus "
            f"~{headroom_gib:.1f} GiB working headroom, but only {have:.1f} GiB {where} "
            f"is available.\n"
            f"Options: (a) run on a machine with more memory, (b) pass "
            f"--load-in-4bit (CUDA only, {four_bit_hint}), or (c) use a smaller "
            f"model via --model for a pipeline check."
        )
        if not allow_insufficient:
            raise MemoryError(msg)
        log.warning("%s\n--force given: continuing anyway.", msg)
    return need


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def resolve_hf_token(explicit: str | None = None) -> str | None:
    """Token from argument, then the usual env vars. huggingface_hub also reads
    ~/.cache/huggingface/token on its own, so None is a fine result."""
    return explicit or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _best_attn_implementation(device: DeviceInfo) -> str:
    """FlashAttention-2 where it is genuinely available, else SDPA.

    SDPA is the safe default: it is pure PyTorch and preserves the standard
    module structure that our later hook-based activation capture depends on.
    """
    if device.kind == "cuda":
        try:
            import flash_attn  # noqa: F401

            return "flash_attention_2"
        except ImportError:
            pass
    return "sdpa"


def load_model_and_tokenizer(
    model_id: str,
    *,
    device: DeviceInfo | None = None,
    dtype_override: str | None = None,
    token: str | None = None,
    load_in_4bit: bool = False,
    trust_remote_code: bool = False,
    skip_preflight: bool = False,
    force: bool = False,
    attn_implementation: str | None = None,
) -> tuple[Any, Any, dict]:
    """Load a causal LM + tokenizer and return them with a provenance dict.

    The provenance dict is written alongside every artefact this project
    produces, so that an activation tensor on disk can always be traced back to
    the exact model, dtype and revision that generated it.
    """
    device = device or detect_device()
    dtype = pick_dtype(device, dtype_override)
    token = resolve_hf_token(token)

    if load_in_4bit and device.kind != "cuda":
        raise RuntimeError("--load-in-4bit requires a CUDA GPU (bitsandbytes has no usable CPU path).")

    if not skip_preflight and not load_in_4bit:
        preflight_memory(model_id, device, dtype, token=token, allow_insufficient=force)

    log.info("Loading tokenizer: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token, trust_remote_code=trust_remote_code)
    # Llama has no pad token. Generation needs one for batching; reusing EOS is
    # the standard choice and is masked out by attention_mask anyway.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only models must be left-padded or generation reads padding as context.
    tokenizer.padding_side = "left"

    kwargs: dict[str, Any] = {
        "dtype": dtype,
        "token": token,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
        "attn_implementation": attn_implementation or _best_attn_implementation(device),
    }

    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": 0}
    elif device.kind == "cuda":
        kwargs["device_map"] = {"": 0}  # single GPU, explicit -> no surprise offload

    log.info("Loading model: %s (dtype=%s, device=%s)", model_id, dtype, device.kind)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    if "device_map" not in kwargs:
        model = model.to(device.kind if device.kind != "cpu" else "cpu")

    model.eval()
    model.config.use_cache = True

    provenance = {
        "model_id": model_id,
        "dtype": str(dtype),
        "device": device.kind,
        "device_name": device.name,
        "load_in_4bit": load_in_4bit,
        "attn_implementation": kwargs["attn_implementation"],
        "num_hidden_layers": model.config.num_hidden_layers,
        "hidden_size": model.config.hidden_size,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
    }
    log.info(
        "Loaded %s: %d layers, hidden_size=%d, %.2fB params.",
        model_id,
        provenance["num_hidden_layers"],
        provenance["hidden_size"],
        provenance["num_parameters"] / 1e9,
    )
    return model, tokenizer, provenance


def free_disk_gib(path: str = ".") -> float:
    return shutil.disk_usage(path).free / BYTES_PER_GIB
