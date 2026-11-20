"""Shared runtime helpers: where models live, loading by short name, timing, JSONL I/O.

Model weights are fetched with `scripts/fetch_model.py` into a plain folder,
`$AFFECT_MODELS_DIR/<org>--<name>/` (default `~/models`). `resolve_model()` turns a hub id
into that local path when it exists, so every script works offline and identically.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

import torch

from .model_loader import detect_device, load_model_and_tokenizer

MODELS_DIR = Path(os.path.expanduser(os.environ.get("AFFECT_MODELS_DIR", "~/models")))

# Short names used on the command line.
ALIASES = {
    "llama-1b": "unsloth/Llama-3.2-1B-Instruct",
    "llama-8b": "unsloth/Llama-3.1-8B-Instruct",
    "harmbench-cls": "cais/HarmBench-Llama-2-13b-cls",
    "harmbench-cls-7b": "cais/HarmBench-Mistral-7b-val-cls",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-14b": "Qwen/Qwen3-14B",
    "gemma2-9b": "unsloth/gemma-2-9b-it",
}


def resolve_model(name: str) -> str:
    hub_id = ALIASES.get(name, name)
    local = MODELS_DIR / hub_id.replace("/", "--")
    if (local / "config.json").exists():
        return str(local)
    return hub_id


def load(name: str, dtype: str | None = "bfloat16", **kw: Any):
    """Load a model + tokenizer by alias, hub id or path. Returns (model, tok, provenance)."""
    path = resolve_model(name)
    model, tok, prov = load_model_and_tokenizer(
        path, device=detect_device(), dtype_override=dtype, skip_preflight=True, **kw
    )
    # The shipped generation_config sets max_length=131072, which makes every generate()
    # call warn when max_new_tokens is passed. Generation length is always explicit here.
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.max_length = None
    prov["model_name"] = ALIASES.get(name, name)
    prov["model_path"] = path
    return model, tok, prov


def sync(device: torch.device | str | None = None) -> None:
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def mem_gib() -> dict:
    out = {}
    if torch.backends.mps.is_available():
        out["mps_allocated_gib"] = torch.mps.current_allocated_memory() / 1024**3
        out["mps_driver_gib"] = torch.mps.driver_allocated_memory() / 1024**3
    return out


class Timer:
    def __enter__(self):
        sync()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        sync()
        self.seconds = time.perf_counter() - self.t0


def write_jsonl(path: Path | str, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_jsonl(path: Path | str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def write_json(path: Path | str, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    os.replace(tmp, path)
