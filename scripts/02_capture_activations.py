#!/usr/bin/env python
"""MILESTONE 2: capture and save the model's hidden activations.

    .venv/bin/python scripts/02_capture_activations.py --dev          # verify locally
    .venv/bin/python scripts/02_capture_activations.py                # the real 8B
    .venv/bin/python scripts/02_capture_activations.py --sites resid_post attn_out mlp_out
    .venv/bin/python scripts/02_capture_activations.py --prompts my_prompts.txt

Writes data/activations/<run>/{activations.safetensors,metadata.json}.

Activations are taken at the final prompt token of a single prefill pass -- the
position whose residual stream determines the first generated token, and the
standard readout point for steering-vector and refusal-direction work.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src import config  # noqa: E402
from src.activations import (  # noqa: E402
    SITES,
    TOKEN_POSITIONS,
    CaptureConfig,
    collect_activations,
    load_activation_set,
    save_activation_set,
)
from src.generate import build_chat_prompt  # noqa: E402
from src.model_loader import detect_device, load_model_and_tokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None)
    p.add_argument("--dev", action="store_true", help=f"use the small stand-in ({config.DEV_MODEL_ID})")
    p.add_argument("--prompts", type=Path, default=None, help="text file, one user prompt per line")
    p.add_argument("--system", default=config.DEFAULT_SYSTEM_PROMPT)
    p.add_argument("--sites", nargs="+", default=["resid_post"], choices=list(SITES))
    p.add_argument("--layers", nargs="+", type=int, default=None, help="layer indices (default: all)")
    p.add_argument("--token-position", default="last", choices=list(TOKEN_POSITIONS))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--run-name", default=None)
    p.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None)
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default=None)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--token", default=None)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def load_prompts(args: argparse.Namespace) -> tuple[list[str], list[dict]]:
    """Return (user_prompts, per-prompt metadata)."""
    if args.prompts:
        lines = [ln.strip() for ln in args.prompts.read_text().splitlines()]
        users = [ln for ln in lines if ln and not ln.startswith("#")]
        if not users:
            raise SystemExit(f"No usable prompts in {args.prompts}")
        return users, [{"name": f"prompt_{i:04d}", "source": str(args.prompts)} for i in range(len(users))]

    return (
        [item.user for item in config.SMOKE_TEST_PROMPTS],
        [{"name": item.name, "source": "smoke_test"} for item in config.SMOKE_TEST_PROMPTS],
    )


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")

    model_id = args.model or (config.DEV_MODEL_ID if args.dev else config.PRIMARY_MODEL_ID)
    device = detect_device(args.device)

    print(f"\n{'=' * 70}\n  MILESTONE 2: capture hidden activations\n{'=' * 70}")
    print(f"  model           {model_id}")
    print(f"  device          {device}")
    print(f"  sites           {args.sites}")
    print(f"  token position  {args.token_position}")
    print(f"{'=' * 70}\n")

    model, tokenizer, provenance = load_model_and_tokenizer(
        model_id,
        device=device,
        dtype_override=args.dtype,
        token=args.token,
        load_in_4bit=args.load_in_4bit,
        force=args.force,
    )

    users, prompt_meta = load_prompts(args)
    # Chat-format every prompt. Capturing raw text would put the activations
    # off-distribution and invalidate everything built on them.
    formatted = [build_chat_prompt(tokenizer, user=u, system=args.system) for u in users]
    for m, u in zip(prompt_meta, users):
        m["user"] = u
        m["system"] = args.system

    capture_config = CaptureConfig(
        sites=tuple(args.sites),
        layers=tuple(args.layers) if args.layers else None,
        token_position=args.token_position,
    )

    print(f"Capturing over {len(formatted)} prompts (batch size {args.batch_size})...")
    acts = collect_activations(
        model,
        tokenizer,
        formatted,
        config=capture_config,
        batch_size=args.batch_size,
        provenance=provenance,
        prompt_metadata=prompt_meta,
    )

    print(f"\n{acts.summary()}\n")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or f"{model_id.split('/')[-1]}_{args.token_position}_{stamp}"
    out_dir = save_activation_set(acts, config.ACTIVATIONS_DIR / run_name)

    # Read it straight back. A save format that cannot round-trip is worse than
    # no save at all, and the failure would otherwise surface days later.
    reloaded = load_activation_set(out_dir)
    for site in acts.activations:
        original, restored = acts.activations[site], reloaded.activations[site]
        assert original.shape == restored.shape, f"{site}: shape changed on reload"
        assert torch.equal(original, restored), f"{site}: values changed on reload"
    assert reloaded.layer_indices == acts.layer_indices
    assert len(reloaded.prompts) == len(acts.prompts)
    print("Round-trip verified: saved activations reload bit-identical.")

    total_mib = sum(t.numel() * t.element_size() for t in acts.activations.values()) / 1024**2
    print(f"\n{'=' * 70}")
    print(f"  MILESTONE 2 COMPLETE -- {total_mib:.1f} MiB captured.")
    print(f"  Output: {out_dir.relative_to(config.PROJECT_ROOT)}")
    print(f"{'=' * 70}\n")

    print("Read them back with:")
    print("  from src.activations import load_activation_set")
    print(f"  acts = load_activation_set('{out_dir.relative_to(config.PROJECT_ROOT)}')")
    print("  acts.layer('resid_post', 15)   # -> [n_prompts, hidden]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
