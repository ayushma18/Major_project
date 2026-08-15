#!/usr/bin/env python
"""MILESTONE 1: load Llama 3.1 8B Instruct and generate a normal response.

    .venv/bin/python scripts/01_load_and_generate.py                 # the 8B
    .venv/bin/python scripts/01_load_and_generate.py --dev           # small stand-in
    .venv/bin/python scripts/01_load_and_generate.py --load-in-4bit  # CUDA, ~6 GiB
    .venv/bin/python scripts/01_load_and_generate.py -p "your prompt"

Every run writes a JSON record to outputs/generations/ containing the prompts,
the completions, and full provenance (model id, dtype, device, versions). That
record is the evidence the milestone was met, and it is the same provenance
format the activation-capture stage will reuse.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.generate import generate_chat  # noqa: E402
from src.model_loader import detect_device, load_model_and_tokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None, help=f"HF repo id (default: {config.PRIMARY_MODEL_ID})")
    p.add_argument("--dev", action="store_true", help=f"use the small stand-in ({config.DEV_MODEL_ID})")
    p.add_argument("-p", "--prompt", default=None, help="single prompt instead of the smoke-test set")
    p.add_argument("-s", "--system", default=config.DEFAULT_SYSTEM_PROMPT)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--sample", action="store_true", help="sample instead of greedy decoding")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None)
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default=None)
    p.add_argument("--load-in-4bit", action="store_true", help="4-bit NF4 quantisation (CUDA only)")
    p.add_argument("--token", default=None, help="HF token (else HF_TOKEN env or cached login)")
    p.add_argument("--force", action="store_true", help="load even if the memory preflight says it will not fit")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    model_id = args.model or (config.DEV_MODEL_ID if args.dev else config.PRIMARY_MODEL_ID)
    device = detect_device(args.device)

    print(f"\n{'=' * 70}\n  MILESTONE 1: load + generate\n{'=' * 70}")
    print(f"  model   {model_id}")
    print(f"  device  {device}")
    print(f"{'=' * 70}\n")

    model, tokenizer, provenance = load_model_and_tokenizer(
        model_id,
        device=device,
        dtype_override=args.dtype,
        token=args.token,
        load_in_4bit=args.load_in_4bit,
        force=args.force,
    )

    # Sanity-check the architecture when we believe we loaded the real 8B, so a
    # silently-substituted model cannot slip into the results.
    if "8B" in model_id:
        assert provenance["num_hidden_layers"] == config.LLAMA31_8B_NUM_LAYERS, (
            f"Expected {config.LLAMA31_8B_NUM_LAYERS} layers, got {provenance['num_hidden_layers']}"
        )
        assert provenance["hidden_size"] == config.LLAMA31_8B_HIDDEN_SIZE

    if args.prompt:
        prompts = [config.SmokeTestPrompt(name="custom", system=args.system, user=args.prompt)]
    else:
        prompts = config.SMOKE_TEST_PROMPTS

    records = []
    for i, item in enumerate(prompts, start=1):
        print(f"\n{'-' * 70}\n[{i}/{len(prompts)}] {item.name}\n{'-' * 70}")
        print(f"USER: {item.user}\n")

        result = generate_chat(
            model,
            tokenizer,
            user=item.user,
            system=item.system,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.sample,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
        )

        print(f"ASSISTANT: {result.completion}\n")
        print(
            f"[{result.num_prompt_tokens} prompt + {result.num_completion_tokens} new tokens "
            f"in {result.seconds:.1f}s = {result.tokens_per_second:.1f} tok/s]"
        )

        records.append(
            {
                "name": item.name,
                "system": item.system,
                "user": item.user,
                "prompt_text": result.prompt_text,
                "completion": result.completion,
                "num_prompt_tokens": result.num_prompt_tokens,
                "num_completion_tokens": result.num_completion_tokens,
                "seconds": round(result.seconds, 3),
                "tokens_per_second": round(result.tokens_per_second, 2),
            }
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = config.GENERATIONS_DIR / f"milestone1_{model_id.split('/')[-1]}_{stamp}.json"
    out_path.write_text(
        json.dumps(
            {
                "created_utc": stamp,
                "provenance": provenance,
                "decoding": {
                    "do_sample": args.sample,
                    "temperature": args.temperature if args.sample else None,
                    "top_p": args.top_p if args.sample else None,
                    "max_new_tokens": args.max_new_tokens,
                    "seed": args.seed,
                },
                "results": records,
            },
            indent=2,
        )
    )

    print(f"\n{'=' * 70}")
    print(f"  MILESTONE 1 COMPLETE -- {model_id} loaded and generating.")
    print(f"  Record: {out_path.relative_to(config.PROJECT_ROOT)}")
    print(f"{'=' * 70}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
