#!/usr/bin/env python
"""Report the environment and whether this machine can host Llama 3.1 8B.

Run this first. It answers three questions:
  1. Are the libraries installed and mutually compatible?
  2. What compute do we have, and what will the 8B actually need on it?
  3. Is the Hugging Face token present and does it have access to the gated repo?
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src import config  # noqa: E402
from src.model_loader import (  # noqa: E402
    BYTES_PER_GIB,
    _available_ram_gib,
    detect_device,
    estimate_4bit_gib,
    estimate_weight_gib,
    free_disk_gib,
    pick_dtype,
    resolve_hf_token,
)


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> int:
    section("1. LIBRARIES")
    import transformers

    print(f"  python        {sys.version.split()[0]}")
    print(f"  torch         {torch.__version__}")
    print(f"  transformers  {transformers.__version__}")
    try:
        import accelerate

        print(f"  accelerate    {accelerate.__version__}")
    except ImportError:
        print("  accelerate    MISSING (needed for device_map / offload)")
    try:
        import huggingface_hub

        print(f"  huggingface_hub {huggingface_hub.__version__}")
    except ImportError:
        print("  huggingface_hub MISSING")

    section("2. COMPUTE")
    device = detect_device()
    dtype = pick_dtype(device)
    print(f"  backend         {device.kind}")
    print(f"  device          {device.name}")
    print(f"  memory          {device.total_memory_gib:.1f} GiB total")
    if device.kind == "cpu":
        print(f"  available now   {_available_ram_gib():.1f} GiB")
    print(f"  bf16 supported  {device.supports_bf16}")
    print(f"  chosen dtype    {dtype}")
    print(f"  free disk       {free_disk_gib(str(config.PROJECT_ROOT)):.1f} GiB")
    if device.kind == "cuda":
        free_b, total_b = torch.cuda.mem_get_info()
        print(f"  free VRAM       {free_b / BYTES_PER_GIB:.1f} / {total_b / BYTES_PER_GIB:.1f} GiB")

    section("3. HUGGING FACE ACCESS")
    token = resolve_hf_token()
    stored = Path.home() / ".cache" / "huggingface" / "token"
    if token:
        print("  token           found in environment (HF_TOKEN)")
    elif stored.exists():
        print(f"  token           found at {stored}")
    else:
        print("  token           NOT FOUND")
        print("\n  To fix:")
        print("    1. Accept the licence at")
        print(f"       https://huggingface.co/{config.PRIMARY_MODEL_ID}")
        print("       (Meta approval is usually granted within minutes to a few hours.)")
        print("    2. Create a READ token at https://huggingface.co/settings/tokens")
        print("    3. Run:  .venv/bin/hf auth login")

    print(f"\n  target model    {config.PRIMARY_MODEL_ID}")
    # model_info() succeeds anonymously for gated repos -- it only exposes public
    # metadata. The real test is whether we can pull a file, so download the
    # smallest one (config.json) and see if the gate lets us through.
    access_ok = False
    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(config.PRIMARY_MODEL_ID, filename="config.json", token=token)
        print("  access          OK -- gated repo is unlocked for this token")
        access_ok = True
    except Exception as exc:  # noqa: BLE001
        name = type(exc).__name__
        detail = str(exc)
        print(f"  access          DENIED ({name})")
        if "Gated" in name or "401" in detail or "403" in detail or "awaiting" in detail.lower():
            print("                  -> licence not accepted yet, or the token lacks access.")
            print(f"                  -> accept at https://huggingface.co/{config.PRIMARY_MODEL_ID}")
        else:
            print(f"                  -> {detail.splitlines()[0][:140]}")

    section("4. VERDICT")
    need = estimate_weight_gib(config.PRIMARY_MODEL_ID, dtype) if access_ok else None
    need_4bit = estimate_4bit_gib(config.PRIMARY_MODEL_ID) if access_ok else None
    if need is None:
        # Fall back to the published parameter count so we can still advise.
        need = 8.03e9 * dtype.itemsize / BYTES_PER_GIB
        need_4bit = 5.9
        print("  (estimated from the published 8.03B parameter count)")
    have = (
        torch.cuda.mem_get_info()[0] / BYTES_PER_GIB if device.kind == "cuda" else _available_ram_gib()
    )
    print(f"  Llama 3.1 8B in {str(dtype).replace('torch.', '')} needs ~{need:.1f} GiB of weights.")
    print(f"  In 4-bit NF4 it needs ~{need_4bit:.1f} GiB (CUDA only).")
    print(f"  This machine has ~{have:.1f} GiB usable.")

    if need + 0.20 * need <= have:
        print("\n  READY: this machine can load Llama 3.1 8B Instruct in full precision.")
        print("  Next:  .venv/bin/python scripts/01_load_and_generate.py")
        return 0

    if need_4bit and device.kind == "cuda" and need_4bit + 1.5 <= have:
        # The common Colab-T4 case: fp16 weights alone would eat the whole card.
        print("\n  TIGHT: full precision will not fit, but 4-bit will.")
        print("    .venv/bin/python scripts/01_load_and_generate.py --load-in-4bit")
        print("    pip install bitsandbytes   # required for 4-bit")
        print("\n  Caveat for this project: 4-bit quantisation perturbs activations.")
        print("  Directions are still recoverable, but for final numbers prefer a")
        print("  >=24 GiB card (L4 / 3090 / 4090 / A100) and run in fp16.")
        return 0

    print("\n  INSUFFICIENT MEMORY for the 8B on this machine.")
    print("  Workable options, cheapest first:")
    print("    * Colab free T4 (15 GiB): use --load-in-4bit. fp16 weights alone are")
    print("      ~15.0 GiB, so full precision does NOT fit on a T4.")
    print("    * Kaggle 2x T4, 30h/week free, with persistent output storage.")
    print("    * A >=24 GiB card (L4 / 3090 / 4090) for fp16 -- best for final results.")
    print("    * Verify the pipeline here on a small model first:")
    print("        .venv/bin/python scripts/01_load_and_generate.py --dev")
    print("        .venv/bin/python scripts/02_capture_activations.py --dev")
    print("      The same scripts then run the 8B unchanged on real hardware.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
