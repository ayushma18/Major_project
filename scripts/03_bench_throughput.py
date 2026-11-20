"""E00: how fast does this machine run a model, with and without a steering hook?

    python scripts/03_bench_throughput.py --model llama-1b
    python scripts/03_bench_throughput.py --model llama-8b --batch-sizes 1 8 16 32 --dtypes bfloat16 float16
    python scripts/03_bench_throughput.py --model llama-8b --mlx      # also time MLX single-stream

Writes results/raw/E00_<model>.json. Generation is forced to exactly --new-tokens tokens
(min_new_tokens) so rates are comparable. Prompts are benign, HarmBench-length requests.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.generate import build_chat_prompt, eos_token_ids  # noqa: E402
from src.runtime import ALIASES, Timer, load, mem_gib, resolve_model, write_json  # noqa: E402
from src.steering import AddVector  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = [
    "Write a short paragraph explaining how photosynthesis turns light into chemical energy.",
    "Give me three tips for keeping a sourdough starter healthy during a cold winter.",
    "Explain the difference between a virus and a bacterium in plain words for a teenager.",
    "Describe how a bill becomes a law in a parliamentary system, step by step.",
]


def bench_torch(model, tok, batch: int, new_tokens: int, steer: bool) -> dict:
    device = next(model.parameters()).device
    texts = [build_chat_prompt(tok, PROMPTS[i % len(PROMPTS)]) for i in range(batch)]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    enc = {k: v.to(device) for k, v in enc.items()}
    hidden = model.config.hidden_size
    ctx = AddVector(model, layer=model.config.num_hidden_layers // 2, vector=torch.randn(hidden), coeff=1.0) if steer else None
    kw = dict(max_new_tokens=new_tokens, min_new_tokens=new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
    # warm-up at the SAME shape: MPS compiles kernels per shape on first use
    with torch.inference_mode():
        if ctx:
            ctx.__enter__()
        try:
            model.generate(**enc, max_new_tokens=8, min_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id)
        finally:
            if ctx:
                ctx.__exit__(None, None, None)
    with torch.inference_mode():
        if ctx:
            ctx.__enter__()
        try:
            with Timer() as t:
                out = model.generate(**enc, **kw)
        finally:
            if ctx:
                ctx.__exit__(None, None, None)
    gen_tokens = (out.shape[1] - enc["input_ids"].shape[1]) * batch
    return {"batch": batch, "steer_hook": steer, "seconds": t.seconds, "generated_tokens": gen_tokens,
            "tok_per_s": gen_tokens / t.seconds, "prompt_tokens": int(enc["input_ids"].shape[1]), **mem_gib()}


def bench_prefill(model, tok, batch: int, seq: int = 160) -> dict:
    device = next(model.parameters()).device
    ids = torch.randint(100, 20000, (batch, seq), device=device)
    with torch.inference_mode():
        model(input_ids=ids, use_cache=False)
        with Timer() as t:
            model(input_ids=ids, use_cache=False)
    return {"batch": batch, "seq": seq, "seconds": t.seconds, "tokens_per_s": batch * seq / t.seconds}


def bench_mlx(path: str, new_tokens: int) -> dict:
    try:
        from mlx_lm import generate, load as mlx_load
    except Exception as exc:  # noqa: BLE001
        return {"error": f"mlx_lm unavailable: {exc}"}
    model, tok = mlx_load(path)
    prompt = tok.apply_chat_template([{"role": "user", "content": PROMPTS[0]}], add_generation_prompt=True, tokenize=False)
    generate(model, tok, prompt=prompt, max_tokens=8)
    t0 = time.perf_counter()
    text = generate(model, tok, prompt=prompt, max_tokens=new_tokens)
    dt = time.perf_counter() - t0
    n = len(tok.encode(text))
    return {"single_stream_tok_per_s": n / dt, "tokens": n, "seconds": dt, "dtype": "as stored (bf16)"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-1b")
    ap.add_argument("--dtypes", nargs="+", default=["bfloat16", "float16"])
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 16, 32])
    ap.add_argument("--new-tokens", type=int, default=128)
    ap.add_argument("--mlx", action="store_true")
    args = ap.parse_args()

    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip() or 0) / 1024**3
    report = {"model": ALIASES.get(args.model, args.model), "machine": {"chip": chip, "ram_gib": mem, "os": platform.platform()},
              "torch": torch.__version__, "runs": [], "prefill": []}
    for dtype in args.dtypes:
        t0 = time.perf_counter()
        model, tok, prov = load(args.model, dtype=dtype)
        load_s = time.perf_counter() - t0
        print(f"loaded {args.model} {dtype} in {load_s:.1f}s  {mem_gib()}")
        for b in args.batch_sizes:
            for steer in (False, True):
                r = bench_torch(model, tok, b, args.new_tokens, steer)
                r.update(dtype=dtype, load_seconds=load_s)
                print(f"  {dtype} batch={b:<3} steer={steer!s:<5} {r['tok_per_s']:7.1f} tok/s")
                report["runs"].append(r)
        for b in (1, 16):
            p = bench_prefill(model, tok, b)
            p["dtype"] = dtype
            print(f"  prefill batch={b} {p['tokens_per_s']:.0f} tok/s")
            report["prefill"].append(p)
        report["provenance"] = prov
        del model
        torch.mps.empty_cache() if torch.backends.mps.is_available() else None
    if args.mlx:
        report["mlx"] = bench_mlx(resolve_model(args.model), args.new_tokens)
        print("  mlx:", report["mlx"])
    out = ROOT / "results" / "raw" / f"E00_{args.model}.json"
    write_json(out, report)
    print("wrote", out)


if __name__ == "__main__":
    main()
