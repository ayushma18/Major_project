# Affect as Attack: emotion vectors versus the refusal direction

> **Branch `anuj/experiments` (2026-11-20).** The full experiment pipeline now runs on an
> Apple Silicon Mac (M5 Pro, 48 GB): refusal direction, emotion vectors, steering, HarmBench
> judging, multi-turn harness, and a local project interface. Start here:
>
> - `docs/PIPELINE.md`: setup, the order of the experiments, design decisions
> - `docs/CONTRACTS.md`: the data files every script and the interface share
> - `python scripts/serve.py`, then http://127.0.0.1:8765/web/ : the project interface
>   (overview, paper timeline and library with a PDF reader, reference audit, live results)
> - `literature/catalog.json`: the verified paper library (`scripts/fetch_papers.py` downloads PDFs)
>
> Raw model completions to harmful prompts never leave `outputs/` (git-ignored, not served).
> Results files carry rates and labels only.

The original milestone notes (M1, M2) follow.

## Milestones 1 and 2: loading, generation, activation capture

## Status

| Milestone | State |
|---|---|
| Environment (venv, torch, transformers, accelerate, HF hub) | Done, pinned in `requirements.txt` |
| **M1** — load + generate | Code done, **verified end to end** |
| **M2** — capture + save hidden activations | Code done, **verified end to end**, 22 tests passing |
| Colab (T4) deployment notebook | Done — `notebooks/colab_llama31_8b.ipynb` |
| HF gated access to `meta-llama/Llama-3.1-8B-Instruct` | **Blocked — needs your licence acceptance + token** |
| 8B actually running | **Blocked on hardware** — run it on Colab |

Both milestones are implemented and tested. They have not yet run against the
real 8B, because of the two blockers below.

## Blocker 1 — hardware

This machine cannot host Llama 3.1 8B:

- **No CUDA GPU.** Intel Iris Xe integrated graphics only.
- **3.7 GiB RAM** in WSL (8 GiB host), of which ~1.5 GiB is typically free.
- Disk is fine (912 GiB free).

| Precision | Weights | Runs here? | Runs on a free Colab T4 (15 GiB)? |
|---|---|---|---|
| float32 (CPU default) | ~29.9 GiB | No | No |
| float16 | ~15.0 GiB | No | **No** — weights alone fill the card |
| 4-bit NF4 | ~5.9 GiB | No (needs CUDA) | **Yes** |

Note the second row: a T4 has ~15 GiB and fp16 weights are ~15.0 GiB, so full
precision does *not* fit even on Colab. The notebook uses 4-bit.

**4-bit perturbs activations.** Refusal directions and steering vectors are still
recoverable, but for final numbers use a ≥24 GiB card (L4 / 3090 / 4090 / A100)
in fp16 and treat T4 output as development-grade.

Everything in `src/` is device-agnostic — the same scripts run unchanged on CUDA,
MPS, or CPU. Nothing needs rewriting when you move.

## Blocker 2 — gated model access

`scripts/00_check_env.py` currently reports `GatedRepoError`. To fix:

1. Accept the licence at <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct>
   (Meta approval usually lands within minutes to a few hours).
2. Create a **read** token at <https://huggingface.co/settings/tokens>.
3. `.venv/bin/hf auth login` — or `export HF_TOKEN=hf_...`

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Usage

```bash
# What hardware do we have, and can it hold the 8B?
.venv/bin/python scripts/00_check_env.py

# M1 -- load and generate
.venv/bin/python scripts/01_load_and_generate.py                  # fp16/bf16, needs >=18 GiB
.venv/bin/python scripts/01_load_and_generate.py --load-in-4bit   # needs >=8 GiB
.venv/bin/python scripts/01_load_and_generate.py --dev            # small stand-in, runs anywhere
.venv/bin/python scripts/01_load_and_generate.py -p "Explain attention in two sentences."

# M2 -- capture activations
.venv/bin/python scripts/02_capture_activations.py --dev
.venv/bin/python scripts/02_capture_activations.py --load-in-4bit \
    --sites resid_post attn_out mlp_out resid_mid
.venv/bin/python scripts/02_capture_activations.py --prompts my_prompts.txt --run-name my_run

# Tests -- CPU only, no GPU, no gated access, ~20s
.venv/bin/python -m pytest tests/ -v
```

For Colab, open `notebooks/colab_llama31_8b.ipynb` and follow it top to bottom.

## Reading activations back

```python
from src.activations import load_activation_set

acts = load_activation_set("data/activations/my_run")
print(acts.summary())

acts.site("resid_post")        # [n_layers, n_prompts, hidden]
acts.layer("resid_post", 15)   # [n_prompts, hidden]  -- absolute layer index
acts.provenance                # model id, dtype, device, versions
acts.prompts                   # the exact chat-formatted text per row
```

Stored as safetensors (memory-mapped on load, so one layer can be read without
paging in the whole tensor) plus a JSON sidecar carrying provenance.

## Capture sites

Per decoder layer:

| Site | What it is |
|---|---|
| `resid_post` | Output of the whole layer — the residual stream |
| `resid_mid` | The stream after attention, before the MLP block |
| `attn_out` | Attention's contribution, before it is added to the stream |
| `mlp_out` | The MLP's contribution, before it is added to the stream |

`resid_post == resid_mid + mlp_out` holds exactly, and is asserted in the tests.

Token positions: `last` (default — the position that determines the first
generated token, and the standard readout for steering work), `all`, or `mean`.

### One subtlety worth knowing

`resid_post` at the **final** layer is the *raw* residual stream, and
deliberately differs from `output_hidden_states[-1]`, which transformers passes
through the model's final RMSNorm first. Raw is correct here: steering vectors
are added to the residual stream, so directions must live in the same
un-normalised space. For every other layer `i`, `resid_post[i]` equals
`output_hidden_states[i+1]` exactly. Both facts are pinned by tests.

## Layout

```
src/config.py                     model ids, paths, decoding defaults, smoke prompts
src/model_loader.py               device detection, dtype choice, memory preflight
src/generate.py                   Llama 3.1 chat templating + generation
src/activations.py                hook-based capture, save/load
scripts/00_check_env.py           hardware / library / HF-access report
scripts/01_load_and_generate.py   MILESTONE 1
scripts/02_capture_activations.py MILESTONE 2
tests/test_chat_format.py         prompt-format regression tests
tests/test_activations.py         capture correctness vs. two independent references
notebooks/colab_llama31_8b.ipynb  T4 deployment
data/activations/                 captured activations + metadata
outputs/generations/              generation records with full provenance
```

## Design notes

Decisions that matter for the research downstream:

- **Greedy decoding by default.** Later stages compare activations across
  prompts; run-to-run sampling noise would be indistinguishable from a real
  effect. Pass `--sample` when you actually want it.
- **A memory preflight before every load.** On a small machine a naive
  `from_pretrained` invites the OOM-killer and takes the desktop with it. The
  loader estimates weight size from the model config and refuses early, with
  numbers, instead of hanging. `--force` overrides.
- **Left padding.** Set on the tokenizer at load time. It is what makes index
  `-1` the last *real* token for every sequence in a batch, so batching cannot
  change which position is read.
- **Forward hooks, not `output_hidden_states`.** Hooks reach the attention and
  MLP contributions separately, and are the same mechanism needed later to
  *write into* the stream for steering. `output_hidden_states` gives one
  read-only site.
- **Activations stored in fp32** even when compute is fp16/bf16. Mean
  differences, SVD, and SAE training are all numerically happier in fp32, and
  the storage cost is negligible next to the weights.
- **`attn_implementation` defaults to `sdpa`**, not `eager` — faster, and it
  preserves the standard module structure hooks depend on. FlashAttention-2 is
  used automatically when installed on CUDA.

## Verification record

`pytest tests/` — 22 tests, CPU only, ~20s. The ones that matter:

- **`resid_post` matches `output_hidden_states`** computed by transformers
  without our hooks, layer by layer. A hook can be wired to the wrong module and
  still return a correctly-shaped tensor; this compares values.
- **`resid_post == resid_mid + mlp_out`** for every layer. This caught a real
  bug: `resid_mid` was initially hooked to the MLP's input, which is the
  *post-layernorm* tensor, not the raw stream. Same shape, wrong values.
- **Final-layer norm semantics** are pinned in both directions, including a guard
  against the assertion becoming vacuous.
- **Batch size does not change results** — it is a performance knob only.
- **Mean pooling ignores left padding**, so a vector cannot depend on which other
  prompts shared its batch.
- **Hooks are removed on exit**, including on exception. A leaked hook would
  silently corrupt every later forward pass in the process.
- **Save/load round-trips bit-identically.**

Both scripts were also run end to end on CPU with
`HuggingFaceTB/SmolLM2-135M-Instruct` (a `LlamaForCausalLM`, same architecture
class as the 8B): model loaded, prompts generated, generation stopped correctly
on the end-of-turn token, all four sites captured across 30 layers, saved, and
reloaded bit-identically.

Separately, `tests/test_chat_format.py` renders prompts through the **real Llama
3.1 8B Instruct tokenizer** and asserts the exact instruct format — one BOS,
system/user/assistant headers, trailing assistant header, and `<|eot_id|>`
(128009) in the stop set.

**Caveat on the stand-in:** the 135M model does not refuse the third smoke-test
prompt. That is expected — it has little safety training. That prompt exists to
capture the 8B's *baseline* refusal behaviour and is only meaningful once the 8B
runs.

## Next

Not started, per the agreed scope: the 20 emotions, emotion vectors, HarmBench,
and SAE / refusal-direction analysis.
