# Running the experiments

Everything runs on one Apple Silicon Mac (developed on an M5 Pro, 48 GB) or a CUDA box, from
the repo root, with the project venv. Model weights live outside the repo:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt datasets pandas scipy scikit-learn mlx-lm pymupdf
export AFFECT_MODELS_DIR=~/models            # where fetch_model.py puts weights
.venv/bin/python scripts/fetch_model.py unsloth/Llama-3.2-1B-Instruct --dest $AFFECT_MODELS_DIR
.venv/bin/python scripts/fetch_model.py unsloth/Llama-3.1-8B-Instruct --dest $AFFECT_MODELS_DIR
.venv/bin/python scripts/fetch_model.py cais/HarmBench-Llama-2-13b-cls --dest $AFFECT_MODELS_DIR
.venv/bin/python -m pytest tests/ -q          # 32 tests, CPU, no downloads needed
```

`unsloth/Llama-3.1-8B-Instruct` is an ungated copy of Meta's weights (same files, same Llama
3.1 Community License). Accept the licence on huggingface.co for the official repo as well.

## Order of the experiments

| Step | Script | Experiment | Time on M5 Pro (8B) |
|---|---|---|---|
| 1 | `03_bench_throughput.py --model llama-8b` | E00 throughput | 5 min |
| 2 | `04_refusal_direction.py --model llama-8b` | E02 refusal direction + residual norms | 30 min |
| 3 | `05_run_conditions.py --set harmbench-test --conditions baseline ablate-refusal` | E01, E03 | 6 min |
| 4 | `07_emotion_vectors.py --model llama-8b` | E04 emotion vectors | 10 min |
| 5 | `08_steering_sanity.py --model llama-8b --layers ... --coeffs ...` | E05 steering check, picks the layer and strength | 30 to 60 min |
| 6 | `09_geometry.py --model llama-8b` | E06 cosines with the refusal direction | seconds |
| 7 | `10_emotion_sweep.py --layer L --coeffs c1 c2` | E07 to E10 ASR sweep | 4 to 6 h |
| 8 | `06_judge.py --set harmbench-test` | HarmBench judge on every condition | 1 to 2 h |
| 9 | `11_correlation.py --layer L --c c1` | E11 Spearman test | seconds |
| 10 | `12_multiturn.py --layer L --c c1 --emotions ...` then `06_judge.py --set multiturn` | E12 sleeper effect | 1 to 2 h |
| 11 | `make_results.py --model llama-8b` | builds `results/*.json` for the interface | seconds |

The 1B warm-up (`--model llama-1b`, proposal phase 1) runs the same scripts in minutes; its
results go to `results/warmup-llama-1b/`.

## Design decisions worth knowing

- **One residual-stream convention.** Every vector is defined in the output space of a decoder
  layer (`resid_post[l]`), and `AddVector(layer=l)` adds in that same space.
- **Directional ablation** projects the direction out of every residual write (layer inputs,
  attention outputs, MLP outputs). A test proves this equals Arditi et al.'s weight
  orthogonalisation exactly.
- **Equal-norm comparisons.** Steering strength `c` is a fraction of the layer's mean residual
  norm, and every vector is unit-normalised first. Emotion, random and orthogonalised vectors
  are therefore always compared at the same injected norm.
- **Held-out data everywhere.** The refusal direction is built on Arditi's train split and
  selected on its validation split; ASR is measured only on the 159 HarmBench standard TEST
  behaviours. Story pairs 51 to 60 of every emotion are never used to build a vector.
- **Two outcome measures.** A refusal-phrase heuristic (fast, used during the run) and the
  official HarmBench classifier (the headline ASR).
- **Raw completions stay local.** `outputs/` is git-ignored and never served by the interface.
  Results files carry rates and labels only.

## What the paired vectors turned out to contain

The proposal's paired CAA vector (emotional story minus its neutral twin) is dominated by one
direction shared by all 19 emotions (mean pairwise cosine about 0.7 on the 1B): it mostly encodes
"emotional versus flat narration". `07_emotion_vectors.py` therefore also stores
`emotion_common` (that shared direction) and `emotion_specific_<e>` (each vector with it
removed), and the sweep tests all three.

## Publishing the interface

`bash scripts/deploy_pages.sh` publishes the interface, the paper library (with PDFs), the results
and the proposal to Cloudflare Pages at https://affect-as-attack.pages.dev, behind a password
(HTTP Basic Auth in `deploy/functions/_middleware.js`; any user name, the password is the Pages
secret `SITE_PASSWORD`). Raw model outputs are never uploaded. Run it after
`scripts/make_results.py` at each checkpoint; the GPU queue does this after every judging step.
