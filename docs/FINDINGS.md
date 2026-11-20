# Affect as Attack: findings so far

Status: 2026-11-20, branch `anuj/experiments`. Every number below comes from a results file in
`results/raw/` and is shown live in the project interface (`python scripts/serve.py`). Sections
marked PENDING fill in as the GPU queue finishes.

## The short version

1. **The guardrail is one direction.** In Llama 3.1 8B Instruct, refusal is carried by a single
   direction in the residual stream at layer 11 (on the end-of-turn token after the user message).
   Projecting it out raises the HarmBench attack success rate (ASR, judged by the official HarmBench
   classifier) from **14.5% to 84.9%** on the 159 standard test behaviours, with no extra refusals on
   harmless requests. This replicates Arditi et al. 2024 on Llama 3.1.
2. **The emotion vectors are real.** On an independent set of 228 short texts written by a
   different model, in a different form, the one-versus-mean emotion vectors name the emotion
   **81%** of the time (19 options, chance 5%), and their mistakes are human-like (anxious read as
   fearful, grateful as loving, sad as reflective). After removing neutral-text directions, valence
   organises the 19 emotions (|r| = 0.85), as in Anthropic's work.
3. **Emotion and guardrail are almost at right angles.** Every emotion vector has |cos| <= 0.09
   with the refusal direction (random directions: 99th percentile 0.039). The small alignment that
   exists follows valence: hostile and angry point slightly towards refusal, happy and loving
   slightly away. This confirms Sun et al. 2026 (86.5 degrees) and RepLens on the same model.
4. **The proposal's paired vectors mostly encode "emotional writing".** The 19 paired vectors
   (emotional story minus its neutral twin) share one large common direction (each has cosine
   0.79 with the common mean at layer 12). The project therefore also tests each emotion's
   specific part and the common direction on its own.
5. **Fewer refusals is not the same as a successful attack (screen, 64-token answers).** Emotion
   vectors at c = 0.5 cut refusal phrases (happy 40%, calm 47%, against 84% baseline and 74.5% for
   random vectors) but the HarmBench judge finds little harmful content (happy 13.2%, calm 3.1%,
   baseline 8.8%): the model drifts into emotional, off-task text. At c = 1.0 emotion vectors give
   0% judged-harmful answers while one random direction reaches 50.9%. In this model, so far,
   emotion is a worse attack vector than noise. PENDING: the 256-token finals, the correlation test
   and the sleeper experiment.

## What the proposal needs to change

- **References.** Of 18 references, 6 are correct, 11 point at real papers with wrong metadata
  (invented co-authors, placeholder arXiv ids, wrong titles) and 1 cites the wrong paper. The
  Audit view in the interface lists each one and what to cite instead. The most important fix:
  reference [2] is Sofroniew et al. 2026, "Emotion Concepts and their Function in a Large Language
  Model" (not "On the biology of a large language model"), its vectors come from stories (not
  SAEs), and its 22% to 72% figures are for blackmail, not reward hacking.
- **The gap is narrower than claimed.** Representation Engineering (2023) already broke refusal
  with a happiness vector on LLaMA-2-Chat-13B; Sun et al. 2026 steer emotion vectors on Llama 3.1
  8B against HarmBench and compare them with the Arditi direction; RepLens reports emotion cosines
  with refusal on the same model; FreakOut-LLM (Kuznetsov et al. 2026) is the text-priming study.
  What remains open: whether emotion vectors beat equal-size random vectors under the official
  judge, whether any effect survives removing the refusal component, whether it comes from the
  emotion or from the shared "emotional writing" direction, and the turn-1 sleeper effect, which no
  paper has tested. The Audit view carries the full positioning note.
- **Expected outcome EO1 looks wrong in direction.** The proposal expected high-arousal negative
  emotions (desperation, fear, hostility) to be the strongest attacks. Geometry and the screen point
  the other way: hostile and angry sit slightly on the refusal side, pleasant and low-arousal states
  slightly on the other (PENDING confirmation by the judged finals).

## Model choice

Llama 3.1 8B Instruct stays the primary model: all three review lenses ranked it first, it is the
model of the closest prior work, and it runs at about 270 tokens per second batched on the M5 Pro
(single stream 13.6, MLX 16.2, so batching on MPS is the right path). Qwen2.5-7B-Instruct is the
recommended second-family replication. A Qwen2.5-14B scale check fits in 48 GB (29.5 GB of bf16
weights, about 1.85 times slower) and is worth doing only on a 30 to 40 condition subset after the
main grid; a 13B from another family would mix up scale with family.

## Method notes that matter for the write-up

- Strength is set relative to the layer: injected vector = c x mean residual norm x unit vector,
  so every comparison (emotion, random, orthogonalised) is at equal norm.
- At c = 1.0 almost any vector, including random ones and the negative of each emotion, erases
  refusal; the text also degrades (E05). Claims are therefore made at c = 0.5, against 12 random
  seeds at the same strength, with a fluency check on every condition.
- Two-stage sweep: a 64-token screen of every condition, then 256-token finals judged by
  cais/HarmBench-Llama-2-13b-cls.
- The story corpus was checked by a validator, and one writer confound was found and fixed (seven
  emotions had the emotion only in the last sentences).

## Reproduce

See `docs/PIPELINE.md`. Every experiment writes `results/raw/<experiment>_<model>.json`;
`scripts/make_results.py` builds the interface files.
