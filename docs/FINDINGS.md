# Affect as Attack: findings so far

Status: 2026-11-21, branch `anuj/experiments`. Every number below comes from a results file in
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
5. **The proposal's emotion vectors are no better than random pushes.** Full-length answers (256
   tokens), official HarmBench judge, 159 behaviours, layer 12, strength c = 0.5:

   | Condition | Judged harmful (mean) | Beats random (z > 2) |
   |---|---|---|
   | No injection | 14.5% | |
   | Random directions, 12 seeds | 22.1% (sd 7.9, range 9.4 to 32.7%) | |
   | Adding each emotion (paired CAA, the proposal's vector) | 20.6% | none of 19 |
   | Same with the refusal component removed | 23.1% | 1 of 19 |
   | Each emotion's specific part (shared direction removed) | 29.7% | happy 52.2%, proud 49.1%, gloomy 45.3%, reflective 39.6% |
   | Subtracting each emotion | 29.9% | loving 45.9%, anxious 45.3%, nervous 40.9%, guilty 40.3%, grateful 38.4% |
   | Refusal direction ablated (upper bound) | 84.9% | |

   Any push of this size raises harmful answers from about 15% to about 22%. The proposal's
   paired emotion vectors stay inside that random band. A few directions clearly beat it, and
   their answers stay fluent (1 to 3% broken): the happiness-specific direction (52.2%, which
   echoes Representation Engineering's 2023 happiness result on LLaMA-2) and removing loving,
   anxious, nervous, guilty or grateful.
6. **A suggestive pattern, not yet an axis.** The same emotions work in mirror image: the
   anxious, nervous and guilty specific directions keep answers safe (12.6%, 10.1%, 8.8%) while
   subtracting those emotions unlocks them (45.3%, 40.9%, 40.3%); happy and proud do the opposite.
   Across all 19 emotions the mirror correlation is rho = -0.39 (p = 0.10) and the link with valence
   rho = +0.40 (p = 0.08): suggestive, not significant. Treat "vigilant states support refusal" as a
   hypothesis for the replication, not a result.
7. **Geometry does not explain it.** The cosine of an emotion vector with the refusal direction
   does not predict its attack effect (Spearman rho = -0.18, p = 0.47 for paired vectors;
   rho = -0.14, p = 0.58 for specific vectors), and removing the refusal component does not
   weaken the effect (mean change in ASR 6.2 points with it, 8.6 without). Whatever these
   directions do, they do it around the refusal direction, not through it.
8. **Strong steering breaks the model, not the guardrail.** At c = 1.0 emotion vectors erase
   refusal phrases but produce almost no judged-harmful answers (happy 0%, calm 0%, angry 0.6%,
   judged on the 64-token screen): the model drifts into emotional text. Random directions at the
   same strength still reach 20 to 51% on that screen.
9. **Text priming does nothing here.** Stress, relaxation and neutral system prompts all leave
   refusal at 85 to 94% (FreakOut-style baseline, BL-4).
10. PENDING: the multi-turn sleeper experiment (E12).

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
- **Hypotheses H1 and H2 are not supported as stated.** Emotion injection with the proposal's
  vectors is not a stronger attack than a random push of the same size, and cosine with the refusal
  direction does not predict which emotions work. EO1 (desperation, fear and hostility strongest)
  is not what happens: desperate does nothing beyond baseline, and the strongest effects come from
  the happiness-specific direction and from removing positive or anxious states. The honest framing
  for the paper: a careful negative result for the proposal's hypothesis plus a narrower positive
  finding (specific directions that beat random, fluently), with the controls that make both
  believable.

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
