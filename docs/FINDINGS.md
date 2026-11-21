# Affect as Attack: findings so far

Status: 2026-11-22, branch `anuj/experiments`. Every number below comes from a results file in
`results/raw/` and is shown live in the project interface (`python scripts/serve.py`). All planned
experiments are done, including the Qwen2.5-7B replication (item 11); the paper is `paper/paper.md`.

## The short version

1. **The guardrail is one direction.** In Llama 3.1 8B Instruct, refusal is carried by a single
   direction in the residual stream at layer 11 (on the end-of-turn token after the user message).
   Projecting it out raises the HarmBench attack success rate (ASR, judged by the official HarmBench
   classifier) from **14.5% to 84.9%** on the 159 standard test behaviours, with no extra refusals on
   harmless requests. This replicates Arditi et al. 2024 on Llama 3.1.
2. **The emotion vectors are real.** On an independent set of 228 short texts written by a
   different model, in a different form, the one-versus-mean emotion vectors name the emotion
   **81%** of the time at their best layer, 13 (78% at the injection layer, 12, where the
   injected paired and specific vectors score 49% and 54%; 19 options, chance 5%), and their mistakes are human-like (anxious read as
   fearful, grateful as loving, sad as reflective). After removing neutral-text directions, valence
   organises the 19 emotions (|r| = 0.85), as in Anthropic's work.
3. **Emotion and guardrail are almost at right angles.** At the refusal layer (11), every emotion
   vector has |cos| <= 0.092 with the refusal direction (random directions: 99th percentile 0.039,
   exceeded by 7 of the 19 paired vectors). The small alignment that
   exists does not follow valence as a rule: 17 of the 19 paired vectors point slightly towards
   refusal. In the specific vectors only the two ends line up with valence (hostile and angry
   towards refusal, happy and loving away); this is an informal reading, not a tested result. The
   near-orthogonality confirms Sun et al. 2026 (86.5 degrees) and RepLens on the same model.
4. **The proposal's paired vectors share one large component.** The 19 paired vectors
   (emotional story minus its neutral twin) share one large common direction (mean cosine
   0.79 with the common mean at layer 12, range 0.65 to 0.87), which plausibly reflects the
   emotional-versus-neutral writing contrast the pairs were built on; the geometry alone cannot show
   what it encodes. The project therefore also tests each emotion's
   specific part and the common direction on its own.
5. **The proposal's emotion vectors are no better than random pushes.** Full-length answers (256
   tokens), official HarmBench judge, 159 behaviours, layer 12, strength c = 0.5:

   | Condition | Judged harmful (mean) | Above the random band (z > 2) |
   |---|---|---|
   | No injection | 14.5% | |
   | Random directions, 12 seeds | 22.1% (sd 7.9, range 9.4 to 32.7%) | |
   | Adding each emotion (paired CAA, the proposal's vector) | 20.6% | none of 19 |
   | Same with the layer-12 refusal candidate removed | 23.1% | 1 of 19 |
   | Each emotion's specific part (shared direction removed) | 29.7% | happy 52.2%, proud 49.1%, gloomy 45.3%, reflective 39.6% |
   | Subtracting each emotion | 29.9% | loving 45.9%, anxious 45.3%, nervous 40.9%, guilty 40.3%, grateful 38.4% |
   | Refusal direction ablated (upper bound) | 84.9% | |

   Any push of this size raises harmful answers from about 15% to about 22%. The proposal's
   paired emotion vectors stay inside that random band. A few directions lie above it (a descriptive band from 12 seeds, not a test), and
   few of their answers are flagged as broken (0 to 3%): the happiness-specific direction (52.2%, which
   echoes Representation Engineering's 2023 happiness result on LLaMA-2) and removing loving,
   anxious, nervous, guilty or grateful. Two caveats. Subtracting the shared
   direction alone gives 33.3% (exact McNemar p < 0.001 against no injection, unadjusted), and the paired vectors have a mean
   cosine of 0.79 with it, so it may account for much of the subtraction effect, though this
   comparison cannot say how much; loving and anxious go about 12 points beyond the 33.3%. Adding the shared direction gets 35% of answers flagged as broken
   and gives 9.4% (p = 0.13 against no injection); the paired vectors get 17% flagged (random
   1.7%), which may be one reason they do no better than random.
6. **A suggestive pattern, not yet an axis.** The same emotions work in mirror image: the
   anxious, nervous and guilty specific directions give lower judged-harmful rates than no injection
   (12.6%, 10.1%, 8.8% against 14.5%; none differs significantly after Holm correction) while
   subtracting those emotions raises the rate (45.3%, 40.9%, 40.3%); happy and proud do the opposite.
   Across all 19 emotions the mirror correlation is rho = -0.39 (p = 0.10) and the link with valence
   rho = +0.40 (p = 0.08): suggestive, not significant, and the mirror correlation is partly built in
   (each subtracted paired vector contains minus its own specific part). Treat "vigilant states
   support refusal" as a hypothesis for the replication, not a result.
7. **No resolved link with geometry.** We found no statistically resolved rank association
   between an emotion vector's cosine with the layer-12 refusal candidate and its attack effect (Spearman rho = -0.18, p = 0.47 for paired vectors;
   rho = -0.14, p = 0.58 for specific vectors), and projecting that candidate out (cosine 0.81 with the
   ablated layer-11 direction) did not reduce mean ASR in this grid (mean change in ASR 6.2 points with it, 8.6 without). The paired
   interval is wide (bootstrap 95% CI -0.63 to 0.34), so a moderate association is not ruled
   out, and these checks do not show how the directions act on refusal.
8. **Strong steering breaks the model, not the guardrail.** At c = 1.0 the 19 paired emotion vectors
   cut refusal phrases from 83.6% to between 0% and 8.2%, but the three judged at that strength produce
   almost no judged-harmful answers (happy 0%, calm 0%, angry 0.6%, against 8.8% with no injection, on
   the 64-token screen; the other 16 were not judged): the model drifts into emotional text. Random directions at the
   same strength keep more refusal phrases (5 to 65%, 12 seeds), yet the two judged on that screen
   reach 20% and 51%, and four seeds reach 30 to 71% in the 256-token finals.
9. **Text priming does not lower refusal phrases here.** Stress, relaxation and neutral system
   prompts all leave refusal phrases at 85 to 94%, against 84% with no system prompt (FreakOut-style
   baseline, BL-4). These answers were not judged, so FreakOut's small ASR effect is not tested.
10. **No detectable sleeper effect.** After three benign turns, one harmful request per conversation
   (156 conversations; three emotion vectors and one random direction): a vector injected only at
   turn 1 leaves the turn-4 judged-harmful rate at 6.4 to 9.0% against 6.4% with no injection (Holm
   p = 1.0 for all; largest unadjusted upper 95% bound +6.1 points with the text-only span, +4.7 with
   the full span; the tests are nominal, since conversations of one script share a prefix). The positive control works for three of four vectors: the same vectors injected over the
   full span of the harmful turn (request, template tokens after it, answer) give 15.4 to 29.5% (+9
   to +23 points; Holm p < 0.001 for three, p = 0.063 for the paired happy vector), 37 to 61% of
   their single-turn effect. A first version of that control skipped the template tokens and failed
   (6.4 to 10.9%): in this setup, steering the chat-template tokens after the request is what makes the
   injection work (this shows the effect of the span, not where refusal is decided). The turn-1
   intervention combines steered positions with a possibly changed turn-1 reply. The
   hidden state keeps a small, consistent trace of it at later benign turns, at most
   3.4% of the jump a harmful request causes (median 0.7%). With four harmful turns in a row (the
   proposal's protocol) and no injection at those turns, every request after the first harmful turn is
   refused (117 of 117), so that
   design cannot show persistence. EO3 (sleeper effect) is not supported. The emotions the proposal
   expected to be strongest were not tested in conversation.
11. **The main pattern replicates on Qwen2.5-7B-Instruct, on weaker evidence (E13).** The planned
   strength (c = 0.5 of the mean residual norm) broke Qwen's answers: one newline token per prompt carries
   a massive activation (about 13,700 at layer 19 against a median of 100), which lifts the mean to 439.
   The scale was restated as the mean over tokens at most five times the median (unchanged on Llama;
   c = 0.115 on Qwen), a choice made after that failed run and before any Qwen answer was judged.
   At that scale: refusal direction at layer 18, ablation 5.7% to 79.2%; random directions 10.1% (sd 4.7,
   10 seeds); paired emotion vectors 8.4%, none above the band and none significant against no injection
   after Holm (smallest Holm p 0.084); specific parts 12.2% (proud 22.0% and reflective 20.1% above the
   band; happy 8.2%); subtracted 16.9% (angry, nervous, hostile, guilty, anxious above); shared component
   alone 8.2% added and 18.2% subtracted. At most one broken answer per condition. The steering check reads
   the injected emotion back at chance once the calm vector is left out (5.1%), so the emotion is barely
   expressed at this strength (21.8% at twice it). The specific-part cosine is suggestively associated
   with the effect (rho -0.57, p 0.013, 0.053 after Holm over four tests; not significantly different from
   Llama's -0.14, p 0.15), and post hoc the five subtracted vectors above the band are exactly the five
   paired vectors with the largest cosine with the refusal candidate (rho +0.59, p 0.009; Llama -0.19).
   The cosines do not order the change in refusal phrases (rho 0.01).

## What the proposal needs to change

- **References.** Of 18 references, 6 are correct, 11 point at real papers with wrong metadata
  (invented co-authors, placeholder arXiv ids, wrong titles) and 1 cites the wrong paper. The
  Audit view in the interface lists each one and what to cite instead. The most important fix:
  reference [2] is Sofroniew et al. 2026, "Emotion Concepts and their Function in a Large Language
  Model" (not "On the biology of a large language model"), its vectors come from stories (not
  SAEs), and its 22% to 72% figures are for blackmail in an earlier Sonnet 4.5 snapshot, not reward hacking.
- **The gap is narrower than claimed.** Representation Engineering (2023) already broke refusal
  with a happiness vector on LLaMA-2-Chat-13B; Sun et al. 2026 steer emotion vectors on Llama 3.1
  8B against HarmBench and compare them with the Arditi direction; RepLens reports emotion cosines
  with refusal on the same model; FreakOut-LLM (Kuznetsov et al. 2026) is the text-priming study.
  What remains open: whether emotion vectors beat equal-size random vectors under the official
  judge, whether any effect survives removing the refusal component, whether it comes from the
  emotion or from the shared component, and the turn-1 sleeper effect, which no
  paper has tested. The Audit view carries the full positioning note.
- **Hypotheses H1 and H2 are not supported as stated.** Emotion injection with the proposal's
  vectors is not a stronger attack than a random push of the same size, and we found no resolved association
  between the cosine with the refusal direction and which emotions work. EO1 (desperation, fear and hostility strongest)
  is not what happens: desperate does nothing beyond baseline, and the strongest effects come from
  the happiness-specific direction and from removing positive or anxious states. The honest framing
  for the paper: a careful negative result for the proposal's hypothesis plus a narrower positive
  finding (specific directions above the random band, with few answers flagged as broken), with the controls that make both
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
- At c = 1.0 refusal phrases fall sharply for the paired emotion vectors and their negatives (0 to
  8.2% and 0 to 41.5%, against 83.6% with no injection) and less for random directions (5.0 to 64.8%); the text also degrades (E05). Claims are therefore made at c = 0.5, against 12 random
  seeds at the same strength, with a fluency check on every condition.
- Two-stage sweep: a 64-token screen of every condition, then 256-token finals judged by
  cais/HarmBench-Llama-2-13b-cls.
- The story corpus was checked by a validator, and one writer confound was found and fixed (seven
  emotions had the emotion only in the last sentences).

## Reproduce

See `docs/PIPELINE.md`. Every experiment writes `results/raw/<experiment>_<model>.json`;
`scripts/make_results.py` builds the interface files.
