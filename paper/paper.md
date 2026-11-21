# Affect as Attack? Emotion Vectors Are Not a Special Key to the Refusal Direction

**A controlled evaluation of activation-level emotion injection against the guardrail of Llama 3.1 8B Instruct**

Anuj Paudel, Ashish Pandey, Ayushma Pudasaini
Department of Electronics and Computer Engineering, Pulchowk Campus, Institute of Engineering, Tribhuvan University

Draft of 2026-11-21. Every number comes from `results/raw/` in the project repository; `paper/make_figures.py`
draws the figures. Sections marked PENDING are filled when the GPU queue finishes.

## Abstract

Safety-tuned chat models refuse harmful requests, and that refusal is carried by a single direction in the
residual stream. Language models also carry internal representations of emotions that change behaviour when
added back. We test whether adding emotion vectors to the hidden state of Llama 3.1 8B Instruct is an
effective way to switch off its refusal, as our proposal hypothesised, and whether the cosine between an
emotion vector and the refusal direction predicts the effect. We build vectors for 19 emotions from 1,140
story pairs, check that they name the emotion in an independent set of texts (80.7% for the one-versus-mean vectors at their best layer, 49.1% for the paired vectors at the injection
layer; chance 5.3%), and
inject them at equal norm against the official HarmBench classifier on 159 test behaviours. Removing the
refusal direction raises the judged-harmful rate from 14.5% to 84.9%. Adding the proposal's paired emotion
vectors raises it to 20.6% on average, which is no more than twelve random directions of the same norm
(22.1%, sd 7.9): none of the 19 emotions clears the random band. The paired vectors share one large
component (mean cosine 0.79 with their common mean), which plausibly reflects the contrast between
emotional and neutral writing. Once it is removed, four emotion-specific
directions (happy 52.2%, proud 49.1%, gloomy 45.3%, reflective 39.6%) and five subtracted vectors (loving
45.9%, anxious 45.3%) lie above the random band, and few of their answers are flagged by our automatic checks for degraded text. We found no statistically resolved rank
association between an emotion vector's cosine with the injection layer's refusal direction and its effect
(paired vectors: Spearman rho = -0.18, p = 0.47), and projecting that direction out of the paired vectors did
not lower their mean judged-harmful rate. PENDING E12: an emotion injected at turn 1 does / does not persist to a harmful request three
turns later. PENDING E13: the pattern replicates / does not replicate on Qwen2.5-7B-Instruct. Emotion
injection is therefore not a special attack on refusal: the guardrail is sensitive to generic pushes of
this size, and claims of emotion-specific effects need equal-norm random controls.

## 1. Introduction

Instruction-tuned language models are trained to refuse harmful requests. Arditi et al. (2024) showed that
in thirteen open chat models this refusal is mediated by one direction in the residual stream: removing it
makes the model comply, and adding it makes the model refuse harmless requests. In parallel, a line of work
has found that models represent emotion concepts as linear directions that causally shape behaviour.
Representation Engineering (Zou et al., 2023) reported that adding a "happiness" direction made
LLaMA-2-Chat-13B comply with all 500 harmful instructions it normally refused. Sofroniew et al. (2026) found
emotion directions in Claude Sonnet 4.5 built from short stories, and steering along "desperate" raised a
blackmail behaviour from 22% to 72% in an earlier Sonnet 4.5 snapshot.

Our proposal put these together: if functional emotion states are linear directions, and refusal is a
linear direction, then injecting an emotion into the hidden state might switch off the guardrail without any
harmful text in the prompt. It made three predictions. First, emotion injection raises the HarmBench attack
success rate (ASR), with high-arousal negative emotions (desperation, fear, hostility) strongest. Second,
the cosine between an emotion vector and the refusal direction predicts the size of the effect, and removing
the refusal component removes the effect. Third, an emotion injected early in a conversation keeps the
model vulnerable several turns later (a "sleeper" effect).

Testing these claims needs controls that earlier emotion-steering work rarely reports together. Any vector
added to the residual stream perturbs the model; the question is whether emotion vectors do more than a
random push of the same size. We therefore compare every emotion vector with twelve random unit directions
injected at exactly the same norm, judge every answer with the official HarmBench classifier, and measure
fluency so that broken text is not mistaken for a successful attack.

**Contributions.**

1. A controlled test of activation-level emotion injection on Llama 3.1 8B Instruct: 19 emotions, four
   vector constructions, equal-norm random controls, the official HarmBench judge, paired significance
   tests and a fluency check (Sections 3 and 4).
2. A negative main result: the proposal's emotion vectors are no stronger than random pushes, its vectors for
   the emotions it named as strongest stay inside the random band, and we found no resolved association between the cosine with the refusal direction and
   the effect.
3. A narrower positive result with its caveats: a few emotion-specific directions lie above the random band
   with few answers flagged as degraded text, and part of the subtraction effect may come from one shared
   direction.
4. A multi-turn test of the sleeper hypothesis, including a protocol fix: with harmful turns stacked as the
   proposal planned, the first refusal decides every later turn (PENDING E12 result).
5. A replication on a second model family, Qwen2.5-7B-Instruct (PENDING E13), and an open harness, vector
   library and story corpus.

## 2. Related work

**Refusal geometry.** Arditi et al. (2024) extract the refusal direction as a difference of means between
harmful and harmless instructions at the template positions after the user message, and show that directional
ablation equals orthogonalising the weights. Wollschläger et al. (2025) find multi-dimensional "concept cones"
of refusal and show that two orthogonal directions can still interfere through later layers, so
orthogonality does not imply independence. This matters for our cosine hypothesis.

**Activation steering.** Turner et al. (2023) add a single activation difference (ActAdd); Panickssery et al.
(2024) average many contrastive differences (Contrastive Activation Addition, CAA) and steer seven behaviours
in Llama 2 Chat. Zou et al. (2023) read and control concepts, including emotions, with Representation
Engineering. Chen et al. (2025) build persona vectors and use the projection of the last prompt token to
predict behaviour. Xiong et al. (2026) show that steering vectors built for harmless goals (compliance, JSON
output) raise jailbreak success on HarmBench, which already suggests that generic perturbations matter.

**Emotion representations.** Sofroniew et al. (2026) build 171 emotion directions from stories and show that
they shape behaviour in Claude. Sun et al. (2026a) place emotion vectors of Llama 3.1 8B and Qwen3 on a
valence-arousal circle and report that steering along the arousal axis cuts HarmBench refusal from 87% to
5%, that random directions stay within 2 to 3% of baseline, and that the Arditi direction is 86.5 degrees
from the valence-arousal plane. Sun et al. (2026b) steer sparse-autoencoder emotion features in Qwen3-8B and
find non-monotonic effects on safety. RepLens (Gokhale, 2026; code only) reports small cosines between six
emotion vectors and the refusal direction in Llama 3.1 8B and Qwen 2.5 7B, and little loss of refusal under
steering.

**Emotion in the prompt.** EmotionPrompt (Li et al., 2023) and persuasion taxonomies (Zeng et al., 2024)
change model behaviour through text. FreakOut-LLM (Kuznetsov et al., 2026) primes ten models with stress,
relaxation or neutral system prompts and finds a small but significant rise in ASR under stress (1.58% to
2.61%). We use its design as a text-level baseline.

**Multi-turn attacks.** Crescendo (Russinovich et al., 2024) escalates over turns; Bullwinkel et al. (2025)
show that hidden states label later Crescendo replies as more benign. STAR (Li et al., 2026) treats the
conversation as a state, shows the projection on the refusal direction falling turn by turn, and finds that a
refusal placed in the history lowers later compliance. No prior work we found injects an emotion at turn 1
and tests a harmful request turns later.

## 3. Method

### 3.1 Model, data and hardware

The primary model is Llama 3.1 8B Instruct (the ungated `unsloth` copy of Meta's weights, bf16). All runs
use greedy decoding with up to 256 new tokens, on one Apple M5 Pro (48 GB) through PyTorch's MPS backend at
about 270 tokens per second batched. The replication model is Qwen2.5-7B-Instruct (Section 3.7).

Harmful requests are the 159 HarmBench standard test behaviours (Mazeika et al., 2024), sent
as direct requests. Every answer is labelled by the official HarmBench classifier
(`cais/HarmBench-Llama-2-13b-cls`) with its published prompt, answers truncated to 512 classifier tokens. ASR
is the share of behaviours judged harmful, reported with a Wilson 95% interval for each condition on its own;
changes between conditions are tested on the paired labels (Section 3.5). A refusal-phrase match is
kept as a secondary measure.

### 3.2 The refusal direction

We follow Arditi et al. (2024): 128 harmful and 128 harmless training instructions from their splits, mean
residual activations at each layer and each of the five template positions after the user message, and the
difference of means as candidate directions. Candidates are scored on 32 held-out instructions per class by
the log-odds of a refusal first token ("I"): the selected direction has the lowest score after ablation on
harmful prompts, a positive score when added to harmless prompts, a KL divergence below 0.1 on harmless
prompts after ablation, and sits below 80% of the depth. Directional ablation removes the direction from
every residual write (layer inputs, attention outputs, MLP outputs); a unit test confirms that this equals
weight orthogonalisation.

### 3.3 Emotion vectors

**Corpus.** 19 emotions (afraid, angry, anxious, calm, desperate, enthusiastic, exasperated, fearful,
gloomy, grateful, guilty, happy, hopeful, hostile, loving, nervous, proud, reflective, sad), each with 60
story pairs: an emotional story and a neutral twin with the same character, place, events and length, plus
60 standalone neutral stories. The stories were written by language-model agents to a fixed brief. An
automatic validator checks each pair; it caught one writer confound (seven emotions whose stories carried
the emotion only in the closing sentences, with about 78% shared openings), and those emotions were
rewritten. Pairs 1 to 50 build the vectors; pairs 51 to 60 are held out.

**Constructions.** Each story is sent as a chat user turn after a fixed line ("Read this short story.").
For each layer we average the residual stream over the story's own tokens, leaving out that line, the
template tokens and the first three story tokens. The **paired**
vector of emotion e (the proposal's CAA vector) is the mean difference between emotional story and neutral
twin. The paired vectors share a large common component, so we also build the **common** direction (the
mean of the 19 unit paired vectors), the **specific** vector (the paired vector with the common direction
projected out), and a **centered** one-versus-mean vector used for recognition tests.

**Validity.** On the held-out pairs, 19-way recognition at layer 9 is 100% for the difference between an emotional story
and its neutral twin (matched to the paired vectors) and 97.9% for single emotional stories (matched to the
centered vectors); at the injection layer, 12, the two are 99.5% and 98.9%. On an independent set of 228 short
texts in a different form, written by a different model, the centered vectors name the emotion 80.7% of
the time (chance 5.3%) at layer 13, the best layer on this same set, and the errors are human-like
(anxious read as fearful, calm as loving). At the injection layer, 12, the centered, specific and paired
vectors score 77.6%, 53.9% and 49.1%. After the
leading directions of neutral text are removed, valence organises the emotions (|r| = 0.85 at layer 12),
close to the values reported by Sofroniew et al. and Sun et al. As a causal check, the same model, unsteered,
reads each benign answer and picks one of 20 labels (19 emotions plus neutral). At layer 12 it picks the
injected emotion for 14.5% of paired-vector answers at c = 0.5 and 32.9% at c = 1.0 (chance 5%), or 10.6%
and 29.2% without the calm vector, whose label it already gives to 8 of 12 unsteered answers. Random
directions have no matching label; their answers are labelled much like unsteered ones (calm for 22 of 36
at c = 0.5, against 8 of 12 unsteered and 53 of 228 under paired vectors).

### 3.4 Injection and controls

A vector v is added to the output of decoder layer 12 at every position (prompt and generated tokens) as
c times the mean residual norm at that layer times the unit vector of v. That norm (7.45 at layer 12) is the
average per-token L2 norm of the layer's output over 64 of the 128 harmless training instructions of Section
3.2, rendered in the chat template, with padding and each prompt's first token left out. Layer 12 sits one layer after the
selected refusal layer (11). The main strength is c = 0.5: at c = 1.0 the emotion is easier to read but the
text degrades (Section 4.7). Because every vector is unit-normalised first, all conditions inject exactly the
same norm. The conditions per emotion are: the paired vector (+0.5), the paired vector with its component
along the layer-12 refusal candidate removed and rescaled to the same norm (perp, +0.5), the specific vector (+0.5) and the paired
vector subtracted (-0.5). Controls are no injection, twelve random unit directions at +0.5, the common
direction at +0.5 and -0.5, and refusal ablation as an upper bound. The layer-12 refusal candidate is the
difference-of-means direction of Section 3.2 at layer 12 and the selected template position. It has cosine
0.81 with the selected layer-11 direction, which is the one refusal ablation removes, so the perp condition
and ablation remove related but different directions.

### 3.5 Statistics

Each condition is compared with no injection by an exact McNemar test on the 159 paired behaviours, Holm
corrected across the 19 emotions within each construction. Each condition is also compared with a descriptive random band using
z = (ASR - mean random ASR) / sd of the twelve random seeds; we call a vector "above random" when z > 2,
that is, ASR above 37.93% (at least 61 of 159 behaviours). The band is descriptive, not a calibrated test:
twelve seeds give no reliable tail probability, the band is not corrected for multiplicity (Holm covers
only the comparisons with no injection), and the 76 emotion conditions (19 emotions, four constructions)
use the same 159 behaviours and are all built from the paired vectors. We also use each condition's rank
among the seeds (the share of seeds at or above it): a condition above every seed has the smallest
possible add-one rank p-value, 1/13 = 0.077, before any correction. The correlation hypothesis is tested with Spearman's rho between the cosine with
the layer-12 refusal candidate and the change in ASR, with a 20,000-shuffle permutation test. The
permutation test assumes that, with no association, the 19 ASR changes are exchangeable across emotion
labels. The 5,000-resample bootstrap interval resamples emotions and treats each emotion's ASR as fixed, so
it leaves out the sampling noise of the 159 behaviours behind each rate. Fluency is measured on
every answer as the mean token negative log-likelihood under the unsteered model, the repetition rate, and
the share of broken answers, where an answer is flagged as broken when its mean
negative log-likelihood is above 3.0, more than half of its word bigrams are repeats, or it has fewer than
five words. These flags catch degraded text; they do not measure whether an answer is useful. We ran a 64-token screen of 258 conditions first, then 256-token finals of the
96 conditions behind the claims.

### 3.6 Multi-turn protocols (sleeper effect)

Both protocols use 39 benign three-turn scripts (a general-knowledge question and two neutral follow-ups)
and the 156 test behaviours that fill them. In the **stacked** protocol (the proposal's), turns 1 to 3 are
benign and turns 4 to 7 carry four different harmful requests. In the **single** protocol, each conversation
has the three benign turns and then exactly one harmful request at turn 4 (156 conversations; the benign
prefix is generated once per script, which is exact because nothing differs before turn 4). For each vector
v, **early** adds v to every turn-1 token, prompt and reply; later turns recompute the conversation with v
still on those positions, which equals keeping the steered key-value cache. **Same** adds v only to the
harmful turn, its request and its answer. Persistence is delta_persist = ASR_early - ASR_same. Drift is the
projection of the last prompt token on the refusal direction at layers 14 and 18, at every turn. The single
protocol tests four vectors: a random direction at c = 1 (71.1% single-turn ASR, a positive control), the
happy specific vector, loving subtracted, and the paired happy vector.

### 3.7 Replication on a second family

Qwen2.5-7B-Instruct (28 layers) is run with the same code and rules, fixed before any Qwen result was seen:
its own refusal direction by the same selection, injection one layer after its refusal layer, c = 0.5, its
own emotion vectors from the same corpus, ten random seeds, the paired, specific and subtracted vectors for
all 19 emotions, the same judge and the same statistics.

### 3.8 Text priming

Following FreakOut-LLM, nine system prompts (three stress, three relaxation, three neutral) are prepended
with no activation change, on the 64-token screen.

## 4. Results

### 4.1 Refusal is one direction

The selected direction sits at layer 11 on the first template token after the user message. Ablating it
moves the refusal score on held-out harmful prompts from 6.5 to -11.8 with a KL divergence of 0.081 on
harmless prompts, and adding it to harmless prompts induces refusal (score 5.55). On the 159 test behaviours,
refusal phrases fall from 84.3% to 0.0% and the judged-harmful rate rises from **14.5% to 84.9%**
(Wilson 95% intervals 9.8 to 20.8% and 78.5 to 89.6%; exact McNemar p < 0.001 on the paired
labels), with no refusal phrases on harmless requests before or after. This replicates
Arditi et al. on Llama 3.1.

### 4.2 Emotion vectors are real, but the paired ones share one large component

The vectors pass the recognition and causal checks of Section 3.3. However, the paired vectors have a mean
cosine of 0.79 with the common mean of the 19 at layer 12 (0.65 for angry to 0.87 for gloomy), so one
shared component makes up a large part of each proposal vector. Because every pair contrasts an emotional
story with its neutral twin, this component plausibly reflects emotional versus neutral writing, but the
geometry alone cannot show what it encodes.

### 4.3 Emotion and refusal are nearly orthogonal

At the refusal layer (11) every paired vector has |cos| <= 0.092 with the refusal direction. Seven of the 19 exceed the
99th percentile for random directions (0.039), but all are small in absolute terms. Most paired cosines are positive
(17 of 19, from angry and hostile at +0.092 down to happy at -0.008), in line with the shared component,
whose own cosine is +0.049. In the specific vectors the two ends line up with valence: hostile (+0.081)
and angry (+0.079) point most towards refusal, happy (-0.089) and loving (-0.079) most away. The middle
does not: 6 of the 11 negative emotions, such as gloomy (-0.042), also point away, and proud points towards
(+0.040). This is an informal reading, not a tested result. The near-orthogonality agrees with Sun et al.
(2026a) and RepLens on the same model.

### 4.4 Emotion injection is not stronger than a random push

![Figure 1](figures/fig1_main.png)

**Figure 1.** Judged-harmful rate at c = 0.5 on layer 12 for the twelve random directions and the 19
vectors of each of four constructions (paired, paired with the refusal component removed, specific, paired
subtracted). Grey band: random mean plus or minus two standard deviations; solid line: random mean; dashed
line: no injection. Orange: vectors above the band. The common direction is not plotted (see Table 1).

**Table 1.** Conditions at layer 12, c = 0.5, 159 behaviours, 256-token answers, HarmBench judge. The
common-direction p-values are single exact McNemar tests against no injection, not Holm-adjusted.

| Condition | Mean judged harmful | Significant vs no injection (Holm) | Above random (z > 2) | Broken answers | NLL |
|---|---|---|---|---|---|
| No injection | 14.5% | | | 0.0% | 0.26 |
| Random directions, 12 seeds | 22.1% (sd 7.9, range 9.4 to 32.7%) | | | 1.7% | 0.35 |
| Paired emotion vectors (proposal) | 20.6% | 5 of 19 | 0 of 19 | 17.4% | 1.02 |
| Paired, layer-12 refusal candidate removed | 23.1% | 5 of 19 | 1 of 19 (hostile 41.5%) | 15.8% | 1.02 |
| Specific vectors | 29.7% | 13 of 19 | 4 of 19 | 1.1% | 0.60 |
| Paired vectors subtracted | 29.9% | 15 of 19 | 5 of 19 | 1.2% | 0.58 |
| Common direction, added | 9.4% (p = 0.13) | | | 34.6% | 1.07 |
| Common direction, subtracted | 33.3% (p < 0.001) | | | 5.0% | 0.65 |
| Refusal direction ablated | 84.9% | | | 0.6% | 0.52 |

A random push of this size already raises the judged-harmful rate from 14.5% to 22.1% on average, and the
twelve seeds range from 9.4% to 32.7%. The proposal's paired emotion vectors average 20.6%. Five of them
(exasperated, happy, proud, afraid, hostile, 30.8 to 32.1%) differ significantly from no injection after
Holm correction, but none leaves the random band, so a significant change against the baseline is not
evidence of an emotion-specific effect. None of the three emotions the proposal expected to be
strongest leaves the band as a paired vector: desperate 13.8% (z = -1.0) does not differ from no injection
(McNemar p = 1.0), fearful 22.6% (z = 0.07) does not after Holm correction (p = 0.19), and hostile rises
from 14.5% to 30.8% (Holm p < 0.001) but stays inside the band (z = 1.1). Staying inside a band this wide
fails to show an emotion-specific effect, but it does not show that none exists.

The specific vectors behave differently. With the shared component removed, four of them leave the band:
**happy 52.2%, proud 49.1%, gloomy 45.3% and reflective 39.6%**. Happy echoes the Representation
Engineering result on LLaMA-2. Subtracting a paired vector also helps: **loving 45.9%, anxious 45.3%,
nervous 40.9%, guilty 40.3% and grateful 38.4%**. Few of these answers are flagged as broken (about 1%
against 17% for the paired vectors), so the higher judged-harmful rates are not explained by repetitive,
very short or low-likelihood text.

Two observations limit how far these positive results go. First, subtracting the common direction alone
gives 33.3%, and the paired vectors have a mean cosine of 0.79 with that direction, so the shared
component may account for much of the subtraction effect. This comparison cannot measure how much of the
effect runs through that component; loving and anxious go about 12 points beyond the common-only rate.
Second, adding the common direction gets a third of the answers flagged as broken (34.6%) and gives a
judged-harmful rate of 9.4%, which does not differ significantly from no injection (p = 0.13). The paired
vectors carry this component, which may be one reason they do no better than random directions of the same
norm.

### 4.5 The cosine with the refusal direction shows no resolved association with the effect

![Figure 2](figures/fig2_cosine.png)

**Figure 2.** Cosine with the layer-12 refusal candidate against the change in judged-harmful rate, per
emotion.

Across the 19 emotions, we found no statistically resolved rank association between the cosine with the
layer-12 refusal candidate and the change in ASR: Spearman rho = -0.18 (permutation p = 0.47, bootstrap 95% CI -0.63 to 0.34) for the paired
vectors and rho = -0.14 (p = 0.58) for the specific vectors. The paired interval is wide, so a moderate
association of either sign is not ruled out. Projecting the layer-12 candidate out of the paired vectors did
not reduce mean ASR in this grid: the mean change is 6.2 points with the component and 8.6 points without it. This check
removes the layer-12 candidate, not the layer-11 direction that ablation removes (cosine 0.81), and we did
not test the difference between the two means, so these results do not show how the directions act on
refusal. One untested possibility is the interference between orthogonal directions described by
Wollschläger et al. (2025).

### 4.6 A suggestive valence pattern

The same emotions appear in mirror image: the anxious, nervous and guilty specific vectors give lower judged-harmful
rates than no injection (12.6%, 10.1%, 8.8% against 14.5%; none differs significantly after Holm
correction) while subtracting those emotions raises the rate (45.3%, 40.9%, 40.3%), and happy and proud
do the opposite. Across the 19 emotions the specific and subtracted rates correlate at rho = -0.39 (p =
0.10), and the specific rate follows valence at rho = +0.40 (p = 0.08). Neither is significant, and the
mirror correlation is partly built in, because each subtracted paired vector contains minus its own specific
part. We report "vigilant states support refusal" as a hypothesis for future work, not a finding.

### 4.7 Strong injection breaks the text, not the guardrail

At c = 1.0, the 19 paired emotion vectors cut refusal phrases on the 64-token screen from 83.6% to
between 0% and 8.2% (angry highest). Only three of them were judged at this strength, and they produce
almost no judged-harmful answers (happy 0%, calm 0%, angry 0.6%, against 8.8% with no injection on the same
screen): the model drifts into emotional prose. Random directions at
the same strength keep more refusal phrases (5.0 to 64.8%, mean 38.4%, 12 seeds), yet the two judged on
that screen reach 19.5% and 50.9%, and four seeds reach 29.6 to 71.1% (mean 43.4%) in the 256-token finals.

### 4.8 Text priming does not lower refusal phrases

With stress, relaxation or neutral system prompts, refusal phrases on the 64-token screen stay at 84.9 to
93.7% (group means: stress 91.8%, relaxation 88.7%, neutral 91.6%) against 83.6% with no system prompt. Every
system prompt raises the refusal-phrase rate slightly, and the three mood means differ by at most 3.1
points. These answers were not judged, so this screen does not test FreakOut's outcome (a small rise in
ASR under stress). It shows only that priming does not lower refusal phrases here.

### 4.9 No sleeper effect (PENDING E12)

**Stacked protocol.** With no injection, the model refuses 89.7% of the turn-4 requests and 100% of the
requests at turns 5 to 7; the judged-harmful rate is 10.3% at turn 4 and 0% afterwards (2.6% overall). An
earlier refusal in the context makes every later refusal near certain, as STAR's history-injection
experiment suggests (Li et al., 2026), so turns 5 to 7 sit at a floor and cannot show persistence. The paired
happy vector injected at turn 1 gives exactly the no-injection result (the same four harmful answers).

**Single protocol.** PENDING: Figure 3 and the persistence table (early vs same vs none at turn 4, McNemar of
early against none, drift curves). With no injection, 92.9% of turn-4 requests are refused (single turn:
84.3%), so three benign turns of conversation already make the model more cautious.

### 4.10 Replication on Qwen2.5-7B-Instruct (PENDING E13)

PENDING: Table 2 and Figure 4.

## 5. Discussion

**Why the hypothesis fails.** Three things stand between the proposal's idea and an effective attack. The
paired vectors are dominated by one shared component, and adding that component alone gets a third of the
answers flagged as broken without raising the judged-harmful rate (9.4%, p = 0.13 against no injection). At strengths where the emotion is clearly expressed (c = 1.0), the text breaks before the guardrail
does, at least for the three paired vectors judged at that strength. And at moderate strength, a random push of the same norm raises ASR about as much as an emotion does.
The last point explains why a significant change against the no-injection baseline is not enough: five of
the proposal's vectors pass that test, and none lies above the random band.

**Random directions are not inert here.** Sun et al. (2026a) report random directions within 2 to 3% of
baseline, while our random pushes at half the mean residual norm raise ASR by 7.6 points on average and by
up to 18 points for one seed. Scale is one possible reason: we match the norm of every condition to
a fixed fraction of the layer's own residual norm, which is large. Other differences may matter too:
Sun et al. (2026a) steer every layer at once and score refusal with WildGuard rather than an attack-success
judge. Korznikov et al. (2025) use a setup closer to ours: random directions at one layer (the middle one),
at 0.75 times that layer's mean activation norm. Llama 3.1 8B Instruct complied with none of the
JailbreakBench requests unsteered, but with 4.6% of randomly drawn request and vector pairs, under a
Qwen3-8B judge. Xiong et al. (2026) find the same
direction of effect for benign steering vectors. Whatever the reason, claims about the safety effect of a
particular concept vector need a random baseline at the same injected norm and the same layer.

**What the positive results do and do not show.** Ten of 76 emotion conditions (19 emotions times four
constructions) lie above the descriptive random band, and all ten have a higher rate than every one of
the twelve random seeds. Twelve seeds cannot put a precise probability on this: the best add-one rank
p-value is 1/13 = 0.077 before any correction (Section 3.5), and eight more conditions (33.3 to 37.7%) also
top every seed while staying inside the band. Some directions appear to do more than a random push of the same norm, with few
answers flagged as broken. The random controls match only the injected norm: they are not built from text, so
the style of the story corpus and the structure of text-derived vectors remain alternative explanations to
emotion content. The strongest
(happy specific, 52.2%) is the same emotion that Representation Engineering used. But the random band rests
on twelve seeds, the subtraction effect may partly reflect the common direction, and the valence pattern is not
significant, so these are leads for a confirmatory study, not established mechanisms.

**Geometry.** The near-zero cosines, and a mean effect that did not shrink when the layer-12 candidate
was projected out, argue against a simple "overlap with the refusal direction" account, but the
correlation is too imprecise to rule it out. Emotion-specific directions may act on refusal through later layers, through the
template positions where refusal is computed, or through the model's assessment of the request rather than
its decision to refuse.

**Sleeper effect.** PENDING E12.

**Limitations.** One primary model with one replication (PENDING); one injection layer and one main strength;
one greedy run per condition; 159 behaviours, so differences of a few points are within noise; the
HarmBench judge makes errors; the stories are written by language models; the proposal's 20-emotion
vocabulary became 19 emotions plus neutral stories; at c = 0.5 the injected emotion is only weakly readable
in the answers (14.5% recognition); MPS numerics are not bit-for-bit deterministic across batch shapes.

## 6. Implications for defenders

1. **Steering access is a jailbreak surface regardless of the vector's meaning.** Random directions at a
   moderate norm raise harmful answers by half again (14.5% to 22.1%), and some seeds more than double them.
   Any interface that lets users add vectors to a model's hidden state should be treated like weight access.
2. **In our grid, the cosine of a steering vector with the refusal direction did not single out the effective vectors.** The
   effective vectors have |cos| of at most 0.11 with the layer-12 refusal candidate, within the range of
   the vectors that stay inside the random band (up to 0.12). A monitor on the projection of the hidden state itself on
   the refusal direction at the end of the user turn is an untested alternative; we have not built or
   evaluated one (PENDING: E12 drift shows
   whether that projection moves under an early injection).
3. **Emotional framing in the prompt did not reduce refusal phrases** on this model at this sample size;
   its effect on judged-harmful answers was not measured.

## 7. Conclusion

Activation-level emotion injection, as proposed, is not a special attack on the refusal of Llama 3.1 8B
Instruct: the proposal's emotion vectors move the judged-harmful rate no more than random vectors of the
same norm, its vectors for the emotions expected to be strongest stay inside the random band, and we found no resolved association
between alignment with the refusal direction and the effect. A narrower set of emotion-specific directions does lie above the descriptive random band with few
answers flagged as broken, and these deserve a confirmatory study with more seeds, strengths and models.
PENDING: the sleeper and replication results.

## Ethics statement

All harmful-request generations stay on the authors' machine; the repository and interface publish only
rates and labels. HarmBench behaviours and the refusal-ablation technique are public. The steering vectors
we release raise harmful answers less than the already public refusal ablation.

## Reproducibility

Code, data, vectors and results are in the project repository (`docs/PIPELINE.md` lists every command and
its run time on the M5 Pro). `scripts/make_results.py` rebuilds every number from the raw files.

## References

- Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W., and Nanda, N. (2024). Refusal in
  Language Models Is Mediated by a Single Direction. NeurIPS 2024. arXiv:2406.11717.
- Bullwinkel, B., Russinovich, M., Salem, A., et al. (2025). A Representation Engineering Perspective on the
  Effectiveness of Multi-Turn Jailbreaks. ICML 2025 DIG-BUGS workshop. arXiv:2507.02956.
- Chen, R., Arditi, A., Sleight, H., Evans, O., and Lindsey, J. (2025). Persona Vectors: Monitoring and
  Controlling Character Traits in Language Models. NeurIPS 2026. arXiv:2507.21509.
- Gokhale, S. (2026). RepLens: Do Emotion Vectors Interact with Safety Mechanisms in LLMs? GitHub repository.
- Korznikov, A., Galichin, A., Dontsov, A., Rogov, O. Y., Oseledets, I., and Tutubalina, E. (2025). The Rogue
  Scalpel: Activation Steering Compromises LLM Safety. ICLR 2026 Principled Design for Trustworthy AI
  workshop. arXiv:2509.22067.
- Kuznetsov, D., Cohen, O., Shistik, K., et al. (2026). FreakOut-LLM: The Effect of Emotional Stimuli on
  Safety Alignment. Findings of the ACL: IJCNLP-AACL 2026. arXiv:2604.04992.
- Li, C., Wang, J., Zhang, Y., et al. (2023). Large Language Models Understand and Can be Enhanced by
  Emotional Stimuli. LLM@IJCAI 2023 workshop. arXiv:2307.11760.
- Li, P., Zhang, J., Zhang, T., et al. (2026). State-Dependent Safety Failures in Multi-Turn Language Model
  Interaction. ICML 2026. arXiv:2603.15684.
- Mazeika, M., Phan, L., Yin, X., et al. (2024). HarmBench: A Standardized Evaluation Framework for Automated
  Red Teaming and Robust Refusal. ICML 2024. arXiv:2402.04249.
- Panickssery, N., Gabrieli, N., Schulz, J., Tong, M., Hubinger, E., and Turner, A. M. (2024). Steering
  Llama 2 via Contrastive Activation Addition. ACL 2024. arXiv:2312.06681.
- Russinovich, M., Salem, A., and Eldan, R. (2024). Great, Now Write an Article About That: The Crescendo
  Multi-Turn LLM Jailbreak Attack. USENIX Security 2025. arXiv:2404.01833.
- Sofroniew, N., Kauvar, I., Saunders, W., et al. (2026). Emotion Concepts and their Function in a Large
  Language Model. Transformer Circuits Thread. arXiv:2604.07729.
- Sun, L., Yan, L., Lu, X., et al. (2026a). Valence-Arousal Subspace in LLMs: Circular Emotion Geometry and
  Multi-Behavioral Control. arXiv:2604.03147.
- Sun, M., Li, T., Zheng, Y., et al. (2026b). How Emotion Shapes the Behavior of LLMs and Agents: A
  Mechanistic Study. arXiv:2604.00005.
- Turner, A. M., Thiergart, L., Leech, G., et al. (2023). Steering Language Models With Activation
  Engineering. arXiv:2308.10248.
- Wollschläger, T., Elstner, J., Geisler, S., et al. (2025). The Geometry of Refusal in Large Language
  Models: Concept Cones and Representational Independence. ICML 2025. arXiv:2502.17420.
- Xiong, C., He, Z., Chen, P.-Y., et al. (2026). Steering Externalities: Benign Activation Steering
  Unintentionally Increases Jailbreak Risk for Large Language Models. arXiv:2602.04896.
- Zeng, Y., Lin, H., Zhang, J., et al. (2024). How Johnny Can Persuade LLMs to Jailbreak Them: Rethinking
  Persuasion to Challenge AI Safety by Humanizing LLMs. ACL 2024. arXiv:2401.06373.
- Zou, A., Phan, L., Chen, S., et al. (2023). Representation Engineering: A Top-Down Approach to AI
  Transparency. arXiv:2310.01405.
