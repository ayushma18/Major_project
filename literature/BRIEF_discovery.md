# Literature discovery brief

Goal: a complete, VERIFIED list of the papers this project needs, so three students can study
the field from its foundations to the newest work, and see how it evolved towards our question:
**can an emotion direction injected into a model's residual stream switch off its refusal
(safety guardrail), and does that line up with the refusal direction geometrically?**

Today is 2026-11-20. Include 2026 papers. Prefer arXiv, OpenReview, ACL Anthology, PMLR,
NeurIPS proceedings, transformer-circuits.pub and official lab pages.

## Topics to cover (aim for 50 to 80 papers in total, quality over count)

1. **Emotions inside models**: Anthropic's 2026 work on emotion concepts / functional emotions
   in Claude; emotion representations and emotion vectors in open models (Llama, Mistral,
   Qwen, Gemma); mechanistic interpretability of emotion inference; emotion steering papers.
2. **Emotional prompting and affect in jailbreaks**: EmotionPrompt and NegativePrompt style
   work, emotional manipulation or persuasion jailbreaks (e.g. persuasive adversarial
   prompts), stress or pressure effects on model behaviour, persona modulation.
3. **Refusal geometry**: Arditi et al. 2024 and everything that built on it (refusal cones,
   affine refusal, harmfulness vs refusal encoded separately, abliteration, refusal across
   languages, refusal directions in reasoning models, refusal feature adversarial training).
4. **Activation steering methods**: ActAdd, Contrastive Activation Addition (Rimsky /
   Panickssery), Inference-Time Intervention, Representation Engineering (Zou et al.),
   steering reliability and generalisation studies, steering scaling studies, SAE-based steering.
5. **Steering versus safety**: papers showing steering (including random or benign vectors)
   breaks safety alignment, safety-aware steering, defences such as circuit breakers /
   representation rerouting, latent adversarial training, refusal-direction monitors.
6. **Safety benchmarks and judges**: HarmBench, AdvBench (GCG paper), JailbreakBench,
   StrongREJECT, SORRY-Bench, XSTest (over-refusal), WildJailbreak, the HarmBench classifier.
7. **Multi-turn attacks and drift**: Crescendo, multi-turn human jailbreaks (MHJ), a
   representation engineering perspective on multi-turn jailbreaks (Bullwinkel et al.),
   X-Teaming, multi-turn benchmarks, safety drift over long conversations.
8. **Interpretability tools**: Towards Monosemanticity, Scaling Monosemanticity, Gemma Scope,
   Llama Scope, SAELens, TransformerLens, the linear representation hypothesis papers,
   A Mathematical Framework for Transformer Circuits.
9. **Foundations**: Attention Is All You Need, InstructGPT (RLHF), DPO, Constitutional AI,
   the Llama 3 herd paper (our target model), Llama Guard if relevant.

## Verification is mandatory

For every paper, OPEN its real page (arXiv abs page, OpenReview, ACL Anthology, publisher)
and copy title, authors, date and venue from that page. Never write a paper from memory.
If you cannot find the real page, leave it out. Fabricated or misremembered citations are the
main risk in this project (the proposal already contains several), so a short, true list
beats a long one.

## Output

Write `literature/candidates_gemini.json`: a JSON array, one object per paper:

```json
{
  "id": "arditi2024refusal",
  "title": "exact title from the page",
  "authors": ["Full Name", "..."],
  "date": "YYYY-MM-DD (first version date)",
  "venue": "NeurIPS 2024 | arXiv preprint | ACL 2025 | ...",
  "arxiv": "2406.11717 or null",
  "url": "the page you opened",
  "pdf_url": "direct PDF link or null",
  "code_url": "official code repo or null",
  "categories": ["refusal-geometry"],
  "why": "one plain sentence: why this paper matters to our question",
  "evidence": "what you checked, e.g. 'arXiv abs page, v1 date 2024-06-17'"
}
```

- `id`: first author's surname + year + first meaningful title word, lowercase ascii, e.g.
  `zou2023representation`. Unique.
- `categories` from: emotions, emotional-prompting, refusal-geometry, steering,
  steering-safety, benchmarks, multi-turn, interpretability-tools, foundations.

Then download each available PDF to `literature/pdfs/<id>.pdf` (use curl with a polite
User-Agent and a 1 second pause between arXiv downloads). Check each download is a real PDF
(starts with `%PDF`). Leave papers without a free PDF undownloaded.

Only create `literature/candidates_gemini.json` and files in `literature/pdfs/`. Do not edit
any other file and do not run git commands.
