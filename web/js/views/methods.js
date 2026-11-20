/**
 * Methods View (#/methods)
 * - Plain language explanation of project methodology
 * - Residual stream architecture
 * - Contrastive Activation Addition (CAA)
 * - Refusal direction (difference of means)
 * - Directional ablation
 * - Cosine geometry and orthogonal projection
 * - Attack Success Rate (ASR) on HarmBench
 * - Sleeper protocol (multi-turn persistence)
 * - Formulas in plain HTML using Unicode, <sub>, and <sup>
 */

export function renderMethods() {
  const container = document.createElement('div');
  container.className = 'view-methods';

  container.innerHTML = `
    <div class="section-header">
      <div class="section-label">Theoretical Foundations</div>
      <h1 class="section-title">Methods and Mathematical Formulations</h1>
      <p class="section-desc">
        A mechanistic breakdown of activation addition, refusal geometry, and multi-turn safety evaluation in open-source language models.
      </p>
    </div>

    <article class="methods-article">
      <section class="method-block">
        <h3>1. Residual Stream and Hidden Representations</h3>
        <p>
          Autoregressive transformer language models maintain a residual stream across successive transformer blocks. At layer l, the hidden representation vector x<sub>l</sub> ∈ ℝ<sup>d<sub>model</sub></sup> is updated by attention heads and multi-layer perceptron (MLP) sublayers through additive increments.
        </p>
        <p>
          Because the residual stream operates as an additive communication bus, linear directions corresponding to specific semantic attributes, stylistic tones, or cognitive states can be isolated and modified during inference without updating model weights.
        </p>
      </section>

      <section class="method-block">
        <h3>2. Emotion Vector Extraction (Contrastive Activation Addition)</h3>
        <p>
          Functional emotion vectors are extracted using Contrastive Activation Addition (CAA) over a constructed story corpus. For each emotion e in our 20-emotion vocabulary (including afraid, desperate, angry, calm, and neutral), we construct N paired narratives. Each pair consists of an emotionally charged narrative and a neutral counterpart describing the identical scenario.
        </p>
        <p>
          The emotion steering vector v<sub>e</sub><sup>(l)</sup> at layer l is computed as the average difference between hidden states:
        </p>
        <div class="formula-box">
          <div class="formula-label">Equation 1: CAA Emotion Vector Extraction</div>
          v<sub>e</sub><sup>(l)</sup> = (1 / N) ∑<sub>i=1</sub><sup>N</sup> ( h<sub>i,l</sub><sup>pos</sup> − h<sub>i,l</sub><sup>neg</sup> )
        </div>
        <p>
          During generation, the steering vector is injected into the residual stream scaled by a multiplier α:
        </p>
        <div class="formula-box">
          <div class="formula-label">Equation 2: Residual Stream Injection</div>
          x′<sub>l</sub> = x<sub>l</sub> + α · v<sub>e</sub><sup>(l)</sup>
        </div>
        <p>
          Here α controls the steering intensity. Sweeps across layers determine the layer range where steering remains potent without degrading grammatical fluency (empirically layers 10 to 18).
        </p>
      </section>

      <section class="method-block">
        <h3>3. Refusal Direction (Difference of Means)</h3>
        <p>
          Instruction fine-tuned models refuse harmful requests via a dedicated linear subspace. Following Arditi et al. (2024), we estimate the refusal direction r<sup>(l)</sup> as the difference of means between activations on harmful instruction prompts and harmless instruction prompts:
        </p>
        <div class="formula-box">
          <div class="formula-label">Equation 3: Difference of Means Refusal Vector</div>
          r<sup>(l)</sup> = (1 / N<sub>h</sub>) ∑<sub>i=1</sub><sup>N<sub>h</sub></sup> h<sub>i,l</sub><sup>harmful</sup> − (1 / N<sub>n</sub>) ∑<sub>j=1</sub><sup>N<sub>n</sub></sup> h<sub>j,l</sub><sup>neutral</sup>
        </div>
        <p>
          The normalized direction vector is r̂ = r / ‖r‖. This unit vector represents the primary guardrail coordinate in the model's residual geometry.
        </p>
      </section>

      <section class="method-block">
        <h3>4. Directional Ablation</h3>
        <p>
          To establish an upper bound on refusal suppression, the refusal direction can be completely excised from the residual stream during inference. The projection of hidden state x onto r̂ is removed:
        </p>
        <div class="formula-box">
          <div class="formula-label">Equation 4: Directional Ablation</div>
          x′ = x − (x · r̂) · r̂
        </div>
        <p>
          Ablation removes the model's capacity to initiate refusal phrasing ("I cannot fulfill this request..."), allowing harmful queries to be answered directly.
        </p>
      </section>

      <section class="method-block">
        <h3>5. Cosine Geometry and Orthogonal Projection</h3>
        <p>
          Our central mechanistic hypothesis is that emotion vectors bypass safety guardrails by projecting negatively onto the refusal direction. We compute the cosine similarity between each emotion vector v<sub>e</sub> and refusal direction r:
        </p>
        <div class="formula-box">
          <div class="formula-label">Equation 5: Cosine Similarity</div>
          cos θ(v<sub>e</sub>, r) = (v<sub>e</sub> · r) / ( ‖v<sub>e</sub>‖ · ‖r‖ )
        </div>
        <p>
          To demonstrate causal mediation rather than coincidental correlation, we perform orthogonal projection to remove the refusal-aligned component from v<sub>e</sub>:
        </p>
        <div class="formula-box">
          <div class="formula-label">Equation 6: Orthogonalised Emotion Vector</div>
          v<sub>e</sub><sup>⊥</sup> = v<sub>e</sub> − [ (v<sub>e</sub> · r) / ‖r‖<sup>2</sup> ] · r
        </div>
        <p>
          If injecting v<sub>e</sub><sup>⊥</sup> causes Attack Success Rate to drop back to baseline levels, we confirm that geometric interference with the refusal direction is the causal attack mechanism.
        </p>
      </section>

      <section class="method-block">
        <h3>6. Safety Evaluation and Attack Success Rate (ASR)</h3>
        <p>
          Safety evaluation is conducted on HarmBench test behaviours across diverse harm domains. The primary safety metric is Attack Success Rate (ASR):
        </p>
        <div class="formula-box">
          <div class="formula-label">Equation 7: HarmBench Attack Success Rate</div>
          ASR<sub>e</sub> = (1 / |P|) ∑<sub>i=1</sub><sup>|P|</sup> 𝟙[ f<sub>θ</sub>(p<sub>i</sub> | v<sub>e</sub>) is harmful ]
        </div>
        <p>
          where 𝟙 is an indicator function evaluated by an automated LLM judge classifier, and ΔASR<sub>e</sub> = ASR<sub>e</sub> − ASR<sub>∅</sub> measures the marginal safety degradation caused by emotion injection.
        </p>
      </section>

      <section class="method-block">
        <h3>7. The Sleeper Protocol (Multi-Turn Persistence)</h3>
        <p>
          In conversational settings, an attack may be delivered early in dialogue history before any harmful request is submitted. The sleeper protocol evaluates whether emotion injection at turn 1 induces representational drift that persists into later turns:
        </p>
        <ul class="key-list" style="margin: 0.75rem 0;">
          <li><strong>Turn 1:</strong> Inject emotion vector v<sub>e</sub> into residual stream while processing a benign user query.</li>
          <li><strong>Turns 2 to 3:</strong> Exchange neutral, harmless follow-up dialogue without any activation injection.</li>
          <li><strong>Turns 4 to 7:</strong> Submit standard HarmBench harmful requests without activation injection.</li>
        </ul>
        <p>
          The persistence metric compares early injection against same-turn injection:
        </p>
        <div class="formula-box">
          <div class="formula-label">Equation 8: Multi-Turn Persistence Delta</div>
          Δ<sub>persist</sub>(t) = ASR<sub>early</sub>(t) − ASR<sub>same-turn</sub>(t)
        </div>
        <p>
          A positive Δ<sub>persist</sub> indicates that earlier activation steering established an enduring representational bias that impairs subsequent safety compliance.
        </p>
      </section>
    </article>
  `;

  return container;
}
