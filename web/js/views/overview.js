/**
 * Overview View (#/)
 * - Research question in one big statement
 * - Four-module pipeline diagram
 * - Current phase and progress from project/status.json
 * - Findings, blockers, and next steps
 */

export function renderOverview(status, catalog, results) {
  const container = document.createElement('div');
  container.className = 'view-overview';

  const titleText = status.one_line || "Can an emotion pushed directly into a language model's hidden state switch off its safety guardrail, even though the prompt looks harmless?";

  container.innerHTML = `
    <section class="overview-hero">
      <div class="section-label">Core Research Question</div>
      <h1 class="research-statement">${escapeHtml(titleText)}</h1>
      <p class="section-desc">
        A mechanistic investigation of activation-level functional emotion steering versus safety alignment guardrails in Llama 3.1 8B Instruct.
      </p>
    </section>

    <section class="pipeline-container">
      <div class="section-header">
        <div class="section-label">Experimental Framework</div>
        <h2 class="section-title">Four-Module Research Pipeline</h2>
        <p class="section-desc">
          The project investigates internal representations from extraction to safety failure and conversation persistence.
        </p>
      </div>

      <div class="pipeline-grid">
        <a href="#/methods" class="pipeline-card">
          <div class="pipeline-num">MODULE 1</div>
          <h3 class="pipeline-title">Emotion Vectors</h3>
          <p class="pipeline-desc">
            Extracting 20 functional emotion directions from residual stream activations using Contrastive Activation Addition across paired stories.
          </p>
          <div class="pipeline-meta">Method: Contrastive Addition</div>
        </a>

        <a href="#/experiments" class="pipeline-card">
          <div class="pipeline-num">MODULE 2</div>
          <h3 class="pipeline-title">Safety Evaluation</h3>
          <p class="pipeline-desc">
            Measuring Attack Success Rate (ASR) on HarmBench harmful behaviours when emotion vectors are injected at varying layers and strengths.
          </p>
          <div class="pipeline-meta">Metric: HarmBench ASR</div>
        </a>

        <a href="#/experiments" class="pipeline-card">
          <div class="pipeline-num">MODULE 3</div>
          <h3 class="pipeline-title">Refusal Geometry</h3>
          <p class="pipeline-desc">
            Quantifying cosine alignment between emotion vectors and the refusal direction. Testing causal mediation via orthogonal projection.
          </p>
          <div class="pipeline-meta">Geometry: Cosine Correlation</div>
        </a>

        <a href="#/experiments" class="pipeline-card">
          <div class="pipeline-num">MODULE 4</div>
          <h3 class="pipeline-title">Persistence</h3>
          <p class="pipeline-desc">
            Evaluating the sleeper effect: checking if early injection at turn 1 induces persistent representational drift across turns 4 to 7.
          </p>
          <div class="pipeline-meta">Protocol: 7-Turn Harness</div>
        </a>
      </div>
    </section>

    <section class="dashboard-grid">
      <div class="phases-column">
        <div class="section-header">
          <div class="section-label">Milestones</div>
          <h2 class="section-title">Project Phases</h2>
        </div>

        <div class="phases-list">
          ${renderPhases(status.phases)}
        </div>
      </div>

      <div class="side-cards">
        <div class="info-card">
          <h3 class="info-card-title">Target Model</h3>
          <p style="font-weight: 700; color: var(--text-ink); margin-bottom: 0.25rem;">
            ${escapeHtml(status.model?.id || 'Llama-3.1-8B-Instruct')}
          </p>
          <p style="font-size: var(--step--1);">${escapeHtml(status.model?.why || 'Standard open weights target for activation studies.')}</p>
        </div>

        <div class="info-card">
          <h3 class="info-card-title">Research Team</h3>
          <ul class="info-card-list">
            ${(status.team || ['Anuj Paudel', 'Ashish Pandey', 'Ayushma Pudasaini']).map(name => `
              <li class="info-item" style="border-left-color: var(--text-ink);">${escapeHtml(name)}</li>
            `).join('')}
          </ul>
        </div>

        <div class="info-card">
          <h3 class="info-card-title">Latest Findings</h3>
          ${renderFindings(status.findings)}
        </div>

        <div class="info-card">
          <h3 class="info-card-title">Active Blockers</h3>
          ${renderBlockers(status.blockers)}
        </div>

        <div class="info-card">
          <h3 class="info-card-title">Immediate Next Steps</h3>
          <ul class="info-card-list">
            ${(status.next || []).map(item => `
              <li class="info-item">${escapeHtml(item)}</li>
            `).join('')}
          </ul>
        </div>
      </div>
    </section>
  `;

  return container;
}

function renderPhases(phases = []) {
  if (phases.length === 0) {
    return '<p class="empty-state">No phases recorded in status file.</p>';
  }

  return phases.map(phase => {
    const pct = Math.round((phase.progress || 0) * 100);
    const statusClass = `badge-status-${phase.status || 'not_started'}`;
    const statusLabel = (phase.status || 'not_started').replace('_', ' ');

    return `
      <div class="phase-item">
        <div class="phase-header">
          <div>
            <span class="badge ${statusClass}">${escapeHtml(statusLabel)}</span>
            <span class="phase-name" style="margin-left: 0.5rem;">${escapeHtml(phase.id)}: ${escapeHtml(phase.label)}</span>
          </div>
          <span style="font-family: var(--font-family-mono); font-size: var(--step--1); font-weight: 700;">${pct}%</span>
        </div>

        <div class="progress-bar-bg">
          <div class="progress-bar-fill ${phase.status === 'in_progress' ? 'accent' : ''}" style="width: ${pct}%;"></div>
        </div>

        ${phase.items?.length ? `
          <ul class="phase-items">
            ${phase.items.map(it => `
              <li>
                <strong>${escapeHtml(it.label)}</strong>
                ${it.note ? `<span style="color: var(--text-tertiary);"> (${escapeHtml(it.note)})</span>` : ''}
              </li>
            `).join('')}
          </ul>
        ` : ''}
      </div>
    `;
  }).join('');
}

function renderFindings(findings = []) {
  if (findings.length === 0) {
    return '<p style="font-size: var(--step--1); color: var(--text-tertiary);">No experiment findings logged yet.</p>';
  }

  return `
    <ul class="info-card-list">
      ${findings.map(f => `
        <li class="info-item">
          <span class="info-item-date">${escapeHtml(f.date)}</span>
          ${escapeHtml(f.text)}
          ${f.experiment ? `<br><a href="#/experiment/${f.experiment}" style="font-family: var(--font-family-mono); font-size: 0.75rem;">View ${f.experiment}</a>` : ''}
        </li>
      `).join('')}
    </ul>
  `;
}

function renderBlockers(blockers = []) {
  if (blockers.length === 0) {
    return '<p style="font-size: var(--step--1); color: var(--status-done);">No active blockers.</p>';
  }

  return `
    <ul class="info-card-list">
      ${blockers.map(b => `
        <li class="info-item blocker">
          <strong>${escapeHtml(b.owner ? `[${b.owner}] ` : '')}</strong>
          ${escapeHtml(b.text)}
        </li>
      `).join('')}
    </ul>
  `;
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
