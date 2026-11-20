/**
 * Experiments View (#/experiments)
 * - Grouped by module
 * - Status badge, headline, and link to detail view
 */

export function renderExperiments(resultsIndex) {
  const container = document.createElement('div');
  container.className = 'view-experiments';

  const experiments = resultsIndex.experiments || [];

  const moduleNames = {
    0: 'Module 0: Setup and Model Selection',
    1: 'Module 1: Emotion Vector Extraction',
    2: 'Module 2: Safety Evaluation (HarmBench)',
    3: 'Module 3: Refusal Geometry and Mechanistic Analysis',
    4: 'Module 4: Multi-Turn Persistence and Sleeper Protocol'
  };

  // Group by module
  const byModule = {};
  [0, 1, 2, 3, 4].forEach(m => byModule[m] = []);

  experiments.forEach(exp => {
    const m = exp.module ?? 0;
    if (!byModule[m]) byModule[m] = [];
    byModule[m].push(exp);
  });

  container.innerHTML = `
    <div class="section-header">
      <div class="section-label">Empirical Evaluation</div>
      <h1 class="section-title">Experiments Index</h1>
      <p class="section-desc">
        Standardised experimental evaluation suite organized across the four core project modules.
      </p>
    </div>

    <div class="experiments-grid">
      ${[0, 1, 2, 3, 4].map(modNum => {
        const list = byModule[modNum];
        if (!list || list.length === 0) return '';

        return `
          <div class="module-group">
            <h2 class="module-group-title">${escapeHtml(moduleNames[modNum] || `Module ${modNum}`)}</h2>
            <div class="exp-cards-row">
              ${list.map(exp => renderExpCard(exp)).join('')}
            </div>
          </div>
        `;
      }).join('')}
    </div>
  `;

  return container;
}

function renderExpCard(exp) {
  const statusClass = `badge-status-${exp.status || 'planned'}`;

  return `
    <a href="#/experiment/${exp.id}" class="exp-card">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <span class="exp-id">${escapeHtml(exp.id)}</span>
        <span class="badge ${statusClass}">${escapeHtml(exp.status || 'planned')}</span>
      </div>

      <h3 class="exp-title">${escapeHtml(exp.title)}</h3>

      <p class="exp-headline">${escapeHtml(exp.headline || 'No headline available.')}</p>

      <div style="margin-top: auto; padding-top: 0.5rem; display: flex; justify-content: space-between; align-items: center; border-top: 1px solid var(--border-subtle);">
        <span style="font-family: var(--font-family-mono); font-size: 0.75rem; color: var(--text-tertiary);">Module ${exp.module ?? 0}</span>
        <span style="font-size: var(--step--1); font-weight: 600; color: var(--text-ink);">View Results →</span>
      </div>
    </a>
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
