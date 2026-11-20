/**
 * Experiment Detail View (#/experiment/<id>)
 * - Metrics cards with CI95
 * - Data tables
 * - Inline SVG charts (bar, line, scatter, heatmap)
 * - Interpretation and caveats
 * - Graceful fallback when experiment file is missing or still planned
 */

import { getExperimentDetail } from '../api.js';
import { renderChart } from '../charts.js';

export async function renderExperimentDetail(expId, resultsIndex) {
  const container = document.createElement('div');
  container.className = 'view-experiment-detail';

  // Look up index entry first
  const indexEntry = (resultsIndex.experiments || []).find(e => e.id === expId);
  const detail = await getExperimentDetail(expId);

  // If detail file is missing or null, show planned / graceful state
  if (!detail) {
    container.innerHTML = `
      <div class="section-header">
        <a href="#/experiments" class="paper-back-btn">← Back to Experiments</a>
        <div style="display: flex; gap: 0.5rem; align-items: center; margin-top: 0.5rem;">
          <span class="exp-id">${escapeHtml(expId)}</span>
          <span class="badge badge-status-planned">${escapeHtml(indexEntry?.status || 'planned')}</span>
        </div>
        <h1 class="section-title">${escapeHtml(indexEntry?.title || expId)}</h1>
        <p class="section-desc">${escapeHtml(indexEntry?.headline || 'No experiment execution record found.')}</p>
      </div>

      <div class="card card-subtle" style="text-align: center; padding: 3rem; margin-top: 2rem;">
        <p style="font-weight: 700; font-size: var(--step-1); color: var(--text-ink); margin-bottom: 0.5rem;">
          Experiment Planned
        </p>
        <p style="color: var(--text-secondary); max-width: 500px; margin: 0 auto 1.5rem auto;">
          This experiment is scheduled in the project plan. Data collection and result files will be generated once execution begins.
        </p>
        <a href="#/experiments" class="btn-pill">Return to Experiments Index</a>
      </div>
    `;
    return container;
  }

  const statusClass = `badge-status-${detail.status || 'planned'}`;

  container.innerHTML = `
    <div class="section-header">
      <a href="#/experiments" class="paper-back-btn">← Back to Experiments</a>

      <div style="display: flex; gap: 0.5rem; align-items: center; margin-top: 0.5rem; flex-wrap: wrap;">
        <span class="exp-id">${escapeHtml(detail.id)}</span>
        <span class="badge ${statusClass}">${escapeHtml(detail.status || 'done')}</span>
        <span class="badge" style="background-color: var(--bg-surface-subtle); color: var(--text-ink);">Module ${detail.module ?? 0}</span>
      </div>

      <h1 class="section-title" style="margin-top: 0.5rem;">${escapeHtml(detail.title)}</h1>

      ${detail.headline ? `
        <p style="font-size: var(--step-1); color: var(--text-ink); font-weight: 600; margin-top: 0.5rem;">
          ${escapeHtml(detail.headline)}
        </p>
      ` : ''}

      ${detail.question ? `
        <p style="font-size: var(--step-0); color: var(--text-secondary); margin-top: 0.5rem;">
          <strong>Research Question:</strong> ${escapeHtml(detail.question)}
        </p>
      ` : ''}

      <div style="display: flex; gap: 1rem; color: var(--text-tertiary); font-size: var(--step--1); margin-top: 0.75rem; flex-wrap: wrap;">
        ${detail.model ? `<span><strong>Model:</strong> ${escapeHtml(detail.model)}</span>` : ''}
        ${detail.started ? `<span><strong>Started:</strong> ${escapeHtml(detail.started)}</span>` : ''}
        ${detail.finished ? `<span><strong>Finished:</strong> ${escapeHtml(detail.finished)}</span>` : ''}
      </div>
    </div>

    ${detail.metrics?.length ? `
      <section class="metrics-grid">
        ${detail.metrics.map(m => `
          <div class="metric-card">
            <div class="metric-label">${escapeHtml(m.label || m.name)}</div>
            <div class="metric-val">${typeof m.value === 'number' ? (m.value < 1 && m.value > 0 ? m.value.toFixed(3) : m.value) : escapeHtml(m.value)}</div>
            ${m.ci95 ? `<div class="metric-ci">95% CI: [${m.ci95[0]}, ${m.ci95[1]}]</div>` : ''}
          </div>
        `).join('')}
      </section>
    ` : ''}

    <div class="charts-section" id="charts-wrapper"></div>

    ${detail.tables?.length ? `
      <section class="detail-section" style="margin-top: var(--space-xl);">
        ${detail.tables.map(tbl => `
          <div style="margin-bottom: var(--space-xl);">
            <h3 style="font-size: var(--step-1); margin-bottom: var(--space-md);">${escapeHtml(tbl.title)}</h3>
            <div class="audit-table-container">
              <table class="data-table">
                <thead>
                  <tr>
                    ${tbl.columns.map(col => `<th>${escapeHtml(col.replace(/_/g, ' '))}</th>`).join('')}
                  </tr>
                </thead>
                <tbody>
                  ${tbl.rows.map(row => `
                    <tr>
                      ${row.map(cell => `<td>${escapeHtml(cell)}</td>`).join('')}
                    </tr>
                  `).join('')}
                </tbody>
              </table>
            </div>
          </div>
        `).join('')}
      </section>
    ` : ''}

    <div class="paper-meta-grid" style="margin-top: var(--space-xl);">
      <div class="detail-main-column">
        <section class="detail-section">
          <h2 class="detail-section-title">Findings & Interpretation</h2>
          ${detail.interpretation?.length ? `
            <ul class="key-list">
              ${detail.interpretation.map(it => `<li>${escapeHtml(it)}</li>`).join('')}
            </ul>
          ` : '<p style="color: var(--text-tertiary);">No interpretation written yet.</p>'}
        </section>
      </div>

      <div class="detail-side-column">
        <section class="detail-section">
          <h2 class="detail-section-title">Caveats & Limitations</h2>
          ${detail.caveats?.length ? `
            <ul class="key-list">
              ${detail.caveats.map(c => `<li>${escapeHtml(c)}</li>`).join('')}
            </ul>
          ` : '<p style="color: var(--text-tertiary);">No caveats noted.</p>'}
        </section>

        ${detail.artifacts?.length ? `
          <section class="detail-section">
            <h2 class="detail-section-title">Output Artifacts</h2>
            <ul class="key-list" style="font-family: var(--font-family-mono); font-size: 0.8rem;">
              ${detail.artifacts.map(art => `<li>${escapeHtml(art)}</li>`).join('')}
            </ul>
          </section>
        ` : ''}
      </div>
    </div>
  `;

  // Render SVG charts
  const chartsWrapper = container.querySelector('#charts-wrapper');
  if (chartsWrapper && detail.charts?.length) {
    detail.charts.forEach(chart => {
      const chartEl = renderChart(chart);
      chartsWrapper.appendChild(chartEl);
    });
  }

  return container;
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
