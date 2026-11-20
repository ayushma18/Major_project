/**
 * Reference Audit View (#/audit)
 * - Checks citations in proposal against verified literature
 * - Categorised by status: real-correct, real-wrong-metadata, wrong-paper, not-found
 */

export function renderAudit(catalog) {
  const container = document.createElement('div');
  container.className = 'view-audit';

  const auditItems = catalog.proposal_reference_audit || [];
  const papersMap = new Map((catalog.papers || []).map(p => [p.id, p]));

  let selectedStatus = 'all';

  function update() {
    const filtered = auditItems.filter(item => {
      if (selectedStatus === 'all') return true;
      return item.status === selectedStatus;
    });

    const tbody = container.querySelector('#audit-tbody');
    if (!tbody) return;

    if (filtered.length === 0) {
      tbody.innerHTML = `
        <tr>
          <td colspan="5" class="empty-state">No reference audit items match the selected filter.</td>
        </tr>
      `;
      return;
    }

    tbody.innerHTML = filtered.map(item => {
      const matchPaper = item.match_id ? papersMap.get(item.match_id) : null;
      const statusClass = `badge-audit-${item.status || 'not-found'}`;

      return `
        <tr>
          <td style="font-family: var(--font-family-mono); font-weight: 700; white-space: nowrap;">
            ${escapeHtml(item.ref)}
          </td>
          <td style="max-width: 320px; line-height: 1.5;">
            ${escapeHtml(item.as_cited)}
          </td>
          <td>
            <span class="badge ${statusClass}">${escapeHtml(item.status)}</span>
          </td>
          <td style="max-width: 340px; line-height: 1.5; color: var(--text-secondary);">
            ${escapeHtml(item.note)}
          </td>
          <td>
            ${matchPaper ? `
              <a href="#/paper/${matchPaper.id}" style="font-weight: 600;">
                ${escapeHtml(matchPaper.title)}
              </a>
            ` : (item.match_id ? `<code>${escapeHtml(item.match_id)}</code>` : '<span style="color: var(--text-tertiary);">None</span>')}
          </td>
        </tr>
      `;
    }).join('');
  }

  container.innerHTML = `
    <div class="section-header">
      <div class="section-label">Verification & Integrity</div>
      <h1 class="section-title">Proposal Reference Audit</h1>
      <p class="section-desc">
        Systematic audit of references cited in the project proposal to detect hallucinated preprints or erroneous metadata.
      </p>
    </div>

    <div class="timeline-controls" style="margin-bottom: var(--space-lg);">
      <div class="filter-group">
        <label for="audit-status-filter" class="filter-label">Filter by Status:</label>
        <select id="audit-status-filter" class="filter-select">
          <option value="all">All Statuses (${auditItems.length})</option>
          <option value="real-correct">real-correct</option>
          <option value="real-wrong-metadata">real-wrong-metadata</option>
          <option value="wrong-paper">wrong-paper</option>
          <option value="not-found">not-found</option>
        </select>
      </div>

      <div style="display: flex; gap: 0.5rem; flex-wrap: wrap; margin-left: auto;">
        <span class="badge badge-audit-real-correct">real-correct</span>
        <span class="badge badge-audit-real-wrong-metadata">real-wrong-metadata</span>
        <span class="badge badge-audit-wrong-paper">wrong-paper</span>
        <span class="badge badge-audit-not-found">not-found</span>
      </div>
    </div>

    <div class="audit-table-container">
      <table class="data-table">
        <thead>
          <tr>
            <th style="width: 80px;">Ref</th>
            <th>As Cited in Proposal</th>
            <th style="width: 140px;">Status</th>
            <th>Audit Notes & Remediation</th>
            <th>Catalog Match</th>
          </tr>
        </thead>
        <tbody id="audit-tbody"></tbody>
      </table>
    </div>
  `;

  const statusSelect = container.querySelector('#audit-status-filter');
  statusSelect.addEventListener('change', (e) => {
    selectedStatus = e.target.value;
    update();
  });

  update();

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
