/**
 * Paper Detail View (#/paper/<id>)
 * - Complete paper metadata
 * - Plain summary, findings, numbers, method, relevance, limitations
 * - Local PDF reader with fallback and link to full-screen reader
 */

import { checkFileExists } from '../api.js';

export async function renderPaper(paperId, catalog) {
  const container = document.createElement('div');
  container.className = 'view-paper-detail';

  const paper = (catalog.papers || []).find(p => p.id === paperId);

  if (!paper) {
    container.innerHTML = `
      <div class="empty-state">
        <h2>Paper Not Found</h2>
        <p>The requested paper identifier "${escapeHtml(paperId)}" does not exist in the library catalog.</p>
        <p style="margin-top: 1rem;"><a href="#/library" class="btn-pill">Back to Library</a></p>
      </div>
    `;
    return container;
  }

  const authors = Array.isArray(paper.authors) ? paper.authors.join(', ') : (paper.authors || 'Unknown');
  const pdfLocalPath = paper.pdf_local ? `../${paper.pdf_local}` : null;
  const hasLocalPdf = pdfLocalPath ? await checkFileExists(pdfLocalPath) : false;

  container.innerHTML = `
    <div class="paper-detail-header">
      <a href="#/library" class="paper-back-btn">← Back to Library</a>

      <div style="display: flex; gap: 0.5rem; align-items: center; margin-bottom: 0.5rem; flex-wrap: wrap;">
        <span class="badge badge-${escapeHtml(paper.priority || 'background')}">${escapeHtml(paper.priority || 'paper')}</span>
        ${paper.cited_in_proposal ? `<span class="badge" style="background-color: var(--bg-surface-subtle); color: var(--text-ink);">Cited ${escapeHtml(paper.cited_in_proposal)}</span>` : ''}
        ${paper.verified?.status === 'verified'
          ? '<span class="badge badge-status-done">VERIFIED</span>'
          : '<span class="badge" style="background-color: var(--status-in-progress-bg); color: var(--status-in-progress);">UNVERIFIED</span>'
        }
      </div>

      <h1 class="section-title" style="font-size: var(--step-3);">${escapeHtml(paper.title)}</h1>

      <p style="font-size: var(--step-0); color: var(--text-secondary); margin-top: 0.25rem;">
        ${escapeHtml(authors)}
      </p>

      <div style="display: flex; gap: 1rem; color: var(--text-tertiary); font-size: var(--step--1); margin-top: 0.5rem; flex-wrap: wrap;">
        <span><strong>Venue:</strong> ${escapeHtml(paper.venue || 'Preprint')}</span>
        <span><strong>Date:</strong> ${escapeHtml(paper.date || paper.year)}</span>
        ${paper.arxiv ? `<span><strong>arXiv:</strong> ${escapeHtml(paper.arxiv)}</span>` : ''}
      </div>

      <div class="pill-group" style="margin-top: 1rem;">
        ${paper.url ? `<a href="${paper.url}" target="_blank" rel="noopener" class="btn-pill">Publisher / URL</a>` : ''}
        ${paper.arxiv ? `<a href="https://arxiv.org/abs/${paper.arxiv}" target="_blank" rel="noopener" class="btn-pill">arXiv Abstract</a>` : ''}
        ${paper.pdf_url ? `<a href="${paper.pdf_url}" target="_blank" rel="noopener" class="btn-pill">arXiv PDF</a>` : ''}
        ${paper.code_url ? `<a href="${paper.code_url}" target="_blank" rel="noopener" class="btn-pill">Source Code</a>` : ''}
        ${hasLocalPdf ? `<a href="#/read/${paper.id}" class="btn-pill btn-pill-primary">Open Fullscreen Reader</a>` : ''}
      </div>
    </div>

    <div class="paper-meta-grid">
      <div class="detail-main-column">
        <section class="detail-section">
          <h2 class="detail-section-title">Plain Summary</h2>
          <p style="font-size: var(--step-0); color: var(--text-ink); line-height: 1.6;">
            ${escapeHtml(paper.summary_simple || 'Summary not yet provided.')}
          </p>
        </section>

        <section class="detail-section">
          <h2 class="detail-section-title">Key Findings</h2>
          ${paper.key_findings?.length ? `
            <ul class="key-list">
              ${paper.key_findings.map(f => `<li>${escapeHtml(f)}</li>`).join('')}
            </ul>
          ` : '<p style="color: var(--text-tertiary);">No key findings recorded yet.</p>'}
        </section>

        <section class="detail-section">
          <h2 class="detail-section-title">Relevance to Our Project</h2>
          <p style="font-size: var(--step--1); line-height: 1.6;">
            ${escapeHtml(paper.relevance || 'Relevance notes not yet written.')}
          </p>
        </section>

        <section class="detail-section">
          <h2 class="detail-section-title">Methodology</h2>
          <p style="font-size: var(--step--1); line-height: 1.6;">
            ${escapeHtml(paper.method || 'Methodology description not yet written.')}
          </p>
        </section>

        <section class="detail-section">
          <h2 class="detail-section-title">Limitations</h2>
          <p style="font-size: var(--step--1); line-height: 1.6;">
            ${escapeHtml(paper.limitations || 'No limitations catalogued yet.')}
          </p>
        </section>
      </div>

      <div class="detail-side-column">
        <section class="detail-section">
          <h2 class="detail-section-title">Key Numbers</h2>
          ${paper.key_numbers?.length ? `
            <ul class="key-list">
              ${paper.key_numbers.map(n => `<li>${escapeHtml(n)}</li>`).join('')}
            </ul>
          ` : '<p style="color: var(--text-tertiary);">No key metrics logged.</p>'}
        </section>

        <section class="detail-section">
          <h2 class="detail-section-title">Categories and Tags</h2>
          <div class="pill-group">
            ${(paper.categories || []).map(c => `<span class="badge badge-default">${escapeHtml(c)}</span>`).join('')}
            ${(paper.tags || []).map(t => `<span class="badge" style="background-color: var(--bg-surface-subtle); color: var(--text-secondary);">${escapeHtml(t)}</span>`).join('')}
          </div>
        </section>

        <section class="detail-section">
          <h2 class="detail-section-title">Verification</h2>
          <p style="font-size: var(--step--1);">
            <strong>Status:</strong> ${escapeHtml(paper.verified?.status || 'unverified')}
          </p>
          ${paper.verified?.how ? `<p style="font-size: var(--step--1); color: var(--text-tertiary); margin-top: 0.25rem;">${escapeHtml(paper.verified.how)}</p>` : ''}
          ${paper.verified?.at ? `<p style="font-size: 0.75rem; color: var(--text-tertiary); margin-top: 0.25rem;">Checked: ${escapeHtml(paper.verified.at)}</p>` : ''}
        </section>
      </div>
    </div>

    <section class="detail-section" style="margin-top: var(--space-xl);">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; flex-wrap: wrap; gap: 0.5rem;">
        <h2 class="detail-section-title" style="margin-bottom: 0;">Embedded Paper Reader</h2>
        ${hasLocalPdf ? `<a href="#/read/${paper.id}" class="btn-pill">View in Full Screen</a>` : ''}
      </div>

      ${hasLocalPdf ? `
        <div class="reader-container">
          <iframe src="${pdfLocalPath}" class="reader-iframe" title="${escapeHtml(paper.title)} PDF reader"></iframe>
        </div>
      ` : `
        <div class="reader-fallback card card-subtle">
          <p style="font-weight: 600; color: var(--text-ink); margin-bottom: 0.5rem;">PDF not downloaded yet</p>
          <p style="font-size: var(--step--1); max-width: 500px; margin: 0 auto 1rem auto;">
            Local copy (${escapeHtml(paper.pdf_local || 'literature/pdfs/' + paper.id + '.pdf')}) is not present in the workspace.
            Run <code>python scripts/fetch_papers.py</code> to download local copies.
          </p>
          ${paper.pdf_url ? `<a href="${paper.pdf_url}" target="_blank" rel="noopener" class="btn-pill btn-pill-primary">Read on arXiv</a>` : ''}
        </div>
      `}
    </section>
  `;

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
