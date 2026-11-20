/**
 * Full-width Reader View (#/read/<id>)
 */

import { checkFileExists } from '../api.js';

export async function renderRead(paperId, catalog) {
  const container = document.createElement('div');
  container.className = 'view-reader';

  const paper = (catalog.papers || []).find(p => p.id === paperId);

  if (!paper) {
    container.innerHTML = `
      <div class="empty-state">
        <h2>Paper Not Found</h2>
        <p><a href="#/library" class="btn-pill">Back to Library</a></p>
      </div>
    `;
    return container;
  }

  const pdfLocalPath = paper.pdf_local ? `../${paper.pdf_local}` : null;
  const hasLocalPdf = pdfLocalPath ? await checkFileExists(pdfLocalPath) : false;
  const targetPdf = hasLocalPdf ? pdfLocalPath : paper.pdf_url;

  container.innerHTML = `
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; flex-wrap: wrap; gap: 0.5rem;">
      <div>
        <a href="#/paper/${paper.id}" class="paper-back-btn">← Back to Paper Summary</a>
        <h1 style="font-size: var(--step-1); margin-top: 0.25rem;">${escapeHtml(paper.title)}</h1>
      </div>

      <div style="display: flex; gap: 0.5rem;">
        ${paper.pdf_url ? `<a href="${paper.pdf_url}" target="_blank" rel="noopener" class="btn-pill">Open arXiv PDF</a>` : ''}
        ${paper.url ? `<a href="${paper.url}" target="_blank" rel="noopener" class="btn-pill">Original Link</a>` : ''}
      </div>
    </div>

    ${targetPdf ? `
      <div style="width: 100%; height: calc(100vh - 12rem); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); overflow: hidden; background: #525659;">
        <iframe src="${targetPdf}" style="width: 100%; height: 100%; border: none;" title="${escapeHtml(paper.title)} Document"></iframe>
      </div>
    ` : `
      <div class="card card-subtle" style="text-align: center; padding: 3rem;">
        <p style="font-weight: 600;">No PDF URL or local PDF available for this paper.</p>
        <p style="margin-top: 1rem;"><a href="#/paper/${paper.id}" class="btn-pill">Return to details</a></p>
      </div>
    `}
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
