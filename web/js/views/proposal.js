/**
 * Proposal View (#/proposal)
 * - Renders private/proposal.pdf and private/proposal-presentation.pdf
 * - Checks availability via HEAD request and hides or shows fallbacks if 404
 */

import { checkFileExists } from '../api.js';

export async function renderProposal() {
  const container = document.createElement('div');
  container.className = 'view-proposal';

  const proposalPdfPath = '../private/proposal.pdf';
  const slidesPdfPath = '../private/proposal-presentation.pdf';

  const [hasProposal, hasSlides] = await Promise.all([
    checkFileExists(proposalPdfPath),
    checkFileExists(slidesPdfPath)
  ]);

  let activeTab = hasProposal ? 'proposal' : (hasSlides ? 'slides' : 'none');

  function renderContent() {
    if (!hasProposal && !hasSlides) {
      return `
        <div class="card card-subtle" style="text-align: center; padding: 3rem;">
          <h2 style="font-size: var(--step-1); margin-bottom: 0.5rem;">Proposal Files Not Located</h2>
          <p style="color: var(--text-secondary); max-width: 500px; margin: 0 auto;">
            The project proposal and presentation PDFs are stored in the private directory.
            Ensure <code>private/proposal.pdf</code> or <code>private/proposal-presentation.pdf</code> are present in your workspace.
          </p>
        </div>
      `;
    }

    const currentPdf = activeTab === 'proposal' ? proposalPdfPath : slidesPdfPath;
    const currentTitle = activeTab === 'proposal' ? 'Major Project Proposal Document' : 'Project Proposal Presentation Slides';

    return `
      <div class="proposal-tabs">
        ${hasProposal ? `
          <button type="button" class="btn-pill ${activeTab === 'proposal' ? 'btn-pill-primary' : ''}" id="tab-proposal">
            Proposal Document
          </button>
        ` : ''}
        ${hasSlides ? `
          <button type="button" class="btn-pill ${activeTab === 'slides' ? 'btn-pill-primary' : ''}" id="tab-slides">
            Presentation Slides
          </button>
        ` : ''}
        <a href="${currentPdf}" target="_blank" rel="noopener" class="btn-pill" style="margin-left: auto;">
          Open in New Window
        </a>
      </div>

      <div class="proposal-frame-card">
        <iframe src="${currentPdf}" style="width: 100%; height: 100%; border: none;" title="${escapeHtml(currentTitle)}"></iframe>
      </div>
    `;
  }

  container.innerHTML = `
    <div class="section-header">
      <div class="section-label">Institutional Documentation</div>
      <h1 class="section-title">Major Project Proposal</h1>
      <p class="section-desc">
        Department of Electronics & Computer Engineering, Pulchowk Campus, Tribhuvan University.
      </p>
    </div>

    <div id="proposal-content-wrapper">
      ${renderContent()}
    </div>
  `;

  // Tab switching
  container.addEventListener('click', (e) => {
    if (e.target.id === 'tab-proposal') {
      activeTab = 'proposal';
      const wrapper = container.querySelector('#proposal-content-wrapper');
      if (wrapper) wrapper.innerHTML = renderContent();
    } else if (e.target.id === 'tab-slides') {
      activeTab = 'slides';
      const wrapper = container.querySelector('#proposal-content-wrapper');
      if (wrapper) wrapper.innerHTML = renderContent();
    }
  });

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
