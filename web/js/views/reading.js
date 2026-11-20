/**
 * Reading Path View (#/reading)
 * - Ordered by reading_order
 * - Checkbox stored in localStorage
 * - Progress tracking
 */

const STORAGE_KEY = 'affect_reading_status_v1';

function getReadStatus() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch (err) {
    return {};
  }
}

function saveReadStatus(status) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(status));
  } catch (err) {
    console.warn('Failed to save reading status to localStorage:', err);
  }
}

export function renderReading(catalog) {
  const container = document.createElement('div');
  container.className = 'view-reading';

  const papers = [...(catalog.papers || [])];

  // Filter papers that have reading_order or include all sorted by reading_order then date
  papers.sort((a, b) => {
    const ordA = a.reading_order ?? 999;
    const ordB = b.reading_order ?? 999;
    if (ordA !== ordB) return ordA - ordB;
    return (a.date || '').localeCompare(b.date || '');
  });

  const readMap = getReadStatus();

  function updateProgress() {
    const total = papers.length;
    let completed = 0;
    papers.forEach(p => {
      if (readMap[p.id]) completed++;
    });
    const pct = total > 0 ? Math.round((completed / total) * 100) : 0;

    const fillEl = container.querySelector('#reading-progress-fill');
    const textEl = container.querySelector('#reading-progress-text');
    if (fillEl) fillEl.style.width = `${pct}%`;
    if (textEl) textEl.textContent = `${completed} of ${total} papers read (${pct}%)`;
  }

  container.innerHTML = `
    <div class="section-header">
      <div class="section-label">Curriculum</div>
      <h1 class="section-title">Sequential Reading Path</h1>
      <p class="section-desc">
        Recommended reading sequence for project researchers and students. Mark off papers as you complete them; your progress is preserved locally.
      </p>
    </div>

    <div class="reading-progress-card">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
        <span style="font-weight: 700; font-size: var(--step-0);">Your Reading Progress</span>
        <span id="reading-progress-text" style="font-family: var(--font-family-mono); font-weight: 700; font-size: var(--step--1);">0 of ${papers.length} papers read (0%)</span>
      </div>
      <div class="progress-bar-bg" style="height: 8px;">
        <div id="reading-progress-fill" class="progress-bar-fill" style="width: 0%;"></div>
      </div>
    </div>

    <div class="reading-list">
      ${papers.map((paper, idx) => {
        const isRead = !!readMap[paper.id];
        const stepNum = paper.reading_order ?? (idx + 1);
        const authors = Array.isArray(paper.authors) ? paper.authors.join(', ') : (paper.authors || '');

        return `
          <div class="reading-item ${isRead ? 'completed' : ''}" data-paper-id="${escapeHtml(paper.id)}">
            <input type="checkbox" class="reading-checkbox" id="check-${escapeHtml(paper.id)}" ${isRead ? 'checked' : ''} aria-label="Mark ${escapeHtml(paper.title)} as read">
            <div class="reading-content">
              <div style="display: flex; gap: 0.5rem; align-items: center; margin-bottom: 0.25rem; flex-wrap: wrap;">
                <span class="badge" style="background-color: var(--text-ink); color: var(--text-inverse);">Step ${stepNum}</span>
                <span class="badge badge-${escapeHtml(paper.priority || 'background')}">${escapeHtml(paper.priority || 'paper')}</span>
                ${paper.cited_in_proposal ? `<span class="badge" style="background-color: var(--bg-surface-subtle); color: var(--text-ink);">Cited ${escapeHtml(paper.cited_in_proposal)}</span>` : ''}
              </div>

              <h2 style="font-size: var(--step-1); font-weight: 700; margin-bottom: 0.25rem;">
                <a href="#/paper/${paper.id}" style="text-decoration: none; color: var(--text-ink);">${escapeHtml(paper.title)}</a>
              </h2>

              <p style="font-size: var(--step--1); color: var(--text-tertiary); margin-bottom: 0.5rem;">
                ${escapeHtml(authors)} • ${escapeHtml(paper.venue || 'Preprint')} (${escapeHtml(paper.year || '')})
              </p>

              <p style="font-size: var(--step--1); color: var(--text-secondary); line-height: 1.5;">
                ${escapeHtml(paper.relevance || paper.summary_simple || 'Summary pending.')}
              </p>

              <div style="margin-top: 0.5rem;">
                <a href="#/paper/${paper.id}" class="btn-pill">Open Details & Notes</a>
              </div>
            </div>
          </div>
        `;
      }).join('')}
    </div>
  `;

  // Attach checkbox listeners
  container.querySelectorAll('.reading-checkbox').forEach(box => {
    box.addEventListener('change', (e) => {
      const itemEl = e.target.closest('.reading-item');
      const paperId = itemEl?.getAttribute('data-paper-id');
      if (!paperId) return;

      if (e.target.checked) {
        readMap[paperId] = true;
        itemEl.classList.add('completed');
      } else {
        delete readMap[paperId];
        itemEl.classList.remove('completed');
      }

      saveReadStatus(readMap);
      updateProgress();
    });
  });

  // Initial calculation
  updateProgress();

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
