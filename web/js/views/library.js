/**
 * Library View (#/library)
 * - Searchable by query
 * - Filterable by category, priority, cited in proposal, verified status
 * - Sortable by date, reading order, priority
 */

export function renderLibrary(catalog) {
  const container = document.createElement('div');
  container.className = 'view-library';

  const categories = catalog.categories || [];
  const papers = catalog.papers || [];

  // Local state
  let searchQuery = '';
  let selectedCategory = 'all';
  let selectedPriority = 'all';
  let citedOnly = false;
  let verifiedOnly = false;
  let sortBy = 'date_desc';

  function update() {
    let result = papers.filter(p => {
      // Query filter
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        const titleMatch = (p.title || '').toLowerCase().includes(q);
        const authorsMatch = (Array.isArray(p.authors) ? p.authors.join(' ') : (p.authors || '')).toLowerCase().includes(q);
        const venueMatch = (p.venue || '').toLowerCase().includes(q);
        const tagsMatch = (p.tags || []).some(t => t.toLowerCase().includes(q));
        const summaryMatch = (p.summary_simple || '').toLowerCase().includes(q);
        if (!titleMatch && !authorsMatch && !venueMatch && !tagsMatch && !summaryMatch) {
          return false;
        }
      }

      // Category filter
      if (selectedCategory !== 'all') {
        if (!p.categories?.includes(selectedCategory)) return false;
      }

      // Priority filter
      if (selectedPriority !== 'all') {
        if (p.priority !== selectedPriority) return false;
      }

      // Cited in proposal
      if (citedOnly) {
        if (!p.cited_in_proposal) return false;
      }

      // Verified status
      if (verifiedOnly) {
        if (p.verified?.status !== 'verified') return false;
      }

      return true;
    });

    // Sorting
    result.sort((a, b) => {
      if (sortBy === 'date_desc') {
        return (b.date || '').localeCompare(a.date || '');
      }
      if (sortBy === 'date_asc') {
        return (a.date || '').localeCompare(b.date || '');
      }
      if (sortBy === 'reading_order') {
        const ordA = a.reading_order ?? 999;
        const ordB = b.reading_order ?? 999;
        return ordA - ordB;
      }
      if (sortBy === 'priority') {
        const weights = { core: 1, important: 2, background: 3 };
        return (weights[a.priority] || 4) - (weights[b.priority] || 4);
      }
      if (sortBy === 'title') {
        return (a.title || '').localeCompare(b.title || '');
      }
      return 0;
    });

    // Update count
    const countEl = container.querySelector('#library-count');
    if (countEl) {
      countEl.textContent = `Showing ${result.length} of ${papers.length} papers`;
    }

    // Render cards
    const gridEl = container.querySelector('#library-grid');
    if (gridEl) {
      if (result.length === 0) {
        gridEl.innerHTML = '<div class="empty-state" style="grid-column: 1 / -1;"><p>No papers match your filters.</p></div>';
      } else {
        gridEl.innerHTML = result.map(paper => renderPaperCard(paper)).join('');
      }
    }
  }

  container.innerHTML = `
    <div class="section-header">
      <div class="section-label">Literature Catalog</div>
      <h1 class="section-title">Paper Library</h1>
      <p class="section-desc">
        Curated papers on activation steering, refusal geometry, and LLM safety red-teaming.
      </p>
    </div>

    <div class="library-toolbar">
      <div class="search-row">
        <input type="search" id="library-search" class="search-input" placeholder="Search by title, author, keyword, or venue..." aria-label="Search papers">
      </div>

      <div class="filter-row">
        <div class="filter-group">
          <label for="filter-category" class="filter-label">Category:</label>
          <select id="filter-category" class="filter-select">
            <option value="all">All Categories</option>
            ${categories.map(c => `<option value="${escapeHtml(c.id)}">${escapeHtml(c.label)}</option>`).join('')}
          </select>
        </div>

        <div class="filter-group">
          <label for="filter-priority" class="filter-label">Priority:</label>
          <select id="filter-priority" class="filter-select">
            <option value="all">All Priorities</option>
            <option value="core">Core</option>
            <option value="important">Important</option>
            <option value="background">Background</option>
          </select>
        </div>

        <div class="filter-group">
          <label for="filter-sort" class="filter-label">Sort By:</label>
          <select id="filter-sort" class="filter-select">
            <option value="date_desc">Date (Newest first)</option>
            <option value="date_asc">Date (Oldest first)</option>
            <option value="reading_order">Reading Order</option>
            <option value="priority">Priority</option>
            <option value="title">Title</option>
          </select>
        </div>

        <div class="filter-group">
          <label class="toggle-label">
            <input type="checkbox" id="toggle-cited">
            <span>Cited in proposal</span>
          </label>
        </div>

        <div class="filter-group">
          <label class="toggle-label">
            <input type="checkbox" id="toggle-verified">
            <span>Verified only</span>
          </label>
        </div>

        <div id="library-count" class="library-count"></div>
      </div>
    </div>

    <div id="library-grid" class="library-grid"></div>
  `;

  // Attach event handlers
  const searchInput = container.querySelector('#library-search');
  searchInput.addEventListener('input', (e) => {
    searchQuery = e.target.value;
    update();
  });

  const catSelect = container.querySelector('#filter-category');
  catSelect.addEventListener('change', (e) => {
    selectedCategory = e.target.value;
    update();
  });

  const prioSelect = container.querySelector('#filter-priority');
  prioSelect.addEventListener('change', (e) => {
    selectedPriority = e.target.value;
    update();
  });

  const sortSelect = container.querySelector('#filter-sort');
  sortSelect.addEventListener('change', (e) => {
    sortBy = e.target.value;
    update();
  });

  const citedCheck = container.querySelector('#toggle-cited');
  citedCheck.addEventListener('change', (e) => {
    citedOnly = e.target.checked;
    update();
  });

  const verifiedCheck = container.querySelector('#toggle-verified');
  verifiedCheck.addEventListener('change', (e) => {
    verifiedOnly = e.target.checked;
    update();
  });

  // Initial render
  update();

  return container;
}

function renderPaperCard(paper) {
  const isVerified = paper.verified?.status === 'verified';
  const authors = Array.isArray(paper.authors) ? paper.authors.join(', ') : (paper.authors || '');
  const summary = paper.summary_simple || (paper.key_findings?.[0]) || 'Summary pending.';

  return `
    <article class="library-card">
      <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 0.5rem;">
        <span class="badge badge-${escapeHtml(paper.priority || 'background')}">${escapeHtml(paper.priority || 'paper')}</span>
        <div style="display: flex; gap: 0.25rem;">
          ${paper.cited_in_proposal ? `<span class="badge" style="background-color: var(--bg-surface-subtle); color: var(--text-ink);" title="Cited as ${paper.cited_in_proposal}">Cited ${paper.cited_in_proposal}</span>` : ''}
          ${!isVerified ? '<span class="badge" style="background-color: var(--status-in-progress-bg); color: var(--status-in-progress);">UNVERIFIED</span>' : ''}
        </div>
      </div>

      <h3 class="library-card-title">
        <a href="#/paper/${paper.id}">${escapeHtml(paper.title)}</a>
      </h3>

      <div class="library-card-authors">${escapeHtml(authors)}</div>

      <div style="font-size: var(--step--1); color: var(--text-tertiary); display: flex; gap: 0.5rem;">
        <span>${escapeHtml(paper.venue || 'Preprint')}</span>
        <span>•</span>
        <span>${escapeHtml(paper.date || paper.year)}</span>
      </div>

      <p class="library-card-summary">${escapeHtml(summary)}</p>

      ${paper.key_numbers?.length ? `
        <div class="pill-group">
          ${paper.key_numbers.map(n => `<span class="badge badge-default">${escapeHtml(n)}</span>`).join('')}
        </div>
      ` : ''}

      <div class="library-card-actions">
        <a href="#/paper/${paper.id}" class="btn-pill">View Paper Details</a>
        ${paper.arxiv ? `<a href="https://arxiv.org/abs/${paper.arxiv}" target="_blank" rel="noopener" class="btn-pill">arXiv</a>` : ''}
        ${paper.code_url ? `<a href="${paper.code_url}" target="_blank" rel="noopener" class="btn-pill">Code</a>` : ''}
      </div>
    </article>
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
