/**
 * Timeline View (#/timeline)
 * - Vertical time axis grouped by year
 * - Era bands with summaries
 * - Category and priority filters
 * - Toggle to show unverified papers
 * - Nodes display title, date, venue, first finding, and link to paper page
 */

import { storyHtml } from '../story.js';

export function renderTimeline(catalog) {
  const container = document.createElement('div');
  container.className = 'view-timeline';

  const categories = catalog.categories || [];
  const eras = catalog.eras || [];
  const allPapers = [...(catalog.papers || [])];

  // State
  let selectedCategory = 'all';
  let selectedPriority = 'all';
  let showUnverified = false;

  function update() {
    // Filter papers
    const filtered = allPapers.filter(p => {
      // Verification check
      const isVerified = p.verified?.status === 'verified';
      if (!isVerified && !showUnverified) return false;

      // Category check
      if (selectedCategory !== 'all') {
        if (!p.categories?.includes(selectedCategory)) return false;
      }

      // Priority check
      if (selectedPriority !== 'all') {
        if (p.priority !== selectedPriority) return false;
      }

      return true;
    });

    // Sort chronologically (oldest to newest for timeline progression)
    filtered.sort((a, b) => (a.date || '').localeCompare(b.date || ''));

    // Group papers by Year
    const byYear = {};
    filtered.forEach(p => {
      const year = p.year || (p.date ? p.date.substring(0, 4) : 'Undated');
      if (!byYear[year]) byYear[year] = [];
      byYear[year].push(p);
    });

    const years = Object.keys(byYear).sort();

    const trackHtml = renderTrack(years, byYear, eras);
    const contentEl = container.querySelector('.timeline-content');
    if (contentEl) {
      contentEl.innerHTML = trackHtml;
    }
  }

  container.innerHTML = `
    <div class="section-header">
      <div class="section-label">Field Evolution</div>
      <h1 class="section-title">Research Timeline</h1>
      <p class="section-desc">
        How the field evolved towards activation-level emotion steering and guardrail geometry.
      </p>
    </div>

    ${catalog.timeline_story ? `
    <details class="story-block" open>
      <summary>How the field got to our question</summary>
      <div class="story-body">${storyHtml(catalog.timeline_story, allPapers)}</div>
    </details>` : ''}

    <div class="timeline-controls">
      <div class="filter-group">
        <label for="timeline-category" class="filter-label">Category:</label>
        <select id="timeline-category" class="filter-select">
          <option value="all">All Categories</option>
          ${categories.map(c => `<option value="${escapeHtml(c.id)}">${escapeHtml(c.label)}</option>`).join('')}
        </select>
      </div>

      <div class="filter-group">
        <label for="timeline-priority" class="filter-label">Priority:</label>
        <select id="timeline-priority" class="filter-select">
          <option value="all">All Priorities</option>
          <option value="core">Core</option>
          <option value="important">Important</option>
          <option value="background">Background</option>
        </select>
      </div>

      <div class="filter-group">
        <label class="toggle-label">
          <input type="checkbox" id="toggle-unverified" ${showUnverified ? 'checked' : ''}>
          <span>Show unverified papers</span>
        </label>
      </div>
    </div>

    <div class="timeline-content"></div>
  `;

  // Attach control listeners
  const catSelect = container.querySelector('#timeline-category');
  catSelect.addEventListener('change', (e) => {
    selectedCategory = e.target.value;
    update();
  });

  const prioSelect = container.querySelector('#timeline-priority');
  prioSelect.addEventListener('change', (e) => {
    selectedPriority = e.target.value;
    update();
  });

  const unverifiedCheck = container.querySelector('#toggle-unverified');
  unverifiedCheck.addEventListener('change', (e) => {
    showUnverified = e.target.checked;
    update();
  });

  // Initial render of timeline track
  update();

  return container;
}

function renderTrack(years, byYear, eras) {
  if (years.length === 0) {
    return `
      <div class="empty-state">
        <p>No papers match the selected criteria.</p>
      </div>
    `;
  }

  let html = `<div class="timeline-track">`;

  // Track which eras have been rendered
  const renderedEras = new Set();

  years.forEach(year => {
    // Check if an era starts in or covers this year
    eras.forEach(era => {
      if (!renderedEras.has(era.id)) {
        const startYear = era.start ? era.start.substring(0, 4) : '';
        if (startYear && year >= startYear) {
          renderedEras.add(era.id);
          html += `
            <div class="era-band">
              <div class="era-dates">${escapeHtml(era.start)} to ${escapeHtml(era.end)}</div>
              <div class="era-label">Era: ${escapeHtml(era.label)}</div>
              <p class="era-summary">${escapeHtml(era.summary)}</p>
            </div>
          `;
        }
      }
    });

    html += `
      <div class="year-group">
        <div class="year-marker">${escapeHtml(year)}</div>
        <div class="year-papers">
          ${byYear[year].map(paper => renderPaperNode(paper)).join('')}
        </div>
      </div>
    `;
  });

  html += `</div>`;
  return html;
}

function renderPaperNode(paper) {
  const isCore = paper.priority === 'core';
  const isVerified = paper.verified?.status === 'verified';
  const firstFinding = paper.key_findings?.[0] || paper.summary_simple || 'No key finding recorded yet.';

  return `
    <article class="paper-node ${isCore ? 'core' : ''} ${!isVerified ? 'unverified' : ''}">
      <div class="paper-node-header">
        <h3 class="paper-node-title">
          <a href="#/paper/${paper.id}">${escapeHtml(paper.title)}</a>
        </h3>
        <div style="display: flex; gap: 0.35rem; align-items: center;">
          <span class="badge badge-${escapeHtml(paper.priority || 'background')}">${escapeHtml(paper.priority || 'paper')}</span>
          ${!isVerified ? '<span class="badge" style="background-color: var(--status-in-progress-bg); color: var(--status-in-progress);">UNVERIFIED</span>' : ''}
        </div>
      </div>

      <div class="paper-node-meta">
        <span>${escapeHtml(paper.date || paper.year)}</span>
        <span>${escapeHtml(paper.venue || 'Preprint')}</span>
        <span>${escapeHtml(Array.isArray(paper.authors) ? paper.authors.join(', ') : paper.authors || '')}</span>
      </div>

      <div class="paper-finding">
        <strong>Key finding:</strong> ${escapeHtml(firstFinding)}
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
