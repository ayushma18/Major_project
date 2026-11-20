/**
 * Main Application Orchestrator & Client-Side Router
 * Affect as Attack Interface
 */

import { getCatalog, getStatus, getResultsIndex, isFixtureMode, setFixtureMode } from './js/api.js';
import { renderOverview } from './js/views/overview.js';
import { renderTimeline } from './js/views/timeline.js';
import { renderLibrary } from './js/views/library.js';
import { renderPaper } from './js/views/paper.js';
import { renderRead } from './js/views/read.js';
import { renderReading } from './js/views/reading.js';
import { renderAudit } from './js/views/audit.js';
import { renderExperiments } from './js/views/experiments.js';
import { renderExperimentDetail } from './js/views/experiment_detail.js';
import { renderMethods } from './js/views/methods.js';
import { renderProposal } from './js/views/proposal.js';

// Global cached data for the session
let catalogData = null;
let statusData = null;
let resultsIndexData = null;

/**
 * Initialize theme handling
 */
function initTheme() {
  const savedTheme = localStorage.getItem('affect_theme') || 'system';
  applyTheme(savedTheme);

  document.querySelectorAll('.theme-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const theme = btn.getAttribute('data-theme');
      localStorage.setItem('affect_theme', theme);
      applyTheme(theme);
    });
  });
}

function applyTheme(theme) {
  const root = document.documentElement;
  if (theme === 'system') {
    root.removeAttribute('data-theme');
  } else {
    root.setAttribute('data-theme', theme);
  }

  document.querySelectorAll('.theme-btn').forEach(btn => {
    if (btn.getAttribute('data-theme') === theme) {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  });
}

/**
 * Initialize fixture controls
 */
function initFixtures() {
  const active = isFixtureMode();
  const toggleBtn = document.getElementById('fixture-toggle');
  const banner = document.getElementById('fixture-banner');
  const exitLink = document.getElementById('exit-fixtures-link');

  if (active) {
    toggleBtn?.classList.add('active');
    toggleBtn?.setAttribute('aria-pressed', 'true');
    banner?.classList.remove('hidden');
  } else {
    toggleBtn?.classList.remove('active');
    toggleBtn?.setAttribute('aria-pressed', 'false');
    banner?.classList.add('hidden');
  }

  toggleBtn?.addEventListener('click', () => {
    setFixtureMode(!active);
  });

  exitLink?.addEventListener('click', (e) => {
    e.preventDefault();
    setFixtureMode(false);
  });
}

/**
 * Update active navigation links
 */
function updateNav(route) {
  const navLinks = document.querySelectorAll('.nav-link');
  navLinks.forEach(link => {
    const linkRoute = link.getAttribute('data-route') || '';
    if (route === linkRoute || (route.startsWith(linkRoute) && linkRoute !== '')) {
      link.classList.add('active');
    } else {
      link.classList.remove('active');
    }
  });
}

/**
 * Router dispatcher
 */
async function router() {
  const hash = window.location.hash.slice(1) || '/';
  const appContainer = document.getElementById('app');
  if (!appContainer) return;

  // Show loading indicator
  appContainer.innerHTML = `
    <div class="loading-state">
      <div class="loading-spinner" aria-hidden="true"></div>
      <p>Loading view...</p>
    </div>
  `;

  // Parse path
  const parts = hash.split('/').filter(Boolean);
  const routeBase = parts[0] || '';
  const param = parts[1] || '';

  updateNav(routeBase);

  try {
    let viewElement = null;

    if (routeBase === '' || routeBase === 'overview') {
      viewElement = renderOverview(statusData, catalogData, resultsIndexData);
    } else if (routeBase === 'timeline') {
      viewElement = renderTimeline(catalogData);
    } else if (routeBase === 'library') {
      viewElement = renderLibrary(catalogData);
    } else if (routeBase === 'paper' && param) {
      viewElement = await renderPaper(param, catalogData);
    } else if (routeBase === 'read' && param) {
      viewElement = await renderRead(param, catalogData);
    } else if (routeBase === 'reading') {
      viewElement = renderReading(catalogData);
    } else if (routeBase === 'audit') {
      viewElement = renderAudit(catalogData);
    } else if (routeBase === 'experiments') {
      viewElement = renderExperiments(resultsIndexData);
    } else if (routeBase === 'experiment' && param) {
      viewElement = await renderExperimentDetail(param, resultsIndexData);
    } else if (routeBase === 'methods') {
      viewElement = renderMethods();
    } else if (routeBase === 'proposal') {
      viewElement = await renderProposal();
    } else {
      // 404
      viewElement = document.createElement('div');
      viewElement.className = 'empty-state';
      viewElement.innerHTML = `
        <h2>Page Not Found</h2>
        <p>The route <code>#/${routeBase}</code> was not recognized.</p>
        <p style="margin-top: 1rem;"><a href="#/" class="btn-pill">Return to Overview</a></p>
      `;
    }

    appContainer.innerHTML = '';
    appContainer.appendChild(viewElement);

    // Scroll to top on navigation
    window.scrollTo(0, 0);
  } catch (err) {
    console.error('Routing error:', err);
    appContainer.innerHTML = `
      <div class="empty-state">
        <h2>Rendering Error</h2>
        <p style="color: var(--color-accent);">${err.message || 'An unexpected error occurred while loading this view.'}</p>
        <p style="margin-top: 1rem;"><a href="#/" class="btn-pill">Back to Safety</a></p>
      </div>
    `;
  }
}

/**
 * Bootstrap Application
 */
async function init() {
  initTheme();
  initFixtures();

  // Load datasets concurrently
  try {
    const [catalog, status, resultsIndex] = await Promise.all([
      getCatalog(),
      getStatus(),
      getResultsIndex()
    ]);

    catalogData = catalog;
    statusData = status;
    resultsIndexData = resultsIndex;

    // Attach hash change event
    window.addEventListener('hashchange', router);

    // Initial route
    await router();
  } catch (err) {
    console.error('Fatal initialization error:', err);
    const appContainer = document.getElementById('app');
    if (appContainer) {
      appContainer.innerHTML = `
        <div class="empty-state">
          <h2>Data Initialization Failed</h2>
          <p>Could not load the research repository data.</p>
        </div>
      `;
    }
  }
}

// Kickoff
window.addEventListener('DOMContentLoaded', init);
