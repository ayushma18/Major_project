/**
 * API layer for loading literature catalog, project status, and experiment results.
 * Respects the ?fixtures=1 URL parameter to swap real repository data with fixtures.
 * Handles missing files, null fields, and PDF availability checks via HEAD.
 */

export function isFixtureMode() {
  const urlParams = new URLSearchParams(window.location.search);
  return urlParams.get('fixtures') === '1';
}

export function setFixtureMode(enabled) {
  const url = new URL(window.location.href);
  if (enabled) {
    url.searchParams.set('fixtures', '1');
  } else {
    url.searchParams.delete('fixtures');
  }
  window.location.href = url.toString();
}

/**
 * Fetch JSON with cache-busting and error fallback
 */
async function fetchJson(path, fallback = null) {
  try {
    const res = await fetch(path, { cache: 'no-store' });
    if (!res.ok) {
      return fallback;
    }
    return await res.json();
  } catch (err) {
    console.warn(`Could not fetch ${path}:`, err);
    return fallback;
  }
}

/**
 * Check if a local file exists using HEAD request
 */
export async function checkFileExists(path) {
  if (!path) return false;
  try {
    const res = await fetch(path, { method: 'HEAD', cache: 'no-store' });
    return res.ok;
  } catch (err) {
    return false;
  }
}

/**
 * Fetch literature catalog
 */
export async function getCatalog() {
  const isFixture = isFixtureMode();
  const path = isFixture ? './fixtures/catalog.json' : '../literature/catalog.json';
  const data = await fetchJson(path, { version: 1, updated: '', categories: [], eras: [], papers: [], proposal_reference_audit: [] });
  return {
    ...data,
    categories: Array.isArray(data.categories) ? data.categories : [],
    eras: Array.isArray(data.eras) ? data.eras : [],
    papers: Array.isArray(data.papers) ? data.papers : [],
    proposal_reference_audit: Array.isArray(data.proposal_reference_audit) ? data.proposal_reference_audit : []
  };
}

/**
 * Fetch project status
 */
export async function getStatus() {
  const isFixture = isFixtureMode();
  const path = isFixture ? './fixtures/status.json' : '../project/status.json';
  const data = await fetchJson(path, {
    updated: '',
    title: 'Affect as Attack',
    one_line: '',
    team: [],
    model: { id: '', why: '' },
    phases: [],
    findings: [],
    blockers: [],
    next: []
  });
  return {
    ...data,
    phases: Array.isArray(data.phases) ? data.phases : [],
    findings: Array.isArray(data.findings) ? data.findings : [],
    blockers: Array.isArray(data.blockers) ? data.blockers : [],
    next: Array.isArray(data.next) ? data.next : [],
    team: Array.isArray(data.team) ? data.team : []
  };
}

/**
 * Fetch experiment results index
 */
export async function getResultsIndex() {
  const isFixture = isFixtureMode();
  const path = isFixture ? './fixtures/results/index.json' : '../results/index.json';
  const data = await fetchJson(path, { updated: '', experiments: [] });
  return {
    ...data,
    experiments: Array.isArray(data.experiments) ? data.experiments : []
  };
}

/**
 * Fetch single experiment detail
 */
export async function getExperimentDetail(expId) {
  const isFixture = isFixtureMode();
  const path = isFixture
    ? `./fixtures/results/${expId}.json`
    : `../results/${expId}.json`;
  return await fetchJson(path, null);
}
