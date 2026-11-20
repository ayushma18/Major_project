/** Plain text with [paper-id] or [id1, id2] references -> paragraphs with links to the paper pages. */
export function storyHtml(text, papers) {
  if (!text) return '';
  const byId = Object.fromEntries((papers || []).map(p => [p.id, p]));
  const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const link = id => {
    const p = byId[id];
    if (!p) return esc(id);
    const label = esc((p.authors?.[0] || '').split(' ').slice(-1)[0] + ' ' + (p.year || ''));
    return `<a href="#/paper/${id}" class="story-ref" title="${esc(p.title)}">${label}</a>`;
  };
  return text.split(/\n\s*\n/)
    .filter(Boolean)
    .map(par => '<p>' + esc(par.trim()).replace(/\[([a-z0-9]+(?:,\s*[a-z0-9]+)*)\]/g, (m, ids) =>
      '(' + ids.split(/,\s*/).map(link).join(', ') + ')') + '</p>')
    .join('');
}
