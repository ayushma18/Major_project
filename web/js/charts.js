/**
 * Inline SVG Chart Generator
 * Supports bar, line, scatter, and heatmap chart types without external libraries.
 * WCAG AA accessible contrast, responsive viewBox, hover value tooltips, and legends.
 */

// Tooltip handler
let tooltipEl = null;

function getTooltip() {
  if (!tooltipEl) {
    tooltipEl = document.getElementById('chart-tooltip');
  }
  return tooltipEl;
}

export function showTooltip(e, text) {
  const tip = getTooltip();
  if (!tip) return;
  tip.textContent = text;
  tip.classList.add('visible');
  tip.setAttribute('aria-hidden', 'false');
  updateTooltipPosition(e);
}

export function updateTooltipPosition(e) {
  const tip = getTooltip();
  if (!tip || !tip.classList.contains('visible')) return;
  const x = e.clientX + 12;
  const y = e.clientY + 12;
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
}

export function hideTooltip() {
  const tip = getTooltip();
  if (!tip) return;
  tip.classList.remove('visible');
  tip.setAttribute('aria-hidden', 'true');
}

/**
 * Main chart dispatcher
 */
export function renderChart(chart) {
  if (!chart || !chart.type) {
    return '<div class="empty-state">No chart data available</div>';
  }

  const container = document.createElement('div');
  container.className = 'chart-container';

  const header = document.createElement('div');
  header.className = 'chart-header';
  const title = document.createElement('h4');
  title.className = 'chart-title';
  title.textContent = chart.title || 'Chart';
  header.appendChild(title);
  container.appendChild(header);

  const wrapper = document.createElement('div');
  wrapper.className = 'chart-svg-wrapper';

  let svgHtml = '';
  switch (chart.type) {
    case 'bar':
      svgHtml = renderBarChart(chart);
      break;
    case 'line':
      svgHtml = renderLineChart(chart);
      break;
    case 'scatter':
      svgHtml = renderScatterChart(chart);
      break;
    case 'heatmap':
      svgHtml = renderHeatmapChart(chart);
      break;
    default:
      svgHtml = `<p class="empty-state">Unsupported chart type: ${chart.type}</p>`;
  }

  wrapper.innerHTML = svgHtml;
  container.appendChild(wrapper);

  // Attach hover events
  wrapper.querySelectorAll('[data-tooltip]').forEach(el => {
    el.addEventListener('mouseenter', (e) => {
      showTooltip(e, el.getAttribute('data-tooltip'));
    });
    el.addEventListener('mousemove', (e) => {
      updateTooltipPosition(e);
    });
    el.addEventListener('mouseleave', () => {
      hideTooltip();
    });
  });

  return container;
}

/**
 * Bar Chart SVG
 */
function renderBarChart(chart) {
  const width = 640;
  const height = 320;
  const padding = { top: 30, right: 30, bottom: 60, left: 60 };
  const plotW = width - padding.left - padding.right;
  const plotH = height - padding.top - padding.bottom;

  const categories = chart.x?.values || [];
  const series = chart.series || [];
  if (categories.length === 0 || series.length === 0) {
    return '<p class="empty-state">Empty bar chart series</p>';
  }

  // Find max value
  let maxVal = 0;
  series.forEach(s => {
    s.values.forEach(v => {
      if (v > maxVal) maxVal = v;
    });
  });
  maxVal = maxVal === 0 ? 1 : maxVal * 1.15;

  const groupCount = categories.length;
  const seriesCount = series.length;
  const groupWidth = plotW / groupCount;
  const barWidth = Math.min(36, (groupWidth * 0.7) / seriesCount);

  const colors = ['#B3261E', '#0E0E0E', '#565652', '#8F5B00'];

  let svg = `<svg viewBox="0 0 ${width} ${height}" class="chart-svg" role="img" aria-label="${chart.title}">`;

  // Grid lines and Y axis
  const yTicks = 4;
  for (let i = 0; i <= yTicks; i++) {
    const val = (maxVal / yTicks) * i;
    const yPos = padding.top + plotH - (plotH / yTicks) * i;
    svg += `
      <line x1="${padding.left}" y1="${yPos}" x2="${width - padding.right}" y2="${yPos}" stroke="var(--border-subtle)" stroke-dasharray="3,3" />
      <text x="${padding.left - 10}" y="${yPos + 4}" fill="var(--text-tertiary)" font-family="var(--font-family-mono)" font-size="11" text-anchor="end">${val.toFixed(2)}</text>
    `;
  }

  // Axis labels
  if (chart.y?.label) {
    svg += `<text x="15" y="${height / 2}" fill="var(--text-secondary)" font-family="var(--font-family-base)" font-size="12" font-weight="600" text-anchor="middle" transform="rotate(-90 15 ${height / 2})">${chart.y.label}${chart.y.unit ? ` (${chart.y.unit})` : ''}</text>`;
  }

  // Bars
  categories.forEach((cat, gIdx) => {
    const groupX = padding.left + gIdx * groupWidth + (groupWidth - (barWidth * seriesCount)) / 2;

    series.forEach((s, sIdx) => {
      const val = s.values[gIdx] || 0;
      const barH = (val / maxVal) * plotH;
      const barX = groupX + sIdx * barWidth;
      const barY = padding.top + plotH - barH;
      const color = colors[sIdx % colors.length];

      svg += `
        <rect x="${barX}" y="${barY}" width="${barWidth}" height="${barH}" rx="3" fill="${color}"
              data-tooltip="${s.name ? `${s.name}: ` : ''}${cat} = ${val}"
              style="cursor: pointer; transition: opacity 0.15s;" />
      `;
    });

    // X label
    svg += `
      <text x="${padding.left + gIdx * groupWidth + groupWidth / 2}" y="${height - padding.bottom + 22}" fill="var(--text-secondary)" font-family="var(--font-family-base)" font-size="11" font-weight="500" text-anchor="middle">${cat}</text>
    `;
  });

  // Legend if multi-series
  if (series.length > 1) {
    let legendX = padding.left;
    const legendY = 16;
    svg += `<g class="chart-legend">`;
    series.forEach((s, i) => {
      const color = colors[i % colors.length];
      svg += `
        <rect x="${legendX}" y="${legendY - 8}" width="10" height="10" rx="2" fill="${color}" />
        <text x="${legendX + 14}" y="${legendY}" fill="var(--text-secondary)" font-family="var(--font-family-base)" font-size="11">${s.name}</text>
      `;
      legendX += 130;
    });
    svg += `</g>`;
  }

  svg += `</svg>`;
  return svg;
}

/**
 * Line Chart SVG
 */
function renderLineChart(chart) {
  const width = 640;
  const height = 320;
  const padding = { top: 35, right: 40, bottom: 60, left: 60 };
  const plotW = width - padding.left - padding.right;
  const plotH = height - padding.top - padding.bottom;

  const categories = chart.x?.values || [];
  const series = chart.series || [];
  if (categories.length === 0 || series.length === 0) {
    return '<p class="empty-state">Empty line chart series</p>';
  }

  let maxVal = 0;
  series.forEach(s => {
    s.values.forEach(v => {
      if (v > maxVal) maxVal = v;
    });
  });
  maxVal = maxVal === 0 ? 1 : maxVal * 1.15;

  const colors = ['#B3261E', '#0E0E0E', '#565652', '#8F5B00'];
  let svg = `<svg viewBox="0 0 ${width} ${height}" class="chart-svg" role="img" aria-label="${chart.title}">`;

  // Grid
  const yTicks = 4;
  for (let i = 0; i <= yTicks; i++) {
    const val = (maxVal / yTicks) * i;
    const yPos = padding.top + plotH - (plotH / yTicks) * i;
    svg += `
      <line x1="${padding.left}" y1="${yPos}" x2="${width - padding.right}" y2="${yPos}" stroke="var(--border-subtle)" stroke-dasharray="3,3" />
      <text x="${padding.left - 10}" y="${yPos + 4}" fill="var(--text-tertiary)" font-family="var(--font-family-mono)" font-size="11" text-anchor="end">${val.toFixed(2)}</text>
    `;
  }

  // Y Label
  if (chart.y?.label) {
    svg += `<text x="15" y="${height / 2}" fill="var(--text-secondary)" font-family="var(--font-family-base)" font-size="12" font-weight="600" text-anchor="middle" transform="rotate(-90 15 ${height / 2})">${chart.y.label}</text>`;
  }

  const stepX = plotW / (categories.length - 1 || 1);

  // X Labels
  categories.forEach((cat, idx) => {
    const x = padding.left + idx * stepX;
    svg += `
      <text x="${x}" y="${height - padding.bottom + 22}" fill="var(--text-secondary)" font-family="var(--font-family-base)" font-size="11" text-anchor="middle">${cat}</text>
    `;
  });

  // Series paths
  series.forEach((s, sIdx) => {
    const color = colors[sIdx % colors.length];
    const points = s.values.map((v, idx) => {
      const x = padding.left + idx * stepX;
      const y = padding.top + plotH - (v / maxVal) * plotH;
      return { x, y, val: v, cat: categories[idx] };
    });

    const pathData = points.reduce((acc, p, idx) => {
      return idx === 0 ? `M ${p.x} ${p.y}` : `${acc} L ${p.x} ${p.y}`;
    }, '');

    svg += `<path d="${pathData}" fill="none" stroke="${color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" />`;

    // Points
    points.forEach(p => {
      svg += `
        <circle cx="${p.x}" cy="${p.y}" r="4.5" fill="${color}" stroke="var(--bg-surface)" stroke-width="2"
                data-tooltip="${s.name}: ${p.cat} = ${p.val}"
                style="cursor: pointer;" />
      `;
    });
  });

  // Legend
  if (series.length > 1) {
    let legendX = padding.left;
    const legendY = 16;
    svg += `<g class="chart-legend">`;
    series.forEach((s, i) => {
      const color = colors[i % colors.length];
      svg += `
        <circle cx="${legendX + 4}" cy="${legendY - 3}" r="4" fill="${color}" />
        <text x="${legendX + 14}" y="${legendY}" fill="var(--text-secondary)" font-family="var(--font-family-base)" font-size="11">${s.name}</text>
      `;
      legendX += 160;
    });
    svg += `</g>`;
  }

  svg += `</svg>`;
  return svg;
}

/**
 * Scatter Chart SVG
 */
function renderScatterChart(chart) {
  const width = 640;
  const height = 360;
  const padding = { top: 35, right: 40, bottom: 60, left: 65 };
  const plotW = width - padding.left - padding.right;
  const plotH = height - padding.top - padding.bottom;

  const points = chart.points || [];
  if (points.length === 0) {
    return '<p class="empty-state">No scatter points available</p>';
  }

  // Find min and max
  let minX = 0, maxX = 0, minY = 0, maxY = 0;
  points.forEach((p, idx) => {
    if (idx === 0) {
      minX = maxX = p.x;
      minY = maxY = p.y;
    } else {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }
  });

  // Add margin
  const rangeX = maxX - minX || 1;
  const rangeY = maxY - minY || 1;
  minX -= rangeX * 0.1;
  maxX += rangeX * 0.1;
  minY = Math.min(0, minY - rangeY * 0.1);
  maxY += rangeY * 0.15;

  let svg = `<svg viewBox="0 0 ${width} ${height}" class="chart-svg" role="img" aria-label="${chart.title}">`;

  // Grid Y
  const yTicks = 4;
  for (let i = 0; i <= yTicks; i++) {
    const val = minY + ((maxY - minY) / yTicks) * i;
    const yPos = padding.top + plotH - (plotH / yTicks) * i;
    svg += `
      <line x1="${padding.left}" y1="${yPos}" x2="${width - padding.right}" y2="${yPos}" stroke="var(--border-subtle)" stroke-dasharray="3,3" />
      <text x="${padding.left - 10}" y="${yPos + 4}" fill="var(--text-tertiary)" font-family="var(--font-family-mono)" font-size="11" text-anchor="end">${val.toFixed(2)}</text>
    `;
  }

  // Grid X
  const xTicks = 4;
  for (let i = 0; i <= xTicks; i++) {
    const val = minX + ((maxX - minX) / xTicks) * i;
    const xPos = padding.left + (plotW / xTicks) * i;
    svg += `
      <line x1="${xPos}" y1="${padding.top}" x2="${xPos}" y2="${padding.top + plotH}" stroke="var(--border-subtle)" stroke-dasharray="3,3" />
      <text x="${xPos}" y="${height - padding.bottom + 22}" fill="var(--text-tertiary)" font-family="var(--font-family-mono)" font-size="11" text-anchor="middle">${val.toFixed(2)}</text>
    `;
  }

  // Zero lines if crossing zero
  if (minX <= 0 && maxX >= 0) {
    const zeroX = padding.left + ((0 - minX) / (maxX - minX)) * plotW;
    svg += `<line x1="${zeroX}" y1="${padding.top}" x2="${zeroX}" y2="${padding.top + plotH}" stroke="var(--border-strong)" stroke-width="1.5" />`;
  }
  if (minY <= 0 && maxY >= 0) {
    const zeroY = padding.top + plotH - ((0 - minY) / (maxY - minY)) * plotH;
    svg += `<line x1="${padding.left}" y1="${zeroY}" x2="${width - padding.right}" y2="${zeroY}" stroke="var(--border-strong)" stroke-width="1.5" />`;
  }

  // Axis Labels
  if (chart.x?.label) {
    svg += `<text x="${width / 2}" y="${height - 12}" fill="var(--text-secondary)" font-family="var(--font-family-base)" font-size="12" font-weight="600" text-anchor="middle">${chart.x.label}</text>`;
  }
  if (chart.y?.label) {
    svg += `<text x="18" y="${height / 2}" fill="var(--text-secondary)" font-family="var(--font-family-base)" font-size="12" font-weight="600" text-anchor="middle" transform="rotate(-90 18 ${height / 2})">${chart.y.label}</text>`;
  }

  // Points
  points.forEach(p => {
    const cx = padding.left + ((p.x - minX) / (maxX - minX)) * plotW;
    const cy = padding.top + plotH - ((p.y - minY) / (maxY - minY)) * plotH;
    const isHighRisk = p.y > 0.5;
    const color = isHighRisk ? 'var(--color-accent)' : 'var(--text-ink)';

    svg += `
      <g style="cursor: pointer;" data-tooltip="${p.label}: (${p.x.toFixed(3)}, ${p.y.toFixed(3)})">
        <circle cx="${cx}" cy="${cy}" r="6" fill="${color}" stroke="var(--bg-surface)" stroke-width="1.5" />
        <text x="${cx + 8}" y="${cy + 4}" fill="var(--text-secondary)" font-family="var(--font-family-base)" font-size="10" font-weight="500">${p.label}</text>
      </g>
    `;
  });

  svg += `</svg>`;
  return svg;
}

/**
 * Heatmap Chart SVG
 */
function renderHeatmapChart(chart) {
  const xLabels = chart.x?.values || [];
  const yLabels = chart.y?.values || [];
  const matrix = chart.z || [];

  if (xLabels.length === 0 || yLabels.length === 0 || matrix.length === 0) {
    return '<p class="empty-state">No heatmap data available</p>';
  }

  const width = 640;
  const padding = { top: 30, right: 30, bottom: 50, left: 90 };
  const cellW = (width - padding.left - padding.right) / xLabels.length;
  const cellH = 42;
  const height = padding.top + padding.bottom + yLabels.length * cellH;

  // Find min/max in matrix
  let minZ = 1, maxZ = 0;
  matrix.forEach(row => {
    row.forEach(val => {
      if (val < minZ) minZ = val;
      if (val > maxZ) maxZ = val;
    });
  });

  let svg = `<svg viewBox="0 0 ${width} ${height}" class="chart-svg" role="img" aria-label="${chart.title}">`;

  // Draw cells
  yLabels.forEach((yLab, rIdx) => {
    const row = matrix[rIdx] || [];
    const cellY = padding.top + rIdx * cellH;

    // Y row label
    svg += `
      <text x="${padding.left - 12}" y="${cellY + cellH / 2 + 4}" fill="var(--text-secondary)" font-family="var(--font-family-base)" font-size="11" font-weight="600" text-anchor="end">${yLab}</text>
    `;

    xLabels.forEach((xLab, cIdx) => {
      const val = row[cIdx] ?? 0;
      const cellX = padding.left + cIdx * cellW;

      // Color interpolation: deep red (#B3261E) based on intensity
      const norm = maxZ > minZ ? (val - minZ) / (maxZ - minZ) : val;
      const r = Math.round(242 - norm * (242 - 179));
      const g = Math.round(242 - norm * (242 - 38));
      const b = Math.round(240 - norm * (240 - 30));
      const textFill = norm > 0.55 ? '#FFFFFF' : '#0E0E0E';

      svg += `
        <rect x="${cellX + 2}" y="${cellY + 2}" width="${cellW - 4}" height="${cellH - 4}" rx="4"
              fill="rgb(${r}, ${g}, ${b})" stroke="var(--border-subtle)" stroke-width="1"
              data-tooltip="${yLab}, ${xLab}: ${val}"
              style="cursor: pointer;" />
        <text x="${cellX + cellW / 2}" y="${cellY + cellH / 2 + 4}" fill="${textFill}"
              font-family="var(--font-family-mono)" font-size="11" font-weight="600" text-anchor="middle" pointer-events="none">${val.toFixed(2)}</text>
      `;
    });
  });

  // X column labels
  xLabels.forEach((xLab, cIdx) => {
    const cellX = padding.left + cIdx * cellW + cellW / 2;
    svg += `
      <text x="${cellX}" y="${height - padding.bottom + 22}" fill="var(--text-secondary)" font-family="var(--font-family-base)" font-size="11" font-weight="600" text-anchor="middle">${xLab}</text>
    `;
  });

  svg += `</svg>`;
  return svg;
}
