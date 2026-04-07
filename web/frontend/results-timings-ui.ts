import {
  formatSeconds,
  TIMING_LABELS,
  TIMING_ORDER,
} from './results-constants';

export function renderTimings(
  timingsTotalLine: HTMLDivElement | null,
  timingsRows: HTMLDivElement | null,
  timings: Record<string, unknown> | undefined,
): void {
  if (!timingsTotalLine || !timingsRows) return;
  timingsTotalLine.innerHTML = '';
  timingsRows.innerHTML = '';

  const totalRaw = timings?.total;
  const total = typeof totalRaw === 'number' ? totalRaw : NaN;
  if (!timings || !Number.isFinite(total) || total <= 0) {
    timingsTotalLine.textContent = 'No timing data in result.';
    return;
  }

  const totalEl = document.createElement('div');
  totalEl.className = 'timings-total-badge';
  totalEl.innerHTML = `<span class="timings-total-value">${formatSeconds(total)}</span><span class="timings-total-caption">total wall time</span>`;
  timingsTotalLine.appendChild(totalEl);

  const ordered: string[] = [];
  const seen = new Set<string>();
  for (const k of TIMING_ORDER) {
    if (k in timings && k !== 'total') {
      ordered.push(k);
      seen.add(k);
    }
  }
  for (const k of Object.keys(timings)) {
    if (k !== 'total' && !seen.has(k)) ordered.push(k);
  }

  for (const key of ordered) {
    const v = timings[key];
    if (typeof v !== 'number' || key === 'total') continue;
    const pct = Math.min(100, Math.max(0.5, (v / total) * 100));
    const row = document.createElement('div');
    row.className = 'timing-row';
    row.style.animationDelay = `${ordered.indexOf(key) * 35}ms`;
    const label = TIMING_LABELS[key] ?? key.replace(/_/g, ' ');
    row.innerHTML = `
      <div class="timing-row-top">
        <span class="timing-label">${label}</span>
        <span class="timing-value">${formatSeconds(v)}</span>
      </div>
      <div class="timing-bar-track"><div class="timing-bar-fill" style="width:${pct}%"></div></div>
    `;
    timingsRows.appendChild(row);
  }
}
