import {
  formatSeconds,
  TIMING_LABELS,
  TIMING_ORDER,
} from './results-constants';

export function timingsTextLines(timings: Record<string, unknown> | undefined): string[] {
  const lines: string[] = [];
  if (!timings) {
    lines.push('(none)');
    return lines;
  }
  const totalRaw = timings.total;
  const total = typeof totalRaw === 'number' ? totalRaw : NaN;
  if (Number.isFinite(total) && total > 0) {
    lines.push(`Total wall time: ${formatSeconds(total)}`);
  } else {
    lines.push('Total wall time: —');
  }
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
    const label = TIMING_LABELS[key] ?? key.replace(/_/g, ' ');
    lines.push(`  ${label}: ${formatSeconds(v)}`);
  }
  return lines;
}

export function buildTextExport(
  jobId: string | null,
  result: Record<string, unknown>,
): string {
  const lines: string[] = [
    'Oblisk — analysis result export',
    jobId ? `Job ID: ${jobId}` : 'Job ID: —',
    '',
    'Timings',
    '-------',
    ...timingsTextLines(result.timings as Record<string, unknown> | undefined),
    '',
    'Classification',
    '--------------',
  ];
  const classified = result.classified as Array<Record<string, unknown>> | undefined;
  if (classified?.length) {
    classified.forEach((item, idx) => {
      const aStr =
        typeof item.a === 'number' && Number.isFinite(item.a)
          ? item.a.toExponential(3)
          : String(item.a ?? '—');
      lines.push(`${idx + 1}. ${String(item.label ?? '?')} — a = ${aStr}`);
      const candidates = item.candidates as Array<Record<string, unknown>> | undefined;
      if (candidates?.length) {
        candidates.forEach((c) => {
          const rel =
            typeof c.rel_err === 'number' && Number.isFinite(c.rel_err)
              ? `${(c.rel_err * 100).toFixed(2)}%`
              : '?';
          lines.push(`     ${String(c.name ?? '?')}  rel. error ${rel}`);
        });
      }
    });
  } else {
    lines.push('(none)');
  }
  lines.push('');
  lines.push('For the full structured payload, use the JSON export.');
  return lines.join('\n');
}
