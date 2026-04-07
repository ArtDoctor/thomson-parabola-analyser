import { isRecord, logEntryByKind } from './results-constants';

function highlightCard(title: string, lines: string[]): HTMLElement {
  const card = document.createElement('article');
  card.className = 'trace-hl-card';
  const h = document.createElement('h4');
  h.className = 'trace-hl-title';
  h.textContent = title;
  card.appendChild(h);
  const ul = document.createElement('ul');
  ul.className = 'trace-hl-list';
  for (const line of lines) {
    const li = document.createElement('li');
    li.textContent = line;
    ul.appendChild(li);
  }
  card.appendChild(ul);
  return card;
}

function fmtNum(n: unknown, digits = 4): string {
  return typeof n === 'number' && Number.isFinite(n) ? n.toFixed(digits) : '—';
}

export function renderTraceHighlights(
  traceHighlightsEl: HTMLDivElement | null,
  log: unknown[] | undefined,
): void {
  if (!traceHighlightsEl) return;
  traceHighlightsEl.innerHTML = '';
  if (!log?.length) {
    traceHighlightsEl.textContent = 'No algorithm trace entries.';
    return;
  }

  const cards: HTMLElement[] = [];

  const flags = logEntryByKind(log, 'run_flags');
  if (flags) {
    const u = flags.use_denoise_unet === true;
    const h = flags.use_experimental_a_for_hydrogen === true;
    cards.push(highlightCard('Run flags', [
      `Denoising: ${u ? 'U-Net' : 'classical'}`,
      `H reference: ${h ? 'experimental a' : 'standard'}`,
    ]));
  }

  const crop = logEntryByKind(log, 'detector_crop');
  if (crop) {
    const x1 = crop.x1;
    const y1 = crop.y1;
    const x2 = crop.x2;
    const y2 = crop.y2;
    const yc = crop.yolo_confidence;
    const lines: string[] = [
      `BBox (${x1}, ${y1}) → (${x2}, ${y2})`,
    ];
    if (typeof yc === 'number') lines.push(`Detector confidence ${(yc * 100).toFixed(1)}%`);
    cards.push(highlightCard('Detector crop', lines));
  }

  const orient = logEntryByKind(log, 'orientation');
  const parity = logEntryByKind(log, 'parity');
  if (parity) {
    const parts: string[] = [];
    if (parity.mirrored_fan_detected === true) parts.push('Mirrored fan detected');
    if (parity.horizontal_flip_applied === true) parts.push('Horizontal flip');
    const reo = parity.reorientation_after_horizontal_flip;
    if (isRecord(reo) && typeof reo.rotation_applied === 'string' && reo.rotation_applied !== 'none') {
      parts.push(`Then: ${String(reo.rotation_applied)}`);
    }
    if (parity.final_vertical_flip_applied === true) parts.push('Vertical flip (final)');
    if (!parts.length) parts.push('No parity corrections');
    cards.push(highlightCard('Orientation & parity', parts));
  } else if (orient) {
    const oxy = orient.origin_xy;
    const dq = orient.dominant_quadrant;
    const rot = orient.rotation_applied;
    const lines: string[] = [];
    if (Array.isArray(oxy) && oxy.length >= 2) {
      lines.push(`Origin (${oxy[0]}, ${oxy[1]})`);
    }
    if (typeof dq === 'string') lines.push(`Dominant quadrant ${dq}`);
    if (typeof rot === 'string' && rot !== 'none') lines.push(`Rotation ${rot}`);
    cards.push(highlightCard('Orientation', lines));
  }

  const den = logEntryByKind(log, 'denoise');
  if (den) {
    const method = den.method;
    const lines: string[] = [];
    if (typeof method === 'string') lines.push(`Method: ${method}`);
    const sc = den.unet_resize_scale;
    if (typeof sc === 'number') lines.push(`U-Net resize scale ${fmtNum(sc, 3)}`);
    cards.push(highlightCard('Denoise', lines));
  }

  const peaks = logEntryByKind(log, 'peak_scan');
  if (peaks) {
    const lines: string[] = [];
    const spot = peaks.brightest_spot_yx;
    if (Array.isArray(spot) && spot.length >= 2) {
      lines.push(`Brightest spot (y, x) = (${spot[0]}, ${spot[1]})`);
    }
    const r0 = peaks.peak_extraction_start_row;
    const r1 = peaks.peak_extraction_end_row;
    if (typeof r0 === 'number' && typeof r1 === 'number') {
      lines.push(`Peak rows ${r0} … ${r1}`);
    }
    cards.push(highlightCard('Peak scan', lines));
  }

  const lp = logEntryByKind(log, 'line_pipeline');
  if (lp) {
    cards.push(highlightCard('Line pipeline', [
      `After build: ${lp.num_lines_after_build ?? '—'} lines`,
      `After merge: ${lp.num_lines_after_merge ?? '—'}`,
      `After smooth: ${lp.num_lines_after_smooth ?? '—'}`,
    ].map(String)));
  }

  const gf = logEntryByKind(log, 'global_fit');
  if (gf) {
    const curvs = gf.per_line_curvatures_from_shared_vertex;
    const nCurv = Array.isArray(curvs) ? curvs.length : 0;
    cards.push(highlightCard('Global vertex fit', [
      `Vertex (${fmtNum(gf.x0_fit)}, ${fmtNum(gf.y0_fit)})`,
      `θ = ${fmtNum(gf.theta_fit, 5)} rad · γ = ${fmtNum(gf.gamma_fit, 5)} · δ = ${fmtNum(gf.delta_fit, 5)}`,
      `${nCurv} line curvature sample(s)`,
    ]));
  }

  const cp = logEntryByKind(log, 'curvature_peaks');
  if (cp) {
    const gav = cp.good_a_values;
    const n = typeof gav === 'object' && gav !== null && 'length' in gav ? (gav as unknown[]).length : 0;
    cards.push(highlightCard('Selected curvatures (a)', [
      `${n} peak(s) after scoring`,
      typeof cp.score_grid_size === 'number' ? `Score grid ${cp.score_grid_size} pts` : '',
      typeof cp.prominence_rel === 'number' && typeof cp.height_rel === 'number'
        ? `Prominence ${(Number(cp.prominence_rel) * 100).toFixed(1)}% · height ${(Number(cp.height_rel) * 100).toFixed(1)}% (rel.)`
        : '',
    ].filter(Boolean).map(String)));
  }

  const xp = logEntryByKind(log, 'xp_span');
  if (xp && typeof xp.xp_min === 'number' && typeof xp.xp_max === 'number') {
    const w = xp.xp_max - xp.xp_min;
    cards.push(highlightCard('m/q sampling span', [
      `xp ∈ [${fmtNum(xp.xp_min, 2)}, ${fmtNum(xp.xp_max, 2)}]`,
      `Width ${fmtNum(w, 2)} (px along dispersion)`,
    ]));
  }

  const href = logEntryByKind(log, 'hydrogen_reference');
  if (href) {
    cards.push(highlightCard('Hydrogen reference', [
      `Mode: ${String(href.mode ?? '—')}`,
      `a_H = ${typeof href.hydrogen_a === 'number' ? href.hydrogen_a.toExponential(3) : '—'}`,
      typeof href.classification_match_tolerance === 'number'
        ? `Match tolerance ±${(Number(href.classification_match_tolerance) * 100).toFixed(0)}%`
        : '',
    ].filter(Boolean)));
  }

  const sint = logEntryByKind(log, 'spectra_integration');
  if (sint) {
    const wins = sint.integration_windows_a;
    const nw = Array.isArray(wins) ? wins.length : 0;
    const lines = [
      `${sint.integration_a_samples ?? '—'} samples per species`,
      `${nw} integration window(s) in a`,
    ];
    const bx0 = sint.background_roi_x0;
    const bx1 = sint.background_roi_x1;
    const by0 = sint.background_roi_y0;
    const by1 = sint.background_roi_y1;
    if (typeof bx0 === 'number' && typeof bx1 === 'number' && typeof by0 === 'number' && typeof by1 === 'number') {
      lines.push(`Background ROI x[${bx0}, ${bx1}] y[${by0}, ${by1}]`);
    }
    cards.push(highlightCard('Spectra integration', lines.map(String)));
  }

  const ds = logEntryByKind(log, 'detection_settings');
  if (ds) {
    cards.push(highlightCard('Detection settings', [
      `Window ${ds.window ?? '—'} · prominence ${ds.prominence ?? '—'} · distance ${ds.distance ?? '—'}`,
      `Line length ≥${ds.min_line_length_1 ?? '—'} / ≥${ds.min_line_length_2 ?? '—'} · max gap ${ds.max_x_gap ?? '—'}`,
    ].map(String)));
  }

  for (const c of cards) {
    traceHighlightsEl.appendChild(c);
  }
}
