// results.ts – polls /status/:id and updates the results page

import {
  initRecentAnalysesPanel,
  recordAnalysis,
  speciesSummaryFromResult,
  refreshRecentButtonVisibility,
} from './analysis-history';
import { humanisedLabel } from './results-constants';
import { renderTimings } from './results-timings-ui';
import { renderTraceHighlights } from './results-trace-highlights';
import { buildTextExport } from './results-text-export';
import { InteractiveViewer } from './interactive-viewer';

const API_BASE = import.meta?.env?.VITE_API_URL ?? '';
const POLL_INTERVAL_MS = 2_500;

// ---- DOM refs
const statusBanner  = document.getElementById('status-banner')  as HTMLDivElement;
const statusSpinner = document.getElementById('status-spinner') as HTMLDivElement;
const statusText    = document.getElementById('status-text')    as HTMLSpanElement;
const progressBar   = document.getElementById('progress-bar')   as HTMLDivElement;
const plotsSub      = document.getElementById('plots-sub')      as HTMLParagraphElement;
const plotsFileName = document.getElementById('plots-file-name') as HTMLParagraphElement | null;
const plotsGrid     = document.getElementById('plots-grid')     as HTMLDivElement;
const resultSection = document.getElementById('result-section') as HTMLElement;
const runSummary    = document.getElementById('run-summary')    as HTMLElement;
const timingsTotalLine = document.getElementById('timings-total-line') as HTMLDivElement;
const timingsRows   = document.getElementById('timings-rows')   as HTMLDivElement;
const traceHighlightsEl = document.getElementById('trace-highlights') as HTMLDivElement;
const resultCards   = document.getElementById('result-cards')   as HTMLDivElement;
const algorithmLogDetails = document.getElementById('algorithm-log-details') as HTMLDetailsElement;
const algorithmLogEl = document.getElementById('algorithm-log') as HTMLDivElement;
const rawJson       = document.getElementById('raw-json')       as HTMLPreElement;
const errorBanner   = document.getElementById('error-banner')   as HTMLDivElement;
const errorDetail   = document.getElementById('error-detail')   as HTMLParagraphElement;
const lightbox      = document.getElementById('lightbox')       as HTMLDivElement;
const lightboxImg   = document.getElementById('lightbox-img')   as HTMLImageElement;
const lightboxCaption = document.getElementById('lightbox-caption') as HTMLParagraphElement;
const lightboxClose = document.getElementById('lightbox-close') as HTMLButtonElement;
const exportJsonBtn = document.getElementById('export-json') as HTMLButtonElement | null;
const exportTxtBtn = document.getElementById('export-txt') as HTMLButtonElement | null;
const ivSection = document.getElementById('iv-section') as HTMLElement | null;

// ---- State
const shownPlots = new Set<string>();
let jobId: string | null = null;
let pollTimer: ReturnType<typeof setTimeout> | null = null;
let lastResult: Record<string, unknown> | null = null;
let interactiveViewer: InteractiveViewer | null = null;

function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.rel = 'noopener';
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function exportFilename(ext: string): string {
  const id = jobId ?? 'result';
  return `oblisk-result-${id}.${ext}`;
}

function onExportJson(): void {
  if (!lastResult) return;
  const body = JSON.stringify(lastResult, null, 2);
  downloadBlob(new Blob([body], { type: 'application/json;charset=utf-8' }), exportFilename('json'));
}

function onExportTxt(): void {
  if (!lastResult) return;
  const body = buildTextExport(jobId, lastResult);
  downloadBlob(new Blob([body], { type: 'text/plain;charset=utf-8' }), exportFilename('txt'));
}

if (exportJsonBtn) exportJsonBtn.addEventListener('click', onExportJson);
if (exportTxtBtn) exportTxtBtn.addEventListener('click', onExportTxt);

// ---- Extract job id from URL: /results/<uuid>
function extractJobId(): string | null {
  const parts = window.location.pathname.split('/').filter(Boolean);
  // parts: ['results', '<uuid>']  OR  ['<uuid>'] if served under /results/
  for (let i = parts.length - 1; i >= 0; i--) {
    if (parts[i].match(/^[0-9a-f-]{36}$/i)) return parts[i];
  }
  return null;
}

// ---- Lightbox
function openLightbox(fullSrc: string, caption: string) {
  lightboxImg.classList.add('lightbox-img--loading');
  lightboxImg.onload = (): void => {
    lightboxImg.classList.remove('lightbox-img--loading');
    lightboxImg.onload = null;
  };
  lightboxImg.onerror = (): void => {
    lightboxImg.classList.remove('lightbox-img--loading');
    lightboxImg.onerror = null;
  };
  lightboxImg.src = fullSrc;
  lightboxCaption.textContent = caption;
  lightbox.classList.remove('hidden');
}

lightboxClose.addEventListener('click', () => lightbox.classList.add('hidden'));
lightbox.addEventListener('click', (e) => {
  if (e.target === lightbox) lightbox.classList.add('hidden');
});
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') lightbox.classList.add('hidden');
});

// ---- Add a plot card
function addPlotCard(plotUrl: string) {
  if (shownPlots.has(plotUrl)) return;
  shownPlots.add(plotUrl);

  const label = humanisedLabel(plotUrl);
  const card = document.createElement('div');
  card.className = 'plot-card';
  card.style.animationDelay = `${(shownPlots.size - 1) * 60}ms`;

  const baseUrl = plotUrl.split('?')[0];
  const img = document.createElement('img');
  img.className = 'plot-img';
  img.src = `${baseUrl}?thumb=1&t=${Date.now()}`;
  img.alt = label;
  img.loading = 'lazy';
  img.decoding = 'async';

  const lbl = document.createElement('div');
  lbl.className = 'plot-label';
  lbl.textContent = label;

  card.appendChild(img);
  card.appendChild(lbl);
  card.addEventListener('click', () => openLightbox(`${baseUrl}?t=${Date.now()}`, label));
  plotsGrid.appendChild(card);
}

// ---- Render classification results
function renderResults(result: Record<string, unknown>) {
  if (!result) return;
  lastResult = result;
  resultSection.classList.remove('hidden');
  runSummary.classList.remove('hidden');

  const timings = result.timings as Record<string, unknown> | undefined;
  renderTimings(timingsTotalLine, timingsRows, timings);

  const algoLog = result.algorithm_log as unknown[] | undefined;
  renderTraceHighlights(traceHighlightsEl, algoLog);

  const classified = result.classified as Array<Record<string, unknown>> | undefined;
  if (classified?.length) {
    resultCards.innerHTML = '';
    classified.forEach((item, idx) => {
      const card = document.createElement('div');
      card.className = 'result-card';
      card.style.animationDelay = `${idx * 80}ms`;

      const labelEl = document.createElement('div');
      labelEl.className = 'result-label';
      labelEl.textContent = String(item.label ?? '?');

      const aEl = document.createElement('div');
      aEl.className = 'result-a';
      aEl.textContent = `a = ${typeof item.a === 'number' ? item.a.toExponential(3) : item.a}`;

      card.appendChild(labelEl);
      card.appendChild(aEl);

      const candidates = item.candidates as Array<Record<string, unknown>> | undefined;
      if (candidates?.length) {
        const cList = document.createElement('div');
        cList.className = 'result-candidates';
        candidates.forEach(c => {
          const row = document.createElement('div');
          row.className = 'result-candidate';
          const relErr = typeof c.rel_err === 'number' ? (c.rel_err * 100).toFixed(2) + '%' : '?';
          row.innerHTML = `<span class="cname">${c.name}</span><span class="cerr">${relErr}</span>`;
          cList.appendChild(row);
        });
        card.appendChild(cList);
      }

      resultCards.appendChild(card);
    });
  }

  rawJson.textContent = JSON.stringify(result, null, 2);

  // ---- Interactive Viewer
  if (ivSection && result.classified && result.geometry && !interactiveViewer) {
    ivSection.classList.remove('hidden');
    interactiveViewer = new InteractiveViewer('iv-container');
    // Find the raw cropped image (no matplotlib frame) for accurate overlay,
    // falling back to the matplotlib figure if unavailable.
    const plotBase = shownPlots.size > 0
      ? [...shownPlots].find(u => u.includes('01_cropped_standardized'))?.replace(/01_cropped_standardized.*/, '')
      : null;
    const rawUrl = plotBase ? `${plotBase}00_raw_cropped.png?t=${Date.now()}` : '';
    const fallbackUrl = shownPlots.size > 0
      ? [...shownPlots].find(u => u.includes('01_cropped_standardized'))
      : null;
    const imgUrl = rawUrl || (fallbackUrl ? `${fallbackUrl.split('?')[0]}?t=${Date.now()}` : '');
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    interactiveViewer.load(result as any, imgUrl);
  }

  if (algoLog?.length && algorithmLogDetails && algorithmLogEl) {
    algorithmLogDetails.classList.remove('hidden');
    algorithmLogEl.innerHTML = '';
    algoLog.forEach((entry, idx) => {
      const row = document.createElement('div');
      row.className = 'algorithm-log-row';
      row.style.animationDelay = `${idx * 40}ms`;
      if (entry && typeof entry === 'object' && !Array.isArray(entry)) {
        const rec = entry as Record<string, unknown>;
        const kind = String(rec.kind ?? 'record');
        const head = document.createElement('div');
        head.className = 'algorithm-log-kind';
        head.textContent = kind;
        const pre = document.createElement('pre');
        pre.className = 'algorithm-log-body';
        const { kind: _k, ...rest } = rec;
        pre.textContent = JSON.stringify(rest, null, 2);
        row.appendChild(head);
        row.appendChild(pre);
      } else {
        const pre = document.createElement('pre');
        pre.className = 'algorithm-log-body';
        pre.textContent = JSON.stringify(entry, null, 2);
        row.appendChild(pre);
      }
      algorithmLogEl.appendChild(row);
    });
  } else if (algorithmLogDetails) {
    algorithmLogDetails.classList.add('hidden');
  }
}

// ---- Main polling loop
async function poll() {
  if (!jobId) return;

  try {
    const resp = await fetch(`${API_BASE}/status/${jobId}`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

    const data = await resp.json();
    const state: string       = data.state;
    const plots: string[]     = data.plots ?? [];
    const totalPlots: number  = data.total_plots ?? 10;
    const result              = data.result;
    const sourceFilename      = data.source_filename;

    if (plotsFileName) {
      if (typeof sourceFilename === 'string' && sourceFilename.length > 0) {
        plotsFileName.textContent = `Source file: ${sourceFilename}`;
        plotsFileName.classList.remove('hidden');
      } else {
        plotsFileName.textContent = '';
        plotsFileName.classList.add('hidden');
      }
    }

    // update plot cards
    plots.forEach(addPlotCard);

    // update progress
    const pct = totalPlots > 0 ? Math.round((shownPlots.size / totalPlots) * 100) : 0;
    progressBar.style.width = pct + '%';

    if (state === 'running') {
      statusText.textContent = `Processing… (${shownPlots.size} / ${totalPlots} plots ready)`;
      plotsSub.textContent   = `${shownPlots.size} of ${totalPlots} plots available`;
      pollTimer = setTimeout(poll, POLL_INTERVAL_MS);

    } else if (state === 'done') {
      statusSpinner.style.display = 'none';
      statusText.textContent = `✓ Done — ${shownPlots.size} plots generated`;
      progressBar.style.width = '100%';
      plotsSub.textContent   = `${shownPlots.size} plots`;
      if (jobId) {
        const label =
          typeof sourceFilename === 'string' && sourceFilename.length > 0
            ? sourceFilename
            : 'Result';
        const speciesSummary = result
          ? speciesSummaryFromResult(result)
          : undefined;
        recordAnalysis(jobId, label, speciesSummary);
        refreshRecentButtonVisibility();
      }
      if (result) renderResults(result);
      // done – no more polling

    } else if (state === 'error') {
      stopWithError(data.error ?? 'Unknown processing error');
    }

  } catch (err) {
    stopWithError(err instanceof Error ? err.message : String(err));
  }
}

function stopWithError(msg: string) {
  if (pollTimer) clearTimeout(pollTimer);
  statusBanner.classList.add('hidden');
  errorBanner.classList.remove('hidden');
  errorDetail.textContent = msg;
}

// ---- Boot
jobId = extractJobId();
if (!jobId) {
  stopWithError('No job ID found in URL.');
} else {
  poll();
}

initRecentAnalysesPanel();
