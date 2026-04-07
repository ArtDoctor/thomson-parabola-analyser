// Persist recent analysis job ids + source filenames; index/results Recent panel.

const STORAGE_KEY = 'oblisk_analysis_history';
const MAX_ENTRIES = 40;
const SPECIES_SUMMARY_MAX = 120;

const PREVIEW_PLOT = '10_detected_parabolas.png';

export type AnalysisHistoryEntry = {
  id: string;
  name: string;
  at: number;
  species?: string;
};

/** Comma-separated classification labels (e.g. C⁺, O²⁺) for history summaries. */
export function speciesSummaryFromResult(result: unknown): string | undefined {
  if (result === null || typeof result !== 'object' || Array.isArray(result)) {
    return undefined;
  }
  const classified = (result as Record<string, unknown>).classified;
  if (!Array.isArray(classified) || classified.length === 0) {
    return undefined;
  }
  const labels: string[] = [];
  for (const item of classified) {
    if (item !== null && typeof item === 'object' && !Array.isArray(item)) {
      const lab = (item as Record<string, unknown>).label;
      if (typeof lab === 'string') {
        const t = lab.trim();
        if (t) {
          labels.push(t);
        }
      }
    }
  }
  if (labels.length === 0) {
    return undefined;
  }
  let s = labels.join(', ');
  if (s.length > SPECIES_SUMMARY_MAX) {
    s = `${s.slice(0, SPECIES_SUMMARY_MAX - 1)}…`;
  }
  return s;
}

function isUuid(jobId: string): boolean {
  return /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(
    jobId,
  );
}

function parseStored(raw: string): AnalysisHistoryEntry[] {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw) as unknown;
  } catch {
    return [];
  }
  if (!Array.isArray(parsed)) {
    return [];
  }
  const out: AnalysisHistoryEntry[] = [];
  for (const item of parsed) {
    if (item === null || typeof item !== 'object' || Array.isArray(item)) {
      continue;
    }
    const rec = item as Record<string, unknown>;
    const id = rec.id;
    const name = rec.name;
    const at = rec.at;
    if (typeof id !== 'string' || !isUuid(id)) {
      continue;
    }
    if (typeof name !== 'string') {
      continue;
    }
    const atNum = typeof at === 'number' && Number.isFinite(at) ? at : 0;
    const speciesRaw = rec.species;
    let species: string | undefined;
    if (typeof speciesRaw === 'string' && speciesRaw.trim()) {
      const t = speciesRaw.trim();
      species =
        t.length > SPECIES_SUMMARY_MAX ? `${t.slice(0, SPECIES_SUMMARY_MAX - 1)}…` : t;
    }
    const entry: AnalysisHistoryEntry = { id, name, at: atNum };
    if (species) {
      entry.species = species;
    }
    out.push(entry);
  }
  return out;
}

export function readHistory(): AnalysisHistoryEntry[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return [];
    }
    return parseStored(raw);
  } catch {
    return [];
  }
}

export function recordAnalysis(
  jobId: string,
  sourceName: string,
  species?: string,
): void {
  if (!isUuid(jobId)) {
    return;
  }
  const name = sourceName.trim() || 'Unknown';
  try {
    const existing = readHistory();
    const prev = existing.find((e) => e.id === jobId);
    let speciesStored: string | undefined;
    const trimmed = species?.trim();
    if (trimmed) {
      speciesStored =
        trimmed.length > SPECIES_SUMMARY_MAX
          ? `${trimmed.slice(0, SPECIES_SUMMARY_MAX - 1)}…`
          : trimmed;
    } else if (prev?.species) {
      speciesStored = prev.species;
    }
    let list = existing.filter((e) => e.id !== jobId);
    const entry: AnalysisHistoryEntry = { id: jobId, name, at: Date.now() };
    if (speciesStored) {
      entry.species = speciesStored;
    }
    list.unshift(entry);
    list = list.slice(0, MAX_ENTRIES);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(list));
  } catch {
    /* quota / private mode */
  }
}

export function previewThumbUrl(jobId: string): string {
  const apiBase = import.meta?.env?.VITE_API_URL ?? '';
  return `${apiBase}/data/results/${jobId}/plots/${PREVIEW_PLOT}?thumb=1`;
}

export function resultsPagePath(jobId: string): string {
  return `/results/${jobId}`;
}

export function resultsPageAbsoluteUrl(jobId: string): string {
  return new URL(resultsPagePath(jobId), window.location.origin).href;
}

function closePanel(
  sidebar: HTMLElement,
  overlay: HTMLElement,
  btn: HTMLButtonElement,
): void {
  sidebar.classList.remove('recent-sidebar--open');
  overlay.classList.add('hidden');
  overlay.setAttribute('aria-hidden', 'true');
  sidebar.setAttribute('aria-hidden', 'true');
  btn.setAttribute('aria-expanded', 'false');
}

function openPanel(
  sidebar: HTMLElement,
  overlay: HTMLElement,
  btn: HTMLButtonElement,
): void {
  sidebar.classList.add('recent-sidebar--open');
  overlay.classList.remove('hidden');
  overlay.setAttribute('aria-hidden', 'false');
  sidebar.setAttribute('aria-hidden', 'false');
  btn.setAttribute('aria-expanded', 'true');
}

/** Run after the slide-in transform finishes (or timeout); avoids black thumbs until click. */
function afterRecentSidebarEntered(
  sidebar: HTMLElement,
  fn: () => void,
): void {
  let ran = false;
  const run = (): void => {
    if (ran) {
      return;
    }
    ran = true;
    requestAnimationFrame(() => {
      requestAnimationFrame(fn);
    });
  };

  const onEnd = (ev: TransitionEvent): void => {
    if (ev.propertyName !== 'transform') {
      return;
    }
    sidebar.removeEventListener('transitionend', onEnd);
    run();
  };

  sidebar.addEventListener('transitionend', onEnd);
  window.setTimeout(() => {
    sidebar.removeEventListener('transitionend', onEnd);
    run();
  }, 420);
}

function replaceThumbWithPlaceholder(img: HTMLImageElement): void {
  const ph = document.createElement('div');
  ph.className = 'recent-card-thumb recent-card-thumb--placeholder';
  ph.textContent = 'No preview';
  img.replaceWith(ph);
}

/** Retries: cold Docker / 503 while plot thumb is generated; 0×0 decode races. */
function wireRecentThumb(img: HTMLImageElement, urlBase: string): void {
  let attempt = 0;
  const maxAttempts = 6;

  const bumpPaint = (): void => {
    if (img.naturalWidth < 1 || !img.isConnected) {
      return;
    }
    const kick = (): void => {
      if (img.isConnected) {
        img.classList.add('recent-card-thumb--ready');
      }
    };
    void img.decode().then(kick).catch(kick);
    window.setTimeout(kick, 48);
  };

  const startAttempt = (): void => {
    img.onload = (): void => {
      if (img.naturalWidth < 1) {
        if (attempt < maxAttempts - 1) {
          attempt += 1;
          window.setTimeout(startAttempt, 200 + attempt * 120);
        } else {
          replaceThumbWithPlaceholder(img);
        }
        return;
      }
      bumpPaint();
    };
    img.onerror = (): void => {
      if (attempt < maxAttempts - 1) {
        attempt += 1;
        window.setTimeout(startAttempt, 280 + attempt * 150);
      } else {
        replaceThumbWithPlaceholder(img);
      }
    };
    const sep = urlBase.includes('?') ? '&' : '?';
    img.src = `${urlBase}${sep}_cb=${Date.now()}&n=${attempt}`;
  };

  img.loading = 'eager';
  img.decoding = 'sync';
  startAttempt();
}

function renderList(container: HTMLDivElement): void {
  const entries = readHistory();
  container.innerHTML = '';
  for (const e of entries) {
    const card = document.createElement('article');
    card.className = 'recent-card';

    const previewLink = document.createElement('a');
    previewLink.className = 'recent-card-preview-link';
    previewLink.href = resultsPagePath(e.id);
    previewLink.setAttribute('aria-label', `Open analysis: ${e.name}`);

    const previewWrap = document.createElement('div');
    previewWrap.className = 'recent-card-preview';
    const img = document.createElement('img');
    img.className = 'recent-card-thumb';
    img.alt = '';
    const thumbBase = `${previewThumbUrl(e.id)}&t=${e.at}`;
    wireRecentThumb(img, thumbBase);
    previewWrap.appendChild(img);
    previewLink.appendChild(previewWrap);

    const body = document.createElement('div');
    body.className = 'recent-card-body';

    const title = document.createElement('div');
    title.className = 'recent-card-name';
    title.textContent = e.name;

    const idRow = document.createElement('div');
    idRow.className = 'recent-card-id';
    idRow.textContent = e.id;

    const link = document.createElement('a');
    link.className = 'recent-card-link';
    link.href = resultsPagePath(e.id);
    link.textContent = resultsPageAbsoluteUrl(e.id);

    body.appendChild(title);
    if (e.species) {
      const speciesRow = document.createElement('div');
      speciesRow.className = 'recent-card-species';
      speciesRow.textContent = e.species;
      speciesRow.title = e.species;
      body.appendChild(speciesRow);
    }
    body.appendChild(idRow);
    body.appendChild(link);

    card.appendChild(previewLink);
    card.appendChild(body);
    container.appendChild(card);
  }
}

export function refreshRecentButtonVisibility(): void {
  const btn = document.getElementById(
    'btn-recent-analyses',
  ) as HTMLButtonElement | null;
  if (!btn) {
    return;
  }
  if (readHistory().length > 0) {
    btn.classList.remove('hidden');
  } else {
    btn.classList.add('hidden');
  }
}

/** Wire Recent analyses sidebar; safe to call if DOM nodes are missing. */
export function initRecentAnalysesPanel(): void {
  const btn = document.getElementById(
    'btn-recent-analyses',
  ) as HTMLButtonElement | null;
  const sidebar = document.getElementById('recent-sidebar');
  const overlay = document.getElementById('recent-overlay');
  const closeBtn = document.getElementById('recent-sidebar-close');
  const list = document.getElementById('recent-list') as HTMLDivElement | null;

  refreshRecentButtonVisibility();

  if (!btn || !sidebar || !overlay || !list) {
    return;
  }

  const open = (): void => {
    openPanel(sidebar, overlay, btn);
    afterRecentSidebarEntered(sidebar, () => {
      renderList(list);
    });
  };

  const close = (): void => {
    closePanel(sidebar, overlay, btn);
  };

  btn.addEventListener('click', () => {
    if (sidebar.classList.contains('recent-sidebar--open')) {
      close();
    } else {
      open();
    }
  });

  overlay.addEventListener('click', close);
  if (closeBtn) {
    closeBtn.addEventListener('click', close);
  }

  document.addEventListener('keydown', (ev) => {
    if (ev.key === 'Escape' && sidebar.classList.contains('recent-sidebar--open')) {
      close();
    }
  });
}
