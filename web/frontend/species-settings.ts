import {
  PERIODIC_CELLS,
  PERIODIC_TOTAL_COLS,
  PERIODIC_TOTAL_ROWS,
  type PeriodicCell,
} from './periodic-layout';

const STORAGE_KEY = 'oblisk_classification_elements';

const DEFAULT_SYMBOLS: readonly string[] = ['H', 'C', 'O', 'Si'];

const SYMBOL_TO_Z: ReadonlyMap<string, number> = new Map(
  PERIODIC_CELLS.map((c) => [c.sym, c.z]),
);

function sortSymbols(symbols: string[]): string[] {
  return [...symbols].sort(
    (a, b) => (SYMBOL_TO_Z.get(a) ?? 999) - (SYMBOL_TO_Z.get(b) ?? 999),
  );
}

function canonicalSym(raw: string): string {
  const t = raw.trim();
  if (t.length === 0) {
    return '';
  }
  if (t.length === 1) {
    return t.toUpperCase();
  }
  if (t.length === 2) {
    return t[0].toUpperCase() + t[1].toLowerCase();
  }
  return t[0].toUpperCase() + t.slice(1).toLowerCase();
}

function readStoredSymbols(): string[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw === null) {
      return [...DEFAULT_SYMBOLS];
    }
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      return [...DEFAULT_SYMBOLS];
    }
    const strings = parsed.map((x) => String(x).trim()).filter(Boolean);
    const canonical = new Set<string>();
    for (const s of strings) {
      const sym = canonicalSym(s);
      if (sym) {
        canonical.add(sym);
      }
    }
    const known = [...canonical].filter((x) => SYMBOL_TO_Z.has(x));
    if (known.length === 0) {
      return [...DEFAULT_SYMBOLS];
    }
    return sortSymbols(known);
  } catch {
    return [...DEFAULT_SYMBOLS];
  }
}

function writeStoredSymbols(symbols: string[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(sortSymbols(symbols)));
  } catch {
    /* ignore */
  }
}

let selection: Set<string> = new Set(readStoredSymbols());

export function getClassificationElements(): string[] {
  return sortSymbols([...selection]);
}

/** Restore species selection to built-in defaults (H, C, O, Si). */
export function resetClassificationToDefaults(): void {
  selection = new Set(sortSymbols([...DEFAULT_SYMBOLS]));
  writeStoredSymbols([...selection]);
  refreshAll();
}

/** Replace species selection from an imported preset. Returns false if no valid symbols. */
export function applyClassificationElements(symbols: string[]): boolean {
  const canonical = new Set<string>();
  for (const s of symbols) {
    const sym = canonicalSym(String(s));
    if (sym) {
      canonical.add(sym);
    }
  }
  const known = [...canonical].filter((x) => SYMBOL_TO_Z.has(x));
  if (known.length === 0) {
    return false;
  }
  selection = new Set(sortSymbols(known));
  writeStoredSymbols([...selection]);
  refreshAll();
  return true;
}

function applyAriaExpanded(open: boolean): void {
  const edit = document.getElementById('btn-species-edit') as HTMLButtonElement | null;
  if (edit) {
    edit.setAttribute('aria-expanded', open ? 'true' : 'false');
  }
}

function renderChips(container: HTMLElement): void {
  container.replaceChildren();
  const ordered = sortSymbols([...selection]);
  for (const sym of ordered) {
    const el = document.createElement('span');
    el.className = 'species-chip';
    el.textContent = sym;
    el.title = `Z = ${SYMBOL_TO_Z.get(sym) ?? '?'}`;
    container.append(el);
  }
}

function toggleWideDialog(wide: boolean): void {
  settingsDialogEl?.classList.toggle('settings-dialog-wide', wide);
}

function maybeToggleSymbol(sym: string): void {
  if (selection.has(sym)) {
    if (selection.size <= 1) {
      return;
    }
    selection.delete(sym);
  } else {
    selection.add(sym);
  }
  writeStoredSymbols([...selection]);
  refreshAll();
}

function refreshAll(): void {
  const chips = document.getElementById('species-chips');
  if (chips) {
    renderChips(chips);
  }
  document.querySelectorAll<HTMLButtonElement>('.pt-cell').forEach((btn) => {
    const sym = btn.dataset.sym;
    if (!sym) {
      return;
    }
    const on = selection.has(sym);
    btn.classList.toggle('pt-cell-selected', on);
    btn.setAttribute('aria-pressed', on ? 'true' : 'false');
  });
}

let settingsDialogEl: HTMLDialogElement | null = null;

function chargeRangeHint(z: number): string {
  if (z <= 1) {
    return '1+';
  }
  return `1+–${z}+`;
}

export function initSpeciesSettings(): void {
  selection = new Set(readStoredSymbols());
  settingsDialogEl = document.getElementById('settings-dialog') as HTMLDialogElement | null;

  const chips = document.getElementById('species-chips');
  const picker = document.getElementById('species-picker');
  const editBtn = document.getElementById('btn-species-edit') as HTMLButtonElement | null;
  const grid = document.getElementById('periodic-grid');

  if (!chips || !picker || !editBtn || !grid) {
    return;
  }

  const byPos = new Map<string, PeriodicCell>();
  for (const c of PERIODIC_CELLS) {
    byPos.set(`${c.row},${c.col}`, c);
  }

  grid.replaceChildren();
  grid.style.gridTemplateColumns = `repeat(${PERIODIC_TOTAL_COLS}, minmax(0, 1fr))`;
  grid.style.gridTemplateRows = `repeat(${PERIODIC_TOTAL_ROWS}, auto)`;

  for (let row = 0; row < PERIODIC_TOTAL_ROWS; row++) {
    for (let col = 0; col < PERIODIC_TOTAL_COLS; col++) {
      const c = byPos.get(`${row},${col}`);
      if (c === undefined) {
        const placeholder = document.createElement('div');
        placeholder.className = 'pt-empty';
        placeholder.style.gridRow = String(row + 1);
        placeholder.style.gridColumn = String(col + 1);
        grid.append(placeholder);
        continue;
      }
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'pt-cell';
      btn.dataset.sym = c.sym;
      btn.style.gridRow = String(row + 1);
      btn.style.gridColumn = String(col + 1);
      btn.title = `${c.sym}: charge states ${chargeRangeHint(c.z)}`;
      btn.setAttribute('aria-pressed', selection.has(c.sym) ? 'true' : 'false');

      const zSpan = document.createElement('span');
      zSpan.className = 'pt-z';
      zSpan.textContent = String(c.z);
      const symSpan = document.createElement('span');
      symSpan.className = 'pt-sym';
      symSpan.textContent = c.sym;
      const qSpan = document.createElement('span');
      qSpan.className = 'pt-q';
      qSpan.textContent = chargeRangeHint(c.z);
      btn.append(zSpan, symSpan, qSpan);
      btn.addEventListener('click', () => {
        maybeToggleSymbol(c.sym);
      });
      grid.append(btn);
    }
  }

  let pickerOpen = false;
  editBtn.addEventListener('click', () => {
    pickerOpen = !pickerOpen;
    picker.classList.toggle('hidden', !pickerOpen);
    toggleWideDialog(pickerOpen);
    applyAriaExpanded(pickerOpen);
    refreshAll();
  });

  settingsDialogEl?.addEventListener('close', () => {
    pickerOpen = false;
    picker.classList.add('hidden');
    toggleWideDialog(false);
    applyAriaExpanded(false);
  });

  renderChips(chips);
  refreshAll();
}
