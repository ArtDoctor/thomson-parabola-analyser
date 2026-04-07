// spectrometer-settings.ts – interactive spectrometer geometry dialog

const STORAGE_KEY = 'oblisk_spectrometer_params';

export interface SpectrometerParams {
  /** Electric field strength (kV/m) */
  E_kVm: number;
  /** Electric plate length (cm) */
  LiE_cm: number;
  /** Electric drift to detector (cm) */
  LfE_cm: number;
  /** Magnetic field strength (mT) */
  B_mT: number;
  /** Magnetic region length (cm) */
  LiB_cm: number;
  /** Magnetic drift to detector (cm) */
  LfB_cm: number;
  /** Detector size (cm) */
  detector_size_cm: number;
}

const DEFAULTS: SpectrometerParams = {
  E_kVm: 130,
  LiE_cm: 8,
  LfE_cm: 20.5,
  B_mT: 153,
  LiB_cm: 8,
  LfB_cm: 29.5,
  detector_size_cm: 6,
};

/** Exported for preset import validation defaults. */
export const DEFAULT_SPECTROMETER_PARAMS: SpectrometerParams = { ...DEFAULTS };

function readParams(): SpectrometerParams {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...DEFAULTS };
    const parsed = JSON.parse(raw) as Partial<SpectrometerParams>;
    return { ...DEFAULTS, ...parsed };
  } catch {
    return { ...DEFAULTS };
  }
}

function writeParams(params: SpectrometerParams): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(params));
  } catch { /* ignore */ }
}

let spectrometerInputsBound = false;

let currentParams = readParams();

export function getSpectrometerParams(): SpectrometerParams {
  return { ...currentParams };
}

export function applySpectrometerParams(
  params: Partial<SpectrometerParams>,
): void {
  currentParams = {
    E_kVm: asFiniteParam(params.E_kVm, DEFAULTS.E_kVm),
    LiE_cm: asFiniteParam(params.LiE_cm, DEFAULTS.LiE_cm),
    LfE_cm: asFiniteParam(params.LfE_cm, DEFAULTS.LfE_cm),
    B_mT: asFiniteParam(params.B_mT, DEFAULTS.B_mT),
    LiB_cm: asFiniteParam(params.LiB_cm, DEFAULTS.LiB_cm),
    LfB_cm: asFiniteParam(params.LfB_cm, DEFAULTS.LfB_cm),
    detector_size_cm: asFiniteParam(
      params.detector_size_cm,
      DEFAULTS.detector_size_cm,
    ),
  };
  writeParams(currentParams);
  if (spectrometerInputsBound) {
    syncInputsFromParams();
  }
}

function asFiniteParam(value: number | undefined, fallback: number): number {
  if (value !== undefined && isFinite(value) && value > 0) {
    return value;
  }
  return fallback;
}

// Input element references
let inputE: HTMLInputElement;
let inputLiE: HTMLInputElement;
let inputLfE: HTMLInputElement;
let inputB: HTMLInputElement;
let inputLiB: HTMLInputElement;
let inputLfB: HTMLInputElement;
let inputDetSize: HTMLInputElement;

function syncInputsFromParams(): void {
  inputE.value = String(currentParams.E_kVm);
  inputLiE.value = String(currentParams.LiE_cm);
  inputLfE.value = String(currentParams.LfE_cm);
  inputB.value = String(currentParams.B_mT);
  inputLiB.value = String(currentParams.LiB_cm);
  inputLfB.value = String(currentParams.LfB_cm);
  inputDetSize.value = String(currentParams.detector_size_cm);
}

function readInputsToParams(): void {
  const parse = (el: HTMLInputElement, fallback: number) => {
    const v = parseFloat(el.value);
    return isFinite(v) && v > 0 ? v : fallback;
  };
  currentParams = {
    E_kVm: parse(inputE, DEFAULTS.E_kVm),
    LiE_cm: parse(inputLiE, DEFAULTS.LiE_cm),
    LfE_cm: parse(inputLfE, DEFAULTS.LfE_cm),
    B_mT: parse(inputB, DEFAULTS.B_mT),
    LiB_cm: parse(inputLiB, DEFAULTS.LiB_cm),
    LfB_cm: parse(inputLfB, DEFAULTS.LfB_cm),
    detector_size_cm: parse(inputDetSize, DEFAULTS.detector_size_cm),
  };
  writeParams(currentParams);
}

function setupSchematicInteractivity(): void {
  const svg = document.querySelector('.spec-schematic') as SVGElement | null;
  if (!svg) return;

  const paramGroups = document.querySelectorAll<HTMLElement>('.spec-param-group');

  // Map data-part names to their SVG groups and field line groups
  const parts = svg.querySelectorAll<SVGGElement>('.spec-part');
  const fieldLinesE = svg.querySelector('.spec-field-lines-e') as SVGGElement | null;
  const fieldLinesB = svg.querySelector('.spec-field-lines-b') as SVGGElement | null;

  // Hover on SVG parts highlights corresponding param group
  parts.forEach((part) => {
    const partName = part.dataset.part;

    part.addEventListener('mouseenter', () => {
      part.classList.add('spec-part--hover');

      // Show field lines
      if (partName === 'electric' && fieldLinesE) {
        fieldLinesE.style.transition = 'opacity 0.4s ease';
        fieldLinesE.style.opacity = '1';
      }
      if (partName === 'magnetic' && fieldLinesB) {
        fieldLinesB.style.transition = 'opacity 0.4s ease';
        fieldLinesB.style.opacity = '1';
      }

      // Highlight corresponding param group
      paramGroups.forEach((g) => {
        if (g.dataset.highlight === partName) {
          g.classList.add('spec-param-group--active');
        }
      });
    });

    part.addEventListener('mouseleave', () => {
      part.classList.remove('spec-part--hover');

      if (partName === 'electric' && fieldLinesE) {
        fieldLinesE.style.opacity = '0';
      }
      if (partName === 'magnetic' && fieldLinesB) {
        fieldLinesB.style.opacity = '0';
      }

      paramGroups.forEach((g) => {
        g.classList.remove('spec-param-group--active');
      });
    });
  });

  // Hover on param groups highlights SVG parts
  paramGroups.forEach((group) => {
    const highlightName = group.dataset.highlight;

    group.addEventListener('mouseenter', () => {
      group.classList.add('spec-param-group--active');
      parts.forEach((p) => {
        if (p.dataset.part === highlightName) {
          p.classList.add('spec-part--hover');
        }
      });
      if (highlightName === 'electric' && fieldLinesE) {
        fieldLinesE.style.transition = 'opacity 0.4s ease';
        fieldLinesE.style.opacity = '1';
      }
      if (highlightName === 'magnetic' && fieldLinesB) {
        fieldLinesB.style.transition = 'opacity 0.4s ease';
        fieldLinesB.style.opacity = '1';
      }
    });

    group.addEventListener('mouseleave', () => {
      group.classList.remove('spec-param-group--active');
      parts.forEach((p) => {
        p.classList.remove('spec-part--hover');
      });
      if (highlightName === 'electric' && fieldLinesE) {
        fieldLinesE.style.opacity = '0';
      }
      if (highlightName === 'magnetic' && fieldLinesB) {
        fieldLinesB.style.opacity = '0';
      }
    });
  });
}

function setupInfoTooltips(): void {
  const rows = document.querySelectorAll<HTMLElement>('.spec-param-row');
  let activeTooltip: HTMLElement | null = null;
  let hideTimer: number | null = null;

  const removeTooltip = () => {
    if (hideTimer != null) {
      window.clearTimeout(hideTimer);
      hideTimer = null;
    }
    if (activeTooltip) {
      activeTooltip.remove();
      activeTooltip = null;
    }
  };

  const showForRow = (row: HTMLElement): void => {
    const icon = row.querySelector<HTMLElement>('.spec-info');
    const text = row.dataset.tooltip ?? icon?.dataset.tooltip;
    if (!text) return;

    removeTooltip();
    const tip = document.createElement('div');
    tip.className = 'spec-tooltip';
    tip.textContent = text;
    const host = row.closest('dialog') ?? document.body;
    host.appendChild(tip);
    activeTooltip = tip;

    const rect = row.getBoundingClientRect();
    tip.style.left = `${rect.left + rect.width / 2}px`;
    tip.style.top = `${rect.bottom + 8}px`;
    tip.style.transform = 'translateX(-50%) translateY(4px)';

    requestAnimationFrame(() => {
      const tipRect = tip.getBoundingClientRect();
      let left = rect.left + rect.width / 2 - tipRect.width / 2;
      if (left + tipRect.width > window.innerWidth - 12) {
        left = window.innerWidth - tipRect.width - 12;
      }
      if (left < 12) {
        left = 12;
      }
      tip.style.left = `${left}px`;
      tip.style.transform = 'translateY(0)';
      tip.classList.add('spec-tooltip--visible');
    });
  };

  rows.forEach((row) => {
    row.addEventListener('mouseenter', () => {
      if (hideTimer != null) {
        window.clearTimeout(hideTimer);
        hideTimer = null;
      }
      showForRow(row);
    });
    row.addEventListener('mouseleave', () => {
      hideTimer = window.setTimeout(removeTooltip, 120);
    });
  });
}

export function initSpectrometerSettings(): void {
  const dialog = document.getElementById('spec-dialog') as HTMLDialogElement | null;
  const btnOpen = document.getElementById('btn-spectrometer') as HTMLButtonElement | null;
  const btnClose = document.getElementById('btn-spec-close') as HTMLButtonElement | null;
  const btnReset = document.getElementById('btn-spec-reset') as HTMLButtonElement | null;

  if (!dialog || !btnOpen) return;

  inputE = document.getElementById('spec-E') as HTMLInputElement;
  inputLiE = document.getElementById('spec-LiE') as HTMLInputElement;
  inputLfE = document.getElementById('spec-LfE') as HTMLInputElement;
  inputB = document.getElementById('spec-B') as HTMLInputElement;
  inputLiB = document.getElementById('spec-LiB') as HTMLInputElement;
  inputLfB = document.getElementById('spec-LfB') as HTMLInputElement;
  inputDetSize = document.getElementById('spec-det-size') as HTMLInputElement;

  syncInputsFromParams();

  // Open
  btnOpen.addEventListener('click', () => {
    syncInputsFromParams();
    dialog.showModal();
  });

  // Close
  btnClose?.addEventListener('click', () => {
    readInputsToParams();
    dialog.close();
  });

  // Backdrop click
  dialog.addEventListener('click', (e) => {
    if (e.target === dialog) {
      readInputsToParams();
      dialog.close();
    }
  });

  // Live-save on input change
  const allInputs = [inputE, inputLiE, inputLfE, inputB, inputLiB, inputLfB, inputDetSize];
  allInputs.forEach((inp) => {
    inp.addEventListener('change', readInputsToParams);
  });

  // Reset
  btnReset?.addEventListener('click', () => {
    currentParams = { ...DEFAULTS };
    writeParams(currentParams);
    syncInputsFromParams();
  });

  setupSchematicInteractivity();
  setupInfoTooltips();

  spectrometerInputsBound = true;
}
