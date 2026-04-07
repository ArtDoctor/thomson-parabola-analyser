// upload.ts – handles file selection and upload on the index page

import * as UTIF from 'utif';
import type { TiffPage } from 'utif';

import {
  applyClassificationElements,
  getClassificationElements,
  initSpeciesSettings,
  resetClassificationToDefaults,
} from './species-settings';
import {
  applySpectrometerParams,
  DEFAULT_SPECTROMETER_PARAMS,
  getSpectrometerParams,
  initSpectrometerSettings,
} from './spectrometer-settings';
import { buildExportPayload, parseImportPayload } from './input-settings-preset';
import { initRecentAnalysesPanel, recordAnalysis } from './analysis-history';
import { initLandingPipelineSection } from './landing-pipeline';

const API_BASE = import.meta?.env?.VITE_API_URL ?? '';

const STORAGE_DENOISE_UNET = 'oblisk_use_denoise_unet';
const STORAGE_INNER_MARGIN_CROP = 'oblisk_inner_margin_crop';
const STORAGE_PRESET_NAME = 'oblisk_settings_preset_name';

const dropZone = document.getElementById('drop-zone') as HTMLDivElement;
const fileInput = document.getElementById('file-input') as HTMLInputElement;
const btnBrowse = document.getElementById('btn-browse') as HTMLButtonElement;
const btnClear = document.getElementById('btn-clear') as HTMLButtonElement;
const btnUpload = document.getElementById('btn-upload') as HTMLButtonElement;
const btnSpinner = document.getElementById('btn-spinner') as HTMLSpanElement;
const selectedFile = document.getElementById('selected-file') as HTMLDivElement;
const fileName = document.getElementById('file-name') as HTMLSpanElement;
const filePreview = document.getElementById('file-preview') as HTMLImageElement;
const errorMsg = document.getElementById('error-msg') as HTMLParagraphElement;
const btnSettings = document.getElementById('btn-settings') as HTMLButtonElement;
const settingsDialog = document.getElementById('settings-dialog') as HTMLDialogElement;
const btnSettingsClose = document.getElementById('btn-settings-close') as HTMLButtonElement;
const denoiseUnetRadio = document.getElementById('denoise-unet') as HTMLInputElement;
const denoiseMorphRadio = document.getElementById('denoise-morph') as HTMLInputElement;
const innerMarginCropCheckbox = document.getElementById(
  'inner-margin-crop',
) as HTMLInputElement;
const presetNameInput = document.getElementById(
  'settings-preset-name',
) as HTMLInputElement;
const btnSettingsExport = document.getElementById(
  'btn-settings-export',
) as HTMLButtonElement;
const btnSettingsImport = document.getElementById(
  'btn-settings-import',
) as HTMLButtonElement;
const settingsImportFile = document.getElementById(
  'settings-import-file',
) as HTMLInputElement;
const btnSettingsReset = document.getElementById(
  'btn-settings-reset',
) as HTMLButtonElement;

let chosenFile: File | null = null;
let previewObjectUrl: string | null = null;
let previewGeneration = 0;

const MAX_PREVIEW_PIXELS = 4_000_000;

function fileLooksLikeTiff(file: File): boolean {
  const ext = file.name.split('.').pop()?.toLowerCase() ?? '';
  if (ext === 'tif' || ext === 'tiff') {
    return true;
  }
  return file.type === 'image/tiff' || file.type === 'image/x-tiff';
}

function pickTiffPreviewPage(ifds: TiffPage[]): TiffPage {
  let vsns: TiffPage[] = ifds;
  const first = ifds[0];
  if (first?.subIFD && first.subIFD.length > 0) {
    vsns = vsns.concat(first.subIFD);
  }
  let maxArea = 0;
  let page: TiffPage = vsns[0];
  for (let i = 0; i < vsns.length; i++) {
    const img = vsns[i];
    const t258 = img.t258;
    if (t258 == null || t258.length < 3) {
      continue;
    }
    const w = img.t256?.[0] ?? 0;
    const h = img.t257?.[0] ?? 0;
    const ar = w * h;
    if (ar > maxArea) {
      maxArea = ar;
      page = img;
    }
  }
  return page;
}

async function createPreviewObjectUrl(file: File): Promise<string> {
  if (!fileLooksLikeTiff(file)) {
    return URL.createObjectURL(file);
  }
  const buffer = await file.arrayBuffer();
  const ifds = UTIF.decode(buffer);
  if (ifds.length === 0) {
    throw new Error('Invalid TIFF');
  }
  const page = pickTiffPreviewPage(ifds);
  UTIF.decodeImage(buffer, page, ifds);
  const rgba = UTIF.toRGBA8(page);
  const w = page.width;
  const h = page.height;
  if (w <= 0 || h <= 0) {
    throw new Error('Invalid TIFF dimensions');
  }
  const fullCanvas = document.createElement('canvas');
  fullCanvas.width = w;
  fullCanvas.height = h;
  const fullCtx = fullCanvas.getContext('2d');
  if (!fullCtx) {
    throw new Error('Cannot create canvas context');
  }
  const imageData = fullCtx.createImageData(w, h);
  imageData.data.set(rgba);
  fullCtx.putImageData(imageData, 0, 0);

  let outW = w;
  let outH = h;
  if (w * h > MAX_PREVIEW_PIXELS) {
    const scale = Math.sqrt(MAX_PREVIEW_PIXELS / (w * h));
    outW = Math.max(1, Math.round(w * scale));
    outH = Math.max(1, Math.round(h * scale));
  }
  const outCanvas = document.createElement('canvas');
  outCanvas.width = outW;
  outCanvas.height = outH;
  const outCtx = outCanvas.getContext('2d');
  if (!outCtx) {
    throw new Error('Cannot create canvas context');
  }
  outCtx.drawImage(fullCanvas, 0, 0, outW, outH);

  const blob: Blob | null = await new Promise((resolve) => {
    outCanvas.toBlob((b) => resolve(b), 'image/png');
  });
  if (!blob) {
    throw new Error('Preview encoding failed');
  }
  return URL.createObjectURL(blob);
}

function readUseDenoiseUnet(): boolean {
  try {
    const v = localStorage.getItem(STORAGE_DENOISE_UNET);
    if (v === null) {
      return true;
    }
    return v === '1' || v === 'true';
  } catch {
    return true;
  }
}

function writeUseDenoiseUnet(value: boolean): void {
  try {
    localStorage.setItem(STORAGE_DENOISE_UNET, value ? 'true' : 'false');
  } catch {
    /* ignore quota / private mode */
  }
}

function readInnerMarginCrop(): boolean {
  try {
    const v = localStorage.getItem(STORAGE_INNER_MARGIN_CROP);
    if (v === null) {
      return true;
    }
    return v === '1' || v === 'true';
  } catch {
    return true;
  }
}

function writeInnerMarginCrop(value: boolean): void {
  try {
    localStorage.setItem(STORAGE_INNER_MARGIN_CROP, value ? 'true' : 'false');
  } catch {
    /* ignore quota / private mode */
  }
}

let useDenoiseUnet = readUseDenoiseUnet();
let innerMarginCrop = readInnerMarginCrop();

function syncDenoiseRadios(): void {
  if (useDenoiseUnet) {
    denoiseUnetRadio.checked = true;
  } else {
    denoiseMorphRadio.checked = true;
  }
}

syncDenoiseRadios();

function syncInnerMarginCropCheckbox(): void {
  innerMarginCropCheckbox.checked = innerMarginCrop;
}

syncInnerMarginCropCheckbox();

function readStoredPresetName(): string {
  try {
    return localStorage.getItem(STORAGE_PRESET_NAME) ?? '';
  } catch {
    return '';
  }
}

function writeStoredPresetName(name: string): void {
  try {
    const trimmed = name.trim();
    if (trimmed) {
      localStorage.setItem(STORAGE_PRESET_NAME, trimmed);
    } else {
      localStorage.removeItem(STORAGE_PRESET_NAME);
    }
  } catch {
    /* ignore */
  }
}

initSpeciesSettings();
initSpectrometerSettings();

presetNameInput.value = readStoredPresetName();

presetNameInput.addEventListener('blur', () => {
  writeStoredPresetName(presetNameInput.value);
});

function downloadSettingsJson(filename: string, jsonText: string): void {
  const blob = new Blob([jsonText], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.rel = 'noopener';
  document.body.append(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

btnSettingsExport.addEventListener('click', () => {
  const name = presetNameInput.value.trim() || 'Untitled';
  const payload = buildExportPayload(
    name,
    useDenoiseUnet,
    innerMarginCrop,
    getClassificationElements(),
    getSpectrometerParams(),
  );
  const safe =
    name.replace(/[^a-zA-Z0-9._-]+/g, '-').replace(/^-+|-+$/g, '') ||
    'untitled';
  downloadSettingsJson(`oblisk-settings-${safe}.json`, JSON.stringify(payload, null, 2));
  writeStoredPresetName(name);
});

btnSettingsImport.addEventListener('click', () => {
  settingsImportFile.click();
});

settingsImportFile.addEventListener('change', () => {
  const file = settingsImportFile.files?.[0];
  settingsImportFile.value = '';
  if (!file) {
    return;
  }
  void (async (): Promise<void> => {
    try {
      const text = await file.text();
      let parsed: unknown;
      try {
        parsed = JSON.parse(text) as unknown;
      } catch {
        errorMsg.textContent = 'Settings file is not valid JSON.';
        errorMsg.classList.remove('hidden');
        return;
      }
      const result = parseImportPayload(parsed);
      if (!result.ok) {
        errorMsg.textContent = result.error;
        errorMsg.classList.remove('hidden');
        return;
      }
      const d = result.data;
      useDenoiseUnet = d.useDenoiseUnet;
      writeUseDenoiseUnet(useDenoiseUnet);
      innerMarginCrop = d.innerMarginCrop;
      writeInnerMarginCrop(innerMarginCrop);
      presetNameInput.value = d.name;
      writeStoredPresetName(d.name);
      if (!applyClassificationElements(d.classificationElements)) {
        errorMsg.textContent =
          'Imported file has no valid element symbols for species.';
        errorMsg.classList.remove('hidden');
        return;
      }
      applySpectrometerParams(d.spectrometer);
      syncDenoiseRadios();
      syncInnerMarginCropCheckbox();
      errorMsg.classList.add('hidden');
    } catch {
      errorMsg.textContent = 'Could not read settings file.';
      errorMsg.classList.remove('hidden');
    }
  })();
});

function revokePreview(): void {
  if (previewObjectUrl) {
    URL.revokeObjectURL(previewObjectUrl);
    previewObjectUrl = null;
  }
}

async function setFile(file: File): Promise<void> {
  previewGeneration += 1;
  const gen = previewGeneration;
  revokePreview();
  chosenFile = file;
  fileName.textContent = file.name;
  filePreview.alt = `Preview: ${file.name}`;
  filePreview.classList.add('hidden');
  selectedFile.classList.remove('hidden');
  dropZone.classList.add('hidden');
  btnUpload.disabled = false;
  errorMsg.classList.add('hidden');

  filePreview.onerror = () => {
    filePreview.classList.add('hidden');
  };
  filePreview.onload = () => {
    filePreview.classList.remove('hidden');
  };

  try {
    const url = await createPreviewObjectUrl(file);
    if (gen !== previewGeneration) {
      URL.revokeObjectURL(url);
      return;
    }
    previewObjectUrl = url;
    filePreview.src = url;
  } catch {
    if (gen !== previewGeneration) {
      return;
    }
    filePreview.removeAttribute('src');
    filePreview.classList.add('hidden');
  }
}

function clearFile(): void {
  previewGeneration += 1;
  revokePreview();
  chosenFile = null;
  fileInput.value = '';
  filePreview.removeAttribute('src');
  filePreview.alt = '';
  filePreview.classList.add('hidden');
  filePreview.onerror = null;
  filePreview.onload = null;
  selectedFile.classList.add('hidden');
  dropZone.classList.remove('hidden');
  btnUpload.disabled = true;
}

btnBrowse.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', () => {
  if (fileInput.files?.length) {
    void setFile(fileInput.files[0]);
  }
});

dropZone.addEventListener('click', (e) => {
  if ((e.target as HTMLElement).id !== 'btn-browse') {
    fileInput.click();
  }
});

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer?.files?.[0];
  if (file) {
    void setFile(file);
  }
});

btnClear.addEventListener('click', clearFile);

btnSettings.addEventListener('click', () => {
  syncDenoiseRadios();
  syncInnerMarginCropCheckbox();
  settingsDialog.showModal();
});

btnSettingsReset.addEventListener('click', () => {
  useDenoiseUnet = true;
  writeUseDenoiseUnet(true);
  innerMarginCrop = true;
  writeInnerMarginCrop(true);
  presetNameInput.value = '';
  writeStoredPresetName('');
  resetClassificationToDefaults();
  applySpectrometerParams({ ...DEFAULT_SPECTROMETER_PARAMS });
  syncDenoiseRadios();
  syncInnerMarginCropCheckbox();
});

btnSettingsClose.addEventListener('click', () => {
  settingsDialog.close();
});

settingsDialog.addEventListener('click', (e) => {
  if (e.target === settingsDialog) {
    settingsDialog.close();
  }
});

document.querySelectorAll('input[name="denoise-method"]').forEach((el) => {
  el.addEventListener('change', () => {
    const input = el as HTMLInputElement;
    if (!input.checked) {
      return;
    }
    useDenoiseUnet = input.value === 'unet';
    writeUseDenoiseUnet(useDenoiseUnet);
  });
});

innerMarginCropCheckbox.addEventListener('change', () => {
  innerMarginCrop = innerMarginCropCheckbox.checked;
  writeInnerMarginCrop(innerMarginCrop);
});

btnUpload.addEventListener('click', async () => {
  if (!chosenFile) {
    return;
  }

  btnUpload.disabled = true;
  btnSpinner.classList.remove('hidden');
  errorMsg.classList.add('hidden');

  const form = new FormData();
  form.append('file', chosenFile);
  form.append('use_denoise_unet', useDenoiseUnet ? 'true' : 'false');
  form.append('inner_margin_crop', innerMarginCrop ? 'true' : 'false');
  form.append('species_json', JSON.stringify(getClassificationElements()));
  form.append('spectrometer_json', JSON.stringify(getSpectrometerParams()));

  try {
    const resp = await fetch(`${API_BASE}/upload`, { method: 'POST', body: form });
    if (!resp.ok) {
      const body = (await resp.json()) as { detail?: string };
      const detail = body.detail ?? 'Upload failed';
      throw new Error(detail);
    }
    const { job_id } = (await resp.json()) as { job_id: string };
    recordAnalysis(job_id, chosenFile.name);
    window.location.href = `/results/${job_id}`;
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    errorMsg.textContent = msg;
    errorMsg.classList.remove('hidden');
    btnUpload.disabled = false;
  } finally {
    btnSpinner.classList.add('hidden');
  }
});

initRecentAnalysesPanel();
initLandingPipelineSection();
