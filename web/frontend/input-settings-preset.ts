// Versioned bundle for export/import of all web input settings.

import type { SpectrometerParams } from './spectrometer-settings';
import { DEFAULT_SPECTROMETER_PARAMS } from './spectrometer-settings';

export const OBLISK_INPUT_SETTINGS_KIND = 1 as const;
export const OBLISK_INPUT_SETTINGS_VERSION = 1 as const;

export interface ObliskInputSettingsV1 {
  obliskInputSettings: typeof OBLISK_INPUT_SETTINGS_KIND;
  version: typeof OBLISK_INPUT_SETTINGS_VERSION;
  name: string;
  useDenoiseUnet: boolean;
  innerMarginCrop: boolean;
  classificationElements: string[];
  spectrometer: SpectrometerParams;
}

function asFiniteNumber(value: unknown, fallback: number): number {
  if (typeof value === 'number' && isFinite(value) && value > 0) {
    return value;
  }
  if (typeof value === 'string' && value.trim() !== '') {
    const n = parseFloat(value);
    if (isFinite(n) && n > 0) {
      return n;
    }
  }
  return fallback;
}

function asBool(value: unknown, defaultValue: boolean): boolean {
  if (typeof value === 'boolean') {
    return value;
  }
  if (value === 'true' || value === 1 || value === '1') {
    return true;
  }
  if (value === 'false' || value === 0 || value === '0') {
    return false;
  }
  return defaultValue;
}

function parseSpectrometer(raw: unknown): SpectrometerParams {
  const base = { ...DEFAULT_SPECTROMETER_PARAMS };
  if (raw === null || typeof raw !== 'object') {
    return base;
  }
  const o = raw as Record<string, unknown>;
  return {
    E_kVm: asFiniteNumber(o.E_kVm, base.E_kVm),
    LiE_cm: asFiniteNumber(o.LiE_cm, base.LiE_cm),
    LfE_cm: asFiniteNumber(o.LfE_cm, base.LfE_cm),
    B_mT: asFiniteNumber(o.B_mT, base.B_mT),
    LiB_cm: asFiniteNumber(o.LiB_cm, base.LiB_cm),
    LfB_cm: asFiniteNumber(o.LfB_cm, base.LfB_cm),
    detector_size_cm: asFiniteNumber(o.detector_size_cm, base.detector_size_cm),
  };
}

export function buildExportPayload(
  name: string,
  useDenoiseUnet: boolean,
  innerMarginCrop: boolean,
  classificationElements: string[],
  spectrometer: SpectrometerParams,
): ObliskInputSettingsV1 {
  return {
    obliskInputSettings: OBLISK_INPUT_SETTINGS_KIND,
    version: OBLISK_INPUT_SETTINGS_VERSION,
    name: name.trim() || 'Untitled',
    useDenoiseUnet,
    innerMarginCrop,
    classificationElements: [...classificationElements],
    spectrometer: { ...spectrometer },
  };
}

export function parseImportPayload(
  raw: unknown,
):
  | { ok: true; data: ObliskInputSettingsV1 }
  | { ok: false; error: string } {
  if (raw === null || typeof raw !== 'object') {
    return { ok: false, error: 'File must contain a JSON object.' };
  }
  const o = raw as Record<string, unknown>;
  if (o.obliskInputSettings !== OBLISK_INPUT_SETTINGS_KIND) {
    return {
      ok: false,
      error:
        'Not an Oblisk input settings file (expected obliskInputSettings: 1).',
    };
  }
  if (o.version !== OBLISK_INPUT_SETTINGS_VERSION) {
    return {
      ok: false,
      error: `Unsupported settings version: ${String(o.version)}.`,
    };
  }

  const name =
    typeof o.name === 'string' ? o.name.trim() || 'Untitled' : 'Untitled';

  let classificationElements: string[] = [];
  if (Array.isArray(o.classificationElements)) {
    classificationElements = o.classificationElements.map((x) => String(x));
  }

  const spectrometer = parseSpectrometer(o.spectrometer);

  const data: ObliskInputSettingsV1 = {
    obliskInputSettings: OBLISK_INPUT_SETTINGS_KIND,
    version: OBLISK_INPUT_SETTINGS_VERSION,
    name,
    useDenoiseUnet: asBool(o.useDenoiseUnet, true),
    innerMarginCrop: asBool(o.innerMarginCrop, true),
    classificationElements,
    spectrometer,
  };

  return { ok: true, data };
}
