// Shared plot / timing labels and small helpers for the results page.

export const PLOT_LABELS: Record<string, string> = {
  '01_cropped_standardized': '01 · Cropped & standardized',
  '02_morphological': '02 · Original vs denoised',
  '03_peaks_overlay': '03 · Row-wise peaks',
  '05_smoothed_lines': '05 · Smoothed parabola tracks',
  '06_rotated_with_bg': '06 · Rotated (with image)',
  '07_rotated_nobg': '07 · Rotated (no background)',
  '08_fit_rmse_vs_iter': '08 · Fit RMSE vs iteration',
  '09_a_score_peaks': '09 · Parabola score vs curvature a',
  '10_detected_parabolas': '10 · Detected parabolas',
  '11_classified': '11 · Classified ion species',
  '12_sampling_overlay': '12 · Sampling overlay',
  '15_numbered_log_spectra': '14 · Numbered log spectra',
  '16_linear_energy_logy': '15 · Linear energy / log-Y spectra',
};

export function plotKey(url: string): string {
  const m = url.match(/(\d+_[^/]+?)\.png$/);
  return m ? m[1] : url;
}

export function humanisedLabel(url: string): string {
  return PLOT_LABELS[plotKey(url)] ?? plotKey(url);
}

export const TIMING_ORDER: string[] = [
  'load_crop',
  'denoise',
  'peak_extraction',
  'build_lines',
  'merge_lines',
  'smooth_lines',
  'fit_origin',
  'score_parabolas',
  'detect_good_a',
  'get_xp_range',
  'classify',
  'build_spectra',
];

export const TIMING_LABELS: Record<string, string> = {
  load_crop: 'Load & crop',
  denoise: 'Denoise',
  peak_extraction: 'Peak extraction',
  build_lines: 'Build lines',
  merge_lines: 'Merge lines',
  smooth_lines: 'Smooth lines',
  fit_origin: 'Fit origin (global)',
  score_parabolas: 'Score parabolas',
  detect_good_a: 'Curvature peaks',
  get_xp_range: 'm/q span',
  classify: 'Classify species',
  build_spectra: 'Spectra integration',
};

export function formatSeconds(sec: number): string {
  if (!Number.isFinite(sec)) return '—';
  if (sec === 0) return '0 ms';
  if (sec < 0.01) return `${(sec * 1000).toFixed(1)} ms`;
  if (sec < 1) return `${(sec * 1000).toFixed(0)} ms`;
  return `${sec.toFixed(2)} s`;
}

export function isRecord(x: unknown): x is Record<string, unknown> {
  return x !== null && typeof x === 'object' && !Array.isArray(x);
}

export function logEntryByKind(
  log: unknown[] | undefined,
  kind: string,
): Record<string, unknown> | undefined {
  if (!log) return undefined;
  for (const e of log) {
    if (isRecord(e) && e.kind === kind) return e;
  }
  return undefined;
}
