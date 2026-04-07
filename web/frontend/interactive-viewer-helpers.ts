import type { OverlayPoint, RawOverlayPoint, SpectrumEntry } from './interactive-viewer-types';

export const HIT_TOLERANCE_PX = 12;
/**
 * PNG row index where the displayed slice starts (y increases downward in the file).
 * With the canvas Y-flip, small PNG y maps to the bottom of the viewer; we drop that
 * band and keep rows from (roughly y0_fit − this) through the bottom so the fan
 * from the bright spot upward on screen stays in view.
 */
export const ORIGIN_CONTEXT_ROWS_ABOVE_PX = 50;
export const LABEL_FONT = '600 11px "IBM Plex Sans", system-ui, sans-serif';

// Color palette for parabolas (teal/cyan family matching site design)
export const PARABOLA_COLORS = [
  '#2dd4bf', '#5eead4', '#99f6e4', '#67e8f9', '#a5f3fc',
  '#93c5fd', '#c4b5fd', '#f0abfc', '#fda4af', '#fcd34d',
  '#86efac', '#6ee7b7',
];

export function distToPolyline(px: number, py: number, pts: { x: number; y: number }[]): number {
  let minD = Infinity;
  for (let i = 0; i < pts.length - 1; i++) {
    const ax = pts[i].x, ay = pts[i].y;
    const bx = pts[i + 1].x, by = pts[i + 1].y;
    const dx = bx - ax, dy = by - ay;
    const len2 = dx * dx + dy * dy;
    let t = len2 > 0 ? ((px - ax) * dx + (py - ay) * dy) / len2 : 0;
    t = Math.max(0, Math.min(1, t));
    const cx = ax + t * dx, cy = ay + t * dy;
    const d = Math.hypot(px - cx, py - cy);
    if (d < minD) minD = d;
  }
  return minD;
}

export function spectrumStats(sp: SpectrumEntry): { minE: number; maxE: number; meanE: number; peakE: number } | null {
  const e = sp.energies_keV;
  const w = sp.weights;
  if (!e?.length || !w?.length) return null;
  let minE = Infinity, maxE = -Infinity;
  let sumW = 0, sumWE = 0, peakW = -Infinity, peakE = 0;
  for (let i = 0; i < e.length; i++) {
    if (!isFinite(e[i]) || !isFinite(w[i]) || w[i] <= 0) continue;
    if (e[i] < minE) minE = e[i];
    if (e[i] > maxE) maxE = e[i];
    sumW += w[i];
    sumWE += w[i] * e[i];
    if (w[i] > peakW) { peakW = w[i]; peakE = e[i]; }
  }
  if (sumW <= 0) return null;
  return { minE, maxE, meanE: sumWE / sumW, peakE };
}

export function normalizeOverlayPoint(point: RawOverlayPoint): OverlayPoint | null {
  if (Array.isArray(point)) {
    if (point.length < 2) return null;
    const [x, y] = point;
    if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
    return { x, y };
  }
  if (!point || !Number.isFinite(point.x) || !Number.isFinite(point.y)) {
    return null;
  }
  return { x: point.x, y: point.y };
}
