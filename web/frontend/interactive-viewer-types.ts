// Types for the interactive parabola / spectrum viewer.

export interface Geometry {
  x0_fit: number;
  y0_fit: number;
  common_vertex_x?: number;
  common_vertex_y?: number;
  theta_fit: number;
  gamma_fit: number;
  delta_fit: number;
  k1_fit: number;
  k2_fit: number;
  xp_min?: number;
  xp_max?: number;
}

export interface Candidate {
  name: string;
  mq_target: number;
  rel_err: number;
}

export interface ClassifiedEntry {
  a: number;
  label: string;
  mq_meas: number;
  candidates: Candidate[];
}

export interface SpectrumEntry {
  label: string;
  energies_keV: number[];
  weights: number[];
}

export interface OverlayPoint {
  x: number;
  y: number;
}

export type RawOverlayPoint = OverlayPoint | [number, number];

export interface ClassifiedOverlayCurve {
  entry_index: number;
  segments: RawOverlayPoint[][];
  label_anchor: [number, number] | null;
}

export interface OverlayBundle {
  classified?: {
    curves: ClassifiedOverlayCurve[];
  };
}

export interface ResultData {
  classified: ClassifiedEntry[];
  geometry: Geometry;
  spectra: SpectrumEntry[];
  overlays?: OverlayBundle;
}

/** When `enableSpectrumInteraction` is false, clicking and the spectrum aside are disabled and the tooltip omits energy stats (landing embed). */
export interface InteractiveViewerOptions {
  enableSpectrumInteraction?: boolean;
}

export interface DrawnParabola {
  entry: ClassifiedEntry;
  segments: OverlayPoint[][];
  color: string;
  labelPos: OverlayPoint | null;
}
