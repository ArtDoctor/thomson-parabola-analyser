// interactive-viewer.ts – Canvas-based interactive parabola viewer
//
// Background: cropped detector image
// Overlay: fitted parabola curves with hover/click interaction

import { formatSpeciesLabelHtml } from './species-label-format';
import {
  distToPolyline,
  HIT_TOLERANCE_PX,
  normalizeOverlayPoint,
  ORIGIN_CONTEXT_ROWS_ABOVE_PX,
  PARABOLA_COLORS,
  spectrumStats,
} from './interactive-viewer-helpers';
import { renderInteractiveParabolaFrame } from './interactive-viewer-render';
import { drawSpectrumEnergyChart } from './interactive-viewer-spectrum-chart';
import type {
  DrawnParabola,
  InteractiveViewerOptions,
  OverlayPoint,
  ResultData,
} from './interactive-viewer-types';

export type { InteractiveViewerOptions, ResultData } from './interactive-viewer-types';

export class InteractiveViewer {
  private container: HTMLElement;
  private plotWrap: HTMLElement;
  private spectrumAside: HTMLElement;
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private tooltip: HTMLElement;
  private spectrumPanel: HTMLElement;
  private spectrumCanvas: HTMLCanvasElement;
  private spectrumCtx: CanvasRenderingContext2D;
  private spectrumTitle: HTMLElement;
  private spectrumClose: HTMLElement;
  private spectrumStats: HTMLElement;

  private img: HTMLImageElement | null = null;
  private imgW = 0;
  /** Visible height (source slice height in PNG rows). */
  private imgH = 0;
  /** First PNG row included in the slice (0 = top of file). */
  private imgCropY = 0;
  /** Full natural height (distortion / corner probes match pipeline). */
  private fullImgH = 0;
  private data: ResultData | null = null;
  private drawn: DrawnParabola[] = [];
  private hovered: DrawnParabola | null = null;
  private selected: DrawnParabola | null = null;
  private dpr = 1;
  private animFrame = 0;
  private glowPhase = 0;
  private panelOpen = false;
  private readonly enableSpectrumInteraction: boolean;

  constructor(containerId: string, options: InteractiveViewerOptions = {}) {
    this.enableSpectrumInteraction = options.enableSpectrumInteraction !== false;
    this.container = document.getElementById(containerId)!;

    const layout = document.createElement('div');
    layout.className = 'iv-layout';

    this.plotWrap = document.createElement('div');
    this.plotWrap.className = 'iv-plot-wrap';

    this.canvas = document.createElement('canvas');
    this.canvas.className = 'iv-canvas';
    this.plotWrap.appendChild(this.canvas);
    this.ctx = this.canvas.getContext('2d')!;

    this.tooltip = document.createElement('div');
    this.tooltip.className = 'iv-tooltip';
    this.plotWrap.appendChild(this.tooltip);

    this.spectrumAside = document.createElement('aside');
    this.spectrumAside.className = 'iv-spectrum-aside';
    this.spectrumAside.setAttribute('aria-label', 'Energy spectrum');

    this.spectrumPanel = document.createElement('div');
    this.spectrumPanel.className = 'iv-spectrum-panel';
    this.spectrumPanel.innerHTML = `
      <div class="iv-spectrum-header">
        <span class="iv-spectrum-title"></span>
        <button class="iv-spectrum-close">&times;</button>
      </div>
      <div class="iv-spectrum-stats"></div>
      <canvas class="iv-spectrum-canvas"></canvas>
    `;
    this.spectrumAside.appendChild(this.spectrumPanel);

    layout.appendChild(this.plotWrap);
    layout.appendChild(this.spectrumAside);
    this.container.appendChild(layout);

    if (!this.enableSpectrumInteraction) {
      this.spectrumAside.style.display = 'none';
    }

    this.spectrumAside.addEventListener('transitionend', (e: TransitionEvent) => {
      if (!this.panelOpen || !this.selected) return;
      if (e.target !== this.spectrumAside) return;
      if (e.propertyName !== 'max-width' && e.propertyName !== 'max-height') return;
      const sp = this.data?.spectra.find((s) => s.label === this.selected!.entry.label);
      if (sp && sp.energies_keV.length > 0) {
        drawSpectrumEnergyChart(
          this.spectrumCanvas,
          this.spectrumCtx,
          this.dpr,
          this.spectrumChartDisplayWidth(),
          sp,
          this.selected.color,
        );
      }
    });

    this.spectrumTitle = this.spectrumPanel.querySelector('.iv-spectrum-title')!;
    this.spectrumClose = this.spectrumPanel.querySelector('.iv-spectrum-close')!;
    this.spectrumStats = this.spectrumPanel.querySelector('.iv-spectrum-stats')!;
    this.spectrumCanvas = this.spectrumPanel.querySelector('.iv-spectrum-canvas')!;
    this.spectrumCtx = this.spectrumCanvas.getContext('2d')!;

    this.canvas.addEventListener('mousemove', (e) => this.onMouseMove(e));
    this.canvas.addEventListener('mouseleave', () => this.onMouseLeave());
    if (this.enableSpectrumInteraction) {
      this.canvas.addEventListener('click', (e) => this.onClick(e));
    }
    this.spectrumClose.addEventListener('click', () => this.closePanel());

    document.addEventListener('keydown', (e) => {
      if (this.enableSpectrumInteraction && e.key === 'Escape' && this.panelOpen) this.closePanel();
    });

    const ro = new ResizeObserver(() => {
      if (this.imgW > 0) {
        this.resize();
      }
    });
    ro.observe(this.container);
    const mainPage = this.container.closest('.results-page');
    if (mainPage) {
      ro.observe(mainPage);
    }
  }

  async load(resultData: ResultData, imageUrl: string): Promise<void> {
    this.data = resultData;

    await new Promise<void>((resolve) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        this.img = img;
        this.imgW = img.naturalWidth;
        this.fullImgH = img.naturalHeight;
        const g = this.data?.geometry;
        const y0 = g?.common_vertex_y ?? g?.y0_fit;
        if (y0 !== undefined && Number.isFinite(y0)) {
          this.imgCropY = Math.max(0, Math.floor(y0) - ORIGIN_CONTEXT_ROWS_ABOVE_PX);
          this.imgH = Math.max(1, this.fullImgH - this.imgCropY);
        } else {
          this.imgCropY = 0;
          this.imgH = this.fullImgH;
        }
        resolve();
      };
      img.onerror = () => {
        this.imgW = 800;
        this.fullImgH = 600;
        this.imgCropY = 0;
        this.imgH = 600;
        resolve();
      };
      img.src = imageUrl;
    });

    this.buildParabolas();
    this.resize();
    this.startAnimation();

    this.container.classList.add('iv-loaded');
  }

  relayout(): void {
    this.resize();
  }

  private buildParabolas(): void {
    if (!this.data) return;
    this.drawn = [];

    const overlayCurves = this.data.overlays?.classified?.curves ?? [];
    for (const overlayCurve of overlayCurves) {
      const entry = this.data.classified[overlayCurve.entry_index];
      if (!entry) continue;

      const segments = overlayCurve.segments
        .map((segment) => segment.map((point) => normalizeOverlayPoint(point)).filter((point): point is OverlayPoint => point !== null))
        .filter((segment) => segment.length >= 2);
      if (segments.length === 0) continue;

      this.drawn.push({
        entry,
        segments,
        color: PARABOLA_COLORS[overlayCurve.entry_index % PARABOLA_COLORS.length],
        labelPos: overlayCurve.label_anchor
          ? { x: overlayCurve.label_anchor[0], y: overlayCurve.label_anchor[1] }
          : null,
      });
    }
  }

  private mainColumnContentWidthPx(): number {
    const main = this.container.closest('.results-page');
    if (!main) {
      let w = Math.round(this.container.clientWidth);
      if (w < 2) {
        const card = this.container.closest('.landing-viz-card');
        if (card) {
          const cs = getComputedStyle(card);
          const pad = (parseFloat(cs.paddingLeft) || 0) + (parseFloat(cs.paddingRight) || 0);
          w = Math.round(card.clientWidth - pad);
        }
      }
      return Math.max(1, w);
    }
    const cs = getComputedStyle(main);
    const pl = parseFloat(cs.paddingLeft) || 0;
    const pr = parseFloat(cs.paddingRight) || 0;
    return Math.max(1, Math.round(main.clientWidth - pl - pr));
  }

  private resize(): void {
    this.dpr = window.devicePixelRatio || 1;
    const baseW = this.mainColumnContentWidthPx();
    this.plotWrap.style.width = `${baseW}px`;
    const aspect = this.imgH / Math.max(this.imgW, 1);
    const displayW = baseW;
    const displayH = Math.round(baseW * aspect);

    this.canvas.style.width = displayW + 'px';
    this.canvas.style.height = displayH + 'px';
    this.canvas.width = Math.round(displayW * this.dpr);
    this.canvas.height = Math.round(displayH * this.dpr);
  }

  private startAnimation(): void {
    const tick = () => {
      this.glowPhase += 0.02;
      this.render();
      this.animFrame = requestAnimationFrame(tick);
    };
    tick();
  }

  private canvasToImg(cx: number, cy: number): [number, number] {
    const rect = this.canvas.getBoundingClientRect();
    const sx = (cx - rect.left) / rect.width;
    const sy = (cy - rect.top) / rect.height;
    return [sx * this.imgW, (1 - sy) * this.imgH + this.imgCropY];
  }

  private render(): void {
    renderInteractiveParabolaFrame({
      ctx: this.ctx,
      cw: this.canvas.width,
      ch: this.canvas.height,
      img: this.img,
      imgW: this.imgW,
      imgH: this.imgH,
      imgCropY: this.imgCropY,
      drawn: this.drawn,
      hovered: this.hovered,
      selected: this.selected,
      glowPhase: this.glowPhase,
      dpr: this.dpr,
      data: this.data,
    });
  }

  private onMouseMove(e: MouseEvent): void {
    const [imgX, imgY] = this.canvasToImg(e.clientX, e.clientY);

    const rect = this.canvas.getBoundingClientRect();
    const canvasScale = this.imgW / rect.width;
    const tolerance = HIT_TOLERANCE_PX * canvasScale;

    let closest: DrawnParabola | null = null;
    let closestDist = Infinity;

    for (const p of this.drawn) {
      for (const segment of p.segments) {
        const d = distToPolyline(imgX, imgY, segment);
        if (d < tolerance && d < closestDist) {
          closestDist = d;
          closest = p;
        }
      }
    }

    const prevHovered = this.hovered;
    this.hovered = closest;
    if (prevHovered !== closest) {
      this.canvas.style.cursor =
        closest && this.enableSpectrumInteraction ? 'pointer' : 'crosshair';
    }

    if (closest) {
      this.showTooltip(closest, e.clientX, e.clientY);
    } else {
      this.tooltip.classList.remove('iv-tooltip--visible');
    }
  }

  private onMouseLeave(): void {
    this.hovered = null;
    this.tooltip.classList.remove('iv-tooltip--visible');
    this.canvas.style.cursor = 'crosshair';
  }

  private onClick(e: MouseEvent): void {
    if (!this.enableSpectrumInteraction) return;
    if (!this.hovered) {
      if (this.panelOpen) this.closePanel();
      return;
    }
    this.selected = this.hovered;
    this.openSpectrumPanel(this.selected);

    e.stopPropagation();
  }

  private showTooltip(p: DrawnParabola, cx: number, cy: number): void {
    const entry = p.entry;
    const sp = this.data?.spectra.find(s => s.label === entry.label);
    const stats =
      this.enableSpectrumInteraction && sp ? spectrumStats(sp) : null;

    const titleHtml =
      entry.label !== '?'
        ? formatSpeciesLabelHtml(entry.label)
        : 'Unidentified';
    let html = `<div class="iv-tt-title" style="color:${p.color}">${titleHtml}</div>`;
    html += `<div class="iv-tt-row"><span class="iv-tt-dim">a</span> ${entry.a.toExponential(3)}</div>`;
    html += `<div class="iv-tt-row"><span class="iv-tt-dim">m/q</span> ${entry.mq_meas.toFixed(3)}</div>`;

    if (entry.candidates.length > 0) {
      html += `<div class="iv-tt-sep"></div>`;
      for (const c of entry.candidates.slice(0, 3)) {
        const pct = (c.rel_err * 100).toFixed(1);
        html += `<div class="iv-tt-row"><span class="iv-tt-cand">${formatSpeciesLabelHtml(c.name)}</span><span class="iv-tt-err">${pct}%</span></div>`;
      }
    }

    if (stats) {
      html += `<div class="iv-tt-sep"></div>`;
      html += `<div class="iv-tt-row"><span class="iv-tt-dim">E<sub>min</sub></span> ${stats.minE.toFixed(1)} keV</div>`;
      html += `<div class="iv-tt-row"><span class="iv-tt-dim">E<sub>max</sub></span> ${stats.maxE.toFixed(1)} keV</div>`;
      html += `<div class="iv-tt-row"><span class="iv-tt-dim">E<sub>mean</sub></span> ${stats.meanE.toFixed(1)} keV</div>`;
      html += `<div class="iv-tt-row"><span class="iv-tt-dim">E<sub>peak</sub></span> ${stats.peakE.toFixed(1)} keV</div>`;
    }

    if (this.enableSpectrumInteraction) {
      html += `<div class="iv-tt-hint">Click to view spectrum</div>`;
    }

    this.tooltip.innerHTML = html;
    this.tooltip.classList.add('iv-tooltip--visible');

    const plotRect = this.plotWrap.getBoundingClientRect();
    let tx = cx - plotRect.left + 16;
    let ty = cy - plotRect.top - 10;

    const tw = this.tooltip.offsetWidth;
    const th = this.tooltip.offsetHeight;
    const pw = this.plotWrap.clientWidth;
    const ph = this.plotWrap.clientHeight;
    if (tx + tw > pw - 8) tx = cx - plotRect.left - tw - 16;
    if (ty + th > ph - 8) ty = ph - th - 8;
    if (ty < 8) ty = 8;

    this.tooltip.style.left = tx + 'px';
    this.tooltip.style.top = ty + 'px';
  }

  private openSpectrumPanel(p: DrawnParabola): void {
    const entry = p.entry;
    const sp = this.data?.spectra.find(s => s.label === entry.label);

    this.spectrumTitle.innerHTML =
      entry.label !== '?'
        ? `${formatSpeciesLabelHtml(entry.label)} Energy Spectrum`
        : 'Unidentified Parabola';
    (this.spectrumTitle as HTMLElement).style.color = p.color;

    const st = sp ? spectrumStats(sp) : null;
    if (st) {
      this.spectrumStats.innerHTML = `
        <div class="iv-sp-stat"><span class="iv-sp-stat-label">E<sub>min</sub></span><span class="iv-sp-stat-value">${st.minE.toFixed(1)} keV</span></div>
        <div class="iv-sp-stat"><span class="iv-sp-stat-label">E<sub>max</sub></span><span class="iv-sp-stat-value">${st.maxE.toFixed(1)} keV</span></div>
        <div class="iv-sp-stat"><span class="iv-sp-stat-label">E<sub>mean</sub></span><span class="iv-sp-stat-value">${st.meanE.toFixed(1)} keV</span></div>
        <div class="iv-sp-stat"><span class="iv-sp-stat-label">E<sub>peak</sub></span><span class="iv-sp-stat-value">${st.peakE.toFixed(1)} keV</span></div>
        <div class="iv-sp-stat"><span class="iv-sp-stat-label">m/q</span><span class="iv-sp-stat-value">${entry.mq_meas.toFixed(3)}</span></div>
        <div class="iv-sp-stat"><span class="iv-sp-stat-label">a</span><span class="iv-sp-stat-value">${entry.a.toExponential(3)}</span></div>
      `;
    } else {
      this.spectrumStats.innerHTML = `<div class="iv-sp-no-data">No spectrum data available</div>`;
    }

    this.panelOpen = true;
    this.spectrumAside.classList.add('iv-spectrum-aside--open');

    if (sp && sp.energies_keV.length > 0) {
      drawSpectrumEnergyChart(
        this.spectrumCanvas,
        this.spectrumCtx,
        this.dpr,
        this.spectrumChartDisplayWidth(),
        sp,
        p.color,
      );
    } else {
      const sc = this.spectrumCanvas;
      const w = Math.max(160, this.spectrumChartDisplayWidth());
      const h = 200;
      sc.style.width = w + 'px';
      sc.style.height = h + 'px';
      sc.width = Math.round(w * this.dpr);
      sc.height = Math.round(h * this.dpr);
      const sctx = this.spectrumCtx;
      sctx.fillStyle = '#0c0d10';
      sctx.fillRect(0, 0, sc.width, sc.height);
      sctx.fillStyle = '#6b7280';
      sctx.font = `${13 * this.dpr}px "IBM Plex Sans", system-ui, sans-serif`;
      sctx.textAlign = 'center';
      sctx.fillText('No spectrum data', sc.width / 2, sc.height / 2);
    }
  }

  private closePanel(): void {
    this.panelOpen = false;
    this.selected = null;
    this.spectrumAside.classList.remove('iv-spectrum-aside--open');
  }

  private spectrumChartDisplayWidth(): number {
    const inner = this.spectrumPanel.clientWidth;
    if (inner > 48) {
      return Math.min(inner - 32, 520);
    }
    const cw = this.container.clientWidth;
    const narrow = typeof window !== 'undefined' && window.matchMedia('(max-width: 640px)').matches;
    const estAside = narrow ? cw : Math.min(420, cw * 0.45);
    return Math.min(Math.max(estAside - 32, 160), 520);
  }

  destroy(): void {
    if (this.animFrame) cancelAnimationFrame(this.animFrame);
  }
}
