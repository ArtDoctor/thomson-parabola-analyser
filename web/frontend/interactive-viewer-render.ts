import type { DrawnParabola, OverlayPoint, ResultData } from './interactive-viewer-types';
import { formatSpeciesLabelCanvas } from './species-label-format';
import { LABEL_FONT } from './interactive-viewer-helpers';

export interface ParabolaRenderParams {
  ctx: CanvasRenderingContext2D;
  cw: number;
  ch: number;
  img: HTMLImageElement | null;
  imgW: number;
  imgH: number;
  imgCropY: number;
  drawn: DrawnParabola[];
  hovered: DrawnParabola | null;
  selected: DrawnParabola | null;
  glowPhase: number;
  dpr: number;
  data: ResultData | null;
}

function strokeCurve(
  ctx: CanvasRenderingContext2D,
  pts: { x: number; y: number }[],
  sx: number, sy: number,
  imgCropY: number,
  imgW: number,
  imgH: number,
): void {
  if (pts.length < 2) return;
  const oy = imgCropY;
  const maxGap = Math.max(imgW, imgH) * 0.15;
  ctx.beginPath();
  ctx.moveTo(pts[0].x * sx, (pts[0].y - oy) * sy);
  for (let i = 1; i < pts.length; i++) {
    const dx = pts[i].x - pts[i - 1].x;
    const dy = pts[i].y - pts[i - 1].y;
    if (dx * dx + dy * dy > maxGap * maxGap) {
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(pts[i].x * sx, (pts[i].y - oy) * sy);
    } else {
      ctx.lineTo(pts[i].x * sx, (pts[i].y - oy) * sy);
    }
  }
  ctx.stroke();
}

function strokeSegments(
  ctx: CanvasRenderingContext2D,
  segments: OverlayPoint[][],
  sx: number, sy: number,
  imgCropY: number,
  imgW: number,
  imgH: number,
): void {
  for (const segment of segments) {
    strokeCurve(ctx, segment, sx, sy, imgCropY, imgW, imgH);
  }
}

function drawLabel(
  ctx: CanvasRenderingContext2D,
  dpr: number,
  text: string, x: number, y: number,
  color: string, highlight: boolean,
): void {
  ctx.save();
  ctx.font = LABEL_FONT.replace('11px', `${Math.round(11 * dpr)}px`);
  const displayText = formatSpeciesLabelCanvas(text);
  const m = ctx.measureText(displayText);
  const pw = 7 * dpr;
  const ph = 4 * dpr;
  const bw = m.width + pw * 2;
  const bh = (11 * dpr) + ph * 2;
  const bx = x - bw / 2;
  const by = y - bh - 4 * dpr;

  const r = 4 * dpr;
  ctx.globalAlpha = highlight ? 0.95 : 0.82;
  ctx.fillStyle = highlight ? color : 'rgba(5,6,7,0.88)';
  ctx.beginPath();
  ctx.roundRect(bx, by, bw, bh, r);
  ctx.fill();
  if (!highlight) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1 * dpr;
    ctx.globalAlpha = 0.6;
    ctx.stroke();
  }

  ctx.globalAlpha = 1;
  ctx.fillStyle = highlight ? '#050607' : color;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(displayText, x, by + bh / 2);

  ctx.restore();
}

export function renderInteractiveParabolaFrame(p: ParabolaRenderParams): void {
  const {
    ctx, cw, ch, img, imgW, imgH, imgCropY, drawn, hovered, selected,
    glowPhase, dpr, data,
  } = p;

  ctx.clearRect(0, 0, cw, ch);

  const scaleX = cw / imgW;
  const scaleY = ch / imgH;

  ctx.save();
  ctx.translate(0, ch);
  ctx.scale(1, -1);

  if (img) {
    ctx.drawImage(img, 0, imgCropY, imgW, imgH, 0, 0, cw, ch);
  } else {
    ctx.fillStyle = '#08090b';
    ctx.fillRect(0, 0, cw, ch);
  }

  for (const parab of drawn) {
    const isHovered = parab === hovered;
    const isSel = parab === selected;

    ctx.save();

    if (isHovered || isSel) {
      ctx.strokeStyle = parab.color;
      ctx.lineWidth = (isSel ? 5 : 4) * dpr;
      ctx.globalAlpha = 0.3 + 0.1 * Math.sin(glowPhase * 3);
      ctx.shadowColor = parab.color;
      ctx.shadowBlur = 16 * dpr;
      strokeSegments(ctx, parab.segments, scaleX, scaleY, imgCropY, imgW, imgH);
    }

    ctx.shadowColor = 'transparent';
    ctx.shadowBlur = 0;
    ctx.strokeStyle = parab.color;
    ctx.lineWidth = (isHovered || isSel ? 2.5 : 1.5) * dpr;
    ctx.globalAlpha = isHovered || isSel ? 1.0 : 0.7;
    ctx.setLineDash(isSel ? [] : [6 * dpr, 4 * dpr]);
    strokeSegments(ctx, parab.segments, scaleX, scaleY, imgCropY, imgW, imgH);
    ctx.setLineDash([]);

    ctx.restore();
  }

  ctx.restore();

  const screenY = (iy: number): number => ch - iy * scaleY;

  for (const parab of drawn) {
    if (parab.entry.label === '?' || parab.labelPos == null) continue;
    const lx = parab.labelPos.x * scaleX;
    const ly = screenY(parab.labelPos.y - imgCropY);
    drawLabel(ctx, dpr, parab.entry.label, lx, ly, parab.color, parab === hovered || parab === selected);
  }

  if (data) {
    const vx = data.geometry.common_vertex_x ?? data.geometry.x0_fit;
    const vy = data.geometry.common_vertex_y ?? data.geometry.y0_fit;
    const ox = vx * scaleX;
    const oy0 = screenY(vy - imgCropY);
    ctx.save();
    ctx.strokeStyle = '#2dd4bf';
    ctx.lineWidth = 1.5 * dpr;
    ctx.globalAlpha = 0.6;
    const sz = 6 * dpr;
    ctx.beginPath();
    ctx.moveTo(ox - sz, oy0 - sz); ctx.lineTo(ox + sz, oy0 + sz);
    ctx.moveTo(ox + sz, oy0 - sz); ctx.lineTo(ox - sz, oy0 + sz);
    ctx.stroke();
    ctx.restore();
  }

  ctx.save();
  ctx.fillStyle = '#9ca3af';
  ctx.font = `${Math.round(10 * dpr)}px "IBM Plex Sans", system-ui, sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'bottom';
  ctx.fillText('x (pixels)', cw / 2, ch - 4 * dpr);
  ctx.translate(16 * dpr, ch / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.font = `500 ${Math.round(11 * dpr)}px "IBM Plex Sans", system-ui, sans-serif`;
  ctx.fillText('y (pixels)', 0, 0);
  ctx.restore();
}
