import type { SpectrumEntry } from './interactive-viewer-types';

export function drawSpectrumEnergyChart(
  sc: HTMLCanvasElement,
  ctx: CanvasRenderingContext2D,
  dpr: number,
  displayW: number,
  sp: SpectrumEntry,
  color: string,
): void {
  const displayH = 220;
  sc.style.width = displayW + 'px';
  sc.style.height = displayH + 'px';
  sc.width = Math.round(displayW * dpr);
  sc.height = Math.round(displayH * dpr);

  const w = sc.width, h = sc.height;
  const pad = { left: 55 * dpr, right: 16 * dpr, top: 16 * dpr, bottom: 40 * dpr };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;

  ctx.fillStyle = '#0c0d10';
  ctx.fillRect(0, 0, w, h);

  const energies = sp.energies_keV;
  const weights = sp.weights;
  const numBins = 80;

  const finiteE: number[] = [];
  for (let i = 0; i < energies.length; i++) {
    if (isFinite(energies[i]) && isFinite(weights[i]) && weights[i] > 0 && energies[i] > 0) {
      finiteE.push(energies[i]);
    }
  }
  if (finiteE.length < 2) return;

  finiteE.sort((a, b) => a - b);
  const eMin = 0;
  const eMax = finiteE[Math.floor(finiteE.length * 0.995)];
  const binW = (eMax - eMin) / numBins;

  const bins = new Float64Array(numBins);
  for (let i = 0; i < energies.length; i++) {
    if (!isFinite(energies[i]) || !isFinite(weights[i]) || weights[i] <= 0) continue;
    const idx = Math.floor((energies[i] - eMin) / binW);
    if (idx >= 0 && idx < numBins) bins[idx] += weights[i];
  }

  let maxBin = 0;
  for (let i = 0; i < numBins; i++) if (bins[i] > maxBin) maxBin = bins[i];
  if (maxBin <= 0) return;

  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const gy = pad.top + plotH * (1 - i / 4);
    ctx.beginPath();
    ctx.moveTo(pad.left, gy);
    ctx.lineTo(pad.left + plotW, gy);
    ctx.stroke();
  }

  const gradient = ctx.createLinearGradient(0, pad.top, 0, pad.top + plotH);
  gradient.addColorStop(0, color);
  gradient.addColorStop(1, color + '33');

  const barGap = 1 * dpr;
  const barW = Math.max(1, plotW / numBins - barGap);

  for (let i = 0; i < numBins; i++) {
    const val = bins[i] / maxBin;
    const bh = val * plotH;
    const bx = pad.left + (i / numBins) * plotW;
    const by = pad.top + plotH - bh;

    ctx.fillStyle = gradient;
    ctx.globalAlpha = 0.85;
    ctx.fillRect(bx, by, barW, bh);
  }
  ctx.globalAlpha = 1;

  ctx.strokeStyle = 'rgba(255,255,255,0.15)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, pad.top + plotH);
  ctx.lineTo(pad.left + plotW, pad.top + plotH);
  ctx.stroke();

  ctx.fillStyle = '#9ca3af';
  ctx.font = `${Math.round(10 * dpr)}px "IBM Plex Sans", system-ui, sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  const nxTicks = 5;
  for (let i = 0; i <= nxTicks; i++) {
    const eTick = eMin + (eMax - eMin) * i / nxTicks;
    const tx = pad.left + plotW * i / nxTicks;
    ctx.fillText(eTick.toFixed(0), tx, pad.top + plotH + 6 * dpr);
  }
  ctx.font = `500 ${Math.round(11 * dpr)}px "IBM Plex Sans", system-ui, sans-serif`;
  ctx.fillText('Energy (keV)', pad.left + plotW / 2, pad.top + plotH + 22 * dpr);

  ctx.save();
  ctx.translate(14 * dpr, pad.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.font = `500 ${Math.round(11 * dpr)}px "IBM Plex Sans", system-ui, sans-serif`;
  ctx.fillStyle = '#9ca3af';
  ctx.fillText('Signal (a.u.)', 0, 0);
  ctx.restore();
}
