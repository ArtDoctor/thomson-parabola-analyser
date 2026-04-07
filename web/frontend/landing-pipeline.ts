// landing-pipeline.ts – synthetic demo + interactive viewer on the home page

import { InteractiveViewer, type ResultData } from './interactive-viewer';

/** Landing demo only: swap C3/C4 display strings (curves stay tied by `entry_index`). */
function swapLandingDemoC3C4Labels(data: ResultData): void {
  const mapLabel = (label: string): string => {
    if (label === 'C3') return 'C4';
    if (label === 'C4') return 'C3';
    return label;
  };
  for (const e of data.classified) {
    e.label = mapLabel(e.label);
  }
  for (const s of data.spectra) {
    s.label = mapLabel(s.label);
  }
}

async function initLandingInteractiveDemo(): Promise<void> {
  const el = document.getElementById('landing-iv-container');
  if (!el) return;

  try {
    const res = await fetch('/synthetic/demo-landing-result.json');
    if (!res.ok) throw new Error(String(res.status));
    const data = (await res.json()) as ResultData;
    swapLandingDemoC3C4Labels(data);
    const viewer = new InteractiveViewer('landing-iv-container', {
      enableSpectrumInteraction: false,
    });
    await viewer.load(data, '/synthetic/demo-landing-detector.png');
    const bumpLayout = (): void => {
      viewer.relayout();
    };
    requestAnimationFrame(() => {
      bumpLayout();
      requestAnimationFrame(bumpLayout);
    });
  } catch {
    el.textContent = '';
    const p = document.createElement('p');
    p.className = 'landing-iv-fallback';
    p.textContent = 'Demo viewer could not load. After you run an analysis, the results page shows the full interactive detector view.';
    el.appendChild(p);
  }
}

export function initLandingPipelineSection(): void {
  void initLandingInteractiveDemo();
}
