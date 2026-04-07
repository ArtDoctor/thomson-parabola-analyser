const path = require('path');
const { test, expect } = require('@playwright/test');

const SAMPLE_IMAGE = path.resolve(
  __dirname,
  '..',
  '..',
  '..',
  'eval',
  '62545.tif',
);

const REQUIRED_PLOT_LABELS = [
  '01 · Cropped & standardized',
  '10 · Detected parabolas',
  '11 · Classified ion species',
  '12 · Sampling overlay',
  '14 · Numbered log spectra',
  '15 · Linear energy / log-Y spectra',
];

async function imageElementStats(locator) {
  return locator.evaluate((node) => {
    const img = /** @type {HTMLImageElement} */ (node);
    const width = img.naturalWidth;
    const height = img.naturalHeight;
    const canvas = document.createElement('canvas');
    const maxEdge = 256;
    const scale = Math.min(1, maxEdge / Math.max(width, height, 1));
    canvas.width = Math.max(1, Math.round(width * scale));
    canvas.height = Math.max(1, Math.round(height * scale));
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      return null;
    }
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    return {
      width,
      height,
      ...window.__obliskImageStats(imageData),
    };
  });
}

async function canvasElementStats(locator) {
  return locator.evaluate((node) => {
    const canvas = /** @type {HTMLCanvasElement} */ (node);
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      return null;
    }
    const sampleWidth = Math.max(1, Math.min(canvas.width, 512));
    const sampleHeight = Math.max(1, Math.min(canvas.height, 512));
    const tmp = document.createElement('canvas');
    tmp.width = sampleWidth;
    tmp.height = sampleHeight;
    const tmpCtx = tmp.getContext('2d');
    if (!tmpCtx) {
      return null;
    }
    tmpCtx.drawImage(canvas, 0, 0, sampleWidth, sampleHeight);
    const imageData = tmpCtx.getImageData(0, 0, sampleWidth, sampleHeight).data;
    return {
      width: canvas.width,
      height: canvas.height,
      ...window.__obliskImageStats(imageData),
    };
  });
}

async function findInteractiveHoverPoint(page, canvas) {
  const fromOverlay = await canvas.evaluate(async (node) => {
    const pointXY = (point) => {
      if (Array.isArray(point) && point.length >= 2) {
        return { x: Number(point[0]), y: Number(point[1]) };
      }
      if (point && Number.isFinite(point.x) && Number.isFinite(point.y)) {
        return { x: Number(point.x), y: Number(point.y) };
      }
      return null;
    };

    const canvasEl = /** @type {HTMLCanvasElement} */ (node);
    const rawJson = document.getElementById('raw-json')?.textContent ?? '';
    if (!rawJson) {
      return null;
    }

    /** @type {{ geometry?: { y0_fit?: number }, overlays?: { classified?: { curves?: Array<{ segments?: Array<Array<{ x?: number, y?: number } | number[]>> }> } } }} */
    const result = JSON.parse(rawJson);
    const curves = result.overlays?.classified?.curves ?? [];
    const longest = curves
      .flatMap((curve) => curve.segments ?? [])
      .filter((segment) => Array.isArray(segment) && segment.length >= 4)
      .sort((left, right) => right.length - left.length)[0];
    if (!longest) {
      return null;
    }

    const point = pointXY(longest[Math.floor(longest.length * 0.6)]);
    if (!point) {
      return null;
    }

    const jobId = window.location.pathname.split('/').filter(Boolean).at(-1);
    if (!jobId) {
      return null;
    }

    const img = await new Promise((resolve) => {
      const image = new Image();
      image.onload = () => resolve(image);
      image.onerror = () => resolve(null);
      image.src = `/data/results/${jobId}/plots/00_raw_cropped.png`;
    });
    if (!img) {
      return null;
    }

    const fullImgH = img.naturalHeight;
    const imgW = img.naturalWidth;
    const y0 = Number(result.geometry?.y0_fit ?? 0);
    const cropY = Number.isFinite(y0) ? Math.max(0, Math.floor(y0) - 50) : 0;
    const imgH = Math.max(1, fullImgH - cropY);
    const rect = canvasEl.getBoundingClientRect();
    const sx = point.x / imgW;
    const sy = 1 - ((point.y - cropY) / imgH);

    return {
      x: rect.left + sx * rect.width,
      y: rect.top + sy * rect.height,
    };
  });

  if (fromOverlay) {
    return fromOverlay;
  }

  return null;
}

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    window.__obliskImageStats = (imageData) => {
      let opaquePixels = 0;
      let colorfulPixels = 0;
      let minLuma = 255;
      let maxLuma = 0;
      let sum = 0;
      let sumSq = 0;

      for (let i = 0; i < imageData.length; i += 4) {
        const r = imageData[i];
        const g = imageData[i + 1];
        const b = imageData[i + 2];
        const a = imageData[i + 3];
        if (a === 0) {
          continue;
        }
        opaquePixels += 1;
        const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        minLuma = Math.min(minLuma, luma);
        maxLuma = Math.max(maxLuma, luma);
        sum += luma;
        sumSq += luma * luma;
        if (Math.max(r, g, b) - Math.min(r, g, b) >= 18) {
          colorfulPixels += 1;
        }
      }

      if (opaquePixels === 0) {
        return {
          opaquePixels: 0,
          colorfulPixels: 0,
          lumaStdDev: 0,
          lumaRange: 0,
        };
      }

      const mean = sum / opaquePixels;
      const variance = Math.max(0, sumSq / opaquePixels - mean * mean);
      return {
        opaquePixels,
        colorfulPixels,
        lumaStdDev: Math.sqrt(variance),
        lumaRange: maxLuma - minLuma,
      };
    };

    localStorage.setItem('oblisk_inner_margin_crop', 'true');
  });
});

test('upload flow renders plots and interactive viewer end-to-end', async ({ page }) => {
  const pageErrors = [];
  page.on('pageerror', (error) => {
    pageErrors.push(error.message);
  });

  await page.goto('/');
  await expect(page.locator('#btn-upload')).toBeDisabled();

  await page.locator('#file-input').setInputFiles(SAMPLE_IMAGE);
  await expect(page.locator('#selected-file')).toBeVisible();
  await expect(page.locator('#file-name')).toContainText('62545.tif');
  await expect(page.locator('#btn-upload')).toBeEnabled();

  await page.locator('#btn-upload').click();
  await page.waitForURL(/\/results\/[0-9a-f-]{36}$/i, { timeout: 20_000 });

  await expect(page.locator('#status-text')).toContainText('Done', {
    timeout: 4 * 60 * 1000,
  });
  await expect(page.locator('#error-banner')).toHaveClass(/hidden/);

  const plotCards = page.locator('.plot-card');
  await expect.poll(async () => plotCards.count(), {
    timeout: 30_000,
    message: 'expected diagnostic plots to appear',
  }).toBeGreaterThanOrEqual(10);

  const plotLabels = await page.locator('.plot-label').allTextContents();
  expect(plotLabels).toEqual(expect.arrayContaining(REQUIRED_PLOT_LABELS));

  const keyPlotLabels = ['10 · Detected parabolas', '11 · Classified ion species', '12 · Sampling overlay'];
  for (const label of keyPlotLabels) {
    const card = page.locator('.plot-card').filter({ hasText: label });
    const thumb = card.locator('img.plot-img');
    await expect(thumb).toBeVisible();
    await expect.poll(async () => {
      return thumb.evaluate((img) => img.complete && img.naturalWidth > 0 && img.naturalHeight > 0);
    }, {
      timeout: 15_000,
      message: `expected thumbnail for ${label} to load`,
    }).toBe(true);

    const stats = await imageElementStats(thumb);
    expect(stats).not.toBeNull();
    expect(stats.width).toBeGreaterThan(100);
    expect(stats.height).toBeGreaterThan(100);
    expect(stats.opaquePixels).toBeGreaterThan(5_000);
    expect(stats.lumaRange).toBeGreaterThan(15);
    expect(stats.lumaStdDev).toBeGreaterThan(8);
  }

  await expect(page.locator('#iv-section')).not.toHaveClass(/hidden/);
  const interactiveCanvas = page.locator('#iv-container canvas.iv-canvas');
  await interactiveCanvas.scrollIntoViewIfNeeded();
  await expect(interactiveCanvas).toBeVisible();
  const interactiveStats = await canvasElementStats(interactiveCanvas);
  expect(interactiveStats).not.toBeNull();
  expect(interactiveStats.width).toBeGreaterThan(400);
  expect(interactiveStats.height).toBeGreaterThan(250);
  expect(interactiveStats.opaquePixels).toBeGreaterThan(20_000);
  expect(interactiveStats.lumaRange).toBeGreaterThan(20);
  expect(interactiveStats.colorfulPixels).toBeGreaterThan(5);

  const hoverPoint = await findInteractiveHoverPoint(page, interactiveCanvas);
  expect(hoverPoint).not.toBeNull();
  await page.mouse.move(hoverPoint.x, hoverPoint.y, { steps: 4 });
  await page.waitForTimeout(150);
  await expect(page.locator('.iv-tooltip.iv-tooltip--visible')).toBeVisible();
  await expect(page.locator('.iv-tooltip.iv-tooltip--visible')).toContainText('m/q');
  await expect(page.locator('.iv-tooltip.iv-tooltip--visible')).toContainText('Click to view spectrum');

  await page.mouse.click(hoverPoint.x, hoverPoint.y);
  await expect(page.locator('.iv-spectrum-aside')).toHaveClass(/iv-spectrum-aside--open/);
  await expect(page.locator('.iv-spectrum-title')).not.toHaveText(/^$/);

  const spectrumCanvas = page.locator('.iv-spectrum-canvas');
  await expect(spectrumCanvas).toBeVisible();
  await expect.poll(async () => {
    const stats = await canvasElementStats(spectrumCanvas);
    return stats ? stats.opaquePixels : 0;
  }, {
    timeout: 20_000,
    message: 'expected the spectrum panel to render content',
  }).toBeGreaterThan(5_000);

  await expect.poll(async () => page.locator('.result-card').count(), {
    timeout: 10_000,
    message: 'expected classification cards to render',
  }).toBeGreaterThan(0);
  await expect(page.locator('#raw-json')).toContainText('"classified"');
  expect(pageErrors).toEqual([]);
});
