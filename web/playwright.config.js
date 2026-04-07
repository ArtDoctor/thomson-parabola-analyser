const fs = require('fs');
const path = require('path');
const { defineConfig } = require('@playwright/test');

const repoRoot = path.resolve(__dirname, '..');
const chromiumExecutableCandidates = [
  process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE,
  path.join(process.env.HOME || '', '.cache', 'ms-playwright', 'chromium-1208', 'chrome-linux64', 'chrome'),
  path.join(process.env.HOME || '', '.cache', 'ms-playwright', 'chromium-1169', 'chrome-linux', 'chrome'),
  '/usr/bin/chromium-browser',
];
const chromiumExecutable = chromiumExecutableCandidates.find((candidate) => {
  return candidate && fs.existsSync(candidate);
}) || '/usr/bin/chromium-browser';

module.exports = defineConfig({
  testDir: path.join(__dirname, 'tests', 'e2e'),
  fullyParallel: false,
  workers: 1,
  timeout: 6 * 60 * 1000,
  expect: {
    timeout: 20 * 1000,
  },
  reporter: [
    ['list'],
    ['html', { open: 'never' }],
  ],
  use: {
    baseURL: 'http://127.0.0.1:32212',
    headless: true,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
    launchOptions: {
      executablePath: chromiumExecutable,
      args: ['--no-sandbox'],
    },
  },
  webServer: {
    cwd: repoRoot,
    command: `bash -lc "mkdir -p /tmp/oblisk-playwright-data && cd web/frontend && npm run build && cd ../.. && DATA_DIR=/tmp/oblisk-playwright-data ./venv/bin/python -m uvicorn web.backend.main:app --host 127.0.0.1 --port 32212"`,
    url: 'http://127.0.0.1:32212',
    reuseExistingServer: !process.env.CI,
    timeout: 5 * 60 * 1000,
  },
});
