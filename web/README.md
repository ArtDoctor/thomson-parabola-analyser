# Oblisk Web Application

Upload a Thomson parabola detector image → automated species classification + energy spectra plots.

---

## Architecture

Single container: **FastAPI (uvicorn)** serves both the REST API and the compiled Vite frontend.

```
web/
├── backend/
│   └── main.py             FastAPI app  (POST /upload, GET /status/{id})
├── frontend/
│   ├── index.html          Upload page  (/)
│   ├── results/index.html  Results page (/results/<job-id>)
│   ├── style.css           Dark glassmorphic design
│   ├── upload.ts           Upload page logic
│   ├── results.ts          Polling / plot rendering logic
│   ├── vite.config.ts
│   ├── tsconfig.json
│   └── package.json
├── Dockerfile              Multi-stage: Node builds frontend → Python serves it
├── docker-compose.yml      Single service, port 32212
└── README.md
```

---

## Plots produced (in order)

| # | Filename | Description |
|---|----------|-------------|
| 1 | `02_morphological` | Original vs denoised image |
| 2 | `05_smoothed_lines` | Smoothed parabola tracks |
| 3 | `07_rotated_nobg` | Rotated, no-background view |
| 4 | `09_a_score_peaks` | Parabola intensity score vs curvature |
| 5 | `10_detected_parabolas` | Detected parabolas overlaid on image |
| 6 | `11_classified` | Classified ion species |
| 7 | `12_sampling_overlay` | Integration sampling regions |
| 8 | `14_log_spectra` | Log spectra (shared absolute) |
| 9 | `15_numbered_log_spectra` | Numbered log spectra |
| 10 | `16_linear_energy_logy` | Linear energy / log-Y spectra |

---

## Deploy with Coolify (recommended)

Point Coolify at the repository.  Set:

| Setting | Value |
|---------|-------|
| **Dockerfile path** | `web/Dockerfile` |
| **Build context** | `/` (repo root) |
| **Port** | `32212` |
| **Environment variable** | `DATA_DIR=/app/data` |
| **Persistent volume** | `/app/data` (uploads + results) |

Coolify handles TLS, routing, and hot-reloads on push.

---

## Run locally with Docker Compose

```bash
# from the oblisk project root
cd web
docker compose up --build
# → http://localhost:32212
```

---

## Local development (no Docker)

**Backend**
```bash
# from oblisk root
source venv/bin/activate
pip install fastapi "uvicorn[standard]" python-multipart
uvicorn web.backend.main:app --reload --port 32212
```

**Frontend** (separate terminal)
```bash
cd web/frontend
npm install
npm run dev        # Vite dev server on :5173, proxies API calls to :32212
```

Open **http://localhost:5173** for hot-reloading, or **http://localhost:32212** to use the production-built frontend.

---

## End-to-end tests

Playwright coverage lives at the `web/` level and drives the real FastAPI app:

```bash
cd web
npm install
npm --prefix frontend install
npm run e2e
```

The suite builds `web/frontend`, starts the backend on `127.0.0.1:32212`, uploads a real sample detector image, waits for processing to finish, verifies the key plots load with non-empty pixel content, and exercises the interactive detector view via hover and click. The assertions are intentionally coarse so fitting changes do not break the test unless rendering or interaction is substantially broken.
