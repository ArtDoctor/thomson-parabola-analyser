# Web Application

The Oblisk web application provides a user-friendly GUI to interact with the Thomson parabola analysis pipeline. It allows users to upload raw spectrometer images, automatically run the full analysis, and view the intermediate steps and final physical spectra.

## Architecture

The application uses a unified Docker container holding both the frontend and backend.

- **Backend:** FastAPI (Python), located in `web/backend/main.py`. It exposes REST endpoints (`POST /upload`, `GET /status/{id}`).
- **Frontend:** Vite, TypeScript, and HTML/CSS (Vanilla CSS with a dark glassmorphic design), located in `web/frontend/`.

## Workflow

1. **Upload:** The user uploads a `.tif`, `.png`, or `.jpg` file via the web interface.
2. **Processing (Backend):** 
   - The image is saved to a persistent `DATA_DIR`.
   - The backend spawns a background task running the core pipeline:
     1. YOLO Cropping
     2. UNet Denoising
     3. Line detection & smoothing
     4. Geometry fitting
     5. Energy sampling
   - The backend continually updates a status JSON with progress and generated plots.
3. **Visualization (Frontend):** 
   - The `results.html` page polls the backend for updates.
   - Images and matplotlib plots (saved as files) are rendered in a structured timeline:
     - `02_morphological` (Denoising)
     - `05_smoothed_lines` (Line tracking)
     - `07_rotated_nobg` (Frame alignment)
     - `11_classified` (Species tagging)
     - `14_log_spectra` (Energy spectra)

## Deployment

The application is fully containerized (`web/Dockerfile`).

### Using Docker Compose (Local)
```bash
cd web
docker compose up --build
```
The application will be accessible at `http://localhost:32212`.

### Production (Coolify)
For production deployments, Coolify is recommended. Point Coolify to the repository root, select the `web/Dockerfile`, set the port to `32212`, and ensure `/app/data` is mounted as a persistent volume to preserve uploads and results between container restarts.
