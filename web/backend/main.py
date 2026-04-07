"""FastAPI backend for Thomson parabola image analysis.

Serves:
  POST /upload              – accept image, start background processing
  GET  /status/{job_id}     – poll for state + available plots
  GET  /data/*              – static result files (plots, JSON)
  GET  .../plots/*.png?thumb=1 – downscaled preview for the results grid
  GET  /                    – SPA index
  GET  /results/{job_id}    – SPA results page
  GET  /synthetic/* – landing demo assets from Vite public/synthetic/ (shipped in dist)
  GET  /spectrometer-landing.svg – landing diagram from Vite public/
"""
import io
import os
import json
import shutil
import threading
import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageFile

OBLISK_ROOT = Path(__file__).resolve().parent.parent.parent

import torch

from oblisk.analysis.species import normalize_classification_elements
from oblisk.web_service import preload_web_models, run_web_analysis

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Oblisk – Thomson Parabola Analyser")


@app.on_event("startup")
def _preload_unet_denoiser() -> None:
    preload_web_models(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path(os.environ.get("DATA_DIR", str(OBLISK_ROOT / "data")))
UPLOADS_DIR = DATA_DIR / "uploads"
RESULTS_DIR = DATA_DIR / "results"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Matplotlib pyplot and parts of the stack are not thread-safe. The web server runs
# each job in a background thread; serialize pipeline execution so concurrent
# analyses complete correctly (jobs may wait on each other but do not corrupt).
_WEB_PIPELINE_LOCK = threading.Lock()
_THUMB_PIL_LOCK = threading.Lock()

# Max edge length for grid previews (full PNG served in lightbox only).
PLOT_THUMB_MAX_PX = 512

# ---------------------------------------------------------------------------
# Ordered list of plots produced by main.py
# ---------------------------------------------------------------------------
PLOT_NAMES = [
    "01_cropped_standardized",
    "02_morphological",
    "03_peaks_overlay",
    "05_smoothed_lines",
    "07_rotated_nobg",
    "09_a_score_peaks",
    "10_detected_parabolas",
    "11_classified",
    "12_sampling_overlay",
    "15_numbered_log_spectra",
    "16_linear_energy_logy",
]

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _status_path(job_dir: Path) -> Path:
    return job_dir / "status.json"


def _write_status(
    job_dir: Path,
    state: str,
    error: str | None = None,
    *,
    source_filename: str | None = None,
) -> None:
    path = _status_path(job_dir)
    payload: dict[str, object] = {}
    if path.exists():
        try:
            with open(path) as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError):
            payload = {}
    payload["state"] = state
    if error is not None:
        payload["error"] = error
    else:
        payload.pop("error", None)
    if source_filename is not None:
        payload["source_filename"] = source_filename
    # Write via a temp file + replace so readers never see a truncated status.json
    # (open("w") truncates immediately; concurrent GET /status can JSONDecodeError).
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f)
    os.replace(tmp_path, path)


def _atomic_json_write(path: Path, payload: object, *, indent: int | None) -> None:
    """Write JSON atomically so concurrent readers never see a truncated file."""
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=indent)
    os.replace(tmp_path, path)


def _plot_png_path(job_id: str, filename: str) -> Path | None:
    """Resolve a plot PNG under results/<job_id>/plots/ or None if invalid."""
    try:
        safe_job = str(uuid.UUID(job_id))
    except ValueError:
        return None
    if Path(filename).name != filename:
        return None
    if not filename.lower().endswith(".png"):
        return None
    path = (RESULTS_DIR / safe_job / "plots" / filename).resolve()
    expected_dir = (RESULTS_DIR / safe_job / "plots").resolve()
    if path.parent != expected_dir or not path.is_file():
        return None
    return path


def _png_thumbnail_bytes(path: Path, max_edge: int) -> bytes:
    """Build a downscaled PNG; tolerate partially written files (pipeline race)."""
    buf = io.BytesIO()
    with _THUMB_PIL_LOCK:
        prev_truncated = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            with Image.open(path) as im:
                im.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
                im.save(buf, format="PNG", optimize=True)
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = prev_truncated
    return buf.getvalue()


# Registered before /data StaticFiles so plot paths can serve optional previews.
@app.get("/data/results/{job_id}/plots/{filename}", response_model=None)
def serve_result_plot(
    job_id: str,
    filename: str,
    thumb: bool = Query(False, description="Return a smaller PNG for grid previews"),
) -> FileResponse | Response:
    plot_path = _plot_png_path(job_id, filename)
    if plot_path is None:
        raise HTTPException(status_code=404, detail="Plot not found")
    if thumb:
        try:
            body = _png_thumbnail_bytes(plot_path, PLOT_THUMB_MAX_PX)
        except OSError as exc:
            raise HTTPException(
                status_code=503,
                detail="Plot preview is not ready or the file is unreadable; retry shortly.",
            ) from exc
        return Response(content=body, media_type="image/png")
    return FileResponse(plot_path)


# Result files served as static (plot PNGs are handled above).
app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

def _form_bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    use_denoise_unet: str = Form("true"),
    inner_margin_crop: str = Form("true"),
    species_json: str = Form('["H","C","O","Si"]'),
    spectrometer_json: str = Form("{}"),
) -> JSONResponse:
    """Accept an image, start background processing, return a job id."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported extension '{suffix}'. "
                "Use: PNG, JPEG, TIFF or BMP."
            ),
        )

    job_id = str(uuid.uuid4())
    job_dir = RESULTS_DIR / job_id
    job_dir.mkdir(parents=True)

    image_path = UPLOADS_DIR / f"{job_id}{suffix}"
    with open(image_path, "wb") as fout:
        shutil.copyfileobj(file.file, fout)

    use_unet = _form_bool(use_denoise_unet)
    margin_px = 50 if _form_bool(inner_margin_crop) else 0

    try:
        parsed_species = json.loads(species_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail="species_json must be a JSON array of symbols.",
        ) from exc
    if not isinstance(parsed_species, list):
        raise HTTPException(
            status_code=400,
            detail="species_json must be a JSON array of element symbols.",
        )
    try:
        species_list = normalize_classification_elements(
            [str(x) for x in parsed_species],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        parsed_spec = json.loads(spectrometer_json)
    except json.JSONDecodeError:
        parsed_spec = {}
    if not isinstance(parsed_spec, dict):
        parsed_spec = {}
    spec_params: dict[str, float] | None = parsed_spec if parsed_spec else None

    display_name = Path(file.filename or "").name or "upload"
    _write_status(job_dir, "running", source_filename=display_name)

    def _process() -> None:
        try:
            with _WEB_PIPELINE_LOCK:
                result = run_web_analysis(
                    image_path=image_path,
                    output_dir=job_dir,
                    use_denoise_unet=use_unet,
                    inner_margin_crop=margin_px > 0,
                    classification_element_symbols=species_list,
                    spectrometer_params=spec_params,
                )
            result_path = job_dir / "res.json"
            _atomic_json_write(result_path, result, indent=2)
            _write_status(job_dir, "done")
        except Exception:
            _write_status(job_dir, "error", traceback.format_exc())

    threading.Thread(target=_process, daemon=True).start()

    return JSONResponse({"job_id": job_id})


@app.get("/status/{job_id}")
def get_status(job_id: str) -> dict[str, object]:
    """Return processing state + list of available plot URLs."""
    job_dir = RESULTS_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    status_file = _status_path(job_dir)
    if not status_file.exists():
        raise HTTPException(status_code=404, detail="Status file missing")

    with open(status_file) as f:
        status = json.load(f)

    plot_dir = job_dir / "plots"
    available_plots: list[str] = []
    if plot_dir.exists():
        for name in PLOT_NAMES:
            p = plot_dir / f"{name}.png"
            if p.exists():
                available_plots.append(
                    f"/data/results/{job_id}/plots/{name}.png"
                )

    result_json = None
    res_path = job_dir / "res.json"
    if res_path.exists():
        try:
            with open(res_path) as f:
                result_json = json.load(f)
        except (json.JSONDecodeError, OSError):
            result_json = None

    return {
        "job_id": job_id,
        "state": status.get("state", "unknown"),
        "error": status.get("error"),
        "source_filename": status.get("source_filename"),
        "plots": available_plots,
        "total_plots": len(PLOT_NAMES),
        "result": result_json,
    }


# ---------------------------------------------------------------------------
# Serve the compiled Vite SPA for every non-API / non-data route
# ---------------------------------------------------------------------------
STATIC_DIR = Path(__file__).parent.parent / "frontend" / "dist"

if STATIC_DIR.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=str(STATIC_DIR / "assets")),
        name="assets",
    )

    @app.get("/spectrometer-landing.svg")
    def serve_spectrometer_landing_svg() -> FileResponse:
        path = STATIC_DIR / "spectrometer-landing.svg"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="spectrometer-landing.svg not in frontend dist")
        return FileResponse(path)

    synthetic_public = STATIC_DIR / "synthetic"
    if synthetic_public.is_dir():
        app.mount(
            "/synthetic",
            StaticFiles(directory=str(synthetic_public)),
            name="synthetic_public",
        )

    @app.get("/")
    def serve_index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/results/{job_id}")
    def serve_results_page(job_id: str) -> FileResponse:
        del job_id
        return FileResponse(STATIC_DIR / "results" / "index.html")

else:
    @app.get("/")
    def serve_index_dev() -> JSONResponse:
        return JSONResponse({
            "message": (
                "Frontend not built. Run: cd web/frontend && npm install && "
                "npm run build"
            ),
        })
