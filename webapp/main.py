"""Minimal FastAPI app: upload a plan, download vectorized outputs.

This UI is intentionally tiny:
  - POST /  -> accepts multipart upload (plan / optional overlay / optional mapping)
  - GET  /  -> simple HTML form
  - GET  /jobs/{job_id}            -> results page with download links + preview
  - GET  /jobs/{job_id}/files/{name} -> serve a generated artifact
  - GET  /jobs/{job_id}/download   -> zip of all artifacts
  - GET  /healthz                  -> health probe for Railway

Jobs live under ``JOBS_DIR`` (default ``/tmp/vectorize_jobs``). A background
cleanup task trims old jobs to prevent disk bloat on the Railway instance.
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import shutil
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from vectorize_floor.models import PipelineConfig
from vectorize_floor.pipeline import Pipeline, PipelineInputs


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

BASE_DIR = Path(__file__).resolve().parent
JOBS_DIR = Path(os.environ.get("JOBS_DIR", "/tmp/vectorize_jobs"))
JOBS_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 30 * 1024 * 1024))  # 30 MB
JOB_TTL_SECONDS = int(os.environ.get("JOB_TTL_SECONDS", 60 * 60 * 6))  # 6 h
ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
ALLOWED_MAPPING_SUFFIXES = {".csv", ".json"}

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(
    title="Floor-plan vectorizer",
    description="Upload a floor-plan image and download generated SVG/JSON lots.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


# ---------------------------------------------------------------------------
# Job state
# ---------------------------------------------------------------------------

@dataclass
class JobStatus:
    state: str = "pending"  # pending | running | done | error
    message: str = ""
    summary: Optional[Dict] = None


# In-memory status map. Fine for a single-worker Railway deployment.
# For multi-worker setups, move to Redis.
_job_status: Dict[str, JobStatus] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def _validate_suffix(filename: str, allowed: set, kind: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported {kind} format '{suffix}'. Allowed: {sorted(allowed)}",
        )
    return suffix


def _parse_optional_int(value: Optional[str], field_name: str) -> Optional[int]:
    """Parse an HTML-form string field into Optional[int].

    Treats empty strings and whitespace-only values as ``None`` — browsers
    submit ``""`` for untouched ``<input type="number">`` fields, which is
    not the same as "omitted".
    """
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return int(stripped)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Field '{field_name}' must be an integer, got '{value}'.",
        )


def _parse_optional_float(value: Optional[str], field_name: str) -> Optional[float]:
    """Parse an HTML-form string field into Optional[float]."""
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Field '{field_name}' must be a number, got '{value}'.",
        )


async def _save_upload(upload: UploadFile, dest: Path) -> None:
    size = 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        while chunk := await upload.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"File '{upload.filename}' exceeds "
                           f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
                )
            f.write(chunk)


def _cleanup_old_jobs() -> None:
    """Remove job directories older than JOB_TTL_SECONDS."""
    now = time.time()
    for d in JOBS_DIR.iterdir():
        if not d.is_dir():
            continue
        try:
            if now - d.stat().st_mtime > JOB_TTL_SECONDS:
                shutil.rmtree(d, ignore_errors=True)
                _job_status.pop(d.name, None)
                logger.info("Removed expired job %s", d.name)
        except OSError:
            continue


async def _periodic_cleanup() -> None:
    while True:
        try:
            _cleanup_old_jobs()
        except Exception:  # noqa: BLE001
            logger.exception("Cleanup pass failed")
        await asyncio.sleep(30 * 60)  # twice per hour


def _run_pipeline_sync(job_id: str, cfg: PipelineConfig, inputs: PipelineInputs) -> None:
    status = _job_status[job_id]
    status.state = "running"
    try:
        summary = Pipeline(cfg).run(inputs)
        status.state = "done"
        status.summary = summary
        status.message = "OK"
    except Exception as e:  # noqa: BLE001
        logger.exception("Job %s failed", job_id)
        status.state = "error"
        status.message = str(e)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def _on_startup() -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    asyncio.create_task(_periodic_cleanup())
    logger.info("Floor-plan vectorizer web UI started. JOBS_DIR=%s", JOBS_DIR)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/", response_class=HTMLResponse)
async def submit(
    background_tasks: BackgroundTasks,
    plan: UploadFile = File(..., description="Floor plan image"),
    overlay: Optional[UploadFile] = File(None, description="Overlay image (optional)"),
    mapping: Optional[UploadFile] = File(None, description="Mapping CSV/JSON (optional)"),
    floor_id: str = Form("floor"),
    ocr_fallback: Optional[str] = Form(None),
    debug: Optional[str] = Form(None),
    closing_kernel: Optional[str] = Form(None),
    simplify_tolerance: Optional[str] = Form(None),
):
    if not plan or not plan.filename:
        raise HTTPException(status_code=400, detail="Plan image is required.")

    closing_kernel_val = _parse_optional_int(closing_kernel, "closing_kernel")
    simplify_tolerance_val = _parse_optional_float(simplify_tolerance, "simplify_tolerance")

    job_id = uuid.uuid4().hex[:12]
    job_dir = _job_dir(job_id)
    input_dir = job_dir / "input"
    output_dir = job_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    plan_suffix = _validate_suffix(plan.filename, ALLOWED_IMAGE_SUFFIXES, "plan")
    plan_path = input_dir / f"plan{plan_suffix}"
    await _save_upload(plan, plan_path)

    overlay_path: Optional[Path] = None
    if overlay and overlay.filename:
        overlay_suffix = _validate_suffix(overlay.filename, ALLOWED_IMAGE_SUFFIXES, "overlay")
        overlay_path = input_dir / f"overlay{overlay_suffix}"
        await _save_upload(overlay, overlay_path)

    mapping_path: Optional[Path] = None
    if mapping and mapping.filename:
        mapping_suffix = _validate_suffix(
            mapping.filename, ALLOWED_MAPPING_SUFFIXES, "mapping"
        )
        mapping_path = input_dir / f"mapping{mapping_suffix}"
        await _save_upload(mapping, mapping_path)

    cfg = PipelineConfig(
        floor_id=floor_id or "floor",
        ocr_fallback=bool(ocr_fallback),
        debug=bool(debug),
    )
    if closing_kernel_val is not None:
        cfg.closing_kernel_size = closing_kernel_val
    if simplify_tolerance_val is not None:
        cfg.simplify_tolerance = simplify_tolerance_val

    inputs = PipelineInputs(
        plan_path=plan_path,
        overlay_path=overlay_path,
        mapping_path=mapping_path,
        out_dir=output_dir,
    )

    _job_status[job_id] = JobStatus(state="pending")
    background_tasks.add_task(_run_pipeline_sync, job_id, cfg, inputs)

    return RedirectResponse(url=f"/jobs/{job_id}", status_code=303)


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
async def job_page(request: Request, job_id: str):
    if job_id not in _job_status and not _job_dir(job_id).exists():
        raise HTTPException(status_code=404, detail="Job not found.")
    status = _job_status.get(job_id, JobStatus(state="unknown"))
    output_dir = _job_dir(job_id) / "output"
    files = []
    if output_dir.exists():
        for p in sorted(output_dir.iterdir()):
            if p.is_file():
                files.append({"name": p.name, "size": p.stat().st_size})
    return templates.TemplateResponse(
        "job.html",
        {
            "request": request,
            "job_id": job_id,
            "status": status,
            "files": files,
        },
    )


@app.get("/jobs/{job_id}/status")
async def job_status(job_id: str):
    status = _job_status.get(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {
        "job_id": job_id,
        "state": status.state,
        "message": status.message,
        "summary": status.summary,
    }


@app.get("/jobs/{job_id}/files/{filename}")
async def job_file(job_id: str, filename: str):
    # Path-traversal guard.
    safe_name = Path(filename).name
    path = _job_dir(job_id) / "output" / safe_name
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    media_type = "application/octet-stream"
    suffix = path.suffix.lower()
    if suffix == ".svg":
        media_type = "image/svg+xml"
    elif suffix == ".json":
        media_type = "application/json"
    elif suffix == ".png":
        media_type = "image/png"
    return FileResponse(path, media_type=media_type, filename=safe_name)


@app.get("/jobs/{job_id}/download")
async def job_download(job_id: str):
    output_dir = _job_dir(job_id) / "output"
    if not output_dir.exists():
        raise HTTPException(status_code=404, detail="Job outputs not found.")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in output_dir.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(output_dir))
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{job_id}.zip"'},
    )
