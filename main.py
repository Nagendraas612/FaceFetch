"""
main.py
FaceFetch – FastAPI entry point.

Endpoints:
  GET  /                         → Serve index.html
  GET  /health                   → DB + app health
  GET  /auth/login               → Start Google OAuth
  GET  /auth/callback            → OAuth callback
  GET  /auth/logout              → Clear session
  GET  /auth/me                  → Current user info
  POST /upload-reference         → Upload face photo; encoding saved to DB
  GET  /my-encodings             → List saved encodings for current user
  DELETE /my-encodings           → Delete all saved encodings
  DELETE /delete-reference/{id}  → Delete a specific encoding
  POST /search                   → Kick off deep Drive search (SSE streaming)
  POST /search-local             → Kick off local file search (SSE streaming)
  GET  /download/{search_id}     → Download result ZIP
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import secrets
import time
import uuid
import zipfile
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import uvicorn
from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import (
    HTMLResponse,
    Response,
    StreamingResponse,
)
from starlette.middleware.sessions import SessionMiddleware

import auth
import database
import engine

# ── Suppress noisy library logs ───────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="face_recognition_models")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logging.getLogger("pymongo").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("watchfiles").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger("main")

# ── Bootstrap ─────────────────────────────────────────────────────────────────
load_dotenv()

ENVIRONMENT   = os.getenv("ENVIRONMENT", "development")
IS_PRODUCTION = ENVIRONMENT.lower() in ("production", "prod")

SESSION_SECRET = os.getenv("SESSION_SECRET", "")
if not SESSION_SECRET:
    if IS_PRODUCTION:
        raise EnvironmentError(
            "SESSION_SECRET must be set in production. "
            "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    SESSION_SECRET = secrets.token_hex(32)
    logger.warning("⚠ No SESSION_SECRET set — using random key (sessions won't survive restarts)")

BASE_DIR = Path(__file__).parent

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_UPLOAD_SIZE             = 10 * 1024 * 1024   # 10 MB
MAX_LOCAL_FILES             = 2000
ALLOWED_MIME_PREFIXES       = (
    "image/jpeg", "image/png", "image/webp",
    "image/bmp", "image/tiff", "image/heic",
)
ZIP_TTL_SECONDS             = 1800               # 30 minutes
ZIP_MAX_STORE_MB            = 500
RATE_LIMIT_UPLOADS_PER_MIN  = 20
RATE_LIMIT_SCANS_PER_HOUR   = 100

# Tolerance bounds — widened upper bound to 0.70 so power users can
# intentionally trade precision for recall (e.g. very diverse photo sets).
TOLERANCE_MIN = 0.35
TOLERANCE_MAX = 0.70

# ── In-memory stores ─────────────────────────────────────────────────────────
_zip_store:    dict[str, dict]          = {}
_upload_rate:  dict[str, list[float]]   = defaultdict(list)
_scan_rate:    dict[str, list[float]]   = defaultdict(list)


# ── Rate limiting ─────────────────────────────────────────────────────────────
def _check_rate_limit(
    store: dict[str, list[float]],
    user_id: str,
    max_count: int,
    window_seconds: int,
):
    now    = time.time()
    cutoff = now - window_seconds
    store[user_id] = [t for t in store[user_id] if t > cutoff]
    if len(store[user_id]) >= max_count:
        raise HTTPException(
            429,
            f"Rate limit exceeded. Max {max_count} requests per "
            f"{window_seconds // 60} minute(s)."
        )
    store[user_id].append(now)


# ── ZIP store helpers ─────────────────────────────────────────────────────────
def _cleanup_zip_store():
    now     = time.time()
    expired = [
        sid for sid, entry in _zip_store.items()
        if now - entry["created_at"] > ZIP_TTL_SECONDS
    ]
    for sid in expired:
        del _zip_store[sid]
    if expired:
        logger.info("🗑 Cleaned up %d expired ZIP(s)", len(expired))


def _zip_store_size_mb() -> float:
    return sum(len(e["data"]) for e in _zip_store.values()) / (1024 * 1024)


async def _periodic_cleanup():
    while True:
        await asyncio.sleep(300)
        _cleanup_zip_store()


# ── App lifecycle ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    ok = await database.ping()
    if ok:
        logger.info("✓ MongoDB connected")
    else:
        logger.error("✗ MongoDB NOT reachable — check MONGO_URI")

    await database.ensure_indexes()
    cleanup_task = asyncio.create_task(_periodic_cleanup())

    yield

    cleanup_task.cancel()
    database.get_client().close()
    logger.info("MongoDB client closed")


app = FastAPI(title="FaceFetch", version="2.2.0", lifespan=lifespan)

app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    session_cookie="facefetch_session",
    max_age=3600 * 8,
    same_site="lax",
    https_only=IS_PRODUCTION,
)

app.include_router(auth.router)


# ── Static + HTML ─────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    try:
        return HTMLResponse((BASE_DIR / "index.html").read_text(encoding="utf-8"))
    except Exception as e:
        logger.error("Error loading index.html: %s", e)
        raise HTTPException(status_code=500, detail="index.html missing")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "mongo":  await database.ping(),
        "store": {
            "zip_count":   len(_zip_store),
            "zip_size_mb": round(_zip_store_size_mb(), 2),
        },
    }


# ── Reference face upload ─────────────────────────────────────────────────────
@app.post("/upload-reference")
async def upload_reference(
    request:     Request,
    file:        UploadFile = File(...),
    num_jitters: int        = Form(5),    # default increased to match REF_NUM_JITTERS
    model:       str        = Form("large"),
):
    user    = auth.require_user(request)
    user_id = user["sub"]

    _check_rate_limit(_upload_rate, user_id, RATE_LIMIT_UPLOADS_PER_MIN, 60)

    content_type = file.content_type or ""
    if not any(content_type.startswith(p) for p in ALLOWED_MIME_PREFIXES):
        raise HTTPException(
            415,
            f"Unsupported file type: {content_type}. Upload JPG, PNG, or WebP images.",
        )

    image_bytes = await file.read()
    if len(image_bytes) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            413,
            f"File too large ({len(image_bytes) // (1024*1024)} MB). "
            f"Maximum is {MAX_UPLOAD_SIZE // (1024*1024)} MB.",
        )
    if len(image_bytes) == 0:
        raise HTTPException(400, "Empty file uploaded.")

    if model not in ("small", "large"):
        model = "large"
    num_jitters = max(1, min(num_jitters, 10))

    try:
        # encode_reference_image now returns ALL encodings
        # (original + aligned + flipped) — save each one individually so the
        # database holds the full reference cluster for this photo.
        all_encodings = engine.encode_reference_image(
            image_bytes, num_jitters=num_jitters, model=model
        )
        if not all_encodings:
            raise HTTPException(
                400, "No face detected in this photo. Please upload a clear selfie."
            )

        # Save every encoding variant; the DB de-duplicates by ref_id per photo.
        for enc in all_encodings:
            await database.save_face_encoding(user_id, file.filename, enc)

        total = await database.get_encoding_count(user_id)
        return {
            "status":        "success",
            "total_saved":   total,
            "variants_from_this_photo": len(all_encodings),
        }

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error("Upload encoding failed for user %s: %s", user_id, e)
        raise HTTPException(500, "Face encoding failed. Please try a different photo.")


@app.get("/my-encodings")
async def my_encodings(request: Request):
    user = auth.require_user(request)
    refs = await database.get_all_references(user["sub"])
    return {
        "count":      len(refs),
        "references": [
            {"ref_id": str(r["ref_id"]), "filename": r["filename"]}
            for r in refs
        ],
    }


@app.delete("/my-encodings")
async def delete_encodings(request: Request):
    user    = auth.require_user(request)
    deleted = await database.delete_face_encodings(user["sub"])
    return {"deleted": deleted}


@app.delete("/delete-reference/{ref_id}")
async def delete_ref_endpoint(ref_id: str, request: Request):
    user = auth.require_user(request)

    if not re.match(r"^[a-fA-F0-9]{24}$", ref_id):
        raise HTTPException(400, "Invalid reference ID format.")

    success = await database.delete_specific_reference(user["sub"], ref_id)
    if not success:
        raise HTTPException(404, "Reference DNA not found.")

    logger.info("✓ Deleted DNA reference %s for user %s", ref_id, user["sub"])
    return {"success": True}


# ── Input validation ──────────────────────────────────────────────────────────
def _validate_scan_params(tolerance: float, model: str, upsample: int):
    tolerance = max(TOLERANCE_MIN, min(TOLERANCE_MAX, tolerance))
    if model not in ("hog", "cnn"):
        model = "hog"
    upsample = max(0, min(2, upsample))
    return tolerance, model, upsample


def _validate_drive_link(drive_link: str):
    if not drive_link:
        raise HTTPException(400, "Drive link is required.")
    if not re.search(r"drive\.google\.com", drive_link):
        raise HTTPException(400, "Invalid Google Drive link.")


# ── SSE helper ────────────────────────────────────────────────────────────────
def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


# ── Deep search (SSE streaming) ───────────────────────────────────────────────
@app.post("/search")
async def search(
    request:    Request,
    drive_link: str   = Form(...),
    tolerance:  float = Form(0.50),   # raised slightly — CLAHE + alignment recover precision
    model:      str   = Form("hog"),
    upsample:   int   = Form(1),
):
    user    = auth.require_user(request)
    user_id = user["sub"]

    _check_rate_limit(_scan_rate, user_id, RATE_LIMIT_SCANS_PER_HOUR, 3600)

    tolerance, model, upsample = _validate_scan_params(tolerance, model, upsample)
    _validate_drive_link(drive_link)

    _cleanup_zip_store()
    if _zip_store_size_mb() > ZIP_MAX_STORE_MB:
        raise HTTPException(503, "Server is busy. Please try again in a few minutes.")

    drv_token     = request.session.get("drive_token", "")
    refresh_token = await database.get_refresh_token(user_id)

    if not drv_token and not refresh_token and not engine.GOOGLE_API_KEY:
        raise HTTPException(
            401,
            "No valid Google credentials. Log in with Google for private folders, "
            "or add GOOGLE_API_KEY to .env for public folders.",
        )

    known_encodings = await database.load_face_encodings(user_id)
    if not known_encodings:
        raise HTTPException(400, "No reference face found. Upload a reference photo first.")

    async def event_stream() -> AsyncGenerator[str, None]:
        search_id      = str(uuid.uuid4())
        progress_queue: asyncio.Queue = asyncio.Queue()
        loop           = asyncio.get_running_loop()
        scan_start     = time.time()

        def progress_cb(current, total, filename, matched_count=0):
            loop.call_soon_threadsafe(
                progress_queue.put_nowait,
                {"current": current, "total": total,
                 "filename": filename, "matched": matched_count},
            )

        future = loop.run_in_executor(
            None,
            lambda: engine.run_deep_search(
                drive_link, drv_token, refresh_token,
                known_encodings, tolerance, model, upsample, progress_cb,
            ),
        )

        while not future.done():
            try:
                prog = await asyncio.wait_for(progress_queue.get(), timeout=0.5)
                yield _sse({"type": "progress", **prog})
            except asyncio.TimeoutError:
                if time.time() - scan_start > 1800:
                    future.cancel()
                    yield _sse({"type": "error", "message": "Scan timed out after 30 minutes."})
                    return
                yield ": keep-alive\n\n"

        while not progress_queue.empty():
            yield _sse({"type": "progress", **progress_queue.get_nowait()})

        try:
            zip_bytes     = await future
            scan_duration = round(time.time() - scan_start, 1)

            try:
                with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                    matched = len(zf.namelist())
            except zipfile.BadZipFile:
                matched = 0

            if matched == 0:
                yield _sse({
                    "type":      "done",
                    "search_id": "",
                    "matched":   0,
                    "duration":  scan_duration,
                    "message":   (
                        "No matching photos found. "
                        "Try uploading more reference selfies or raising the tolerance."
                    ),
                })
            else:
                _zip_store[search_id] = {"data": zip_bytes, "created_at": time.time()}
                logger.info(
                    "✓ ZIP stored: id=%s, size=%d bytes, files=%d",
                    search_id, len(zip_bytes), matched,
                )
                yield _sse({
                    "type":      "done",
                    "search_id": search_id,
                    "matched":   matched,
                    "duration":  scan_duration,
                })
        except Exception as exc:
            logger.error("Search failed: %s", exc)
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Local files search (SSE streaming) ───────────────────────────────────────
@app.post("/search-local")
async def search_local(
    request:   Request,
    files:     list[UploadFile] = File(...),
    tolerance: float            = Form(0.50),
    model:     str              = Form("hog"),
    upsample:  int              = Form(1),
):
    user    = auth.require_user(request)
    user_id = user["sub"]

    _check_rate_limit(_scan_rate, user_id, RATE_LIMIT_SCANS_PER_HOUR, 3600)

    tolerance, model, upsample = _validate_scan_params(tolerance, model, upsample)

    if len(files) > MAX_LOCAL_FILES:
        raise HTTPException(400, f"Too many files ({len(files)}). Maximum is {MAX_LOCAL_FILES}.")

    _cleanup_zip_store()
    if _zip_store_size_mb() > ZIP_MAX_STORE_MB:
        raise HTTPException(503, "Server is busy. Please try again in a few minutes.")

    known_encodings = await database.load_face_encodings(user_id)
    if not known_encodings:
        raise HTTPException(400, "No reference face found. Upload a reference photo first.")

    file_data_list = []
    for f in files:
        content = await f.read()
        if content:
            file_data_list.append((f.filename, content))

    if not file_data_list:
        raise HTTPException(400, "No valid image files provided.")

    async def event_stream() -> AsyncGenerator[str, None]:
        search_id      = str(uuid.uuid4())
        progress_queue: asyncio.Queue = asyncio.Queue()
        loop           = asyncio.get_running_loop()
        scan_start     = time.time()

        def progress_cb(current, total, filename, matched_count=0):
            loop.call_soon_threadsafe(
                progress_queue.put_nowait,
                {"current": current, "total": total,
                 "filename": filename, "matched": matched_count},
            )

        future = loop.run_in_executor(
            None,
            lambda: engine.run_local_search(
                file_data_list, known_encodings, tolerance, model, upsample, progress_cb,
            ),
        )

        while not future.done():
            try:
                prog = await asyncio.wait_for(progress_queue.get(), timeout=0.5)
                yield _sse({"type": "progress", **prog})
            except asyncio.TimeoutError:
                if time.time() - scan_start > 1800:
                    future.cancel()
                    yield _sse({"type": "error", "message": "Scan timed out after 30 minutes."})
                    return
                yield ": keep-alive\n\n"

        while not progress_queue.empty():
            yield _sse({"type": "progress", **progress_queue.get_nowait()})

        try:
            zip_bytes     = await future
            scan_duration = round(time.time() - scan_start, 1)

            try:
                with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                    matched = len(zf.namelist())
            except zipfile.BadZipFile:
                matched = 0

            if matched == 0:
                yield _sse({
                    "type":      "done",
                    "search_id": "",
                    "matched":   0,
                    "duration":  scan_duration,
                    "message":   (
                        "No matching photos found. "
                        "Try uploading more reference selfies or raising the tolerance."
                    ),
                })
            else:
                _zip_store[search_id] = {"data": zip_bytes, "created_at": time.time()}
                logger.info(
                    "✓ ZIP stored: id=%s, size=%d bytes, files=%d",
                    search_id, len(zip_bytes), matched,
                )
                yield _sse({
                    "type":      "done",
                    "search_id": search_id,
                    "matched":   matched,
                    "duration":  scan_duration,
                })
        except Exception as exc:
            logger.error("Search local failed: %s", exc)
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── ZIP download ──────────────────────────────────────────────────────────────
@app.get("/download/{search_id}")
async def download_zip(search_id: str, request: Request):
    auth.require_user(request)

    try:
        uuid.UUID(search_id)
    except ValueError:
        raise HTTPException(400, "Invalid search ID.")

    entry = _zip_store.get(search_id)
    if not entry:
        raise HTTPException(
            404,
            "ZIP not found or expired. Results are kept for 30 minutes after scanning.",
        )

    zip_data = entry["data"]
    filename = f"facefetch_matches_{search_id[:8]}.zip"

    return Response(
        content=zip_data,
        status_code=200,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length":      str(len(zip_data)),
            "Content-Type":        "application/octet-stream",
            "Cache-Control":       "no-store",
        },
    )


# ── Debug endpoint ────────────────────────────────────────────────────────────
@app.get("/debug/face-status")
async def debug_face_status(request: Request):
    user    = auth.require_user(request)
    user_id = user["sub"]
    known   = await database.load_face_encodings(user_id)
    return {
        "user_id":         user_id,
        "reference_count": len(known),
        "has_references":  len(known) > 0,
        "encoding_sample": known[0][:5] if known else None,
    }


# ── Dev server ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
