"""
main.py
EventAI – FastAPI entry point.

Endpoints:
  GET  /                         → Serve index.html
  GET  /health                   → DB + app health
  GET  /auth/login               → Start Google OAuth
  GET  /auth/callback            → OAuth callback
  GET  /auth/logout              → Clear session
  GET  /auth/me                  → Current user info
  POST /upload-reference         → Upload face photo; encoding saved to MongoDB
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

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
IS_PRODUCTION = ENVIRONMENT.lower() in ("production", "prod")

# Session secret: must be stable in production
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
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_LOCAL_FILES = 2000              # Max files in a single local scan
ALLOWED_MIME_PREFIXES = ("image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff", "image/heic")
ZIP_TTL_SECONDS = 1800              # 30 minutes
ZIP_MAX_STORE_MB = 500              # Max total ZIP store size in MB
RATE_LIMIT_UPLOADS_PER_MIN = 20
RATE_LIMIT_SCANS_PER_HOUR = 10

# ── In-memory stores ─────────────────────────────────────────────────────────
# ZIP store with timestamps: { search_id: { "data": bytes, "created_at": float } }
_zip_store: dict[str, dict] = {}

# Rate limiter: { user_id: [timestamp, ...] }
_upload_rate: dict[str, list[float]] = defaultdict(list)
_scan_rate: dict[str, list[float]] = defaultdict(list)


# ── Rate limiting helper ──────────────────────────────────────────────────────
def _check_rate_limit(store: dict[str, list[float]], user_id: str, max_count: int, window_seconds: int):
    """Check if user has exceeded rate limit. Raises HTTPException if so."""
    now = time.time()
    cutoff = now - window_seconds
    # Prune old entries
    store[user_id] = [t for t in store[user_id] if t > cutoff]
    if len(store[user_id]) >= max_count:
        raise HTTPException(
            429,
            f"Rate limit exceeded. Max {max_count} requests per {window_seconds // 60} minute(s)."
        )
    store[user_id].append(now)


# ── ZIP store cleanup ─────────────────────────────────────────────────────────
def _cleanup_zip_store():
    """Remove expired ZIPs from the in-memory store."""
    now = time.time()
    expired = [sid for sid, entry in _zip_store.items()
               if now - entry["created_at"] > ZIP_TTL_SECONDS]
    for sid in expired:
        del _zip_store[sid]
    if expired:
        logger.info("🗑 Cleaned up %d expired ZIP(s) from store", len(expired))


def _zip_store_size_mb() -> float:
    """Get total size of ZIP store in MB."""
    return sum(len(entry["data"]) for entry in _zip_store.values()) / (1024 * 1024)


async def _periodic_cleanup():
    """Background task that cleans up expired ZIPs every 5 minutes."""
    while True:
        await asyncio.sleep(300)  # 5 minutes
        _cleanup_zip_store()


# ── App lifecycle ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    ok = await database.ping()
    if ok:
        logger.info("✓ MongoDB Atlas connected")
    else:
        logger.error("✗ MongoDB Atlas NOT reachable — check MONGO_URI")

    # Ensure indexes exist
    await database.ensure_indexes()

    # Start background cleanup
    cleanup_task = asyncio.create_task(_periodic_cleanup())

    yield

    # Shutdown
    cleanup_task.cancel()
    client = database.get_client()
    client.close()
    logger.info("MongoDB client closed")


app = FastAPI(title="EventAI", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    session_cookie="eventai_session",
    max_age=3600 * 8,
    same_site="lax",
    https_only=IS_PRODUCTION,
)

app.include_router(auth.router)


# ── Static + HTML ─────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    try:
        index_html = (BASE_DIR / "index.html").read_text(encoding="utf-8")
        return HTMLResponse(index_html)
    except Exception as e:
        logger.error("Error loading index.html: %s", e)
        raise HTTPException(status_code=500, detail="index.html missing")


@app.get("/health")
async def health():
    store_info = {
        "zip_count": len(_zip_store),
        "zip_size_mb": round(_zip_store_size_mb(), 2),
    }
    return {"status": "ok", "mongo": await database.ping(), "store": store_info}


# ── Reference face upload ────────────────────────────────────────────────────
@app.post("/upload-reference")
async def upload_reference(
    request: Request,
    file: UploadFile = File(...),
    num_jitters: int = Form(1),
    model: str = Form("large"),
):
    user = auth.require_user(request)
    user_id = user["sub"]

    # Rate limiting
    _check_rate_limit(_upload_rate, user_id, RATE_LIMIT_UPLOADS_PER_MIN, 60)

    # Validate MIME type
    content_type = file.content_type or ""
    if not any(content_type.startswith(prefix) for prefix in ALLOWED_MIME_PREFIXES):
        raise HTTPException(415, f"Unsupported file type: {content_type}. Upload JPG, PNG, or WebP images.")

    # Read with size limit
    image_bytes = await file.read()
    if len(image_bytes) > MAX_UPLOAD_SIZE:
        raise HTTPException(413, f"File too large ({len(image_bytes) // (1024*1024)}MB). Maximum is {MAX_UPLOAD_SIZE // (1024*1024)}MB.")

    if len(image_bytes) == 0:
        raise HTTPException(400, "Empty file uploaded.")

    # Validate model param
    if model not in ("small", "large"):
        model = "large"

    # Clamp num_jitters
    num_jitters = max(1, min(num_jitters, 10))

    try:
        encodings = engine.encode_reference_image(image_bytes, num_jitters=num_jitters, model=model)

        if not encodings:
            raise HTTPException(400, "No face detected in this photo. Please upload a clear selfie with your face visible.")

        await database.save_face_encoding(user_id, file.filename, encodings[0])

        total = await database.get_encoding_count(user_id)
        return {"status": "success", "total_saved": total}

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

    serializable_refs = []
    for ref in refs:
        serializable_refs.append({
            "ref_id": str(ref["ref_id"]),
            "filename": ref["filename"]
        })

    return {"count": len(serializable_refs), "references": serializable_refs}


@app.delete("/my-encodings")
async def delete_encodings(request: Request):
    user = auth.require_user(request)
    deleted = await database.delete_face_encodings(user["sub"])
    return {"deleted": deleted}


@app.delete("/delete-reference/{ref_id}")
async def delete_ref_endpoint(ref_id: str, request: Request):
    user = auth.require_user(request)

    # Validate ref_id format (ObjectId is 24 hex chars)
    if not re.match(r"^[a-fA-F0-9]{24}$", ref_id):
        raise HTTPException(400, "Invalid reference ID format.")

    success = await database.delete_specific_reference(user["sub"], ref_id)

    if not success:
        raise HTTPException(status_code=404, detail="Reference DNA not found.")

    logger.info("Deleted DNA reference %s for user %s", ref_id, user["sub"])
    return {"success": True}


# ── Input validation helper ───────────────────────────────────────────────────
def _validate_scan_params(tolerance: float, model: str, upsample: int):
    """Validate and sanitize scan parameters."""
    tolerance = max(0.35, min(0.65, tolerance))
    if model not in ("hog", "cnn"):
        model = "hog"
    upsample = max(0, min(2, upsample))
    return tolerance, model, upsample


def _validate_drive_link(drive_link: str):
    """Validate Google Drive link format."""
    if not drive_link:
        raise HTTPException(400, "Drive link is required.")
    if not re.search(r"drive\.google\.com", drive_link):
        raise HTTPException(400, "Invalid Google Drive link. Please provide a valid Drive folder URL.")


# ── Deep search (SSE streaming progress) ─────────────────────────────────────
@app.post("/search")
async def search(
    request: Request,
    drive_link: str = Form(...),
    tolerance: float = Form(0.50),
    model: str = Form("hog"),
    upsample: int = Form(0),
):
    user = auth.require_user(request)
    user_id = user["sub"]

    # Rate limiting
    _check_rate_limit(_scan_rate, user_id, RATE_LIMIT_SCANS_PER_HOUR, 3600)

    # Validate inputs
    tolerance, model, upsample = _validate_scan_params(tolerance, model, upsample)
    _validate_drive_link(drive_link)

    # Check ZIP store capacity
    _cleanup_zip_store()
    if _zip_store_size_mb() > ZIP_MAX_STORE_MB:
        raise HTTPException(503, "Server is busy processing other scans. Please try again in a few minutes.")

    drv_token = request.session.get("drive_token", "")
    refresh_token = await database.get_refresh_token(user_id)

    if not drv_token and not refresh_token:
        raise HTTPException(401, "Google Drive token missing. Please log in again.")

    known_encodings = await database.load_face_encodings(user_id)
    if not known_encodings:
        raise HTTPException(400, "No reference face found. Upload a reference photo first.")

    async def event_stream() -> AsyncGenerator[str, None]:
        search_id = str(uuid.uuid4())
        progress_queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        scan_start = time.time()

        def progress_cb(current, total, filename, matched_count=0):
            loop.call_soon_threadsafe(
                progress_queue.put_nowait,
                {
                    "current": current,
                    "total": total,
                    "filename": filename,
                    "matched": matched_count,
                },
            )

        future = loop.run_in_executor(
            None,
            lambda: engine.run_deep_search(
                drive_link, drv_token, refresh_token,
                known_encodings, tolerance, model, upsample, progress_cb
            ),
        )

        while not future.done():
            try:
                prog = await asyncio.wait_for(progress_queue.get(), timeout=0.5)
                yield f"data: {json.dumps({'type': 'progress', **prog})}\n\n"
            except asyncio.TimeoutError:
                # Check for scan timeout (30 minutes)
                if time.time() - scan_start > 1800:
                    future.cancel()
                    yield f"data: {json.dumps({'type': 'error', 'message': 'Scan timed out after 30 minutes.'})}\n\n"
                    return
                yield ": keep-alive\n\n"

        # Drain remaining progress events
        while not progress_queue.empty():
            prog = progress_queue.get_nowait()
            yield f"data: {json.dumps({'type': 'progress', **prog})}\n\n"

        try:
            zip_bytes = await future
            scan_duration = round(time.time() - scan_start, 1)

            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                matched = len(zf.namelist())

            if matched == 0:
                payload = {
                    "type": "done",
                    "search_id": "",
                    "matched": 0,
                    "duration": scan_duration,
                    "message": "No matching photos found. Try uploading more reference selfies or increasing the tolerance."
                }
            else:
                _zip_store[search_id] = {"data": zip_bytes, "created_at": time.time()}
                payload = {
                    "type": "done",
                    "search_id": search_id,
                    "matched": matched,
                    "duration": scan_duration,
                }

            yield f"data: {json.dumps(payload)}\n\n"
        except Exception as exc:
            logger.error("Search failed: %s", exc)
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Local files search (SSE streaming) ────────────────────────────────────────
@app.post("/search-local")
async def search_local(
    request: Request,
    files: list[UploadFile] = File(...),
    tolerance: float = Form(0.50),
    model: str = Form("hog"),
    upsample: int = Form(0),
):
    user = auth.require_user(request)
    user_id = user["sub"]

    # Rate limiting
    _check_rate_limit(_scan_rate, user_id, RATE_LIMIT_SCANS_PER_HOUR, 3600)

    # Validate inputs
    tolerance, model, upsample = _validate_scan_params(tolerance, model, upsample)

    if len(files) > MAX_LOCAL_FILES:
        raise HTTPException(400, f"Too many files ({len(files)}). Maximum is {MAX_LOCAL_FILES}.")

    # Check ZIP store capacity
    _cleanup_zip_store()
    if _zip_store_size_mb() > ZIP_MAX_STORE_MB:
        raise HTTPException(503, "Server is busy processing other scans. Please try again in a few minutes.")

    known_encodings = await database.load_face_encodings(user_id)
    if not known_encodings:
        raise HTTPException(400, "No reference face found. Upload a reference photo first.")

    # Read files into memory
    file_data_list = []
    for f in files:
        content = await f.read()
        if len(content) > 0:
            file_data_list.append((f.filename, content))

    if not file_data_list:
        raise HTTPException(400, "No valid image files provided.")

    async def event_stream() -> AsyncGenerator[str, None]:
        search_id = str(uuid.uuid4())
        progress_queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        scan_start = time.time()

        def progress_cb(current, total, filename, matched_count=0):
            loop.call_soon_threadsafe(
                progress_queue.put_nowait,
                {
                    "current": current,
                    "total": total,
                    "filename": filename,
                    "matched": matched_count,
                },
            )

        future = loop.run_in_executor(
            None,
            lambda: engine.run_local_search(
                file_data_list, known_encodings, tolerance, model, upsample, progress_cb
            ),
        )

        while not future.done():
            try:
                prog = await asyncio.wait_for(progress_queue.get(), timeout=0.5)
                yield f"data: {json.dumps({'type': 'progress', **prog})}\n\n"
            except asyncio.TimeoutError:
                if time.time() - scan_start > 1800:
                    future.cancel()
                    yield f"data: {json.dumps({'type': 'error', 'message': 'Scan timed out after 30 minutes.'})}\n\n"
                    return
                yield ": keep-alive\n\n"

        while not progress_queue.empty():
            prog = progress_queue.get_nowait()
            yield f"data: {json.dumps({'type': 'progress', **prog})}\n\n"

        try:
            zip_bytes = await future
            scan_duration = round(time.time() - scan_start, 1)

            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                matched = len(zf.namelist())

            if matched == 0:
                payload = {
                    "type": "done",
                    "search_id": "",
                    "matched": 0,
                    "duration": scan_duration,
                    "message": "No matching photos found. Try uploading more reference selfies or increasing the tolerance."
                }
            else:
                _zip_store[search_id] = {"data": zip_bytes, "created_at": time.time()}
                payload = {
                    "type": "done",
                    "search_id": search_id,
                    "matched": matched,
                    "duration": scan_duration,
                }

            yield f"data: {json.dumps(payload)}\n\n"
        except Exception as exc:
            logger.error("Search local failed: %s", exc)
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── ZIP download ──────────────────────────────────────────────────────────────
@app.get("/download/{search_id}")
async def download_zip(search_id: str, request: Request):
    auth.require_user(request)

    entry = _zip_store.get(search_id)
    if not entry:
        raise HTTPException(404, "ZIP not found. It may have expired (results expire after 30 minutes).")

    return Response(
        content=entry["data"],
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="eventai_matches_{search_id[:8]}.zip"'
        },
    )


# ── Dev server ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
