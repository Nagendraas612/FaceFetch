"""
main.py
FaceFetch – FastAPI entry point.

Endpoints:
  GET  /                         → Serve index.html
  GET  /privacy                  → Serve privacy.html (Privacy Policy & Biometric Consent)
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
import requests
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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
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
MAX_ZIP_SIZE                = 500 * 1024 * 1024  # 500 MB (ZIP bomb protection)
MAX_FILENAME_LENGTH         = 255
ALLOWED_MIME_PREFIXES       = (
    "image/jpeg", "image/png", "image/webp",
    "image/bmp", "image/tiff", "image/heic",
)
ZIP_TTL_SECONDS             = 1800               # 30 minutes
ZIP_MAX_STORE_MB            = 500
RATE_LIMIT_UPLOADS_PER_MIN  = 20
RATE_LIMIT_SCANS_PER_HOUR   = 100
RATE_LIMIT_DOWNLOADS_PER_HOUR = 50

# Tolerance bounds
TOLERANCE_MIN = 0.35
TOLERANCE_MAX = 0.70

# Security: Allowed origins (configure per deployment)
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# ── In-memory stores ─────────────────────────────────────────────────────────
_zip_store:        dict[str, dict]          = {}
_upload_rate:      dict[str, list[float]]   = defaultdict(list)
_scan_rate:        dict[str, list[float]]   = defaultdict(list)
_download_rate:    dict[str, list[float]]   = defaultdict(list)
_user_zip_access:  dict[str, str]           = {}  # search_id -> user_id mapping


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
        # Clean up access mapping
        if sid in _user_zip_access:
            del _user_zip_access[sid]
    if expired:
        logger.info("🗑 Cleaned up %d expired ZIP(s)", len(expired))


def _zip_store_size_mb() -> float:
    return sum(len(e["data"]) for e in _zip_store.values()) / (1024 * 1024)


def _extract_images_from_zip(zip_bytes: bytes) -> dict[str, bytes]:
    images = {}
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for name in zf.namelist():
                images[name] = zf.read(name)
    except Exception as e:
        logger.error("Error extracting images from ZIP: %s", e)
    return images



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


app = FastAPI(
    title="FaceFetch",
    version="2.3.0",
    lifespan=lifespan,
    docs_url=None if IS_PRODUCTION else "/docs",  # Disable docs in production
    redoc_url=None if IS_PRODUCTION else "/redoc",
)

# ── Security Middleware ───────────────────────────────────────────────────────
# CORS protection
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
    max_age=3600,
)

# Trusted host protection (prevent host header injection)
if IS_PRODUCTION:
    trusted_hosts = os.getenv("TRUSTED_HOSTS", "").split(",")
    if trusted_hosts and trusted_hosts[0]:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=trusted_hosts)

# Session middleware
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    session_cookie="facefetch_session",
    max_age=3600 * 8,
    same_site="lax",
    https_only=IS_PRODUCTION,
)

# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    if IS_PRODUCTION:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net https://fonts.googleapis.com https://apis.google.com; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.jsdelivr.net; "
            "img-src 'self' data: blob: https:; "
            "font-src 'self' https://fonts.gstatic.com; "
            "connect-src 'self' https://accounts.google.com https://oauth2.googleapis.com https://content.googleapis.com https://cdn.jsdelivr.net; "
            "frame-src 'self' https://docs.google.com https://accounts.google.com; "
            "worker-src 'self' blob:; "
            "frame-ancestors 'none'"
        )
    return response

app.include_router(auth.router)


# ── Static + HTML ─────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    try:
        return HTMLResponse((BASE_DIR / "index.html").read_text(encoding="utf-8"))
    except Exception as e:
        logger.error("Error loading index.html: %s", e)
        raise HTTPException(status_code=500, detail="index.html missing")


@app.get("/privacy", response_class=HTMLResponse)
async def privacy_policy(request: Request):
    try:
        return HTMLResponse((BASE_DIR / "privacy.html").read_text(encoding="utf-8"))
    except Exception as e:
        logger.error("Error loading privacy.html: %s", e)
        raise HTTPException(status_code=500, detail="privacy.html missing")



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
    request:      Request,
    file:         UploadFile = File(...),
    num_jitters:  int        = Form(5),
    model:        str        = Form("large"),
    profile_name: str        = Form("default"),
):
    user    = auth.require_user(request)
    user_id = user["sub"]

    _check_rate_limit(_upload_rate, user_id, RATE_LIMIT_UPLOADS_PER_MIN, 60)

    # Validate filename
    if not file.filename:
        raise HTTPException(400, "Filename is required")
    
    safe_filename = _sanitize_filename(file.filename)

    # Validate content type
    content_type = file.content_type or ""
    if not any(content_type.startswith(p) for p in ALLOWED_MIME_PREFIXES):
        raise HTTPException(
            415,
            f"Unsupported file type: {content_type}. Upload JPG, PNG, or WebP images.",
        )

    # Read and validate size
    image_bytes = await file.read()
    if len(image_bytes) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            413,
            f"File too large ({len(image_bytes) // (1024*1024)} MB). "
            f"Maximum is {MAX_UPLOAD_SIZE // (1024*1024)} MB.",
        )
    if len(image_bytes) == 0:
        raise HTTPException(400, "Empty file uploaded.")

    # Validate parameters
    if model not in ("small", "large"):
        raise HTTPException(400, "Invalid model parameter")
    if not isinstance(num_jitters, int) or num_jitters < 1 or num_jitters > 10:
        raise HTTPException(400, "num_jitters must be between 1 and 10")
    
    num_jitters = max(1, min(num_jitters, 10))

    # Check user hasn't exceeded reasonable encoding count per profile (prevent abuse)
    current_count = await database.get_encoding_count(user_id, profile_name)
    if current_count >= 50:  # Reasonable limit
        raise HTTPException(
            429,
            f"Maximum reference photos limit reached (50) for profile '{profile_name}'. Delete some before uploading more."
        )

    try:
        all_encodings = engine.encode_reference_image(
            image_bytes, num_jitters=num_jitters, model=model
        )
        if not all_encodings:
            raise HTTPException(
                400, "No face detected in this photo. Please upload a clear selfie."
            )

        for enc in all_encodings:
            await database.save_face_encoding(user_id, safe_filename, enc, profile_name)

        total = await database.get_encoding_count(user_id, profile_name)
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
async def my_encodings(request: Request, profile_name: str = "default"):
    user = auth.require_user(request)
    refs = await database.get_all_references(user["sub"], profile_name)
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


# ── Input validation & sanitization ──────────────────────────────────────────
def _sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal and other attacks."""
    if not filename:
        return "unnamed"
    
    # Remove path components
    filename = os.path.basename(filename)
    
    # Remove dangerous characters
    filename = re.sub(r'[^\w\s\-\.]', '_', filename)
    
    # Limit length
    if len(filename) > MAX_FILENAME_LENGTH:
        name, ext = os.path.splitext(filename)
        filename = name[:MAX_FILENAME_LENGTH - len(ext)] + ext
    
    return filename or "unnamed"


def _validate_scan_params(tolerance: float, model: str, upsample: int):
    """Validate and sanitize scan parameters."""
    # Validate tolerance (prevent algorithm abuse)
    if not isinstance(tolerance, (int, float)):
        raise HTTPException(400, "Invalid tolerance type")
    tolerance = max(TOLERANCE_MIN, min(TOLERANCE_MAX, float(tolerance)))
    
    # Validate model
    if model not in ("hog", "cnn"):
        raise HTTPException(400, "Invalid model type")
    
    # Validate upsample (prevent resource exhaustion)
    if not isinstance(upsample, int):
        raise HTTPException(400, "Invalid upsample type")
    upsample = max(0, min(2, int(upsample)))
    
    return tolerance, model, upsample


def _validate_drive_link(drive_link: str):
    """Validate Google Drive link to prevent injection attacks."""
    if not drive_link or not isinstance(drive_link, str):
        raise HTTPException(400, "Drive link is required.")
    
    # Limit length (prevent DoS)
    if len(drive_link) > 500:
        raise HTTPException(400, "Drive link too long.")
    
    # Validate format
    if not re.search(r"drive\.google\.com", drive_link):
        raise HTTPException(400, "Invalid Google Drive link.")
    
    # Prevent injection attempts
    if any(char in drive_link for char in ["<", ">", "\"", "'", ";", "(", ")", "{"]):
        raise HTTPException(400, "Invalid characters in Drive link.")


# ── Hybrid Architecture: Lightweight endpoints ───────────────────────────────

@app.post("/api/list-drive-files")
async def list_drive_files(
    request:    Request,
    drive_link: str = Form(...),
):
    """Return a JSON list of image file IDs/names from a Google Drive folder.
    The browser will use this list to download images client-side."""
    user    = auth.require_user(request)
    user_id = user["sub"]

    _check_rate_limit(_scan_rate, user_id, RATE_LIMIT_SCANS_PER_HOUR, 3600)
    _validate_drive_link(drive_link)

    drv_token     = request.session.get("drive_token", "")
    refresh_token = await database.get_refresh_token(user_id)

    if not drv_token and not refresh_token and not engine.GOOGLE_API_KEY:
        raise HTTPException(
            401,
            "No valid Google credentials. Log in with Google for private folders, "
            "or add GOOGLE_API_KEY to .env for public folders.",
        )

    try:
        token_state = {
            "access_token": drv_token,
            "refresh_token": refresh_token or "",
        }
        if (drv_token in ("", "mock_drive_token") and
            refresh_token in ("", "mock_refresh_token", None) and
            engine.GOOGLE_API_KEY):
            token_state["_use_api_key"] = True

        folder_id = engine._folder_id_from_link(drive_link)
        files = list(engine._list_drive_files(folder_id, token_state))
        return {"files": files, "total": len(files)}
    except ValueError as e:
        raise HTTPException(400, str(e))
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response is not None else 500
        detail = "Failed to list Drive folder files."
        if status_code == 404:
            detail = "Google Drive folder not found. Please verify the link and ensure it is shared publically or with your account."
        elif status_code == 403:
            detail = "Access denied to Google Drive folder. Please verify the folder's sharing permissions."
        elif status_code == 401:
            detail = "Your Google login session has expired. Please log out and log in again to refresh access."
        raise HTTPException(status_code=status_code, detail=detail)
    except Exception as e:
        logger.error("list-drive-files failed: %s", e)
        raise HTTPException(500, "Failed to list Drive folder files.")


@app.get("/api/proxy-image/{file_id}")
async def proxy_image(file_id: str, request: Request):
    """Proxy-download a single image from Google Drive for the browser.
    This avoids CORS issues when the browser tries to fetch Drive images directly."""
    user    = auth.require_user(request)
    user_id = user["sub"]

    # Validate file_id format (alphanumeric + dashes/underscores)
    if not re.match(r"^[a-zA-Z0-9_-]{10,80}$", file_id):
        raise HTTPException(400, "Invalid file ID format.")

    drv_token     = request.session.get("drive_token", "")
    refresh_token = await database.get_refresh_token(user_id)

    token_state = {
        "access_token": drv_token,
        "refresh_token": refresh_token or "",
    }
    if (drv_token in ("", "mock_drive_token") and
        refresh_token in ("", "mock_refresh_token", None) and
        engine.GOOGLE_API_KEY):
        token_state["_use_api_key"] = True

    img_bytes = await asyncio.get_running_loop().run_in_executor(
        None, engine._download_image_bytes, file_id, token_state
    )

    if not img_bytes:
        raise HTTPException(404, "Image not found or download failed.")

    return Response(
        content=img_bytes,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.post("/api/match-batch")
async def match_batch(
    request:      Request,
    files:        list[UploadFile] = File(...),
    tolerance:    float            = Form(0.50),
    model:        str              = Form("hog"),
    upsample:     int              = Form(1),
    profile_name: str              = Form("default"),
    search_id:    str              = Form(""),
):
    """Accept a batch of face images from the browser.
    Run dlib encoding + matching against the user's Master DNA.
    Returns matches and updates the accumulated search session ZIP."""
    user    = auth.require_user(request)
    user_id = user["sub"]

    tolerance, model, upsample = _validate_scan_params(tolerance, model, upsample)

    if len(files) > 50:
        raise HTTPException(400, "Too many files in batch. Maximum is 50.")

    known_encodings = await database.load_face_encodings(user_id, profile_name)
    if not known_encodings:
        raise HTTPException(400, f"No reference face found for profile '{profile_name}'. Upload a selfie first.")

    master_dna = engine.prepare_encodings(known_encodings)
    if not master_dna:
        raise HTTPException(400, "Could not prepare reference encodings.")

    matches = []
    match_images = {}

    for f in files:
        img_bytes = await f.read()
        if not img_bytes or len(img_bytes) > MAX_UPLOAD_SIZE:
            continue

        safe_name = _sanitize_filename(f.filename or "unknown.jpg")

        try:
            result_name, result_bytes, _ = await asyncio.get_running_loop().run_in_executor(
                None,
                engine._process_image_bytes,
                safe_name, img_bytes, master_dna, tolerance, model, upsample,
            )
            if result_bytes:
                matches.append(safe_name)
                match_images[safe_name] = result_bytes
        except Exception as exc:
            logger.warning("match-batch error for %s: %s", safe_name, exc)

    if not search_id:
        search_id = str(uuid.uuid4())
    else:
        # Validate format
        try:
            uuid.UUID(search_id)
        except ValueError:
            raise HTTPException(400, "Invalid search ID format.")

    if search_id in _zip_store:
        # Accumulate to existing
        existing_entry = _zip_store[search_id]
        if matches:
            existing_entry["images"].update(match_images)
            zip_data = engine._build_zip(list(existing_entry["images"].items()))
            existing_entry["data"] = zip_data
            existing_entry["created_at"] = time.time()
    else:
        # Initialize new
        if matches:
            zip_data = engine._build_zip(list(match_images.items()))
        else:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED):
                pass
            buf.seek(0)
            zip_data = buf.read()

        _zip_store[search_id] = {
            "data": zip_data,
            "images": match_images,
            "created_at": time.time(),
        }
        _user_zip_access[search_id] = user_id

    return {
        "matches": matches,
        "matched_count": len(matches),
        "search_id": search_id,
        "total_processed": len(files),
    }


# ── SSE helper ────────────────────────────────────────────────────────────────
def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


# ── Deep search (SSE streaming) ───────────────────────────────────────────────
@app.post("/search")
async def search(
    request:      Request,
    drive_link:   str   = Form(...),
    tolerance:    float = Form(0.50),
    model:        str   = Form("hog"),
    upsample:     int   = Form(1),
    profile_name: str   = Form("default"),
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

    known_encodings = await database.load_face_encodings(user_id, profile_name)
    if not known_encodings:
        raise HTTPException(400, f"No reference face found for profile '{profile_name}'. Upload a selfie first.")

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

        def sync_get_cache(file_id: str, modified_time: str):
            coro = database.get_cached_encodings(file_id, modified_time)
            fut = asyncio.run_coroutine_threadsafe(coro, loop)
            return fut.result()

        def sync_save_cache(file_id: str, modified_time: str, encs: list):
            coro = database.save_cached_encodings(file_id, modified_time, encs)
            fut = asyncio.run_coroutine_threadsafe(coro, loop)
            return fut.result()

        future = loop.run_in_executor(
            None,
            lambda: engine.run_deep_search(
                drive_link, drv_token, refresh_token,
                known_encodings, tolerance, model, upsample, progress_cb,
                sync_get_cache, sync_save_cache
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
                _zip_store[search_id] = {
                    "data": zip_bytes,
                    "images": _extract_images_from_zip(zip_bytes),
                    "created_at": time.time()
                }
                _user_zip_access[search_id] = user_id  # Track ownership for authorization
                logger.info(
                    "✓ ZIP stored: id=%s, size=%d bytes, files=%d, user=%s",
                    search_id, len(zip_bytes), matched, user_id,
                )
                filenames = list(_zip_store[search_id]["images"].keys())
                yield _sse({
                    "type":      "done",
                    "search_id": search_id,
                    "matched":   matched,
                    "duration":  scan_duration,
                    "filenames": filenames,
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
    request:      Request,
    files:        list[UploadFile] = File(...),
    tolerance:    float            = Form(0.50),
    model:        str              = Form("hog"),
    upsample:     int              = Form(1),
    profile_name: str              = Form("default"),
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

    known_encodings = await database.load_face_encodings(user_id, profile_name)
    if not known_encodings:
        raise HTTPException(400, f"No reference face found for profile '{profile_name}'. Upload a selfie first.")


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
                _zip_store[search_id] = {
                    "data": zip_bytes,
                    "images": _extract_images_from_zip(zip_bytes),
                    "created_at": time.time()
                }
                _user_zip_access[search_id] = user_id  # Track ownership for authorization
                logger.info(
                    "✓ ZIP stored: id=%s, size=%d bytes, files=%d, user=%s",
                    search_id, len(zip_bytes), matched, user_id,
                )
                filenames = list(_zip_store[search_id]["images"].keys())
                yield _sse({
                    "type":      "done",
                    "search_id": search_id,
                    "matched":   matched,
                    "duration":  scan_duration,
                    "filenames": filenames,
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
    """Download matched photos ZIP with security checks."""
    user = auth.require_user(request)
    user_id = user["sub"]
    
    # Rate limiting for downloads
    _check_rate_limit(_download_rate, user_id, RATE_LIMIT_DOWNLOADS_PER_HOUR, 3600)

    # Validate search_id format (prevent injection)
    try:
        uuid.UUID(search_id)
    except ValueError:
        raise HTTPException(400, "Invalid search ID format.")

    # Check if ZIP exists
    entry = _zip_store.get(search_id)
    if not entry:
        raise HTTPException(
            404,
            "ZIP not found or expired. Results are kept for 30 minutes after scanning.",
        )

    # Authorization check: verify user owns this search result
    if search_id in _user_zip_access:
        if _user_zip_access[search_id] != user_id:
            logger.warning(
                "⚠️ Unauthorized ZIP access attempt: user %s tried to access %s (owned by %s)",
                user_id, search_id, _user_zip_access[search_id]
            )
            raise HTTPException(403, "Access denied. This search belongs to another user.")

    zip_data = entry["data"]
    
    # ZIP bomb protection: check uncompressed size
    if len(zip_data) > MAX_ZIP_SIZE:
        logger.error("🚨 Potential ZIP bomb detected: %d bytes", len(zip_data))
        raise HTTPException(413, "ZIP file too large. This might indicate a problem.")
    
    filename = f"facefetch_matches_{search_id[:8]}.zip"
    # Sanitize filename for download
    safe_filename = _sanitize_filename(filename)

    return Response(
        content=zip_data,
        status_code=200,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_filename}"',
            "Content-Length":      str(len(zip_data)),
            "Content-Type":        "application/octet-stream",
            "Cache-Control":       "no-store, no-cache, must-revalidate, private",
            "Pragma":              "no-cache",
            "Expires":             "0",
        },
    )


# ── Matched results serving ───────────────────────────────────────────────────
@app.get("/result-image/{search_id}/{filename}")
async def get_result_image(search_id: str, filename: str, request: Request):
    """Serve a specific matched photo from the search run."""
    user = auth.require_user(request)
    user_id = user["sub"]

    try:
        uuid.UUID(search_id)
    except ValueError:
        raise HTTPException(400, "Invalid search ID format.")

    if search_id in _user_zip_access:
        if _user_zip_access[search_id] != user_id:
            raise HTTPException(403, "Access denied.")

    entry = _zip_store.get(search_id)
    if not entry or "images" not in entry:
        raise HTTPException(404, "Search results expired or not found.")

    img_bytes = entry["images"].get(filename)
    if not img_bytes:
        raise HTTPException(404, "Image not found in results.")

    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    media_type = "image/jpeg"
    if ext == ".png":
        media_type = "image/png"
    elif ext == ".webp":
        media_type = "image/webp"

    return Response(content=img_bytes, media_type=media_type)


@app.post("/download-selected/{search_id}")
async def download_selected(search_id: str, request: Request, payload: dict = None):
    """Download a subset of matched photos as a ZIP."""
    user = auth.require_user(request)
    user_id = user["sub"]

    try:
        uuid.UUID(search_id)
    except ValueError:
        raise HTTPException(400, "Invalid search ID format.")

    if search_id in _user_zip_access:
        if _user_zip_access[search_id] != user_id:
            raise HTTPException(403, "Access denied.")

    entry = _zip_store.get(search_id)
    if not entry or "images" not in entry:
        raise HTTPException(404, "Search results expired or not found.")

    filenames = (payload or {}).get("filenames", [])
    if not filenames:
        raise HTTPException(400, "No filenames provided.")

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for name in filenames:
            img_bytes = entry["images"].get(name)
            if img_bytes:
                zf.writestr(name, img_bytes)
                
    zip_buf.seek(0)
    custom_zip_data = zip_buf.read()

    filename = f"facefetch_selected_{search_id[:8]}.zip"
    safe_filename = _sanitize_filename(filename)

    return Response(
        content=custom_zip_data,
        status_code=200,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_filename}"',
            "Content-Length":      str(len(custom_zip_data)),
            "Content-Type":        "application/octet-stream",
            "Cache-Control":       "no-store, no-cache, must-revalidate, private",
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
