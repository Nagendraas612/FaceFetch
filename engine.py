"""
engine.py
Deep face-search engine:
  • Streams photos from Google Drive OR Local Uploads.
  • Auto-Refreshes expired Google tokens with retry logic.
  • Uses dynamic tolerance from the UI slider to prevent false positives.
  • Averages Face DNA for massive accuracy boosts.
  • Multi-scale face detection fallback.
"""

from __future__ import annotations

import io
import logging
import os
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Generator

import numpy as np
import requests
import face_recognition
from PIL import Image

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
DRIVE_API_BASE   = "https://www.googleapis.com/drive/v3"
DRIVE_LIST_URL   = f"{DRIVE_API_BASE}/files"
DRIVE_DL_URL     = f"{DRIVE_API_BASE}/files/{{file_id}}?alt=media"
PAGE_SIZE        = 100
MAX_WORKERS      = 6
SUPPORTED_EXTS   = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

# Reference image settings — higher quality for the "master DNA"
REF_MAX_SIZE     = 1024    # Larger for better encoding quality
REF_NUM_JITTERS  = 3       # More jitters = more accurate (but slower)

# Scan image settings — balance speed and detection
SCAN_MAX_SIZE    = 800     # Slightly larger than before for better detection
SCAN_QUALITY     = 85      # JPEG quality for matched photo output

# Token refresh settings
TOKEN_REFRESH_MAX_RETRIES = 3
TOKEN_REFRESH_BACKOFF     = 2  # seconds

# Drive API retry settings
DRIVE_API_MAX_RETRIES     = 3
DRIVE_API_BACKOFF         = 1  # seconds


# ── Reusable HTTP session for connection pooling ──────────────────────────────
_http_session = None

def _get_http_session() -> requests.Session:
    """Get a reusable requests session with connection pooling."""
    global _http_session
    if _http_session is None:
        _http_session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=MAX_WORKERS,
            pool_maxsize=MAX_WORKERS * 2,
            max_retries=0,  # We handle retries manually
        )
        _http_session.mount("https://", adapter)
    return _http_session


# ── Token Auto-Refresher with retry logic ─────────────────────────────────────
def _refresh_access_token(token_state: dict) -> bool:
    """
    Attempt to refresh the access token using the refresh token.
    Retries up to TOKEN_REFRESH_MAX_RETRIES times with exponential backoff.
    Returns True if successful, False otherwise.
    """
    refresh_token = token_state.get("refresh_token")
    if not refresh_token:
        return False

    client_id = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")

    if not client_id or not client_secret:
        logger.error("Cannot refresh token: GOOGLE_CLIENT_ID or GOOGLE_CLIENT_SECRET not set")
        return False

    url = "https://oauth2.googleapis.com/token"
    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }

    for attempt in range(1, TOKEN_REFRESH_MAX_RETRIES + 1):
        try:
            resp = requests.post(url, data=payload, timeout=10)
            if resp.status_code == 200:
                token_state["access_token"] = resp.json()["access_token"]
                logger.info("✅ Access token refreshed (attempt %d)", attempt)
                return True
            else:
                logger.warning(
                    "Token refresh attempt %d/%d failed: %s",
                    attempt, TOKEN_REFRESH_MAX_RETRIES, resp.text[:200]
                )
        except requests.RequestException as e:
            logger.warning("Token refresh attempt %d/%d error: %s", attempt, TOKEN_REFRESH_MAX_RETRIES, e)

        if attempt < TOKEN_REFRESH_MAX_RETRIES:
            backoff = TOKEN_REFRESH_BACKOFF * (2 ** (attempt - 1))
            time.sleep(backoff)

    logger.error("Failed to refresh access token after %d attempts", TOKEN_REFRESH_MAX_RETRIES)
    return False


def _make_drive_request(url: str, token_state: dict, **kwargs) -> requests.Response:
    """
    Make a Drive API request with automatic token refresh and retry on 429/5xx.
    """
    session = _get_http_session()

    for attempt in range(1, DRIVE_API_MAX_RETRIES + 1):
        headers = {"Authorization": f"Bearer {token_state['access_token']}"}
        try:
            resp = session.get(url, headers=headers, **kwargs)

            # Token expired → refresh and retry
            if resp.status_code == 401:
                if _refresh_access_token(token_state):
                    continue
                resp.raise_for_status()

            # Rate limited or server error → backoff and retry
            if resp.status_code in (429, 500, 502, 503):
                if attempt < DRIVE_API_MAX_RETRIES:
                    backoff = DRIVE_API_BACKOFF * (2 ** (attempt - 1))
                    logger.warning(
                        "Drive API %d, retrying in %ds (attempt %d/%d)",
                        resp.status_code, backoff, attempt, DRIVE_API_MAX_RETRIES
                    )
                    time.sleep(backoff)
                    continue

            return resp

        except requests.RequestException as e:
            if attempt < DRIVE_API_MAX_RETRIES:
                backoff = DRIVE_API_BACKOFF * (2 ** (attempt - 1))
                logger.warning("Drive request error, retrying in %ds: %s", backoff, e)
                time.sleep(backoff)
            else:
                raise

    return resp


# ── Google Drive helpers ──────────────────────────────────────────────────────
def _folder_id_from_link(drive_link: str) -> str:
    """Extract folder ID from a Google Drive link."""
    patterns = [
        r"/folders/([a-zA-Z0-9_-]{10,})",
        r"[?&]id=([a-zA-Z0-9_-]{10,})",
    ]
    for pat in patterns:
        m = re.search(pat, drive_link)
        if m:
            return m.group(1)
    raise ValueError(f"Cannot extract folder ID from link: {drive_link!r}")


def _list_drive_files(folder_id: str, token_state: dict) -> Generator[dict, None, None]:
    """List all image files in a Drive folder (paginated)."""
    params = {
        "q": f"'{folder_id}' in parents and mimeType contains 'image/' and trashed = false",
        "fields": "nextPageToken, files(id, name)",
        "pageSize": PAGE_SIZE,
    }
    while True:
        resp = _make_drive_request(DRIVE_LIST_URL, token_state, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        for f in data.get("files", []):
            ext = "." + f["name"].rsplit(".", 1)[-1].lower() if "." in f["name"] else ""
            if ext in SUPPORTED_EXTS:
                yield f

        page_token = data.get("nextPageToken")
        if not page_token:
            break
        params["pageToken"] = page_token


def _download_image_bytes(file_id: str, token_state: dict) -> bytes | None:
    """Download image bytes from Google Drive."""
    url = DRIVE_DL_URL.format(file_id=file_id)
    try:
        resp = _make_drive_request(url, token_state, timeout=20, stream=True)
        resp.raise_for_status()
        return resp.content
    except Exception as exc:
        logger.warning("Failed to download file_id=%s: %s", file_id, exc)
        return None


# ── Face-encoding helpers ─────────────────────────────────────────────────────
def encode_reference_image(
    image_bytes: bytes,
    num_jitters: int = REF_NUM_JITTERS,
    model: str = "large"
) -> list[list[float]]:
    """
    Encode reference image with high accuracy settings.

    Uses multi-scale detection: if no face found at default size,
    retries with upsampling for smaller/distant faces.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Resize for quality reference encoding
    if max(img.size) > REF_MAX_SIZE:
        img.thumbnail((REF_MAX_SIZE, REF_MAX_SIZE), Image.Resampling.LANCZOS)

    arr = np.array(img)

    # Try standard detection first (HOG, fast)
    locations = face_recognition.face_locations(arr, model="hog")

    # Multi-scale fallback: if no face found, try with upsampling
    if not locations:
        logger.info("No face at default scale, retrying with upsample=1...")
        locations = face_recognition.face_locations(arr, number_of_times_to_upsample=2, model="hog")

    # Final fallback: try CNN (slower but more robust)
    if not locations:
        logger.info("Still no face, trying CNN model as last resort...")
        try:
            locations = face_recognition.face_locations(arr, model="cnn")
        except Exception:
            pass  # CNN may not be available on all systems

    if not locations:
        raise ValueError("No face detected in the reference photo. Please upload a clear selfie with your face visible.")

    encodings = face_recognition.face_encodings(
        arr,
        known_face_locations=locations,
        num_jitters=num_jitters,
        model=model,
    )

    if not encodings:
        raise ValueError("Face was detected but encoding failed. Please try a different photo.")

    img.close()
    return [enc.tolist() for enc in encodings]


def prepare_encodings(known_encodings: list) -> list[np.ndarray]:
    """
    Prepare reference encodings for matching.

    If multiple references exist, computes an averaged "Master DNA" embedding
    which significantly boosts accuracy by reducing noise from individual photos.
    """
    np_encodings = [
        np.array(enc) if not isinstance(enc, np.ndarray) else enc
        for enc in known_encodings
    ]

    if len(np_encodings) == 0:
        return []

    if len(np_encodings) > 1:
        # Average all encodings into a single master embedding
        avg_encoding = np.mean(np_encodings, axis=0)
        # Normalize the averaged vector for more consistent distance calculations
        norm = np.linalg.norm(avg_encoding)
        if norm > 0:
            avg_encoding = avg_encoding / norm * np.linalg.norm(np_encodings[0])
        return [avg_encoding]

    return np_encodings


# ── Core Image Processor ─────────────────────────────────────────────────────
def _process_image_bytes(
    filename: str,
    img_bytes: bytes,
    known_encodings: list[np.ndarray],
    tolerance: float,
    model_type: str,
    upsample: int,
) -> tuple[str, bytes | None]:
    """
    Process a single image: detect faces, compare with known encodings.
    Returns (filename, matched_image_bytes_or_None).
    """
    try:
        # 1. Load and resize image
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        if max(img.size) > SCAN_MAX_SIZE:
            img.thumbnail((SCAN_MAX_SIZE, SCAN_MAX_SIZE), Image.Resampling.LANCZOS)

        img_arr = np.array(img)

        # 2. Detect faces
        locations = face_recognition.face_locations(
            img_arr,
            number_of_times_to_upsample=upsample,
            model=model_type,
        )

        if not locations:
            del img_arr
            img.close()
            return filename, None

        # 3. Encode all detected faces
        candidate_encodings = face_recognition.face_encodings(
            img_arr,
            known_face_locations=locations,
            model="large",
        )

        # 4. Compare each candidate face against known encodings
        found_match = False
        best_distance = float('inf')

        for candidate in candidate_encodings:
            distances = face_recognition.face_distance(known_encodings, candidate)
            min_dist = np.min(distances)
            best_distance = min(best_distance, min_dist)

            if min_dist < tolerance:
                found_match = True
                break

        # 5. Build result
        result_bytes = None
        if found_match:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=SCAN_QUALITY)
            result_bytes = buf.getvalue()
            logger.debug("✓ Match: %s (distance: %.4f)", filename, best_distance)

        # Memory cleanup
        del img_arr
        img.close()

        return filename, result_bytes

    except Exception as exc:
        logger.warning("Error processing %s: %s", filename, exc)
        return filename, None


def _process_drive_file(file_meta, token_state, known_encodings, tolerance, model_type, upsample):
    """Download and process a single Drive file."""
    img_bytes = _download_image_bytes(file_meta["id"], token_state)
    if not img_bytes:
        return file_meta["name"], None
    return _process_image_bytes(
        file_meta["name"], img_bytes, known_encodings, tolerance, model_type, upsample
    )


# ── Public APIs ──────────────────────────────────────────────────────────────
def run_deep_search(
    drive_link: str,
    access_token: str,
    refresh_token: str,
    known_encodings: list,
    tolerance: float,
    model_type: str,
    upsample: int,
    progress_callback=None,
) -> bytes:
    """Search for matching faces in a Google Drive folder."""

    token_state = {
        "access_token": access_token,
        "refresh_token": refresh_token,
    }

    folder_id = _folder_id_from_link(drive_link)
    master_dna = prepare_encodings(known_encodings)

    if not master_dna:
        raise ValueError("No valid face encodings found. Please upload reference photos.")

    all_files = list(_list_drive_files(folder_id, token_state))
    total = len(all_files)
    if total == 0:
        raise ValueError("No supported images found in Drive folder. Make sure the folder is shared and contains image files.")

    logger.info("Starting Drive scan: %d files, tolerance=%.2f, model=%s", total, tolerance, model_type)
    if progress_callback:
        progress_callback(0, total, "Starting scan engine...", 0)

    matched: list[tuple[str, bytes]] = []
    processed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_map = {
            pool.submit(
                _process_drive_file, f, token_state, master_dna,
                tolerance, model_type, upsample
            ): f
            for f in all_files
        }
        for future in as_completed(future_map):
            processed += 1
            try:
                filename, img_bytes = future.result()
                if img_bytes:
                    matched.append((filename, img_bytes))
                if progress_callback:
                    progress_callback(processed, total, filename, len(matched))
            except Exception as exc:
                logger.error("Unexpected worker error: %s", exc)

    logger.info("Drive scan complete: %d/%d matched", len(matched), total)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in matched:
            zf.writestr(name, data)
    zip_buf.seek(0)
    return zip_buf.read()


def run_local_search(
    files_data: list[tuple[str, bytes]],
    known_encodings: list,
    tolerance: float,
    model_type: str,
    upsample: int,
    progress_callback=None,
) -> bytes:
    """Search for matching faces in locally uploaded files."""

    master_dna = prepare_encodings(known_encodings)

    if not master_dna:
        raise ValueError("No valid face encodings found. Please upload reference photos.")

    total = len(files_data)
    logger.info("Starting local scan: %d files, tolerance=%.2f, model=%s", total, tolerance, model_type)
    if progress_callback:
        progress_callback(0, total, "Starting local scan...", 0)

    matched = []
    processed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_map = {
            pool.submit(
                _process_image_bytes, fname, fdata, master_dna,
                tolerance, model_type, upsample
            ): fname
            for fname, fdata in files_data
        }
        for future in as_completed(future_map):
            processed += 1
            try:
                filename, out_bytes = future.result()
                if out_bytes:
                    matched.append((filename, out_bytes))
                if progress_callback:
                    progress_callback(processed, total, filename, len(matched))
            except Exception as exc:
                logger.error("Unexpected worker error: %s", exc)

    logger.info("Local scan complete: %d/%d matched", len(matched), total)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in matched:
            zf.writestr(name, data)
    zip_buf.seek(0)
    return zip_buf.read()
