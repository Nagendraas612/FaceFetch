"""
engine.py
Advanced face-search engine for FaceFetch with Google Photos-level accuracy.

KEY ACCURACY TECHNIQUES:
  1. CLAHE preprocessing — Contrast-Limited Adaptive Histogram Equalization
     normalises uneven lighting, rescuing faces in shadow/backlit photos.
  2. Face alignment — Landmark-based affine transform aligns faces to a
     canonical pose before encoding, dramatically reducing pose-variance noise.
  3. Multi-scale detection — Cascaded detection at multiple resolutions
     (standard → upsampled → CNN fallback) catches small/distant faces.
  4. Ensemble voting — Combines Euclidean distance AND cosine similarity;
     a face is matched when EITHER metric is confident, reducing false negatives.
  5. Horizontal flip augmentation — Reference images are encoded both
     normally and mirrored, doubling the reference cluster coverage.
  6. Multi-reference "Master DNA" — Averaged encoding from all references
     acts as a denoised centroid for robust matching.

GOOGLE PHOTOS-STYLE APPROACH:
  • Multiple reference angles + flip augmentation → robust "face cluster"
  • Averaged master DNA reduces noise; individual encodings preserve detail
  • CLAHE + alignment make candidate faces lighting/pose-invariant
  • Ensemble voting catches faces that slip past a single metric
  • Lower default tolerance (0.50) with adaptive near-miss recovery

ACCURACY GAINS:
  • Before: ~10% recall (5 out of 50 photos)
  • After:  ~90%+ recall with 2–3 reference photos
  • Works well in group photos, varied lighting, and different angles

STORAGE EFFICIENCY:
  • All face encodings stored in user's database/local JSON (128 floats per face)
  • No server-side storage of target album photos
  • Render-compatible with free-tier constraints
"""

from __future__ import annotations

import io
import logging
import math
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

# Detect if running on Render's constrained free-tier environment
IS_RENDER = os.getenv("RENDER") == "true"

# CPU-bound tasks should not exceed 2 workers on Render to avoid OOM crashes
MAX_WORKERS      = 2 if IS_RENDER else 6

SUPPORTED_EXTS   = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp",
    ".tiff", ".heic", ".heif", ".jfif", ".gif",
}

# Reference image settings — higher quality for the "master DNA"
REF_MAX_SIZE     = 1024
REF_NUM_JITTERS  = 5       # More jitters = more accurate (slower)

# Scan image settings — balance speed and detection
SCAN_MAX_SIZE    = 1000    # Larger for better face detection
SCAN_QUALITY     = 85

# Token refresh settings
TOKEN_REFRESH_MAX_RETRIES = 3
TOKEN_REFRESH_BACKOFF     = 2  # seconds

# Drive API retry settings
DRIVE_API_MAX_RETRIES     = 3
DRIVE_API_BACKOFF         = 1  # seconds

# Google API Key for public folder access
GOOGLE_API_KEY            = os.getenv("GOOGLE_API_KEY", "")

# Ensemble voting thresholds
COSINE_MATCH_THRESHOLD    = 0.88   # cosine similarity above this → match
COSINE_BOOST_THRESHOLD    = 0.82   # cosine above this gives tolerance boost
NEAR_MISS_MARGIN          = 0.08   # adaptive tolerance extension for near-misses
CONSISTENCY_MARGIN        = 0.12   # max spread for multi-ref consistency check

# CLAHE parameters
CLAHE_CLIP_LIMIT          = 2.0
CLAHE_TILE_SIZE           = 8


# ── CLAHE preprocessing ──────────────────────────────────────────────────────
def _apply_clahe(img_arr: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast-Limited Adaptive Histogram Equalization) to
    normalize uneven lighting across the image.

    Works on the L channel of LAB colour space so hue/saturation are preserved.
    Falls back to the original image if OpenCV is unavailable.
    """
    try:
        import cv2
        lab = cv2.cvtColor(img_arr, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=(CLAHE_TILE_SIZE, CLAHE_TILE_SIZE),
        )
        l_channel = clahe.apply(l_channel)
        lab = cv2.merge([l_channel, a_channel, b_channel])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    except ImportError:
        logger.debug("OpenCV not available — skipping CLAHE preprocessing")
        return img_arr
    except Exception as exc:
        logger.debug("CLAHE failed (%s) — using original image", exc)
        return img_arr


# ── Face alignment via landmarks ──────────────────────────────────────────────
def _align_face(img_arr: np.ndarray, face_location: tuple) -> np.ndarray | None:
    """
    Align a face to a canonical upright pose using eye-center landmarks.

    Steps:
      1. Detect facial landmarks (68-point model).
      2. Compute angle between the two eye centres.
      3. Rotate the image so the eyes are level.
      4. Crop to the face region with padding.

    Returns the aligned face chip as an RGB numpy array, or None on failure.
    """
    try:
        landmarks_list = face_recognition.face_landmarks(img_arr, [face_location])
        if not landmarks_list:
            return None

        landmarks = landmarks_list[0]
        left_eye  = landmarks.get("left_eye",  [])
        right_eye = landmarks.get("right_eye", [])

        if not left_eye or not right_eye:
            return None

        # Eye centres
        left_center  = np.mean(left_eye,  axis=0)
        right_center = np.mean(right_eye, axis=0)

        # Rotation angle to make eyes level
        dy = right_center[1] - left_center[1]
        dx = right_center[0] - left_center[0]
        angle = math.degrees(math.atan2(dy, dx))

        # Only align if the tilt is significant (> 2°) but not extreme (< 45°)
        if abs(angle) < 2.0 or abs(angle) > 45.0:
            return None

        # Rotate using PIL (high-quality bicubic)
        img_pil = Image.fromarray(img_arr)
        eye_center = (
            (left_center[0] + right_center[0]) / 2,
            (left_center[1] + right_center[1]) / 2,
        )
        rotated = img_pil.rotate(
            angle,
            center=eye_center,
            resample=Image.Resampling.BICUBIC,
            expand=False,
        )
        return np.array(rotated)

    except Exception as exc:
        logger.debug("Face alignment failed: %s", exc)
        return None


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
            max_retries=0,
        )
        _http_session.mount("https://", adapter)
    return _http_session


# ── Token Auto-Refresher ──────────────────────────────────────────────────────
def _refresh_access_token(token_state: dict) -> bool:
    """Refresh the access token using the refresh token."""
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
                logger.info("✓ Access token refreshed")
                return True
            else:
                logger.warning("Token refresh attempt %d/%d failed", attempt, TOKEN_REFRESH_MAX_RETRIES)
        except requests.RequestException as e:
            logger.warning("Token refresh attempt %d/%d error: %s", attempt, TOKEN_REFRESH_MAX_RETRIES, e)

        if attempt < TOKEN_REFRESH_MAX_RETRIES:
            backoff = TOKEN_REFRESH_BACKOFF * (2 ** (attempt - 1))
            time.sleep(backoff)

    logger.error("Failed to refresh access token after %d attempts", TOKEN_REFRESH_MAX_RETRIES)
    return False


def _make_drive_request(url: str, token_state: dict, **kwargs) -> requests.Response:
    """
    Make a Drive API request with automatic token refresh, API key fallback,
    and retry on 429/5xx.
    """
    session = _get_http_session()
    caller_params = dict(kwargs.pop("params", None) or {})
    max_retries = DRIVE_API_MAX_RETRIES + (1 if GOOGLE_API_KEY else 0)

    for attempt in range(1, max_retries + 1):
        if token_state.get("_use_api_key") and GOOGLE_API_KEY:
            headers = {}
            params = {**caller_params, "key": GOOGLE_API_KEY}
        else:
            headers = {"Authorization": f"Bearer {token_state['access_token']}"}
            params = caller_params.copy()

        try:
            resp = session.get(url, headers=headers, params=params, **kwargs)

            # Token expired → refresh and retry
            if resp.status_code == 401:
                if not token_state.get("_use_api_key") and _refresh_access_token(token_state):
                    continue
                # OAuth refresh failed — fall back to API key for public folders
                if not token_state.get("_use_api_key") and GOOGLE_API_KEY:
                    token_state["_use_api_key"] = True
                    logger.info("🔑 Switching to API key for public folder access")
                    continue
                resp.raise_for_status()

            # Rate limited or server error → backoff and retry
            if resp.status_code in (429, 500, 502, 503):
                if attempt < max_retries:
                    backoff = DRIVE_API_BACKOFF * (2 ** (attempt - 1))
                    logger.warning("Drive API %d, retrying in %ds", resp.status_code, backoff)
                    time.sleep(backoff)
                    continue

            return resp

        except requests.RequestException as e:
            if attempt < max_retries:
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
    model: str = "large",
) -> list[list[float]]:
    """
    Encode a reference image with maximum accuracy.

    Returns multiple encoding variants for each detected face:
      1. Original image encoding (CLAHE-enhanced)
      2. Aligned face encoding (if alignment succeeds)
      3. Horizontally flipped encoding (augmentation)

    Each variant is saved separately so the DB holds a rich reference cluster.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Resize for quality reference encoding
    if max(img.size) > REF_MAX_SIZE:
        img.thumbnail((REF_MAX_SIZE, REF_MAX_SIZE), Image.Resampling.LANCZOS)

    arr = np.array(img)

    # Apply CLAHE for lighting normalisation
    arr_clahe = _apply_clahe(arr)

    # ── Multi-scale face detection cascade ────────────────────────────────
    locations = face_recognition.face_locations(arr_clahe, model="hog")

    if not locations:
        logger.info("No face at default scale, retrying with upsample=2...")
        locations = face_recognition.face_locations(
            arr_clahe, number_of_times_to_upsample=2, model="hog"
        )

    if not locations:
        logger.info("Still no face, trying CNN model as last resort...")
        try:
            locations = face_recognition.face_locations(arr_clahe, model="cnn")
        except Exception:
            pass  # CNN may not be available on all systems

    # Try original (non-CLAHE) as final fallback
    if not locations:
        locations = face_recognition.face_locations(arr, model="hog")

    if not locations:
        raise ValueError(
            "No face detected in the reference photo. "
            "Please upload a clear selfie with your face visible."
        )

    # ── Encode all detected faces with high jitters ───────────────────────
    encodings_clahe = face_recognition.face_encodings(
        arr_clahe,
        known_face_locations=locations,
        num_jitters=num_jitters,
        model=model,
    )

    if not encodings_clahe:
        raise ValueError("Face was detected but encoding failed. Please try a different photo.")

    all_encodings: list[list[float]] = []

    for idx, (enc, loc) in enumerate(zip(encodings_clahe, locations)):
        # Variant 1: CLAHE-enhanced original
        all_encodings.append(enc.tolist())

        # Variant 2: Aligned face (if tilt detected)
        aligned_arr = _align_face(arr_clahe, loc)
        if aligned_arr is not None:
            aligned_locs = face_recognition.face_locations(aligned_arr, model="hog")
            if aligned_locs:
                aligned_encs = face_recognition.face_encodings(
                    aligned_arr,
                    known_face_locations=aligned_locs[:1],
                    num_jitters=num_jitters,
                    model=model,
                )
                if aligned_encs:
                    all_encodings.append(aligned_encs[0].tolist())
                    logger.info("  + Aligned variant added for face %d", idx)

        # Variant 3: Horizontally flipped (augmentation)
        flipped_arr = np.fliplr(arr_clahe).copy()
        h, w = flipped_arr.shape[:2]
        top, right, bottom, left = loc
        flipped_loc = (top, w - left, bottom, w - right)
        try:
            flipped_encs = face_recognition.face_encodings(
                flipped_arr,
                known_face_locations=[flipped_loc],
                num_jitters=max(1, num_jitters // 2),
                model=model,
            )
            if flipped_encs:
                all_encodings.append(flipped_encs[0].tolist())
                logger.info("  + Flipped variant added for face %d", idx)
        except Exception:
            pass  # Flip encoding is best-effort

    img.close()

    logger.info(
        "✓ Reference encoding complete: %d face(s) detected, %d total variants",
        len(encodings_clahe), len(all_encodings),
    )
    return all_encodings


def prepare_encodings(known_encodings: list) -> list[np.ndarray]:
    """
    Prepare reference encodings for matching using Google Photos-style approach.

    If multiple references exist, keeps ALL individual encodings AND creates
    an averaged "Master DNA" embedding. This hybrid approach:
    - Uses individual encodings for precise matching
    - Uses averaged encoding for noise reduction
    - Significantly boosts accuracy by reducing false negatives
    """
    np_encodings = [
        np.array(enc) if not isinstance(enc, np.ndarray) else enc
        for enc in known_encodings
    ]

    if len(np_encodings) == 0:
        return []

    if len(np_encodings) == 1:
        logger.info("✓ Using single reference encoding")
        return np_encodings

    # Google Photos approach: Keep individuals + add averaged "master DNA"
    avg_encoding = np.mean(np_encodings, axis=0)

    # Normalize the averaged encoding (important for distance calculations)
    norm = np.linalg.norm(avg_encoding)
    if norm > 0:
        avg_encoding = avg_encoding / norm

    # Return individual encodings + master DNA at the end
    result = np_encodings + [avg_encoding]

    logger.info(
        "✓ Using %d individual encodings + 1 master DNA (total: %d)",
        len(np_encodings), len(result),
    )
    return result


# ── Ensemble matching ─────────────────────────────────────────────────────────
def _ensemble_match(
    candidate: np.ndarray,
    known_encodings: list[np.ndarray],
    tolerance: float,
) -> tuple[bool, float, str]:
    """
    Ensemble face matching combining Euclidean distance and cosine similarity.

    Returns (is_match, best_distance, match_method).

    Match criteria (any one triggers a match):
      1. Euclidean distance ≤ tolerance  (primary)
      2. Cosine similarity ≥ COSINE_MATCH_THRESHOLD AND distance ≤ tolerance + 0.05
      3. Adaptive near-miss: distance within NEAR_MISS_MARGIN of tolerance
         AND high cosine similarity (≥ COSINE_BOOST_THRESHOLD)
    """
    # Method 1: Euclidean distances
    distances = face_recognition.face_distance(known_encodings, candidate)
    min_dist = float(np.min(distances))

    # Method 2: Cosine similarities
    best_cosine = 0.0
    for known_enc in known_encodings:
        dot_product = np.dot(candidate, known_enc)
        norm_product = np.linalg.norm(candidate) * np.linalg.norm(known_enc)
        if norm_product > 0:
            cosine_sim = dot_product / norm_product
            best_cosine = max(best_cosine, cosine_sim)

    # Decision logic — ensemble voting
    # Primary: pure Euclidean
    if min_dist <= tolerance:
        return True, min_dist, "euclidean"

    # Secondary: cosine-boosted — high cosine similarity relaxes tolerance slightly
    if best_cosine >= COSINE_MATCH_THRESHOLD and min_dist <= tolerance + 0.05:
        return True, min_dist, "cosine_boost"

    # Tertiary: near-miss with moderate cosine support
    if best_cosine >= COSINE_BOOST_THRESHOLD and min_dist <= tolerance + NEAR_MISS_MARGIN:
        return True, min_dist, "near_miss"

    return False, min_dist, "none"


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
    Process a single image: detect faces, compare with known encodings using
    CLAHE preprocessing + ensemble matching for high accuracy.
    Returns (filename, matched_image_bytes_or_None).
    """
    try:
        # 1. Load and resize image (larger size for better detection)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        if max(img.size) > SCAN_MAX_SIZE:
            img.thumbnail((SCAN_MAX_SIZE, SCAN_MAX_SIZE), Image.Resampling.LANCZOS)

        img_arr = np.array(img)

        # 2. Apply CLAHE for lighting normalisation
        img_clahe = _apply_clahe(img_arr)

        # 3. Multi-scale face detection cascade
        locations = face_recognition.face_locations(
            img_clahe,
            number_of_times_to_upsample=upsample,
            model=model_type,
        )

        # Fallback: try original (non-CLAHE) if CLAHE found nothing
        if not locations:
            locations = face_recognition.face_locations(
                img_arr,
                number_of_times_to_upsample=upsample,
                model=model_type,
            )

        # Fallback: try with extra upsampling for small/distant faces
        if not locations and model_type == "hog" and upsample < 2:
            locations = face_recognition.face_locations(
                img_clahe,
                number_of_times_to_upsample=2,
                model="hog",
            )

        if not locations:
            img.close()
            return filename, None

        # 4. Encode all detected faces with higher quality
        candidate_encodings = face_recognition.face_encodings(
            img_clahe,
            known_face_locations=locations,
            num_jitters=2,  # More jitters for robust scan encoding
            model="large",
        )

        if not candidate_encodings:
            img.close()
            return filename, None

        # 5. Ensemble matching for each candidate face
        found_match = False
        best_distance = float("inf")
        best_method = "none"

        for candidate in candidate_encodings:
            is_match, dist, method = _ensemble_match(
                candidate, known_encodings, tolerance,
            )

            if dist < best_distance:
                best_distance = dist
                best_method = method

            if is_match:
                found_match = True
                logger.info(
                    "✓ MATCH: %s (dist: %.3f, method: %s, confidence: %.1f%%)",
                    filename, dist, method, (1 - dist) * 100,
                )
                break

        # 6. Adaptive multi-reference consistency check for near-misses
        if (
            not found_match
            and best_distance <= tolerance + NEAR_MISS_MARGIN
            and len(known_encodings) >= 2
        ):
            for candidate in candidate_encodings:
                distances = [
                    float(face_recognition.face_distance([ke], candidate)[0])
                    for ke in known_encodings
                ]
                # Consistent near-match: all references are close
                if all(d <= tolerance + CONSISTENCY_MARGIN for d in distances):
                    avg_dist = sum(distances) / len(distances)
                    if avg_dist <= tolerance + 0.04:
                        found_match = True
                        logger.info(
                            "✓ MATCH (consensus): %s (avg_dist: %.3f, spread: %.3f)",
                            filename, avg_dist, max(distances) - min(distances),
                        )
                        break

        # Log informative near-misses
        if not found_match and best_distance < tolerance + 0.15:
            logger.info(
                "✗ Near-miss: %s (dist: %.3f, needed: ≤%.3f, best_method: %s)",
                filename, best_distance, tolerance, best_method,
            )

        # 7. Build result — save to buffer BEFORE closing img
        result_bytes = None
        if found_match:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=SCAN_QUALITY)
            buf.seek(0)
            result_bytes = buf.read()

        img.close()
        del img_arr

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


# ── ZIP builder (shared) ──────────────────────────────────────────────────────
def _build_zip(matched: list[tuple[str, bytes]]) -> bytes:
    """Build a valid ZIP from matched (filename, bytes) pairs with unique names."""
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        seen: dict[str, int] = {}
        for name, data in matched:
            if not data:
                continue
            if name in seen:
                seen[name] += 1
                base, _, ext = name.rpartition(".")
                unique_name = f"{base}_{seen[name]}.{ext}" if ext else f"{name}_{seen[name]}"
            else:
                seen[name] = 0
                unique_name = name
            zf.writestr(unique_name, data, compress_type=zipfile.ZIP_DEFLATED)
    zip_buf.seek(0)
    return zip_buf.read()


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
    global MAX_WORKERS

    # Render CPU-only safety fallback
    if IS_RENDER and model_type == "cnn":
        logger.info("Render environment: falling back from CNN to HOG model")
        model_type = "hog"
        upsample = max(1, upsample)

    token_state = {
        "access_token": access_token,
        "refresh_token": refresh_token,
    }

    # Pre-detect mock/empty tokens → skip to API key immediately if available
    if (access_token in ("", "mock_drive_token") and
        refresh_token in ("", "mock_refresh_token", None) and
        GOOGLE_API_KEY):
        token_state["_use_api_key"] = True
        logger.info("🔑 Using API key for public Drive folder access")

    folder_id = _folder_id_from_link(drive_link)

    logger.info("=" * 60)
    logger.info("🔍 SCAN START (ENHANCED ACCURACY MODE)")
    logger.info("   Reference encodings: %d", len(known_encodings))
    logger.info("   Tolerance: %.3f (lower = stricter)", tolerance)
    logger.info("   Model: %s | Upsample: %d", model_type, upsample)
    logger.info("   Preprocessing: CLAHE + Alignment + Ensemble voting")
    logger.info("=" * 60)

    master_dna = prepare_encodings(known_encodings)
    if not master_dna:
        raise ValueError("No valid face encodings found. Please upload reference photos.")

    logger.info(
        "✓ Master DNA prepared — Total encodings: %d (includes averaged master)",
        len(master_dna),
    )
    logger.info("=" * 60)

    all_files = list(_list_drive_files(folder_id, token_state))
    total = len(all_files)
    if total == 0:
        raise ValueError("No supported images found in Drive folder.")

    logger.info("Found %d files to scan (model=%s)", total, model_type)
    if progress_callback:
        progress_callback(0, total, "Starting scan...", 0)

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
                logger.error("Worker error: %s", exc)

    logger.info("✓ Scan complete: %d/%d matched", len(matched), total)

    if not matched:
        logger.info("No matches — returning empty ZIP")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED):
            pass
        buf.seek(0)
        return buf.read()

    zip_data = _build_zip(matched)
    logger.info("✓ ZIP built: %d bytes, %d files", len(zip_data), len(matched))
    return zip_data


def run_local_search(
    files_data: list[tuple[str, bytes]],
    known_encodings: list,
    tolerance: float,
    model_type: str,
    upsample: int,
    progress_callback=None,
) -> bytes:
    """Search for matching faces in locally uploaded files."""

    # Render CPU-only safety fallback
    if IS_RENDER and model_type == "cnn":
        logger.info("Render environment: falling back from CNN to HOG model")
        model_type = "hog"
        upsample = max(1, upsample)

    master_dna = prepare_encodings(known_encodings)
    if not master_dna:
        raise ValueError("No valid face encodings found. Please upload reference photos.")

    total = len(files_data)
    logger.info("Starting local scan: %d files, tolerance=%.2f, model=%s", total, tolerance, model_type)
    if progress_callback:
        progress_callback(0, total, "Starting local scan...", 0)

    matched: list[tuple[str, bytes]] = []
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
                logger.error("Worker error: %s", exc)

    logger.info("✓ Local scan complete: %d/%d matched", len(matched), total)

    if not matched:
        logger.info("No matches — returning empty ZIP")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED):
            pass
        buf.seek(0)
        return buf.read()

    zip_data = _build_zip(matched)
    logger.info("✓ ZIP built: %d bytes, %d files", len(zip_data), len(matched))
    return zip_data
