"""
engine.py
Advanced face-search engine for FaceFetch with Google Photos-level accuracy.
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

try:
    import cv2
except ImportError:
    pass

try:
    from sklearn.cluster import DBSCAN
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
DRIVE_API_BASE   = "https://www.googleapis.com/drive/v3"
DRIVE_LIST_URL   = f"{DRIVE_API_BASE}/files"
DRIVE_DL_URL     = f"{DRIVE_API_BASE}/files/{{file_id}}?alt=media"
PAGE_SIZE        = 100

IS_RENDER = os.getenv("RENDER") == "true"
MAX_WORKERS      = 1

SUPPORTED_EXTS   = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp",
    ".tiff", ".heic", ".heif", ".jfif", ".gif",
}

REF_MAX_SIZE     = 1024
REF_NUM_JITTERS  = 5

SCAN_MAX_SIZE    = 1600
SCAN_QUALITY     = 85


TOKEN_REFRESH_MAX_RETRIES = 3
TOKEN_REFRESH_BACKOFF     = 2

DRIVE_API_MAX_RETRIES     = 3
DRIVE_API_BACKOFF         = 1

GOOGLE_API_KEY            = os.getenv("GOOGLE_API_KEY", "")

COSINE_MATCH_THRESHOLD    = 0.88
COSINE_BOOST_THRESHOLD    = 0.82
NEAR_MISS_MARGIN          = 0.08
CONSISTENCY_MARGIN        = 0.12

CLAHE_CLIP_LIMIT          = 2.0
CLAHE_TILE_SIZE           = 8


# ── CLAHE preprocessing ──────────────────────────────────────────────────────
def _apply_clahe(img_arr: np.ndarray) -> np.ndarray:
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
    try:
        landmarks_list = face_recognition.face_landmarks(img_arr, [face_location])
        if not landmarks_list:
            return None

        landmarks = landmarks_list[0]
        left_eye  = landmarks.get("left_eye",  [])
        right_eye = landmarks.get("right_eye", [])

        if not left_eye or not right_eye:
            return None

        left_center  = np.mean(left_eye,  axis=0)
        right_center = np.mean(right_eye, axis=0)

        dy = right_center[1] - left_center[1]
        dx = right_center[0] - left_center[0]
        angle = math.degrees(math.atan2(dy, dx))

        if abs(angle) < 2.0 or abs(angle) > 45.0:
            return None

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


# ── Reusable HTTP session ──────────────────────────────────────────────────────
_http_session = None

def _get_http_session() -> requests.Session:
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

            if resp.status_code == 401:
                if not token_state.get("_use_api_key") and _refresh_access_token(token_state):
                    continue
                if not token_state.get("_use_api_key") and GOOGLE_API_KEY:
                    token_state["_use_api_key"] = True
                    logger.info("🔑 Switching to API key for public folder access")
                    continue
                resp.raise_for_status()

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
    """Extract folder ID from a Google Drive link with ReDoS protection."""
    # Limit input length to prevent ReDoS
    if len(drive_link) > 500:
        raise ValueError("Drive link too long")
    
    # Simple, non-backtracking patterns
    patterns = [
        r"/folders/([a-zA-Z0-9_-]{10,50})",  # Limited length
        r"[?&]id=([a-zA-Z0-9_-]{10,50})",    # Limited length
    ]
    for pat in patterns:
        m = re.search(pat, drive_link)  # ReDoS safe via length limits
        if m:
            return m.group(1)
    raise ValueError(f"Cannot extract folder ID from link")


def _list_drive_files(folder_id: str, token_state: dict) -> Generator[dict, None, None]:
    params = {
        "q": f"'{folder_id}' in parents and mimeType contains 'image/' and trashed = false",
        "fields": "nextPageToken, files(id, name, modifiedTime)",
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
    """Download image bytes from Google Drive.

    Tries the official API first.  When that returns 403 (common when using
    an API key on a public folder), falls back to Google's public export URL
    which works for any file shared as "Anyone with the link".

    For larger files Google shows a virus-scan confirmation page — this is
    handled by extracting the confirm token and re-requesting.
    """
    session = _get_http_session()

    # ── Attempt 1: Official Drive API ─────────────────────────────────────
    url = DRIVE_DL_URL.format(file_id=file_id)
    try:
        resp = _make_drive_request(url, token_state, timeout=20, stream=True)
        if resp.status_code == 200:
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                return resp.content
    except Exception:
        pass  # fall through to public fallback

    # ── Attempt 2: Public download URL (bypasses API-key 403) ─────────────
    public_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    try:
        resp = session.get(public_url, timeout=30, allow_redirects=True)
        if resp.status_code != 200 or not resp.content:
            logger.debug("Public URL failed for file_id=%s (status=%s)", file_id, resp.status_code)
            return None

        content_type = resp.headers.get("Content-Type", "")

        # ── Attempt 3: Virus-scan confirmation flow ────────────────────────
        # Google returns an HTML warning page (can be 10–50 KB) for files
        # over ~100 KB.  Extract the confirm token and retry.
        if "text/html" in content_type:
            logger.debug("Got HTML virus-scan page for file_id=%s, attempting confirm", file_id)
            body = resp.text[:50000]  # Limit body size to prevent memory exhaustion
            confirm_token = None

            # Pattern 1: hidden input confirm value (simplified to prevent ReDoS)
            m = re.search(r'name=["\']confirm["\']\s+value=["\']([^"\']{1,100})["\']', body)
            if m:
                confirm_token = m.group(1)

            # Pattern 2: confirm= in URL params (limited length)
            if not confirm_token:
                m = re.search(r'[?&]confirm=([a-zA-Z0-9_\-]{1,50})', body)
                if m:
                    confirm_token = m.group(1)

            # Pattern 3: newer "confirm=t" style
            if not confirm_token and 'confirm=t' in body:
                confirm_token = "t"

            if confirm_token:
                confirm_url = (
                    f"https://drive.google.com/uc?export=download"
                    f"&id={file_id}&confirm={confirm_token}"
                )
                try:
                    resp2 = session.get(confirm_url, timeout=30, allow_redirects=True)
                    if resp2.status_code == 200 and resp2.content:
                        ct2 = resp2.headers.get("Content-Type", "")
                        if "text/html" not in ct2:
                            logger.debug("✓ Confirm flow succeeded for file_id=%s", file_id)
                            return resp2.content
                except Exception as exc:
                    logger.debug("Confirm flow request failed for file_id=%s: %s", file_id, exc)

            # ── Attempt 4: drive.usercontent.google.com endpoint ──────────
            # Newer Google Drive uses this domain for large file downloads
            alt_url = (
                f"https://drive.usercontent.google.com/download"
                f"?id={file_id}&export=download&confirm=t"
            )
            try:
                resp3 = session.get(alt_url, timeout=30, allow_redirects=True)
                if resp3.status_code == 200 and resp3.content:
                    ct3 = resp3.headers.get("Content-Type", "")
                    if "text/html" not in ct3:
                        logger.debug("✓ usercontent fallback succeeded for file_id=%s", file_id)
                        return resp3.content
            except Exception as exc:
                logger.debug("usercontent fallback failed for file_id=%s: %s", file_id, exc)

            logger.debug("Could not bypass virus-scan page for file_id=%s — skipping", file_id)
            return None

        # Direct image response — success
        return resp.content

    except Exception as exc:
        logger.debug("Download failed for file_id=%s: %s", file_id, exc)

    return None


# ── Face-encoding helpers ─────────────────────────────────────────────────────
def encode_reference_image(
    image_bytes: bytes,
    num_jitters: int = REF_NUM_JITTERS,
    model: str = "large",
) -> list[list[float]]:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    if max(img.size) > REF_MAX_SIZE:
        img.thumbnail((REF_MAX_SIZE, REF_MAX_SIZE), Image.Resampling.LANCZOS)

    arr = np.array(img)

    # ── Quality Gateway: Blur Detection ──
    try:
        if 'cv2' in globals():
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            if blur_score < 40.0:
                raise ValueError(f"Image is too blurry (Blur score: {blur_score:.1f}). Please upload a clearer, well-lit selfie.")
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        logger.debug("Blur detection skipped: %s", exc)

    arr_clahe = _apply_clahe(arr)

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
            pass

    if not locations:
        locations = face_recognition.face_locations(arr, model="hog")

    if not locations:
        raise ValueError(
            "No face detected in the reference photo. "
            "Please upload a clear selfie with your face visible."
        )

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
        all_encodings.append(enc.tolist())

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
            pass

    img.close()

    logger.info(
        "✓ Reference encoding complete: %d face(s) detected, %d total variants",
        len(encodings_clahe), len(all_encodings),
    )
    return all_encodings


def prepare_encodings(known_encodings: list) -> list[np.ndarray]:
    np_encodings = [
        np.array(enc) if not isinstance(enc, np.ndarray) else enc
        for enc in known_encodings
    ]

    if len(np_encodings) == 0:
        return []

    if len(np_encodings) == 1:
        logger.info("✓ Using single reference encoding")
        return np_encodings

    # ── DBSCAN Clustering for Outlier Rejection ──
    try:
        if 'DBSCAN' in globals() and len(np_encodings) >= 3:
            clustering = DBSCAN(eps=0.45, min_samples=2, metric='euclidean').fit(np_encodings)
            labels = clustering.labels_
            if len(set(labels)) > 1 or (len(set(labels)) == 1 and list(labels)[0] == -1):
                unique_labels = set(labels)
                largest_cluster = -1
                max_count = 0
                for label in unique_labels:
                    if label != -1:
                        count = list(labels).count(label)
                        if count > max_count:
                            max_count = count
                            largest_cluster = label
                
                if largest_cluster != -1:
                    filtered_encodings = [enc for enc, label in zip(np_encodings, labels) if label == largest_cluster]
                    logger.info("✓ DBSCAN rejected %d outlier faces (kept %d)", len(np_encodings) - len(filtered_encodings), len(filtered_encodings))
                    np_encodings = filtered_encodings
    except Exception as exc:
        logger.warning("DBSCAN clustering failed: %s", exc)

    avg_encoding = np.mean(np_encodings, axis=0)
    norm = np.linalg.norm(avg_encoding)
    if norm > 0:
        avg_encoding = avg_encoding / norm

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
    distances = face_recognition.face_distance(known_encodings, candidate)
    min_dist = float(np.min(distances))

    # Calculate average pairwise distance of reference encodings (ERV - Ensemble Reference Variance)
    if len(known_encodings) > 1:
        ref_distances = []
        for i in range(len(known_encodings)):
            for j in range(i + 1, len(known_encodings)):
                ref_distances.append(float(np.linalg.norm(known_encodings[i] - known_encodings[j])))
        erv = float(np.mean(ref_distances))
    else:
        erv = 0.0

    # Dynamic tolerance boost based on reference variance (max boost +0.06)
    # If references have diverse angles, we trust the model to match wider variations.
    variance_boost = min(0.06, erv * 0.15)
    effective_tolerance = tolerance + variance_boost

    best_cosine = 0.0
    for known_enc in known_encodings:
        dot_product = np.dot(candidate, known_enc)
        norm_product = np.linalg.norm(candidate) * np.linalg.norm(known_enc)
        if norm_product > 0:
            cosine_sim = dot_product / norm_product
            best_cosine = max(best_cosine, cosine_sim)

    # Multi-Algorithm consensus logic
    if min_dist <= effective_tolerance:
        return True, min_dist, f"euclidean_boosted_{variance_boost:.3f}"

    if best_cosine >= COSINE_MATCH_THRESHOLD and min_dist <= effective_tolerance + 0.05:
        return True, min_dist, "cosine_boost"

    if best_cosine >= COSINE_BOOST_THRESHOLD and min_dist <= effective_tolerance + NEAR_MISS_MARGIN:
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
) -> tuple[str, bytes | None, list[list[float]]]:
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        max_size = 1800 if model_type == "cnn" else SCAN_MAX_SIZE
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
 
        img_arr = np.array(img)
        img_clahe = _apply_clahe(img_arr)

        locations = face_recognition.face_locations(
            img_clahe,
            number_of_times_to_upsample=upsample,
            model=model_type,
        )

        if not locations:
            locations = face_recognition.face_locations(
                img_arr,
                number_of_times_to_upsample=upsample,
                model=model_type,
            )

        if not locations and model_type == "hog" and upsample < 2:
            locations = face_recognition.face_locations(
                img_clahe,
                number_of_times_to_upsample=2,
                model="hog",
            )

        if not locations:
            logger.info("✗ No face detected: %s", filename)
            img.close()
            return filename, None, []

        candidate_encodings = face_recognition.face_encodings(
            img_clahe,
            known_face_locations=locations,
            num_jitters=2,
            model="large",
        )

        if not candidate_encodings:
            logger.info("✗ Face encoding failed: %s", filename)
            img.close()
            return filename, None, []

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
                if all(d <= tolerance + CONSISTENCY_MARGIN for d in distances):
                    avg_dist = sum(distances) / len(distances)
                    if avg_dist <= tolerance + 0.04:
                        found_match = True
                        logger.info(
                            "✓ MATCH (consensus): %s (avg_dist: %.3f, spread: %.3f)",
                            filename, avg_dist, max(distances) - min(distances),
                        )
                        break

        if not found_match:
            if best_distance < tolerance + 0.15:
                logger.info(
                    "✗ Near-miss: %s (dist: %.3f, needed: ≤%.3f, best_method: %s)",
                    filename, best_distance, tolerance, best_method,
                )
            else:
                logger.info(
                    "✗ No match: %s (dist: %.3f, needed: ≤%.3f)",
                    filename, best_distance, tolerance,
                )

        # Convert candidate encodings to standard list of floats for serialization
        serializable_encodings = [enc.tolist() for enc in candidate_encodings]

        result_bytes = None
        if found_match:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=SCAN_QUALITY)
            buf.seek(0)
            result_bytes = buf.read()

        img.close()
        del img_arr

        return filename, result_bytes, serializable_encodings

    except Exception as exc:
        logger.warning("Error processing %s: %s", filename, exc)
        return filename, None, []



def _process_drive_file(
    file_meta,
    token_state,
    known_encodings,
    tolerance,
    model_type,
    upsample,
    get_cache_func=None,
    save_cache_func=None,
):
    file_id = file_meta["id"]
    modified_time = file_meta.get("modifiedTime", "default")
    filename = file_meta["name"]

    cached_encodings = None
    if get_cache_func:
        try:
            cached_encodings = get_cache_func(file_id, modified_time)
        except Exception as e:
            logger.warning("Cache fetch error for %s: %s", filename, e)

    if cached_encodings is not None:
        # Convert list of floats back to np.ndarray
        candidate_encodings = [np.array(enc) for enc in cached_encodings]
        
        found_match = False
        for candidate in candidate_encodings:
            is_match, _, _ = _ensemble_match(candidate, known_encodings, tolerance)
            if is_match:
                found_match = True
                break

        if not found_match:
            logger.debug("✓ Cache Hit (No Match): %s", filename)
            return filename, None

        logger.info("✓ Cache Hit (Match Found): %s - downloading image", filename)
        img_bytes = _download_image_bytes(file_id, token_state)
        if not img_bytes:
            return filename, None
        return filename, img_bytes

    # Cache miss
    img_bytes = _download_image_bytes(file_id, token_state)
    if not img_bytes:
        logger.warning("✗ Download failed: %s", filename)
        return filename, None

    filename, out_bytes, extracted_encs = _process_image_bytes(
        filename, img_bytes, known_encodings, tolerance, model_type, upsample
    )

    if save_cache_func and extracted_encs is not None:
        try:
            save_cache_func(file_id, modified_time, extracted_encs)
        except Exception as e:
            logger.warning("Cache write error for %s: %s", filename, e)

    return filename, out_bytes



# ── ZIP builder ──────────────────────────────────────────────────────────────
def _build_zip(matched: list[tuple[str, bytes]]) -> bytes:
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
    get_cache_func=None,
    save_cache_func=None,
) -> bytes:
    global MAX_WORKERS

    if IS_RENDER and model_type == "cnn":
        logger.info("Render environment: falling back from CNN to HOG model")
        model_type = "hog"
        upsample = max(1, upsample)

    token_state = {
        "access_token": access_token,
        "refresh_token": refresh_token,
    }

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
                tolerance, model_type, upsample, get_cache_func, save_cache_func
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
                filename, out_bytes, _ = future.result()
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