"""
database.py
Async MongoDB connection via Motor.
Uses a 'One Document Per User' model for scalability and token persistence.
Includes health-check ping, index management, and retry logic.

Falls back to local JSON file storage when MongoDB is unreachable.
"""

import json
import os
import logging
import asyncio
from pathlib import Path
from typing import Optional

import motor.motor_asyncio
from bson import ObjectId
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

MONGO_URI: str = os.getenv("MONGO_URI", "")
if not MONGO_URI:
    if os.getenv("ENVIRONMENT", "development").lower() not in ("production", "prod"):
        logger.warning("⚠ MONGO_URI is not set. Defaulting to local MongoDB: mongodb://localhost:27017/facefetch")
        MONGO_URI = "mongodb://localhost:27017/facefetch"
    else:
        raise EnvironmentError("MONGO_URI is not set in the .env file.")


# ── Constants ─────────────────────────────────────────────────────────────────
DB_NAME = "facefetch"
USER_COLLECTION = "user_profiles"
CONNECT_RETRY_ATTEMPTS = 3
CONNECT_RETRY_DELAY = 2  # seconds

# ── Fallback JSON storage (used when MongoDB is unavailable) ──────────────────
_FALLBACK_FILE = Path(__file__).parent / "local_data.json"
_mongo_available: Optional[bool] = None  # cached availability flag


def _load_local_data() -> dict:
    """Load local JSON fallback data."""
    if _FALLBACK_FILE.exists():
        try:
            return json.loads(_FALLBACK_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_local_data(data: dict):
    """Persist local JSON fallback data."""
    _FALLBACK_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _is_mongo_up() -> bool:
    """Return cached MongoDB availability (checked at startup)."""
    return _mongo_available is True


# ── Motor client ──────────────────────────────────────────────────────────────
_client: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None
_db = None


def get_client() -> motor.motor_asyncio.AsyncIOMotorClient:
    """Get or create the Motor client with connection pooling."""
    global _client
    if _client is None:
        _client = motor.motor_asyncio.AsyncIOMotorClient(
            MONGO_URI,
            serverSelectionTimeoutMS=8_000,
            maxPoolSize=20,
        )
    return _client


def get_db():
    """Get the database instance."""
    global _db
    if _db is None:
        _db = get_client()[DB_NAME]
    return _db


async def ping() -> bool:
    """Health-check: returns True when MongoDB responds."""
    global _mongo_available
    try:
        await get_client().admin.command("ping")
        _mongo_available = True
        return True
    except Exception as exc:
        logger.error("MongoDB ping failed: %s", exc)
        _mongo_available = False
        return False


async def ensure_indexes():
    """
    Create database indexes for performance.
    Safe to call multiple times (idempotent).
    """
    if not _is_mongo_up():
        logger.warning("⚠ Skipping index creation — using local file storage fallback")
        return
    try:
        db = get_db()
        collection = db[USER_COLLECTION]
        await collection.create_index("user_id", unique=True)
        logger.info("✓ Database indexes verified")
    except Exception as e:
        logger.warning("Index creation warning: %s", e)


async def ping_with_retry() -> bool:
    """
    Try to connect to MongoDB with retry logic.
    Used during app startup for resilience against transient network issues.
    """
    for attempt in range(1, CONNECT_RETRY_ATTEMPTS + 1):
        if await ping():
            return True
        if attempt < CONNECT_RETRY_ATTEMPTS:
            logger.warning(
                "MongoDB connection attempt %d/%d failed, retrying in %ds...",
                attempt, CONNECT_RETRY_ATTEMPTS, CONNECT_RETRY_DELAY
            )
            await asyncio.sleep(CONNECT_RETRY_DELAY)
    return False


# ── Face-encoding & User Management ──────────────────────────────────────────

async def save_face_encoding(
    user_id: str,
    filename: str,
    encoding: list,
    profile_name: str = "default",
) -> str:
    """
    Pushes a new encoding into the user's 'references' array with a profile name.
    Falls back to local JSON file if MongoDB is unavailable.
    """
    if _is_mongo_up():
        try:
            db = get_db()
            ref_id = ObjectId()
            new_reference = {
                "ref_id": ref_id,
                "filename": filename,
                "encoding": [float(x) for x in encoding],
                "profile_name": profile_name,
            }
            await db[USER_COLLECTION].update_one(
                {"user_id": user_id},
                {"$push": {"references": new_reference}},
                upsert=True,
            )
            logger.info("✓ Saved face reference for user %s, profile %s (filename: %s)", user_id, profile_name, filename)
            return str(ref_id)
        except Exception as e:
            logger.warning("MongoDB save failed, falling back to local storage: %s", e)

    # ── Local fallback ────────────────────────────────────────────────────────
    ref_id = str(ObjectId())
    data = _load_local_data()
    user_data = data.setdefault(user_id, {"references": [], "refresh_token": None})
    user_data["references"].append({
        "ref_id": ref_id,
        "filename": filename,
        "encoding": [float(x) for x in encoding],
        "profile_name": profile_name,
    })
    _save_local_data(data)
    logger.info("✓ Saved face reference locally for user %s, profile %s (filename: %s)", user_id, profile_name, filename)
    return ref_id


async def load_face_encodings(user_id: str, profile_name: str = "default") -> list:
    """
    Retrieves all face encodings for a user filtered by profile name.
    Checks both MongoDB and local file, prioritizing whichever has data.
    """
    encodings = []

    # Try MongoDB first if available
    if _is_mongo_up():
        try:
            db = get_db()
            user_profile = await db[USER_COLLECTION].find_one({"user_id": user_id})
            if user_profile and "references" in user_profile:
                encodings = [
                    ref["encoding"]
                    for ref in user_profile["references"]
                    if "encoding" in ref and ref.get("profile_name", "default") == profile_name
                ]
                if encodings:
                    logger.info("✓ Loaded %d encoding(s) from MongoDB for user %s (profile: %s)", len(encodings), user_id, profile_name)
                    return encodings
                else:
                    logger.warning("MongoDB profile found but no encodings for user %s (profile: %s)", user_id, profile_name)
            else:
                logger.warning("No MongoDB profile found for user %s, trying local file", user_id)
        except Exception as e:
            logger.error("MongoDB load failed: %s", e)

    # Fall back to local file
    logger.info("Loading from local_data.json for user %s (profile: %s)", user_id, profile_name)
    data = _load_local_data()
    user_data = data.get(user_id, {})
    encodings = [
        ref["encoding"]
        for ref in user_data.get("references", [])
        if "encoding" in ref and ref.get("profile_name", "default") == profile_name
    ]
    logger.info("✓ Loaded %d encoding(s) from local file for user %s (profile: %s)", len(encodings), user_id, profile_name)
    return encodings



async def get_encoding_count(user_id: str, profile_name: str = "default") -> int:
    """Get the number of saved face encodings for a profile without loading them all."""
    if _is_mongo_up():
        try:
            db = get_db()
            result = await db[USER_COLLECTION].aggregate([
                {"$match": {"user_id": user_id}},
                {"$project": {
                    "references": {
                        "$filter": {
                            "input": "$references",
                            "as": "ref",
                            "cond": {"$eq": [{"$ifNull": ["$$ref.profile_name", "default"]}, profile_name]}
                        }
                    }
                }},
                {"$project": {"count": {"$size": {"$ifNull": ["$references", []]}}}}
            ]).to_list(1)
            if result:
                return result[0].get("count", 0)
            return 0
        except Exception as e:
            logger.warning("MongoDB count failed, falling back to local storage: %s", e)

    # ── Local fallback ────────────────────────────────────────────────────────
    data = _load_local_data()
    refs = data.get(user_id, {}).get("references", [])
    filtered_refs = [r for r in refs if r.get("profile_name", "default") == profile_name]
    return len(filtered_refs)


async def get_all_references(user_id: str, profile_name: str = "default") -> list:
    """
    Get all references for a user filtered by profile name (without encoding data).
    Falls back to local JSON file if MongoDB is unavailable.
    """
    if _is_mongo_up():
        try:
            db = get_db()
            user_profile = await db[USER_COLLECTION].find_one(
                {"user_id": user_id},
                {"references.encoding": 0},
            )
            if not user_profile or "references" not in user_profile:
                return []
            return [
                {"ref_id": str(ref["ref_id"]), "filename": ref["filename"]}
                for ref in user_profile["references"]
                if ref.get("profile_name", "default") == profile_name
            ]
        except Exception as e:
            logger.warning("MongoDB get_all_refs failed, falling back to local storage: %s", e)

    # ── Local fallback ────────────────────────────────────────────────────────
    data = _load_local_data()
    refs = data.get(user_id, {}).get("references", [])
    return [
        {"ref_id": ref["ref_id"], "filename": ref["filename"]}
        for ref in refs
        if ref.get("profile_name", "default") == profile_name
    ]



async def delete_specific_reference(user_id: str, ref_id: str) -> bool:
    """
    Deletes a specific Face DNA reference from the user's document.
    Falls back to local JSON file if MongoDB is unavailable.
    """
    if _is_mongo_up():
        try:
            db = get_db()
            object_id = ObjectId(ref_id)
            result = await db[USER_COLLECTION].update_one(
                {"user_id": user_id},
                {"$pull": {"references": {"ref_id": object_id}}},
            )
            return result.modified_count > 0
        except Exception as e:
            logger.warning("MongoDB delete_ref failed, falling back to local storage: %s", e)

    # ── Local fallback ────────────────────────────────────────────────────────
    data = _load_local_data()
    user_data = data.get(user_id, {})
    refs = user_data.get("references", [])
    new_refs = [r for r in refs if r.get("ref_id") != ref_id]
    if len(new_refs) == len(refs):
        return False
    user_data["references"] = new_refs
    data[user_id] = user_data
    _save_local_data(data)
    return True


async def delete_face_encodings(user_id: str) -> int:
    """Deletes all saved face references for a given user."""
    if _is_mongo_up():
        try:
            db = get_db()
            result = await db[USER_COLLECTION].update_one(
                {"user_id": user_id},
                {"$set": {"references": []}},
            )
            return result.modified_count
        except Exception as e:
            logger.warning("MongoDB delete_encodings failed, falling back to local storage: %s", e)

    # ── Local fallback ────────────────────────────────────────────────────────
    data = _load_local_data()
    if user_id in data and data[user_id].get("references"):
        count = len(data[user_id]["references"])
        data[user_id]["references"] = []
        _save_local_data(data)
        return count
    return 0


# ── Token Persistence ────────────────────────────────────────────────────────

async def save_refresh_token(user_id: str, refresh_token: str):
    """Stores the refresh token for persistent auth during long scans."""
    if _is_mongo_up():
        try:
            db = get_db()
            await db[USER_COLLECTION].update_one(
                {"user_id": user_id},
                {"$set": {"refresh_token": refresh_token}},
                upsert=True,
            )
            return
        except Exception as e:
            logger.warning("MongoDB save_token failed, falling back to local storage: %s", e)

    # ── Local fallback ────────────────────────────────────────────────────────
    data = _load_local_data()
    user_data = data.setdefault(user_id, {"references": [], "refresh_token": None})
    user_data["refresh_token"] = refresh_token
    _save_local_data(data)


async def get_refresh_token(user_id: str) -> Optional[str]:
    """Retrieve the stored refresh token for a user."""
    if _is_mongo_up():
        try:
            db = get_db()
            user = await db[USER_COLLECTION].find_one({"user_id": user_id})
            return user.get("refresh_token") if user else None
        except Exception as e:
            logger.warning("MongoDB get_token failed, falling back to local storage: %s", e)

    # ── Local fallback ────────────────────────────────────────────────────────
    data = _load_local_data()
    return data.get(user_id, {}).get("refresh_token")


# ── Face Encodings Cache ──────────────────────────────────────────────────────
CACHE_COLLECTION = "scanned_cache"

async def get_cached_encodings(file_id: str, modified_time: str) -> Optional[list]:
    """Retrieve cached face encodings for a file if modified_time matches."""
    if _is_mongo_up():
        try:
            db = get_db()
            cached = await db[CACHE_COLLECTION].find_one({"file_id": file_id, "modified_time": modified_time})
            if cached:
                return cached.get("encodings")
            return None
        except Exception as e:
            logger.warning("MongoDB get_cached_encodings failed, falling back to local: %s", e)

    # Local fallback
    data = _load_local_data()
    cache_store = data.get("scanned_cache", {})
    entry = cache_store.get(file_id)
    if entry and entry.get("modified_time") == modified_time:
        return entry.get("encodings")
    return None


async def save_cached_encodings(file_id: str, modified_time: str, encodings: list):
    """Save face encodings for a file to the cache."""
    if _is_mongo_up():
        try:
            db = get_db()
            await db[CACHE_COLLECTION].update_one(
                {"file_id": file_id},
                {"$set": {"modified_time": modified_time, "encodings": encodings}},
                upsert=True,
            )
            return
        except Exception as e:
            logger.warning("MongoDB save_cached_encodings failed, falling back to local: %s", e)

    # Local fallback
    data = _load_local_data()
    cache_store = data.setdefault("scanned_cache", {})
    cache_store[file_id] = {
        "modified_time": modified_time,
        "encodings": encodings
    }
    _save_local_data(data)

