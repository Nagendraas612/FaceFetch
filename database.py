"""
database.py
Async MongoDB connection via Motor.
Uses a 'One Document Per User' model for scalability and token persistence.
Includes health-check ping, index management, and retry logic.
"""

import os
import logging
import asyncio
from typing import Optional

import motor.motor_asyncio
from bson import ObjectId
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

MONGO_URI: str = os.getenv("MONGO_URI", "")
if not MONGO_URI:
    raise EnvironmentError("MONGO_URI is not set in the .env file.")

# ── Constants ─────────────────────────────────────────────────────────────────
DB_NAME = "eventai"
USER_COLLECTION = "user_profiles"
CONNECT_RETRY_ATTEMPTS = 3
CONNECT_RETRY_DELAY = 2  # seconds

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
    """Health-check: returns True when Atlas responds."""
    try:
        await get_client().admin.command("ping")
        return True
    except Exception as exc:
        logger.error("MongoDB ping failed: %s", exc)
        return False


async def ensure_indexes():
    """
    Create database indexes for performance.
    Safe to call multiple times (idempotent).
    """
    try:
        db = get_db()
        collection = db[USER_COLLECTION]

        # Index on user_id for fast lookups
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

async def save_face_encoding(user_id: str, filename: str, encoding: list) -> str:
    """
    Pushes a new encoding into the user's 'references' array.
    Uses 'upsert' to create the user document if it doesn't exist.
    Stores encoding as a plain list of floats (safe — no pickle/binary).
    """
    db = get_db()
    ref_id = ObjectId()

    new_reference = {
        "ref_id": ref_id,
        "filename": filename,
        "encoding": [float(x) for x in encoding],
    }

    await db[USER_COLLECTION].update_one(
        {"user_id": user_id},
        {"$push": {"references": new_reference}},
        upsert=True,
    )

    logger.info("Saved face reference for user %s (filename: %s)", user_id, filename)
    return str(ref_id)


async def load_face_encodings(user_id: str) -> list:
    """
    Retrieves all face encodings for a user.
    Returns a list of plain numeric arrays.
    """
    db = get_db()
    user_profile = await db[USER_COLLECTION].find_one({"user_id": user_id})

    if not user_profile or "references" not in user_profile:
        return []

    encodings = [
        ref["encoding"]
        for ref in user_profile["references"]
        if "encoding" in ref
    ]

    logger.info("Loaded %d encoding(s) for user %s", len(encodings), user_id)
    return encodings


async def get_encoding_count(user_id: str) -> int:
    """Get the number of saved face encodings without loading them all."""
    db = get_db()
    result = await db[USER_COLLECTION].aggregate([
        {"$match": {"user_id": user_id}},
        {"$project": {"count": {"$size": {"$ifNull": ["$references", []]}}}}
    ]).to_list(1)

    if result:
        return result[0].get("count", 0)
    return 0


async def get_all_references(user_id: str) -> list:
    """
    Get all references for a user (without the encoding data, for listing).
    Returns cleaned data with string ref_ids.
    """
    db = get_db()
    user_profile = await db[USER_COLLECTION].find_one(
        {"user_id": user_id},
        {"references.encoding": 0},  # Exclude encoding data for efficiency
    )

    if not user_profile or "references" not in user_profile:
        return []

    cleaned_refs = []
    for ref in user_profile["references"]:
        cleaned_refs.append({
            "ref_id": str(ref["ref_id"]),
            "filename": ref["filename"],
        })
    return cleaned_refs


async def delete_specific_reference(user_id: str, ref_id: str) -> bool:
    """
    Deletes a specific Face DNA reference from the user's document.
    """
    try:
        db = get_db()
        object_id = ObjectId(ref_id)

        result = await db[USER_COLLECTION].update_one(
            {"user_id": user_id},
            {"$pull": {"references": {"ref_id": object_id}}},
        )

        return result.modified_count > 0
    except Exception as e:
        logger.error("Error deleting reference %s: %s", ref_id, e)
        return False


async def delete_face_encodings(user_id: str) -> int:
    """Deletes all saved face references for a given user."""
    db = get_db()
    result = await db[USER_COLLECTION].update_one(
        {"user_id": user_id},
        {"$set": {"references": []}},
    )
    return result.modified_count


# ── Token Persistence ────────────────────────────────────────────────────────

async def save_refresh_token(user_id: str, refresh_token: str):
    """Stores the refresh token for persistent auth during long scans."""
    db = get_db()
    await db[USER_COLLECTION].update_one(
        {"user_id": user_id},
        {"$set": {"refresh_token": refresh_token}},
        upsert=True,
    )


async def get_refresh_token(user_id: str) -> Optional[str]:
    """Retrieve the stored refresh token for a user."""
    db = get_db()
    user = await db[USER_COLLECTION].find_one({"user_id": user_id})
    return user.get("refresh_token") if user else None
