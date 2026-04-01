import json
import redis.asyncio as aioredis
import structlog
from backend.config.settings import REDIS_URL, REDIS_TTL_TRANSCRIPT, REDIS_TTL_SUMMARY, REDIS_TTL_NOTES

log = structlog.get_logger()

# --- Single shared Redis connection pool ---
_redis_client = None

async def get_redis() -> aioredis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = await aioredis.from_url(
            REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )
        log.info("redis.connected", url=REDIS_URL)
    return _redis_client

async def close_redis():
    global _redis_client
    if _redis_client:
        await _redis_client.close()
        _redis_client = None

# --- Transcript ---
async def cache_transcript(video_id: str, data: dict):
    r = await get_redis()
    await r.setex(
        f"transcript:{video_id}",
        REDIS_TTL_TRANSCRIPT,
        json.dumps(data)
    )
    log.info("cache.transcript.set", video_id=video_id)

async def get_cached_transcript(video_id: str):
    r = await get_redis()
    raw = await r.get(f"transcript:{video_id}")
    if raw:
        log.info("cache.transcript.hit", video_id=video_id)
        return json.loads(raw)
    log.info("cache.transcript.miss", video_id=video_id)
    return None

# --- Summary ---
async def cache_summary(video_id: str, summary: str):
    r = await get_redis()
    await r.setex(f"summary:{video_id}", REDIS_TTL_SUMMARY, summary)
    log.info("cache.summary.set", video_id=video_id)

async def get_cached_summary(video_id: str):
    r = await get_redis()
    val = await r.get(f"summary:{video_id}")
    if val:
        log.info("cache.summary.hit", video_id=video_id)
        return val
    log.info("cache.summary.miss", video_id=video_id)
    return None

# --- Notes ---
async def cache_notes(video_id: str, notes: str):
    r = await get_redis()
    await r.setex(f"notes:{video_id}", REDIS_TTL_NOTES, notes)
    log.info("cache.notes.set", video_id=video_id)

async def get_cached_notes(video_id: str):
    r = await get_redis()
    val = await r.get(f"notes:{video_id}")
    if val:
        log.info("cache.notes.hit", video_id=video_id)
        return val
    log.info("cache.notes.miss", video_id=video_id)
    return None

# --- YT Metadata ---
async def cache_yt_metadata(video_id: str, channel: str, title: str):
    r = await get_redis()
    await r.setex(
        f"yt_meta:{video_id}",
        REDIS_TTL_SUMMARY,
        json.dumps({"channel": channel, "title": title})
    )

async def get_cached_yt_metadata(video_id: str):
    r = await get_redis()
    raw = await r.get(f"yt_meta:{video_id}")
    if raw:
        return json.loads(raw)
    return None

# --- Source language ---
async def cache_source_language(video_id: str, lang: str):
    r = await get_redis()
    await r.setex(f"source_lang:{video_id}", REDIS_TTL_TRANSCRIPT, lang)

async def get_cached_source_language(video_id: str):
    r = await get_redis()
    return await r.get(f"source_lang:{video_id}")

# --- English transcript string ---
async def cache_english_transcript(video_id: str, text: str):
    r = await get_redis()
    await r.setex(f"en_transcript:{video_id}", REDIS_TTL_TRANSCRIPT, text)

async def get_cached_english_transcript(video_id: str):
    r = await get_redis()
    return await r.get(f"en_transcript:{video_id}")

# --- Health check ---
async def ping_redis() -> bool:
    try:
        r = await get_redis()
        return await r.ping()
    except Exception:
        return False