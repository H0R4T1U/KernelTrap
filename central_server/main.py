"""
KernelTrap Central Analysis Server

Subscribes to events.{hostname} Redis Streams from all agents,
scores them with Isolation Forest, and publishes results back:

  scores.{hostname}   <- anomaly score per event (for dashboard)
  commands.{hostname} <- pivot commands when threshold is crossed

Run:
    uvicorn central_server.main:app --host 0.0.0.0 --port 8000

Environment variables (or edit CONFIG below):
    REDIS_HOST          Redis host (default: localhost)
    REDIS_PORT          Redis port (default: 6379)
    MODEL_DIR           Path to trained IF model dir
    PIVOT_THRESHOLD     High-severity events to trigger pivot (default: 5)
    PIVOT_WINDOW_SEC    Sliding window in seconds (default: 60)
    WHITELIST_UIDS      Comma-separated UIDs to never pivot (default: 0)
    HOST_DISCOVERY_SEC  How often to scan Redis for new agent streams (default: 30)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from typing import Dict, List, Optional

import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from central_server.scorer import Scorer
from central_server.window import SlidingWindowTracker

# ---------------------------------------------------------------------------
# Configuration (override with env vars)
# ---------------------------------------------------------------------------

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
MODEL_DIR = os.getenv("MODEL_DIR", "masina_invata/isolation_forest/beth_iforest_model_host2tier")
PIVOT_THRESHOLD = int(os.getenv("PIVOT_THRESHOLD", "5"))
PIVOT_WINDOW_SEC = int(os.getenv("PIVOT_WINDOW_SEC", "60"))
WHITELIST_UIDS = set(
    int(u) for u in os.getenv("WHITELIST_UIDS", "0").split(",") if u.strip().isdigit()
)
HOST_DISCOVERY_SEC = int(os.getenv("HOST_DISCOVERY_SEC", "30"))

# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("kerneltrap.central")

app = FastAPI(title="KernelTrap Central Server")

# Shared state (set up in lifespan)
_redis: Optional[aioredis.Redis] = None
_scorer: Optional[Scorer] = None
_tracker: Optional[SlidingWindowTracker] = None

# Last-read stream IDs per host; "0" means read from beginning of stream
_stream_cursors: Dict[str, str] = {}


# ---------------------------------------------------------------------------
# Lifespan: start background tasks when the server boots
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    global _redis, _scorer, _tracker

    _redis = aioredis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    await _redis.ping()
    log.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")

    _scorer = Scorer(MODEL_DIR)
    log.info(f"Loaded Isolation Forest model from '{MODEL_DIR}'")

    _tracker = SlidingWindowTracker(
        pivot_threshold=PIVOT_THRESHOLD,
        window_seconds=PIVOT_WINDOW_SEC,
        whitelist_uids=WHITELIST_UIDS,
    )

    asyncio.create_task(_event_consumer_loop(), name="event-consumer")
    asyncio.create_task(_pivot_publisher_loop(), name="pivot-publisher")
    log.info("Background tasks started")


@app.on_event("shutdown")
async def shutdown():
    if _redis:
        await _redis.aclose()


# ---------------------------------------------------------------------------
# Background Task A: consume events from all agents
# ---------------------------------------------------------------------------

async def _discover_hosts() -> List[str]:
    """Scan Redis for stream keys matching events.* to find connected agents."""
    keys = await _redis.keys("events.*")
    return [k.replace("events.", "") for k in keys]


async def _event_consumer_loop():
    """
    Continuously reads new events from all known agent streams.
    Discovers new agents every HOST_DISCOVERY_SEC seconds.
    """
    last_discovery = 0.0

    while True:
        try:
            now = asyncio.get_event_loop().time()

            # Re-discover agents periodically
            if now - last_discovery > HOST_DISCOVERY_SEC:
                hosts = await _discover_hosts()
                for h in hosts:
                    if h not in _stream_cursors:
                        _stream_cursors[h] = "0"  # read from start of existing stream
                        log.info(f"Discovered new agent host: '{h}'")
                last_discovery = now

            if not _stream_cursors:
                await asyncio.sleep(1)
                continue

            # Build stream map: {stream_key: last_id}
            stream_map = {f"events.{h}": cursor for h, cursor in _stream_cursors.items()}

            # Block up to 500ms waiting for new messages across all streams
            results = await _redis.xread(stream_map, block=500, count=200)

            if not results:
                continue

            for stream_key, messages in results:
                hostname = stream_key.replace("events.", "")
                for msg_id, fields in messages:
                    _stream_cursors[hostname] = msg_id
                    await _process_message(hostname, fields)

        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error(f"Event consumer error: {e}")
            await asyncio.sleep(1)


async def _process_message(hostname: str, fields: Dict[str, str]):
    """Decode one Redis Stream message, score it, publish scores."""
    try:
        events: List[Dict] = json.loads(fields.get("data", "[]"))
    except (json.JSONDecodeError, ValueError):
        return

    if not events:
        return

    score_results = _scorer.score_batch(events, hostname)

    # Feed scores into the sliding window tracker and publish to scores stream
    score_stream_key = f"scores.{hostname}"
    pipe = _redis.pipeline()

    for result in score_results:
        _tracker.feed(hostname, result["userId"], result["severity"])
        pipe.xadd(
            score_stream_key,
            {"data": json.dumps(result)},
            maxlen=50000,
            approximate=True,
        )

    await pipe.execute()


# ---------------------------------------------------------------------------
# Background Task B: publish pivot commands when window threshold is crossed
# ---------------------------------------------------------------------------

async def _pivot_publisher_loop():
    """
    Checks for pivot candidates every second and publishes commands to the
    appropriate agent's commands.{hostname} stream.
    """
    while True:
        try:
            await asyncio.sleep(1)
            pending = _tracker.drain_pivots()

            for hostname, user_id in pending:
                command_key = f"commands.{hostname}"
                cmd = json.dumps({"action": "pivot", "user": str(user_id)})
                await _redis.xadd(
                    command_key,
                    {"data": cmd},
                    maxlen=1000,
                    approximate=True,
                )
                log.warning(
                    f"PIVOT COMMAND sent → host='{hostname}', userId={user_id}"
                )

        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error(f"Pivot publisher error: {e}")


# ---------------------------------------------------------------------------
# HTTP endpoints (used by dashboard and health checks)
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Quick liveness check."""
    return {"status": "ok"}


@app.get("/stats")
async def stats():
    """Return sliding window stats and known hosts."""
    return JSONResponse({
        "hosts": list(_stream_cursors.keys()),
        "tracker": _tracker.stats() if _tracker else {},
    })


@app.get("/scores/{hostname}")
async def recent_scores(hostname: str, count: int = 50):
    """Return the last N score entries for a given host (for dashboard polling)."""
    results = await _redis.xrevrange(f"scores.{hostname}", count=count)
    scores = []
    for msg_id, fields in results:
        try:
            scores.append(json.loads(fields["data"]))
        except (KeyError, json.JSONDecodeError):
            pass
    return JSONResponse({"hostname": hostname, "scores": scores})
