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
import time
from collections import deque
from typing import Any, Dict, List, Optional, Set

import redis.asyncio as aioredis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from central_server.scorer import Scorer
from central_server.window_v3 import SlidingWindowTracker

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
PROCESS_DIVERSITY_THRESHOLD = int(os.getenv("PROCESS_DIVERSITY_THRESHOLD", "20"))
EXEC_FREQUENCY_THRESHOLD = int(os.getenv("EXEC_FREQUENCY_THRESHOLD", "100"))
LOW_SEVERITY_THRESHOLD = int(os.getenv("LOW_SEVERITY_THRESHOLD", "30"))
RECON_BINARY_THRESHOLD = int(os.getenv("RECON_BINARY_THRESHOLD", "5"))

# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("kerneltrap.central")

app = FastAPI(title="KernelTrap Central Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared state (set up in lifespan)
_redis: Optional[aioredis.Redis] = None
_scorer: Optional[Scorer] = None
_tracker: Optional[SlidingWindowTracker] = None

# Last-read stream IDs per host; "0" means read from beginning of stream
_stream_cursors: Dict[str, str] = {}

# Dashboard state
_pivot_history: List[Dict[str, Any]] = []
_ws_clients: Set[WebSocket] = set()
_log_broadcast_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
# hostname -> deque of event timestamps (last 60 s) for events/min calculation
_host_event_times: Dict[str, deque] = {}


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
        process_diversity_threshold=PROCESS_DIVERSITY_THRESHOLD,
        exec_frequency_threshold=EXEC_FREQUENCY_THRESHOLD,
        low_severity_threshold=LOW_SEVERITY_THRESHOLD,
        recon_binary_threshold=RECON_BINARY_THRESHOLD,
    )

    asyncio.create_task(_event_consumer_loop(), name="event-consumer")
    asyncio.create_task(_pivot_publisher_loop(), name="pivot-publisher")
    asyncio.create_task(_registration_loop(), name="registration")
    asyncio.create_task(_ws_broadcaster_loop(), name="ws-broadcaster")
    log.info("Background tasks started")


@app.on_event("shutdown")
async def shutdown():
    if _redis:
        await _redis.aclose()


# ---------------------------------------------------------------------------
# Background Task A: consume events from all agents
# ---------------------------------------------------------------------------

async def _registration_loop():
    """
    Listens on kerneltrap.registrations for instant agent discovery.
    Agents publish a hello message here on startup, so we add them to
    _stream_cursors immediately instead of waiting for the 30s polling cycle.
    """
    last_id = "$"
    while True:
        try:
            result = await _redis.xread({"kerneltrap.registrations": last_id}, block=2000, count=10)
            if not result:
                continue
            for _stream, messages in result:
                for msg_id, fields in messages:
                    last_id = msg_id
                    hostname = fields.get("hostname", "")
                    if hostname and hostname not in _stream_cursors:
                        _stream_cursors[hostname] = "0"
                        log.info(f"Agent registered instantly: '{hostname}'")
        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error(f"Registration loop error: {e}")
            await asyncio.sleep(1)


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

    # Track event timestamps per host for events/min calculation
    now = time.time()
    if hostname not in _host_event_times:
        _host_event_times[hostname] = deque()
    host_dq = _host_event_times[hostname]
    cutoff = now - 60
    while host_dq and host_dq[0] < cutoff:
        host_dq.popleft()
    host_dq.append(now)

    # Feed scores into the sliding window tracker and publish to scores stream
    score_stream_key = f"scores.{hostname}"
    pipe = _redis.pipeline()

    for result in score_results:
        _tracker.feed(
            hostname,
            result["userId"],
            result["severity"],
            process_name=result.get("processName", ""),
            event_name=result.get("eventName", ""),
        )
        pipe.xadd(
            score_stream_key,
            {"data": json.dumps(result)},
            maxlen=50000,
            approximate=True,
        )
        # Broadcast to WebSocket clients (non-blocking, drop if queue full)
        try:
            _log_broadcast_queue.put_nowait(result)
        except asyncio.QueueFull:
            pass

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

            failed = []
            for hostname, user_id, trigger in pending:
                try:
                    command_key = f"commands.{hostname}"
                    cmd = json.dumps({"action": "pivot", "user": str(user_id)})
                    await _redis.xadd(
                        command_key,
                        {"data": cmd},
                        maxlen=1000,
                        approximate=True,
                    )
                    log.warning(
                        f"PIVOT COMMAND sent → host='{hostname}', userId={user_id}, trigger={trigger}"
                    )
                    _pivot_history.append({
                        "timestamp": time.time(),
                        "hostname": hostname,
                        "user_id": user_id,
                        "trigger": trigger,
                    })
                except Exception as e:
                    log.error(f"Failed to send pivot for {hostname}/{user_id}: {e} — will retry")
                    failed.append((hostname, user_id, trigger))

            if failed:
                _tracker.re_queue_pivots(failed)

        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error(f"Pivot publisher error: {e}")


# ---------------------------------------------------------------------------
# Background Task C: broadcast scored events to WebSocket clients
# ---------------------------------------------------------------------------

async def _ws_broadcaster_loop():
    while True:
        try:
            event = await _log_broadcast_queue.get()
            if not _ws_clients:
                continue
            msg = json.dumps(event)
            dead: Set[WebSocket] = set()
            for ws in list(_ws_clients):
                try:
                    await ws.send_text(msg)
                except Exception:
                    dead.add(ws)
            _ws_clients -= dead
        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error(f"WS broadcaster error: {e}")


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


@app.get("/agents")
async def agents():
    """Return connected agents with events/min and last-seen timestamp."""
    now = time.time()
    result = []
    for hostname in _stream_cursors:
        dq = _host_event_times.get(hostname, deque())
        cutoff = now - 60
        events_per_min = sum(1 for t in dq if t >= cutoff)
        last_seen = dq[-1] if dq else None
        result.append({
            "hostname": hostname,
            "events_per_min": events_per_min,
            "last_seen": last_seen,
        })
    return JSONResponse({"agents": result})


@app.get("/users")
async def users():
    """Return all tracked (hostname, userId) pairs with window pressure state."""
    if not _tracker:
        return JSONResponse({"users": []})
    return JSONResponse({"users": _tracker.user_states()})


@app.post("/pivot/{hostname}/{user_id}")
async def manual_pivot(hostname: str, user_id: int):
    """Manually trigger a pivot command for a specific user on a host."""
    if hostname not in _stream_cursors:
        return JSONResponse({"error": f"Unknown host: {hostname}"}, status_code=404)

    command_key = f"commands.{hostname}"
    cmd = json.dumps({"action": "pivot", "user": str(user_id)})
    await _redis.xadd(command_key, {"data": cmd}, maxlen=1000, approximate=True)

    _pivot_history.append({
        "timestamp": time.time(),
        "hostname": hostname,
        "user_id": user_id,
        "trigger": "manual",
    })
    log.warning(f"MANUAL PIVOT sent → host='{hostname}', userId={user_id}")
    return JSONResponse({"status": "pivot_sent", "hostname": hostname, "user_id": user_id})


@app.get("/pivot-history")
async def pivot_history():
    """Return pivot history, most recent first."""
    return JSONResponse({"pivots": list(reversed(_pivot_history))})


@app.websocket("/ws/logs")
async def ws_logs(websocket: WebSocket):
    """Stream scored events to the dashboard in real-time."""
    await websocket.accept()
    _ws_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        _ws_clients.discard(websocket)
    except Exception:
        _ws_clients.discard(websocket)
