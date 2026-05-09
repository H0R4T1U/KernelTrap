"""
KernelTrap Central Analysis Server

Subscribes to events.{hostname} Redis Streams from all agents,
scores them with Isolation Forest, and publishes results back:

  scores.{hostname}   <- anomaly score per event (for dashboard)
  commands.{hostname} <- pivot commands when threshold is crossed

Run:
    uvicorn central_server.main:app --host 0.0.0.0 --port 8000

Environment variables:
    REDIS_HOST          Redis host (default: localhost)
    REDIS_PORT          Redis port (default: 6379)
    MODEL_DIR           Path to trained IF model dir
    PIVOT_THRESHOLD     Severity-2 events in window to trigger pivot (default: 5)
    PIVOT_WINDOW_SEC    Sliding window in seconds (default: 60)
    WHITELIST_UIDS      Extra comma-separated UIDs to never pivot (beyond built-in daemon list)
    HOST_DISCOVERY_SEC  How often to scan Redis for new agent streams (default: 30)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import pwd
import sys
import time
from collections import deque
from typing import Any, Dict, List, Optional, Set

import numpy as np
import redis.asyncio as aioredis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

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
# UIDs that are always ignored — pure system daemons with no interactive session.
# Service accounts that are attacker targets (www-data=33, apache=48, mysql=27,
# postgres=26, redis=999) are intentionally NOT in this list.
_DEFAULT_SYSTEM_UIDS = (
    # root handled by whitelist; 1-32 are kernel/init accounts
    *range(1, 33),
    # common noisy system daemons (distro-specific; adjust if needed)
    65,   # kvm
    100, 101, 102, 103, 104, 105, 106, 107, 108, 109,  # dbus, syslog, uuidd, …
    110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
    120, 121, 122, 123, 124, 125,
    993, 994, 995, 996, 997, 998,  # systemd-network, systemd-resolve, …
)
WHITELIST_UIDS: set[int] = {0}  # root
WHITELIST_UIDS.update(_DEFAULT_SYSTEM_UIDS)
# Allow the operator to add extra UIDs via env var
_extra = os.getenv("WHITELIST_UIDS", "")
WHITELIST_UIDS.update(int(u) for u in _extra.split(",") if u.strip().isdigit())

HOST_DISCOVERY_SEC = int(os.getenv("HOST_DISCOVERY_SEC", "30"))

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

# Bounded sample of recent raw_score values from the scorer. Used by
# /model/health to compare the live distribution of normality scores against
# the percentiles the model was calibrated at — a cheap drift indicator.
_RECENT_SCORE_SAMPLE_MAX = 100_000
_recent_scores: deque = deque(maxlen=_RECENT_SCORE_SAMPLE_MAX)


def _uid_to_username(uid: int) -> str:
    """Resolve a UID to a username. Falls back to str(uid) if not found.

    The honeypot pivot script (Honeypot/hp_pivot_user.sh) operates on usernames,
    not numeric UIDs, so we must resolve before publishing the command.
    """
    try:
        return pwd.getpwuid(uid).pw_name
    except (KeyError, OSError):
        return str(uid)


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
        if "raw_score" in result:
            _recent_scores.append(float(result["raw_score"]))
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
                    username = _uid_to_username(user_id)
                    command_key = f"commands.{hostname}"
                    cmd = json.dumps({"action": "pivot", "user": username})
                    await _redis.xadd(
                        command_key,
                        {"data": cmd},
                        maxlen=1000,
                        approximate=True,
                    )
                    log.warning(
                        f"PIVOT COMMAND sent → host='{hostname}', user='{username}' (uid={user_id}), trigger={trigger}"
                    )
                    _pivot_history.append({
                        "timestamp": time.time(),
                        "hostname": hostname,
                        "user_id": user_id,
                        "username": username,
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
            _ws_clients.difference_update(dead)
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

    username = _uid_to_username(user_id)
    command_key = f"commands.{hostname}"
    cmd = json.dumps({"action": "pivot", "user": username})
    await _redis.xadd(command_key, {"data": cmd}, maxlen=1000, approximate=True)

    _pivot_history.append({
        "timestamp": time.time(),
        "hostname": hostname,
        "user_id": user_id,
        "username": username,
        "trigger": "manual",
    })
    log.warning(f"MANUAL PIVOT sent → host='{hostname}', user='{username}' (uid={user_id})")
    return JSONResponse({"status": "pivot_sent", "hostname": hostname, "user_id": user_id, "username": username})


@app.get("/pivot-history")
async def pivot_history():
    """Return pivot history, most recent first."""
    return JSONResponse({"pivots": list(reversed(_pivot_history))})


@app.get("/model/health")
async def model_health():
    """Drift indicator for the loaded IsolationForest model.

    Compares the live distribution of normality scores (sampled from the last
    ~100k events) against the percentiles the model was calibrated at. If the
    observed severity-1 hit rate is more than 3× the expected percentile, we
    flag the model as drift-suspected — likely time to retrain.

    Pure observability endpoint, no auth, no side effects.
    """
    if not _scorer:
        return JSONResponse({"status": "scorer-not-loaded"}, status_code=503)

    sample = list(_recent_scores)
    n = len(sample)
    out: Dict[str, Any] = {
        "samples": n,
        "global_low_threshold":  _scorer._global_low,
        "global_high_threshold": _scorer._global_high,
        "expected_low_pct":      _scorer._meta.get("low_percentile"),
        "expected_high_pct":     _scorer._meta.get("high_percentile"),
        "n_features":            len(_scorer._meta.get("features", [])),
    }
    if n < 100:
        out["status"] = "warming-up"
        return JSONResponse(out)

    arr = np.asarray(sample, dtype=float)
    observed_low  = float((arr < _scorer._global_low).mean()  * 100.0)
    observed_high = float((arr < _scorer._global_high).mean() * 100.0)
    out["observed_low_pct"]  = observed_low
    out["observed_high_pct"] = observed_high
    out["score_mean"]        = float(arr.mean())
    out["score_std"]         = float(arr.std())
    out["score_min"]         = float(arr.min())
    out["score_max"]         = float(arr.max())

    expected_low = float(out["expected_low_pct"] or 0.0)
    if expected_low > 0 and observed_low > expected_low * 3:
        out["status"] = "drift-suspected"
    else:
        out["status"] = "ok"
    return JSONResponse(out)


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


# ---------------------------------------------------------------------------
# Static file serving — must be mounted AFTER all API routes so the catch-all
# "/" mount does not shadow any endpoint.
# ---------------------------------------------------------------------------

@app.get("/dashboard")
async def dashboard_redirect():
    return RedirectResponse(url="/dashboard/")


_DASHBOARD_DIST = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dashboard", "dist"))
if os.path.isdir(_DASHBOARD_DIST):
    app.mount("/dashboard", StaticFiles(directory=_DASHBOARD_DIST, html=True), name="dashboard")
else:
    log.warning(f"Dashboard dist not found at {_DASHBOARD_DIST} — /dashboard/ will 404. Run: npm --prefix dashboard run build")

_WEBPAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "webpage"))
if os.path.isdir(_WEBPAGE_DIR):
    app.mount("/", StaticFiles(directory=_WEBPAGE_DIR, html=True), name="webpage")
else:
    log.warning(f"Webpage dir not found at {_WEBPAGE_DIR} — landing page will 404")
