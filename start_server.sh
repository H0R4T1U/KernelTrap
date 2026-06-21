#!/usr/bin/env bash
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

MODEL_DIR="${MODEL_DIR:-masina_invata/isolation_forest/beth_iforest_model_host2tier}"
REDIS_PORT="${REDIS_PORT:-6379}"

# ---------------------------------------------------------------------------
# 0. Prerequisite checks
# ---------------------------------------------------------------------------
for cmd in python3 npm docker; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "[!] '$cmd' not found in PATH" >&2; exit 1; }
done

# ---------------------------------------------------------------------------
# 1. Python venv (create + install only on first run)
# ---------------------------------------------------------------------------
if [ ! -d "$REPO_DIR/.venv" ]; then
  echo "[*] Creating .venv (first run)..."
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r central_server/requirements.txt
else
  source .venv/bin/activate
fi

# Verify uvicorn is importable; reinstall if not
if ! python -c "import uvicorn, fastapi, redis" 2>/dev/null; then
  echo "[*] Python deps missing — installing..."
  pip install -r central_server/requirements.txt
fi

# ---------------------------------------------------------------------------
# 2. Dashboard (install on first run, build only when src changed)
# ---------------------------------------------------------------------------
if [ ! -d "$REPO_DIR/dashboard/node_modules" ]; then
  echo "[*] Installing dashboard deps (first run)..."
  npm --prefix "$REPO_DIR/dashboard" install
fi

REBUILD=0
if [ ! -d "$REPO_DIR/dashboard/dist" ]; then
  REBUILD=1
elif [ -n "$(find "$REPO_DIR/dashboard/src" "$REPO_DIR/dashboard/index.html" "$REPO_DIR/dashboard/vite.config.js" -newer "$REPO_DIR/dashboard/dist" 2>/dev/null | head -1)" ]; then
  REBUILD=1
fi
if [ "$REBUILD" = "1" ]; then
  echo "[*] Building dashboard..."
  npm --prefix "$REPO_DIR/dashboard" run build
else
  echo "[*] Dashboard up-to-date — skipping build"
fi

# ---------------------------------------------------------------------------
# 3. Redis
# ---------------------------------------------------------------------------
echo "[*] Starting Redis..."
if docker ps -q -f name=kerneltrap-redis | grep -q .; then
  echo "    kerneltrap-redis already running"
elif docker ps -aq -f name=kerneltrap-redis | grep -q .; then
  docker start kerneltrap-redis >/dev/null
  echo "    kerneltrap-redis restarted"
else
  docker run -d --name kerneltrap-redis -p "${REDIS_PORT}":6379 redis:7 >/dev/null
  echo "    kerneltrap-redis created and started"
fi

# Wait for Redis to actually accept connections
echo -n "    Waiting for Redis"
for i in $(seq 1 20); do
  if docker exec kerneltrap-redis redis-cli ping 2>/dev/null | grep -q PONG; then
    echo " — ready"
    break
  fi
  echo -n "."
  sleep 0.5
  if [ "$i" = "20" ]; then
    echo
    echo "[!] Redis did not become ready in 10s" >&2
    exit 1
  fi
done

# ---------------------------------------------------------------------------
# 4. Central server (uvicorn)
# ---------------------------------------------------------------------------
echo "[*] Starting central server..."
if pgrep -f "uvicorn central_server.main:app" > /dev/null 2>&1; then
  echo "    uvicorn already running — restarting"
  pkill -f "uvicorn central_server.main:app" || true
  sleep 1
fi

mkdir -p logs

# Pivot tuning — override with env vars if needed.
# Extra UIDs to never pivot beyond the built-in system daemon blocklist.
# Service accounts (www-data=33, apache=48, mysql=27, etc.) are tracked by default.
: "${WHITELIST_UIDS:=}"
# severity-2 events floor in 60s window — raised from 5: VOLUME is what separates
# loud recon (linpeas = a sustained flood of sev-2) from a quiet ls/whoami session
# (a handful). Calibrate against /users on a real run. The count floor is the
# false-positive guard now that the rate gate below is loose.
: "${PIVOT_THRESHOLD:=10}"
# rate gate: ≥this fraction of events in the window must be sev-2 — lowered from
# 0.30 so linpeas's diluted concentration (lots of benign syscalls) still passes.
: "${MIN_SEV2_RATE:=0.10}"
# raw-score cutoff for severity 2 (live-tuned, not BETH). LESS negative = MORE
# sensitive. Kept STRICT at -0.15: the model is BETH-trained and scores ordinary
# interactive commands (whoami/ls/bash have processName_freq=0) in the SAME band
# as real recon, so a looser cutoff (-0.10/-0.05) pivots benign ls/whoami sessions.
# Discrimination is delegated to the dual gate above (high count + low rate), not
# to this cutoff. -0.15 is also stricter than the model's calibrated high (-0.0847).
: "${OVERRIDE_HIGH_THRESHOLD:=-0.15}"

MODEL_DIR="$MODEL_DIR" REDIS_HOST=localhost REDIS_PORT="$REDIS_PORT" \
  WHITELIST_UIDS="$WHITELIST_UIDS" \
  PIVOT_THRESHOLD="$PIVOT_THRESHOLD" \
  MIN_SEV2_RATE="$MIN_SEV2_RATE" \
  OVERRIDE_HIGH_THRESHOLD="$OVERRIDE_HIGH_THRESHOLD" \
  nohup uvicorn central_server.main:app --host 0.0.0.0 --port 8000 \
  > logs/server.log 2>&1 &
disown

sleep 2
if pgrep -f "uvicorn central_server.main:app" > /dev/null; then
  echo "    uvicorn started (PID $(pgrep -f 'uvicorn central_server.main:app'))"
else
  echo "[!] uvicorn FAILED to start. Last log lines:" >&2
  tail -20 logs/server.log >&2
  exit 1
fi

IP=$(hostname -I 2>/dev/null | awk '{print $1}')
[ -z "$IP" ] && IP="localhost"
echo
echo "[*] Ready. http://${IP}:8000"
echo "[*] Dashboard: http://${IP}:8000/dashboard/"
echo "[*] To stop: pkill -f 'uvicorn central_server.main:app'"
echo
tail -f logs/server.log
