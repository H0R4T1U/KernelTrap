#!/usr/bin/env bash
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

MODEL_DIR="${MODEL_DIR:-masina_invata/isolation_forest/beth_iforest_model_host2tier}"
REDIS_PORT="${REDIS_PORT:-6379}"

# ---------------------------------------------------------------------------
# 1. Python virtual environment + dependencies
# ---------------------------------------------------------------------------
echo "[*] Setting up Python environment..."
if [ ! -d "$REPO_DIR/.venv" ]; then
  echo "    Creating .venv..."
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q -r central_server/requirements.txt

# ---------------------------------------------------------------------------
# 2. Dashboard (install + build, served by FastAPI at /dashboard)
# ---------------------------------------------------------------------------
echo "[*] Building dashboard..."
npm --prefix "$REPO_DIR/dashboard" install --silent
npm --prefix "$REPO_DIR/dashboard" run build --silent
echo "    dashboard built → /dashboard"

# ---------------------------------------------------------------------------
# 3. Redis
# ---------------------------------------------------------------------------
echo "[*] Starting Redis..."
if docker ps -q -f name=kerneltrap-redis | grep -q .; then
  echo "    kerneltrap-redis already running — skipping"
elif docker ps -aq -f name=kerneltrap-redis | grep -q .; then
  docker start kerneltrap-redis
  echo "    kerneltrap-redis restarted"
else
  docker run -d --name kerneltrap-redis -p "${REDIS_PORT}":6379 redis:7
  echo "    kerneltrap-redis created and started"
fi

# ---------------------------------------------------------------------------
# 4. Central server (uvicorn)
# ---------------------------------------------------------------------------
echo "[*] Starting central server..."
if pgrep -f "uvicorn central_server.main" > /dev/null 2>&1; then
  echo "    uvicorn already running — killing and restarting"
  pkill -f "uvicorn central_server.main" || true
  sleep 1
fi

mkdir -p logs
MODEL_DIR="$MODEL_DIR" REDIS_HOST=localhost REDIS_PORT="$REDIS_PORT" \
  nohup uvicorn central_server.main:app --host 0.0.0.0 --port 8000 \
  > logs/server.log 2>&1 &
disown

sleep 2
if pgrep -f "uvicorn central_server.main" > /dev/null; then
  echo "    uvicorn started (PID $(pgrep -f 'uvicorn central_server.main'))"
else
  echo "    [!] uvicorn FAILED to start. Last log lines:"
  tail -20 logs/server.log
  exit 1
fi

echo
echo "[*] Ready. http://$(hostname -I | awk '{print $1}'):8000"
echo "[*] To stop: pkill -f uvicorn"
echo
tail -f logs/server.log
