#!/usr/bin/env bash
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# Activate Python virtual environment
source .venv/bin/activate

MODEL_DIR="${MODEL_DIR:-masina_invata/isolation_forest/beth_iforest_model_host2tier}"
REDIS_PORT="${REDIS_PORT:-6379}"

# ---------------------------------------------------------------------------
# 1. Redis
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
# 2. Central server (uvicorn)
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

# ---------------------------------------------------------------------------
# 3. Dashboard (Vite dev server)
# ---------------------------------------------------------------------------
echo "[*] Starting dashboard..."
if pgrep -f "vite --port 5173" > /dev/null 2>&1; then
  echo "    dashboard already running — skipping"
else
  if [ ! -d "$REPO_DIR/dashboard/node_modules" ]; then
    echo "    Installing dashboard dependencies (first run)..."
    npm --prefix "$REPO_DIR/dashboard" install --silent
  fi
  nohup npm --prefix "$REPO_DIR/dashboard" run dev -- --port 5173 \
    > logs/dashboard.log 2>&1 &
  disown
  sleep 2
  if pgrep -f "vite --port 5173" > /dev/null; then
    echo "    dashboard started → http://localhost:5173"
  else
    echo "    [!] dashboard FAILED to start. Last log lines:"
    tail -10 logs/dashboard.log
  fi
fi

echo
echo "[*] Server running. Streaming logs (Ctrl+C stops streaming but server keeps running)."
echo "[*] Dashboard:  http://localhost:5173"
echo "[*] API:        http://localhost:8000"
echo "[*] To stop all: pkill -f uvicorn; pkill -f 'vite --port 5173'"
echo
tail -f logs/server.log
