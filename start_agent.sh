#!/usr/bin/env bash
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

SERVER_IP="${1:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"

echo "[*] Agent connecting to server at ${SERVER_IP}:${REDIS_PORT}"

# ---------------------------------------------------------------------------
# Syscall logger (tracee)
# ---------------------------------------------------------------------------
echo "[*] Starting syscall logger..."
if pgrep -f "syscall_logger.py" > /dev/null 2>&1; then
  echo "    syscall_logger already running — killing and restarting"
  sudo pkill -f "syscall_logger.py" || true
  sleep 1
fi

sudo tracee --output json \
  | .venv/bin/python masina_invata/logger/syscall_logger.py \
      --source tracee \
      --redis-host "$SERVER_IP" \
      --redis-port "$REDIS_PORT" &

echo "    tracee pipeline started (PID $!)"
echo
echo "[*] Agent running. To stop:"
echo "    sudo pkill -f syscall_logger; sudo pkill -f tracee"
