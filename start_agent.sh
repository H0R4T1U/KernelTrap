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

mkdir -p logs
sudo nohup bash -c "docker logs -f tracee \
  | .venv/bin/python masina_invata/logger/syscall_logger.py \
      --source tracee \
      --redis-host '$SERVER_IP' \
      --redis-port '$REDIS_PORT'" \
  > logs/agent.log 2>&1 &
disown

sleep 2
if pgrep -f "syscall_logger.py" > /dev/null; then
  echo "    syscall_logger started (PID $(pgrep -f syscall_logger.py))"
else
  echo "    [!] syscall_logger FAILED to start. Last log lines:"
  tail -20 logs/agent.log
  exit 1
fi
echo
echo "[*] Agent running. Streaming logs (Ctrl+C stops streaming but agent keeps running)."
echo "[*] To stop agent: sudo pkill -f syscall_logger"
echo
tail -f logs/agent.log
