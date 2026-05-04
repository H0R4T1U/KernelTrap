#!/usr/bin/env bash
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# Activate Python virtual environment
source .venv/bin/activate

SERVER_IP="${1:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"

echo "[*] Agent connecting to server at ${SERVER_IP}:${REDIS_PORT}"

# ---------------------------------------------------------------------------
# Tracee
# ---------------------------------------------------------------------------
echo "[*] Starting Tracee..."
if docker ps -q -f name=tracee | grep -q .; then
  echo "    tracee already running — skipping"
elif docker ps -aq -f name=tracee | grep -q .; then
  docker start tracee
  echo "    tracee restarted"
else
  docker run -d --name tracee --privileged --pid=host \
    -v /etc/os-release:/etc/os-release-host:ro \
    -v /var/run/docker.sock:/var/run/docker.sock \
    aquasec/tracee:latest --output json
  echo "    tracee created and started"
fi

echo "[*] Waiting for tracee to be running..."
WAIT=0
until docker inspect -f '{{.State.Running}}' tracee 2>/dev/null | grep -q true; do
  sleep 2
  WAIT=$((WAIT+2))
  if [ $WAIT -ge 30 ]; then
    echo "    [!] Tracee did not start in 30s. Aborting."
    exit 1
  fi
done
echo "    tracee is running"

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
sudo nohup bash -c "source '$REPO_DIR/.venv/bin/activate' && docker logs -f tracee \
  | python '$REPO_DIR/masina_invata/logger/syscall_logger.py' \
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
