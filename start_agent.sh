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
echo
echo "[*] Agent running in the foreground. Press Ctrl+C to stop it."
echo "[*] Output is mirrored to logs/agent.log"
echo

# Run the agent in the FOREGROUND, attached to this terminal: Ctrl+C stops it
# cleanly and closing the terminal stops it too. `tee` mirrors output to the
# log file so there is still a persistent record. `exec` replaces this shell so
# the pipeline becomes the script's foreground process (no orphaned wrapper).
exec sudo bash -c "set -o pipefail; source '$REPO_DIR/.venv/bin/activate' && docker logs --tail 0 -f tracee \
  | python '$REPO_DIR/masina_invata/logger/syscall_logger.py' \
      --source tracee \
      --redis-host '$SERVER_IP' \
      --redis-port '$REDIS_PORT' 2>&1 | tee '$REPO_DIR/logs/agent.log'"
