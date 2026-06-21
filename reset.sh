#!/usr/bin/env bash
set -e

echo "[*] Stopping server and agent..."
pkill -f uvicorn 2>/dev/null || true
sudo pkill -f syscall_logger 2>/dev/null || true
sleep 1

echo "[*] Stopping tracee container..."
sudo docker stop tracee 2>/dev/null || true

echo "[*] Clearing Redis streams for $(hostname)..."
sudo docker exec kerneltrap-redis redis-cli DEL \
  "events.$(hostname)" "scores.$(hostname)" "commands.$(hostname)"

echo "[*] Done. Now run: ./start_server.sh  and  ./start_agent.sh"
