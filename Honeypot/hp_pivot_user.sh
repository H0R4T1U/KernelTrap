#!/bin/bash
# Usage: hp_pivot_user <username>
# Sends SIGUSR1 to all interactive bash sessions for that user and bans
# the source IP of each session via iptables.

USER_TARGET="$1"
if [ -z "$USER_TARGET" ]; then
  echo "Usage: $0 <username>" >&2
  exit 1
fi

# Match both login shells (-bash) and non-login shells (bash)
PIDS=$(ps -u "$USER_TARGET" -o pid=,cmd= | awk '/-?bash$/ {print $1}')

if [ -z "$PIDS" ]; then
  echo "No interactive bash sessions found for user $USER_TARGET"
  exit 0
fi

for PID in $PIDS; do
  # Extract source IP from the SSH_CONNECTION env var of this specific session.
  # SSH_CONNECTION format: "client_ip client_port server_ip server_port"
  SSH_CONN=$(tr '\0' '\n' < /proc/$PID/environ 2>/dev/null | grep '^SSH_CONNECTION=' | cut -d= -f2)
  SRC_IP=$(echo "$SSH_CONN" | awk '{print $1}')

  if [ -n "$SRC_IP" ]; then
    # DOCKER-USER is checked before Docker's RELATED,ESTABLISHED rules,
    # so it's the only chain that reliably blocks connections on Docker hosts.
    if ! iptables -C DOCKER-USER -s "$SRC_IP" -j DROP 2>/dev/null; then
      iptables -I DOCKER-USER -s "$SRC_IP" -j DROP
      echo "Banned IP $SRC_IP (session PID $PID, user $USER_TARGET)"
      logger -t kerneltrap "Banned IP $SRC_IP for user $USER_TARGET (PID $PID)"
    else
      echo "IP $SRC_IP already banned"
    fi
  else
    echo "Could not determine source IP for PID $PID (not an SSH session?)"
  fi

  echo "Sending SIGUSR1 to PID $PID (user $USER_TARGET)"
  kill -USR1 "$PID"
done
