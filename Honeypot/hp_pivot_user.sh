#!/bin/bash
# Usage: hp_pivot_user <username>
# Sends SIGUSR1 to all interactive bash sessions for that user and bans
# the source IP of each session via iptables.

IPTABLES=$(command -v iptables || echo /usr/sbin/iptables)
CONNTRACK=$(command -v conntrack || echo /usr/sbin/conntrack)

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

# Provision a matching user in the honeypot container before any pivot fires.
# The key is copied inside the container from the attacker template — no arg needed.
if docker ps --format '{{.Names}}' | grep -q '^hp-shell$'; then
  echo "Provisioning user $USER_TARGET in honeypot container..."
  docker exec hp-shell /usr/local/sbin/provision-user.sh "$USER_TARGET"
fi

for PID in $PIDS; do
  # Extract source IP from the SSH_CONNECTION env var of this specific session.
  # SSH_CONNECTION format: "client_ip client_port server_ip server_port"
  SSH_CONN=$(tr '\0' '\n' < /proc/$PID/environ 2>/dev/null | grep '^SSH_CONNECTION=' | cut -d= -f2)
  SRC_IP=$(echo "$SSH_CONN" | awk '{print $1}')

  if [ -n "$SRC_IP" ]; then
    # Drop ALL traffic from the attacker IP, not just NEW connections — a
    # reverse shell already established needs to die, not just future SSHs.
    if ! $IPTABLES -C INPUT -s "$SRC_IP" -j DROP 2>/dev/null; then
      $IPTABLES -I INPUT -s "$SRC_IP" -j DROP
      echo "Banned all traffic from $SRC_IP"
      logger -t kerneltrap "Banned all traffic from $SRC_IP for user $USER_TARGET (PID $PID)"
    else
      echo "IP $SRC_IP already banned"
    fi
    # Tear down existing conntrack entries so live shells from that IP die now.
    if [ -x "$CONNTRACK" ]; then
      $CONNTRACK -D -s "$SRC_IP" >/dev/null 2>&1 || true
    fi
  else
    echo "Could not determine source IP for PID $PID (not an SSH session?)"
  fi

  echo "Sending SIGUSR1 to PID $PID (user $USER_TARGET)"
  kill -USR1 "$PID"
done
