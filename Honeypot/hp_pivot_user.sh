#!/bin/bash
# Usage: hp_pivot_user <username>
# Sends SIGUSR1 to all interactive bash sessions for that user,
# causing them to pivot into the honeypot container.

USER_TARGET="$1"
if [ -z "$USER_TARGET" ]; then
  echo "Usage: $0 <username>" >&2
  exit 1
fi

PIDS=$(ps -u "$USER_TARGET" -o pid=,cmd= | awk '/bash$/ {print $1}')

if [ -z "$PIDS" ]; then
  echo "No interactive bash sessions found for user $USER_TARGET"
  exit 0
fi

for PID in $PIDS; do
  echo "Sending SIGUSR1 to PID $PID (user $USER_TARGET)"
  kill -USR1 "$PID"
done
