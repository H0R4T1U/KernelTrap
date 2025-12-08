# Honeypot pivot trap for all interactive bash shells

# Only for interactive shells
if [ -n "$PS1" ]; then
  if [ -z "$HPTRAP_INITIALIZED" ]; then
    export HPTRAP_INITIALIZED=1

    hptrap_pivot() {
      # Log pivot
      logger -t hptrap "Pivoting user $USER from host to honeypot (shell PID=$$, from $SSH_CONNECTION)"

      # Exec into honeypot SSH (Docker container on localhost:2222)
      exec ssh \
        -i /etc/hptrap/hp_key \
        -o StrictHostKeyChecking=no \
        attacker@127.0.0.1 -p 2222
    }

    # When this shell receives SIGUSR1, run hptrap_pivot()
    trap hptrap_pivot USR1
  fi
fi
