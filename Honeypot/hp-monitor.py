#!/usr/bin/env python3
"""
hp-monitor.py — Tail /var/log/audit/audit.log and forward syscall events
to the central Redis server under the stream events.<hostname>.
Runs as a background daemon inside the hp-shell honeypot container.
"""
import json
import os
import re
import select
import socket
import subprocess
import time

import redis

REDIS_HOST     = os.environ.get("REDIS_HOST", "host.docker.internal")
REDIS_PORT     = int(os.environ.get("REDIS_PORT", 6379))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD") or None
# Prefer the explicit honeypot hostname so multiple honeypots don't collide on
# the same Redis stream when the container hostname is generic (`hp-shell`).
HOSTNAME       = os.environ.get("HP_HOSTNAME") or socket.gethostname()
STREAM         = f"events.{HOSTNAME}"
AUDIT_LOG      = "/var/log/audit/audit.log"
FLUSH_INTERVAL = 5.0   # seconds — flush even if batch hasn't hit 50 events
BATCH_SIZE     = 50

SYSCALL_MAP = {
    "2":   "open",
    "59":  "execve",
    "257": "openat",
}

_TS_RE = re.compile(r'audit\((\d+\.\d+):')


def parse_audit_line(line: str) -> dict:
    return dict(re.findall(r'(\w+)=(".*?"|\S+)', line))


def extract_timestamp(msg_field: str) -> float:
    m = _TS_RE.search(msg_field)
    return float(m.group(1)) if m else 0.0


def connect_redis() -> redis.Redis:
    while True:
        try:
            r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True,
                            socket_connect_timeout=5)
            r.ping()
            return r
        except Exception:
            time.sleep(10)


def flush(r: redis.Redis, batch: list) -> redis.Redis:
    """Send batch to Redis. On failure reconnect and retry once; drop only if still failing.

    Field key MUST be "data" — central_server/main.py reads
    ``fields.get("data", "[]")`` and silently skips anything else.
    Previously this used "batch" so honeypot events were never scored.
    """
    try:
        r.xadd(STREAM, {"data": json.dumps(batch)})
    except Exception:
        r = connect_redis()
        try:
            r.xadd(STREAM, {"data": json.dumps(batch)})
        except Exception:
            pass  # drop batch if Redis is still unreachable after reconnect
    return r


def main() -> None:
    r = connect_redis()
    r.xadd("kerneltrap.registrations", {"hostname": HOSTNAME})

    proc = subprocess.Popen(
        ["tail", "-F", "-n", "0", AUDIT_LOG],
        stdout=subprocess.PIPE, text=True, bufsize=1
    )

    batch: list = []
    last_flush = time.monotonic()

    while True:
        ready, _, _ = select.select([proc.stdout], [], [], FLUSH_INTERVAL)
        now = time.monotonic()

        if ready:
            raw_line = proc.stdout.readline()
            if not raw_line:
                break  # tail exited — outer loop will restart main()

            if "type=SYSCALL" in raw_line:
                f = parse_audit_line(raw_line)
                syscall_nr   = f.get("syscall", "")
                syscall_name = SYSCALL_MAP.get(syscall_nr, f"syscall_{syscall_nr}")
                event = {
                    "timestamp":       extract_timestamp(f.get("msg", "")),
                    "processId":       int(f.get("pid",  0)),
                    "parentProcessId": int(f.get("ppid", 0)),
                    "userId":          int(f.get("uid",  -1)),
                    "mountNamespace":  0,
                    "eventId":         int(syscall_nr) if syscall_nr.isdigit() else 0,
                    "eventName":       syscall_name,
                    "argsNum":         0,
                    "returnValue":     int(f.get("exit", 0)),
                    "hostName":        HOSTNAME,
                    "sus":             0,
                    "evil":            0,
                    "args":            [],
                }
                batch.append(event)

        # Flush when batch is full or flush interval has elapsed
        if batch and (len(batch) >= BATCH_SIZE or now - last_flush >= FLUSH_INTERVAL):
            r = flush(r, batch)
            batch.clear()
            last_flush = now


if __name__ == "__main__":
    while True:
        try:
            main()
        except Exception:
            time.sleep(5)
