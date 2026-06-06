#!/usr/bin/env bash
# ============================================================================
# perf_collect.sh — Colectează metricile de performanță pentru tab:cap6-perf
#                   din capitolul 6 al lucrării de licență.
#
# Măsoară pentru un proces (agent sau server):
#   - CPU% mediu (idle și sub încărcare)
#   - RAM rezidentă (RSS, în MB)
# Pentru server măsoară suplimentar:
#   - Throughput susținut (evenimente/secundă)
#
# Cerințe pe gazdă:
#   - sysstat (pidstat) — `sudo apt install sysstat`
#   - redis-cli         — `sudo apt install redis-tools`
#
# Usage:
#   ./perf_collect.sh agent          # măsoară agentul local (CPU/RAM idle)
#   ./perf_collect.sh agent --load   # idle 60s + workload 60s + RAM
#   ./perf_collect.sh server         # măsoară serverul local (CPU/RAM + throughput)
#   ./perf_collect.sh --duration N   # durata fiecărei faze în secunde (default 60)
# ============================================================================

set -u
TARGET="${1:-}"
shift || true

DURATION=60
WITH_LOAD=0
REDIS_HOST="${REDIS_HOST:-127.0.0.1}"
REDIS_PORT="${REDIS_PORT:-6379}"

while (( $# > 0 )); do
  case "$1" in
    --duration) DURATION="$2"; shift 2 ;;
    --load)     WITH_LOAD=1;   shift ;;
    *) echo "Opțiune necunoscută: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$TARGET" || ( "$TARGET" != "agent" && "$TARGET" != "server" ) ]]; then
  cat <<'EOF' >&2
Folosire: ./perf_collect.sh {agent|server} [--load] [--duration N]

  agent          — măsoară agentul local (syscall_logger.py)
  server         — măsoară serverul central (uvicorn)
  --load         — adaugă o fază de workload după faza idle (doar pentru agent)
  --duration N   — durata fiecărei faze în secunde (default 60)
EOF
  exit 1
fi

# -------- Detecție PID -------------------------------------------------------
case "$TARGET" in
  agent)
    PID=$(pgrep -f "syscall_logger.py" | head -1)
    PROC_LABEL="Agent (syscall_logger.py)"
    ;;
  server)
    # Caut uvicorn care servește central_server.main:app
    PID=$(pgrep -f "uvicorn.*central_server" | head -1)
    [[ -z "$PID" ]] && PID=$(pgrep -f "central_server.main" | head -1)
    PROC_LABEL="Server central (uvicorn)"
    ;;
esac

if [[ -z "${PID:-}" ]]; then
  echo "EROARE: nu am găsit procesul pentru ${TARGET}." >&2
  echo "Asigură-te că procesul rulează (start_${TARGET}.sh) înainte de a rula benchmark-ul." >&2
  exit 2
fi

echo "=== perf_collect.sh ==="
echo "Țintă       : ${PROC_LABEL}"
echo "PID         : ${PID}"
echo "Durată/fază : ${DURATION} s"
echo

if ! command -v pidstat >/dev/null 2>&1; then
  echo "EROARE: pidstat lipsește. Instalează: sudo apt install sysstat" >&2
  exit 3
fi

# -------- Funcție utilitară: pidstat → CPU% mediu, RAM RSS mediu (MB) --------
measure_pidstat() {
  local pid=$1 duration=$2 label=$3
  local raw
  echo "[${label}] măsurare ${duration}s pe PID ${pid}..."
  # pidstat -u -r -p PID 1 N: 1s sampling, N samples; -h pentru header omogen
  raw=$(pidstat -h -u -r -p "$pid" 1 "$duration" 2>/dev/null)
  if [[ -z "$raw" ]]; then
    echo "[${label}] AVERTISMENT: pidstat n-a întors date. PID-ul mai rulează?"
    return
  fi

  # Mediile sunt deja pe ultima linie "Average:"; le extragem din output -h.
  # Format -h (header pe o linie):
  # #      Time   UID       PID    %usr %system  %guest    %CPU   CPU  ...  RSS   ...  Command
  # Calculez mediile pe câmpul %CPU și RSS (KB).
  local cpu rss_kb
  cpu=$(awk '!/^#/ && NF>0 { sum+=$8; n++ } END { if (n>0) printf "%.2f", sum/n; else print "n/a" }' <<<"$raw")
  rss_kb=$(awk '!/^#/ && NF>0 { sum+=$13; n++ } END { if (n>0) printf "%.0f", sum/n; else print "n/a" }' <<<"$raw")
  local rss_mb="n/a"
  if [[ "$rss_kb" != "n/a" ]]; then
    rss_mb=$(awk -v k="$rss_kb" 'BEGIN { printf "%.1f", k/1024 }')
  fi
  echo "[${label}] CPU%=${cpu}, RSS=${rss_mb} MB"
  # publicăm prin variabile globale pentru sumar
  declare -g "${label^^}_CPU=$cpu"
  declare -g "${label^^}_RSS_MB=$rss_mb"
}

# -------- Faza IDLE ----------------------------------------------------------
echo "[FAZA 1] Repaus — nu rula nimic suplimentar pe gazdă timp de ${DURATION}s."
sleep 2  # mic buffer
measure_pidstat "$PID" "$DURATION" "idle"
echo

# -------- Faza WORKLOAD (opțional, pentru agent) -----------------------------
if (( WITH_LOAD )) && [[ "$TARGET" == "agent" ]]; then
  echo "[FAZA 2] Workload — generez activitate sintetică (stat + ls în buclă)..."
  # Workload simplu: 2 worker-i în paralel care fac stat/ls repetat
  (
    end_time=$(( $(date +%s) + DURATION ))
    while (( $(date +%s) < end_time )); do
      for f in /etc/passwd /etc/hostname /etc/hosts /etc/os-release; do
        stat "$f" >/dev/null 2>&1
      done
      ls /tmp >/dev/null 2>&1
    done
  ) &
  LOAD_PID1=$!
  (
    end_time=$(( $(date +%s) + DURATION ))
    while (( $(date +%s) < end_time )); do
      cat /proc/loadavg >/dev/null 2>&1
      id >/dev/null 2>&1
      uname -a >/dev/null 2>&1
    done
  ) &
  LOAD_PID2=$!
  measure_pidstat "$PID" "$DURATION" "load"
  wait "$LOAD_PID1" "$LOAD_PID2" 2>/dev/null || true
  echo
fi

# -------- Throughput susținut (doar server) ----------------------------------
if [[ "$TARGET" == "server" ]]; then
  if ! command -v redis-cli >/dev/null 2>&1; then
    echo "AVERTISMENT: redis-cli lipsește, sar peste măsurătoarea de throughput."
  else
    HOSTNAME_SHORT=$(hostname -s 2>/dev/null || hostname)
    STREAM="events.${HOSTNAME_SHORT}"
    echo "[FAZA 3] Throughput pe stream-ul Redis ${STREAM} (${DURATION}s)..."
    LEN0=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" XLEN "$STREAM" 2>/dev/null || echo 0)
    sleep "$DURATION"
    LEN1=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" XLEN "$STREAM" 2>/dev/null || echo 0)
    DELTA=$(( LEN1 - LEN0 ))
    RATE=$(awk -v d="$DELTA" -v t="$DURATION" 'BEGIN { printf "%.1f", d/t }')
    echo "[FAZA 3] Δ evenimente=${DELTA}, throughput=${RATE} ev/s"
    THROUGHPUT="$RATE"
  fi
fi

# -------- Sumar pentru tab:cap6-perf -----------------------------------------
echo
echo "================ SUMAR pentru tab:cap6-perf ================"
case "$TARGET" in
  agent)
    echo "CPU agent (idle)              : ${IDLE_CPU:-n/a}%"
    if (( WITH_LOAD )); then
      echo "CPU agent (sub încărcare)     : ${LOAD_CPU:-n/a}%"
      echo "RAM agent                     : ${LOAD_RSS_MB:-${IDLE_RSS_MB:-n/a}} MB"
    else
      echo "RAM agent                     : ${IDLE_RSS_MB:-n/a} MB"
    fi
    ;;
  server)
    echo "RAM server central            : ${IDLE_RSS_MB:-n/a} MB"
    echo "Throughput susținut server    : ${THROUGHPUT:-n/a} ev/s"
    ;;
esac
echo "============================================================"
