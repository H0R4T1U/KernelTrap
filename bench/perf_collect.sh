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
#   - ps, awk (oricum prezente)
#   - redis-cli         — `sudo apt install redis-tools` (doar pentru server)
#
# Usage:
#   ./perf_collect.sh agent           # măsoară agentul local (CPU/RAM idle)
#   ./perf_collect.sh agent --load    # idle 60s + workload 60s
#   ./perf_collect.sh server          # măsoară serverul local (CPU/RAM + throughput)
#   ./perf_collect.sh --duration N    # durata fiecărei faze în secunde (default 60)
#   ./perf_collect.sh server --stream events.NUME  # override numele stream-ului
# ============================================================================

set -u
TARGET="${1:-}"
shift || true

DURATION=60
WITH_LOAD=0
REDIS_HOST="${REDIS_HOST:-127.0.0.1}"
REDIS_PORT="${REDIS_PORT:-6379}"
STREAM_OVERRIDE=""

while (( $# > 0 )); do
  case "$1" in
    --duration) DURATION="$2"; shift 2 ;;
    --load)     WITH_LOAD=1;   shift ;;
    --stream)   STREAM_OVERRIDE="$2"; shift 2 ;;
    *) echo "Opțiune necunoscută: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$TARGET" || ( "$TARGET" != "agent" && "$TARGET" != "server" ) ]]; then
  cat <<'EOF' >&2
Folosire: ./perf_collect.sh {agent|server} [--load] [--duration N] [--stream NAME]

  agent          — măsoară agentul local (syscall_logger.py)
  server         — măsoară serverul central (uvicorn) + throughput Redis
  --load         — adaugă o fază de workload după faza idle (doar pentru agent)
  --duration N   — durata fiecărei faze în secunde (default 60)
  --stream NAME  — numele stream-ului Redis pentru throughput (default: auto-detect)
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

# -------- Funcție: măsoară CPU% + RSS folosind ps într-o buclă (1 Hz) -------
# Avantaj: ps e portabil, fără dependențe (sysstat poate lipsi), output stabil.
# ps -o %cpu= -o rss= -p $PID întoarce: %CPU(float)  RSS(KB)
measure_ps() {
  local pid=$1 duration=$2 label=$3
  local cpu_sum=0 rss_sum=0 n=0
  local out cpu rss

  echo "[${label}] măsurare ${duration}s pe PID ${pid} (ps polling 1 Hz)..."
  for ((i=0; i<duration; i++)); do
    out=$(ps -o %cpu= -o rss= -p "$pid" 2>/dev/null)
    if [[ -z "$out" ]]; then
      echo "[${label}] AVERTISMENT: PID ${pid} nu mai există."
      break
    fi
    cpu=$(awk '{print $1}' <<<"$out")
    rss=$(awk '{print $2}' <<<"$out")
    cpu_sum=$(awk -v a="$cpu_sum" -v b="$cpu" 'BEGIN { print a+b }')
    rss_sum=$(awk -v a="$rss_sum" -v b="$rss" 'BEGIN { print a+b }')
    n=$((n+1))
    sleep 1
  done

  if (( n == 0 )); then
    echo "[${label}] nu s-au colectat eșantioane."
    return
  fi

  local cpu_avg rss_avg_mb
  cpu_avg=$(awk -v s="$cpu_sum" -v n="$n" 'BEGIN { printf "%.2f", s/n }')
  rss_avg_mb=$(awk -v s="$rss_sum" -v n="$n" 'BEGIN { printf "%.1f", s/n/1024 }')
  echo "[${label}] eșantioane=${n}, CPU%=${cpu_avg}, RSS=${rss_avg_mb} MB"

  declare -g "${label^^}_CPU=$cpu_avg"
  declare -g "${label^^}_RSS_MB=$rss_avg_mb"
}

# -------- Faza IDLE ----------------------------------------------------------
echo "[FAZA 1] Repaus — nu rula nimic suplimentar pe gazdă timp de ${DURATION}s."
sleep 2  # mic buffer pentru a evita CPU% reziduale de la pornire
measure_ps "$PID" "$DURATION" "idle"
echo

# -------- Faza WORKLOAD (opțional, pentru agent) -----------------------------
if (( WITH_LOAD )) && [[ "$TARGET" == "agent" ]]; then
  echo "[FAZA 2] Workload — generez activitate sintetică pentru a stresa agentul..."
  # 4 worker-i paraleli care fac sutele de syscall-uri pe secundă fiecare.
  # stat/cat/ls sunt fast-path = mai multe syscall-uri/s = mai multe evenimente
  # capturate de Tracee = mai mult traffic la agent.
  for w in 1 2 3 4; do
    (
      end_time=$(( $(date +%s) + DURATION ))
      while (( $(date +%s) < end_time )); do
        for f in /etc/passwd /etc/hostname /etc/hosts /etc/os-release /etc/group; do
          stat "$f" >/dev/null 2>&1
          cat "$f" >/dev/null 2>&1
        done
        ls /tmp /var /etc >/dev/null 2>&1
        id >/dev/null 2>&1
      done
    ) &
  done
  measure_ps "$PID" "$DURATION" "load"
  wait
  echo
fi

# -------- Throughput susținut (doar server) ----------------------------------
if [[ "$TARGET" == "server" ]]; then
  if ! command -v redis-cli >/dev/null 2>&1; then
    echo "AVERTISMENT: redis-cli lipsește, sar peste măsurătoarea de throughput."
  else
    # Auto-detect numele stream-ului
    if [[ -n "$STREAM_OVERRIDE" ]]; then
      STREAM="$STREAM_OVERRIDE"
      echo "[FAZA 3] Stream override: ${STREAM}"
    else
      # Caut TOATE stream-urile events.* și le iau pe cele cu lungime > 0
      echo "[FAZA 3] Caut stream-uri events.* în Redis..."
      mapfile -t STREAMS < <(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" --scan --pattern 'events.*' 2>/dev/null)
      if (( ${#STREAMS[@]} == 0 )); then
        echo "AVERTISMENT: nu am găsit niciun stream events.* în Redis."
        echo "             Verifică dacă agentul publică (pe agent: pgrep -af syscall_logger.py)."
        STREAM=""
      else
        # Aleg stream-ul cu cele mai multe intrări (cel mai activ agent)
        best_stream=""; best_len=-1
        for s in "${STREAMS[@]}"; do
          len=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" XLEN "$s" 2>/dev/null)
          [[ -z "$len" ]] && len=0
          echo "  - ${s} → XLEN=${len}"
          if (( len > best_len )); then
            best_len="$len"; best_stream="$s"
          fi
        done
        STREAM="$best_stream"
        echo "[FAZA 3] Stream ales (cel mai activ): ${STREAM}"
      fi
    fi

    if [[ -n "$STREAM" ]]; then
      LEN0=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" XLEN "$STREAM" 2>/dev/null || echo 0)
      echo "[FAZA 3] Măsoare throughput ${DURATION}s pe ${STREAM}..."
      sleep "$DURATION"
      LEN1=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" XLEN "$STREAM" 2>/dev/null || echo 0)
      DELTA=$(( LEN1 - LEN0 ))
      RATE=$(awk -v d="$DELTA" -v t="$DURATION" 'BEGIN { printf "%.1f", d/t }')
      echo "[FAZA 3] XLEN start=${LEN0}, end=${LEN1}, Δ=${DELTA}, throughput=${RATE} ev/s"
      THROUGHPUT="$RATE"
    fi
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
      echo "RAM agent (sub încărcare)     : ${LOAD_RSS_MB:-n/a} MB"
    fi
    echo "RAM agent (idle)              : ${IDLE_RSS_MB:-n/a} MB"
    ;;
  server)
    echo "CPU server central            : ${IDLE_CPU:-n/a}%"
    echo "RAM server central            : ${IDLE_RSS_MB:-n/a} MB"
    echo "Throughput susținut server    : ${THROUGHPUT:-n/a} ev/s"
    if [[ -n "${STREAM:-}" ]]; then
      echo "  (măsurat pe stream-ul: ${STREAM})"
    fi
    ;;
esac
echo "============================================================"
