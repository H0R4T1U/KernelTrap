#!/usr/bin/env bash
# ============================================================================
# env_collect.sh — Colectează configurația mediului de testare pentru
#                  tabelul tab:cap6-env din capitolul 6 al lucrării.
#
# Rulează pe AMBELE mașini (gazda monitorizată ȘI serverul central) și
# copiază valorile rezultate direct în LaTeX.
#
# Usage: ./env_collect.sh [--latex]
#   fără argumente : output uman (key: value)
#   --latex        : output gata de inserat în tabelul LaTeX
# ============================================================================

set -u
LATEX_MODE=0
[[ "${1:-}" == "--latex" ]] && LATEX_MODE=1

# -------- Distribuție Linux ----------------------------------------------------
if command -v lsb_release >/dev/null 2>&1; then
  DISTRO=$(lsb_release -ds 2>/dev/null | tr -d '"')
elif [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  DISTRO=$(. /etc/os-release && printf '%s %s' "$NAME" "$VERSION")
else
  DISTRO="Linux (necunoscut)"
fi

# -------- Kernel ---------------------------------------------------------------
KERNEL=$(uname -r)

# -------- CPU ------------------------------------------------------------------
# Model + nr fizic de cores + nr logical threads
if command -v lscpu >/dev/null 2>&1; then
  CPU_MODEL=$(lscpu | awk -F: '/Model name/ {gsub(/^[ \t]+/, "", $2); print $2; exit}')
  CPU_CORES=$(lscpu | awk -F: '/^Core\(s\) per socket/ {gsub(/[ \t]/, "", $2); print $2; exit}')
  CPU_SOCKETS=$(lscpu | awk -F: '/^Socket\(s\)/ {gsub(/[ \t]/, "", $2); print $2; exit}')
  CPU_THREADS=$(lscpu | awk -F: '/^CPU\(s\):/ {gsub(/[ \t]/, "", $2); print $2; exit}')
  CPU_PHYS=$(( ${CPU_CORES:-0} * ${CPU_SOCKETS:-1} ))
  CPU="${CPU_MODEL} (${CPU_PHYS}C/${CPU_THREADS}T)"
else
  # fallback macOS
  if command -v sysctl >/dev/null 2>&1; then
    CPU_MODEL=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "necunoscut")
    CPU_PHYS=$(sysctl -n hw.physicalcpu 2>/dev/null || echo "?")
    CPU_THREADS=$(sysctl -n hw.logicalcpu 2>/dev/null || echo "?")
    CPU="${CPU_MODEL} (${CPU_PHYS}C/${CPU_THREADS}T)"
  else
    CPU="necunoscut"
  fi
fi

# -------- RAM ------------------------------------------------------------------
if command -v free >/dev/null 2>&1; then
  RAM=$(free -h --si | awk '/^Mem:/ {print $2}')
elif command -v sysctl >/dev/null 2>&1; then
  RAM_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
  RAM=$(awk -v b="$RAM_BYTES" 'BEGIN { printf "%.1f GB", b/1024/1024/1024 }')
else
  RAM="necunoscut"
fi

# -------- Tracee (image tag + digest scurt) -----------------------------------
# Încercăm în ordine:
#   1. container Tracee care RULEAZĂ (cel mai precis)
#   2. orice imagine cu "tracee" în nume (descărcată dar oprită)
#   3. binar tracee instalat pe gazdă (rar, dar posibil)
TRACEE="necunoscut — completează manual din comanda care pornește Tracee"

# Auto-detect dacă docker are nevoie de sudo
DOCKER_CMD=""
if command -v docker >/dev/null 2>&1; then
  if docker ps >/dev/null 2>&1; then
    DOCKER_CMD="docker"
  elif sudo -n docker ps >/dev/null 2>&1; then
    DOCKER_CMD="sudo -n docker"
    echo "(folosesc sudo pentru docker — userul nu e în grupul docker)" >&2
  else
    echo "(docker accesibil doar prin sudo cu parolă; ori rulează:" >&2
    echo "    sudo ./bench/env_collect.sh" >&2
    echo " ori adaugă userul în grupul docker: sudo usermod -aG docker \$USER && logout)" >&2
  fi
fi

if [[ -n "$DOCKER_CMD" ]]; then
  # 1. Container care rulează
  RUNNING=$($DOCKER_CMD ps --filter 'ancestor=aquasec/tracee' --format '{{.Image}}' 2>/dev/null | head -1)
  if [[ -z "$RUNNING" ]]; then
    RUNNING=$($DOCKER_CMD ps --format '{{.Image}}\t{{.Names}}' 2>/dev/null | grep -i 'tracee' | head -1 | cut -f1)
  fi
  if [[ -n "$RUNNING" ]]; then
    DIGEST=$($DOCKER_CMD inspect --format='{{.Image}}' "$RUNNING" 2>/dev/null | cut -c1-19)
    [[ -z "$DIGEST" ]] && DIGEST=$($DOCKER_CMD images "$RUNNING" --format '{{.ID}}' 2>/dev/null | head -1)
    TRACEE="${RUNNING} (${DIGEST})"
  else
    # 2. Imagine descărcată dar nu rulează
    IMG=$($DOCKER_CMD images --format '{{.Repository}}:{{.Tag}}\t{{.ID}}' 2>/dev/null | grep -i 'tracee' | head -1)
    if [[ -n "$IMG" ]]; then
      TRACEE_REPO=$(printf '%s' "$IMG" | cut -f1)
      TRACEE_ID=$(printf '%s' "$IMG" | cut -f2)
      TRACEE="${TRACEE_REPO} (${TRACEE_ID}) — imaginea nu rulează acum"
    fi
  fi
fi
# 3. Binar nativ ca fallback
if [[ "$TRACEE" == necunoscut* ]] && command -v tracee >/dev/null 2>&1; then
  TRACEE="tracee binar nativ: $(tracee --version 2>/dev/null | head -1)"
fi

# -------- Python --------------------------------------------------------------
if command -v python3 >/dev/null 2>&1; then
  PYTHON=$(python3 --version 2>&1 | awk '{print $2}')
else
  PYTHON="indisponibil"
fi

# -------- Redis (din docker, nu host) -----------------------------------------
REDIS="7 (imagine oficială Docker)"

# -------- Output ---------------------------------------------------------------
if (( LATEX_MODE )); then
  cat <<EOF
% --- Copy-paste în tab:cap6-env (Scris/capitol6_evaluare.tex L67-83) ---
    Gazda monitorizată       & ${DISTRO}, kernel ${KERNEL}, ${CPU}, ${RAM} RAM \\\\
    Serverul central         & ${DISTRO}, kernel ${KERNEL}, ${CPU}, ${RAM} RAM \\\\
    Container honeypot       & Docker (Ubuntu 22.04), capabilități \\texttt{AUDIT\\_WRITE}, \\texttt{AUDIT\\_CONTROL} \\\\
    Versiune Tracee          & ${TRACEE} \\\\
    Versiune Python          & ${PYTHON} \\\\
    Versiune Redis           & ${REDIS} \\\\
EOF
else
  cat <<EOF
=== Configurație mediu pentru tab:cap6-env ===
Distribuție : ${DISTRO}
Kernel      : ${KERNEL}
CPU         : ${CPU}
RAM         : ${RAM}
Tracee      : ${TRACEE}
Python      : ${PYTHON}
Redis       : ${REDIS}

Notă: rulează scriptul pe AMBELE mașini (gazdă + server central) și
completează rândurile corespunzătoare separat dacă diferă.
EOF
fi
