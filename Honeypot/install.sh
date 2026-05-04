#!/usr/bin/env bash
set -euo pipefail

echo "[*] Honeypot pivot installer starting..."

############################################
# 0. Sanity checks
############################################

if [ "$(id -u)" -ne 0 ]; then
  echo "[!] This script must be run as root (use sudo)." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "$SCRIPT_DIR/hp_pivot_user.sh" ]; then
  echo "[!] hp_pivot_user.sh not found in $SCRIPT_DIR — run install.sh from the Honeypot/ directory." >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "[*] Docker not found, installing docker.io..."
  apt-get update -qq
  apt-get install -y docker.io
  systemctl enable --now docker
fi

HP_HOSTNAME=$(hostname)

############################################
# 1. Build honeypot Docker image
############################################

HP_DIR="/opt/honeypot-ssh"
mkdir -p "$HP_DIR"
cd "$HP_DIR"

echo "[*] Writing Dockerfile..."

cat > Dockerfile << 'EOF'
FROM ubuntu:22.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      openssh-server sudo vim tmux htop curl net-tools iproute2 lsb-release && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# SSH daemon run dir
RUN mkdir -p /run/sshd /var/run/sshd

# decoy user inside the container
RUN useradd -ms /bin/bash attacker && \
    echo 'attacker:Password123!' | chpasswd && \
    usermod -aG sudo attacker

# simple fake dirs
RUN mkdir -p /opt/app /var/www/html /data && \
    touch /opt/app/README.txt /data/log1.log

# Harden sshd a bit
RUN sed -ri 's/#?PermitRootLogin .*/PermitRootLogin no/' /etc/ssh/sshd_config && \
    sed -ri 's/#?PasswordAuthentication .*/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    echo 'ClientAliveInterval 60' >> /etc/ssh/sshd_config && \
    echo 'ClientAliveCountMax 3' >> /etc/ssh/sshd_config

COPY start-sshd.sh /usr/local/sbin/start-sshd.sh
RUN chmod +x /usr/local/sbin/start-sshd.sh

EXPOSE 22
CMD ["/usr/local/sbin/start-sshd.sh"]
EOF

echo "[*] Writing start-sshd.sh..."

cat > start-sshd.sh << 'EOF'
#!/bin/bash
mkdir -p /run/sshd

if [ -n "$HP_HOSTNAME" ]; then
  echo "$HP_HOSTNAME" > /etc/hostname
  hostname "$HP_HOSTNAME"
fi

if [ -n "$HP_ISSUE" ]; then
  printf "%b" "$HP_ISSUE" > /etc/issue
fi

if [ -n "$HP_LSB_RELEASE" ]; then
  printf "%b" "$HP_LSB_RELEASE" > /etc/lsb-release
fi

exec /usr/sbin/sshd -D -e
EOF

echo "[*] Building Docker image ssh-honeypot:latest..."
if ! docker build -t ssh-honeypot:latest . ; then
  echo "[!] Docker build failed." >&2
  exit 1
fi

############################################
# 2. Docker network and container
############################################

echo "[*] Ensuring Docker network 'honeynet' exists..."
if ! docker network ls --format '{{.Name}}' | grep -q '^honeynet$'; then
  docker network create --driver bridge honeynet
fi

echo "[*] Removing any existing hp-shell container..."
docker rm -f hp-shell >/dev/null 2>&1 || true

HOSTNAME_REAL="$(hostname)"
ISSUE_REAL="$(cat /etc/issue 2>/dev/null || echo 'Ubuntu \n \l')"
LSB_REAL="$(cat /etc/lsb-release 2>/dev/null || echo '')"

ISSUE_ESCAPED=$(printf '%s\n' "$ISSUE_REAL" | sed ':a;N;$!ba;s/\n/\\n/g')
LSB_ESCAPED=$(printf '%s\n' "$LSB_REAL" | sed ':a;N;$!ba;s/\n/\\n/g')

echo "[*] Starting hp-shell container on port 2222..."
docker run -d --name hp-shell \
  --network honeynet \
  -p 2222:22 \
  --restart unless-stopped \
  --read-only \
  --tmpfs /run \
  --tmpfs /tmp \
  --hostname "$HOSTNAME_REAL" \
  -v hp_attacker_home:/home/attacker \
  -e HP_HOSTNAME="$HP_HOSTNAME" \
  -e HP_ISSUE="$ISSUE_ESCAPED" \
  -e HP_LSB_RELEASE="$LSB_ESCAPED" \
  ssh-honeypot:latest

echo "[*] Waiting for hp-shell container to be ready..."
WAIT=0
until docker exec hp-shell sh -c "test -S /run/sshd.pid || ss -tlnp | grep -q ':22'" 2>/dev/null; do
  sleep 2
  WAIT=$((WAIT+2))
  if [ $WAIT -ge 30 ]; then
    echo "[!] hp-shell container did not become ready in 30s. Logs:"
    docker logs hp-shell | tail -20
    exit 1
  fi
done
echo "    hp-shell is ready"

############################################
# 3. Generate pivot SSH key and copy pubkey into container
############################################

echo "[*] Creating /etc/hptrap and SSH keypair for pivot..."

mkdir -p /etc/hptrap

if [ ! -f /etc/hptrap/hp_key ]; then
  ssh-keygen -t ed25519 -f /etc/hptrap/hp_key -N '' -C "hptrap-key"
fi

PUB_KEY="$(cat /etc/hptrap/hp_key.pub)"

echo "[*] Installing public key into hp-shell container..."
docker exec hp-shell sh -c "mkdir -p /home/attacker/.ssh && \
  echo '$PUB_KEY' >> /home/attacker/.ssh/authorized_keys && \
  chown -R attacker:attacker /home/attacker/.ssh && \
  chmod 700 /home/attacker/.ssh && \
  chmod 600 /home/attacker/.ssh/authorized_keys"

############################################
# 4. Create hptrap group and set key perms
############################################

echo "[*] Creating group 'hptrap' and setting key permissions..."
if ! getent group hptrap >/dev/null 2>&1; then
  groupadd hptrap
fi

chown root:hptrap /etc/hptrap/hp_key
chmod 640 /etc/hptrap/hp_key

############################################
# 5. Install global trap script for all interactive bash shells
############################################

echo "[*] Installing /etc/profile.d/hptrap.sh..."

cat > /etc/profile.d/hptrap.sh << 'EOF'
# Honeypot pivot trap — loaded for all interactive bash shells
if [ -n "$PS1" ] && [ -z "$HPTRAP_INITIALIZED" ]; then
  export HPTRAP_INITIALIZED=1

  hptrap_pivot() {
    logger -t hptrap "Pivoting user $USER to honeypot (PID=$$, from $SSH_CONNECTION)"
    exec ssh \
      -i /etc/hptrap/hp_key \
      -o StrictHostKeyChecking=no \
      attacker@127.0.0.1 -p 2222
  }

  trap hptrap_pivot USR1
fi
EOF

chmod 644 /etc/profile.d/hptrap.sh

############################################
# 6. Install hp_pivot_user
############################################

echo "[*] Installing /usr/local/sbin/hp_pivot_user..."

cp "$SCRIPT_DIR/hp_pivot_user.sh" /usr/local/sbin/hp_pivot_user
chmod 755 /usr/local/sbin/hp_pivot_user

# Verify installation
if [ ! -x /usr/local/sbin/hp_pivot_user ]; then
  echo "[!] Failed to install /usr/local/sbin/hp_pivot_user" >&2
  exit 1
fi

############################################
# 7. Sudoers entry — allow running hp_pivot_user without password
############################################

echo "[*] Adding sudoers rule for hp_pivot_user..."

SUDOERS_FILE="/etc/sudoers.d/kerneltrap"
cat > "$SUDOERS_FILE" << 'EOF'
# KernelTrap: allow any user to run hp_pivot_user as root without password
ALL ALL=(root) NOPASSWD: /usr/local/sbin/hp_pivot_user
EOF
chmod 440 "$SUDOERS_FILE"

# Validate sudoers syntax
if ! visudo -cf "$SUDOERS_FILE" >/dev/null 2>&1; then
  echo "[!] sudoers file syntax error — removing." >&2
  rm -f "$SUDOERS_FILE"
  exit 1
fi

############################################
# 8. Add all normal users (UID >= 1000) to hptrap group
############################################

echo "[*] Adding non-system users (UID >= 1000) to group 'hptrap'..."

awk -F: '$3>=1000 && $1!="nobody"{print $1}' /etc/passwd | while read -r u; do
  usermod -aG hptrap "$u" || true
done

############################################
# 9. Final verification
############################################

echo
echo "============================================"
echo "[*] Verification"
echo "============================================"

OK=true

if [ -x /usr/local/sbin/hp_pivot_user ]; then
  echo "  [OK] /usr/local/sbin/hp_pivot_user installed"
else
  echo "  [FAIL] /usr/local/sbin/hp_pivot_user missing" && OK=false
fi

if [ -f /etc/hptrap/hp_key ] && [ -f /etc/hptrap/hp_key.pub ]; then
  echo "  [OK] SSH pivot keypair at /etc/hptrap/hp_key"
else
  echo "  [FAIL] SSH pivot keypair missing" && OK=false
fi

if docker ps --format '{{.Names}}' | grep -q '^hp-shell$'; then
  echo "  [OK] hp-shell container running"
else
  echo "  [FAIL] hp-shell container NOT running" && OK=false
fi

if [ -f "$SUDOERS_FILE" ]; then
  echo "  [OK] sudoers rule at $SUDOERS_FILE"
else
  echo "  [FAIL] sudoers rule missing" && OK=false
fi

if [ "$OK" != "true" ]; then
  echo
  echo "[!] Install completed with errors. See FAIL items above." >&2
  exit 1
fi

echo
echo "[*] Install complete."
echo
echo "Honeypot container:  hp-shell (SSH port 2222)"
echo "Pivot script:        /usr/local/sbin/hp_pivot_user"
echo "Pivot key:           /etc/hptrap/hp_key"
echo "Sudoers rule:        $SUDOERS_FILE"
echo
echo "NOTE: Users must log out and log back in for the SIGUSR1 trap"
echo "      (from /etc/profile.d/hptrap.sh) to take effect."
echo
echo "To test pivot manually:  sudo hp_pivot_user <username>"
echo "[*] Done."
