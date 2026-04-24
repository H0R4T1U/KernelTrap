#!/usr/bin/env bash
set -e

echo "[*] Honeypot pivot installer starting..."

############################################
# 0. Sanity checks
############################################

if [ "$(id -u)" -ne 0 ]; then
  echo "[!] This script must be run as root (use sudo)." >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "[*] Docker not found, installing docker.io..."
  apt-get update
  apt-get install -y docker.io
  systemctl enable --now docker
fi

######
# 0.1 Configs
#####


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

# startup script that ensures /run/sshd exists and mimics host hostname (via env)
COPY start-sshd.sh /usr/local/sbin/start-sshd.sh
RUN chmod +x /usr/local/sbin/start-sshd.sh

EXPOSE 22
CMD ["/usr/local/sbin/start-sshd.sh"]
EOF

echo "[*] Writing start-sshd.sh..."

cat > start-sshd.sh << 'EOF'
#!/bin/bash
# Runtime init inside honeypot container

# Ensure /run/sshd exists on tmpfs or writable fs
mkdir -p /run/sshd

# If HP_HOSTNAME env is set, mimic host hostname
if [ -n "$HP_HOSTNAME" ]; then
  echo "$HP_HOSTNAME" > /etc/hostname
  hostname "$HP_HOSTNAME"
fi

# If HP_ISSUE is set, overwrite /etc/issue to mimic host OS banner
if [ -n "$HP_ISSUE" ]; then
  printf "%b" "$HP_ISSUE" > /etc/issue
fi

# If HP_LSB_RELEASE is set, overwrite /etc/lsb-release
if [ -n "$HP_LSB_RELEASE" ]; then
  printf "%b" "$HP_LSB_RELEASE" > /etc/lsb-release
fi

exec /usr/sbin/sshd -D -e
EOF

echo "[*] Building Docker image ssh-honeypot:latest..."
docker build -t ssh-honeypot:latest .

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
ISSUE_REAL="$(cat /etc/issue || echo 'Ubuntu \n \l')"

if [ -f /etc/lsb-release ]; then
  LSB_REAL="$(cat /etc/lsb-release)"
else
  LSB_REAL=""
fi

# Escape newlines for env passing
ISSUE_ESCAPED=$(printf '%s\n' "$ISSUE_REAL" | sed ':a;N;$!ba;s/\n/\\n/g')
LSB_ESCAPED=$(printf '%s\n' "$LSB_REAL" | sed ':a;N;$!ba;s/\n/\\n/g')

echo "[*] Starting hp-shell container on port 2222..."
docker run -d --name hp-shell \
  --network honeynet \
  -p 2222:22 \
  --read-only \
  --tmpfs /run \
  --tmpfs /tmp \
  --hostname $(hostname) \
  -v hp_attacker_home:/home/attacker \
  -e HP_HOSTNAME="$HP_HOSTNAME" \
  -e HP_ISSUE="$ISSUE_ESCAPED" \
  -e HP_LSB_RELEASE="$LSB_ESCAPED" \
  ssh-honeypot:latest

sleep 2

############################################
# 3. Generate pivot SSH key and copy pubkey into container
############################################

echo "[*] Creating /etc/hptrap and SSH keypair for pivot..."

mkdir -p /etc/hptrap

if [ ! -f /etc/hptrap/hp_key ]; then
  ssh-keygen -t ed25519 -f /etc/hptrap/hp_key -N '' -C "hptrap-key"
fi

PUB_KEY="$(cat /etc/hptrap/hp_key.pub)"

echo "[*] Installing public key into docker container's /home/attacker/.ssh/authorized_keys..."
docker exec hp-shell sh -c "mkdir -p /home/attacker/.ssh"
docker exec hp-shell sh -c "echo '$PUB_KEY' >> /home/attacker/.ssh/authorized_keys"
docker exec hp-shell chown -R attacker:attacker /home/attacker/.ssh
docker exec hp-shell chmod 700 /home/attacker/.ssh
docker exec hp-shell chmod 600 /home/attacker/.ssh/authorized_keys

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
EOF

chmod 644 /etc/profile.d/hptrap.sh

############################################
# 6. Helper script hp_pivot_user
############################################

echo "[*] Installing /usr/local/sbin/hp_pivot_user..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cp "$SCRIPT_DIR/hp_pivot_user.sh" /usr/local/sbin/hp_pivot_user
chmod +x /usr/local/sbin/hp_pivot_user

############################################
# 7. Add all normal users to hptrap group
############################################

echo "[*] Adding all non-system users (UID >= 1000) to group 'hptrap'..."

awk -F: '$3>=1000 && $1!="nobody"{print $1}' /etc/passwd | while read -r u; do
  usermod -aG hptrap "$u" || true
done

############################################
# 8. Final info
############################################

echo
echo "[*] Install complete."
echo
echo "Honeypot container:"
echo "  - Name: hp-shell"
echo "  - SSH: attacker@<host>:2222"
echo
echo "Pivot key:"
echo "  - Private: /etc/hptrap/hp_key"
echo "  - Public : /etc/hptrap/hp_key.pub (also installed in container)"
echo
echo "All interactive bash logins now have a SIGUSR1 trap that execs:"
echo "  ssh -i /etc/hptrap/hp_key attacker@127.0.0.1 -p 2222"
echo
echo "To pivot a user currently connected via SSH, run:"
echo "  sudo hp_pivot_user <username>"
echo
echo "NOTE: Users must log out and log back in for the new group (hptrap)"
echo "      and profile script to fully apply to their sessions."
echo
echo "[*] Done."
