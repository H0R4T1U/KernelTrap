#!/bin/bash
# Usage: provision-user.sh <username>
# Called by hp_pivot_user on the host via docker exec before the SIGUSR1 pivot.
# Creates a user matching the real attacker username and plants canary credentials.
set -euo pipefail
USERNAME="$1"

if ! id "$USERNAME" &>/dev/null; then
    useradd -ms /bin/bash "$USERNAME"
    echo "$USERNAME:$USERNAME" | chpasswd
    usermod -aG sudo "$USERNAME"
fi

# Copy the pivot authorized key from the attacker template — avoids passing the
# key as a shell argument (SSH public keys contain spaces and would split).
mkdir -p "/home/$USERNAME/.ssh"
cp /home/attacker/.ssh/authorized_keys "/home/$USERNAME/.ssh/authorized_keys" 2>/dev/null || true
chown -R "$USERNAME:$USERNAME" "/home/$USERNAME/.ssh"
chmod 700 "/home/$USERNAME/.ssh"
chmod 600 "/home/$USERNAME/.ssh/authorized_keys"

# Seed home directory from the attacker template (fake dirs/files)
cp -rn /home/attacker/. "/home/$USERNAME/" 2>/dev/null || true

HOME_DIR="/home/$USERNAME"

# Canary: fake AWS credentials
mkdir -p "$HOME_DIR/.aws"
printf '[default]\naws_access_key_id = AKIAIOSFODNN7EXAMPLE\naws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY\nregion = us-east-1\n' \
  > "$HOME_DIR/.aws/credentials"

# Canary: fake deploy SSH key (only generate once — ssh-keygen prompts to overwrite)
if [ ! -f "$HOME_DIR/.ssh/id_rsa" ]; then
    ssh-keygen -t rsa -b 2048 -f "$HOME_DIR/.ssh/id_rsa" -N '' -C "deploy-key" -q 2>/dev/null || true
fi

# Canary: fake app .env with DB/Redis credentials
printf 'DB_HOST=db-prod.internal\nDB_PORT=5432\nDB_NAME=app_production\nDB_USER=appuser\nDB_PASS=s3cr3tPr0dPassw0rd!\nREDIS_URL=redis://cache.internal:6379\nJWT_SECRET=ey7hK2mN9pQ4wR8vX1zA3bC6dF0gH5jL\n' \
  > "$HOME_DIR/.env"

# Canary: notes pointing to other fake internal systems
printf 'TODO:\n- check backup server 192.168.10.50 (admin/backup2024)\n- rotate db-prod credentials (use .env file)\n- jenkins at http://ci.internal:8080 (token in ~/.jenkins_token)\n' \
  > "$HOME_DIR/notes.txt"

printf '%s_backup_token_%s\n' "$USERNAME" "$(date +%s)" > "$HOME_DIR/.jenkins_token"

# Add auditd watches on canary files so reads appear in the event stream
auditctl -w "$HOME_DIR/.aws/credentials" -p r -k canary_aws   2>/dev/null || true
auditctl -w "$HOME_DIR/.ssh/id_rsa"      -p r -k canary_ssh   2>/dev/null || true
auditctl -w "$HOME_DIR/.env"             -p r -k canary_env   2>/dev/null || true
auditctl -w "$HOME_DIR/notes.txt"        -p r -k canary_notes 2>/dev/null || true

chown -R "$USERNAME:$USERNAME" "$HOME_DIR"
