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

# Start auditd for in-container syscall monitoring
mkdir -p /var/log/audit
service auditd start 2>/dev/null || auditd -b 256 -f 0 &
sleep 1
# Track all execve calls and canary file reads
auditctl -a always,exit -F arch=b64 -S execve -k exec_track 2>/dev/null || true

# Forward audit events to the central Redis server
python3 /usr/local/sbin/hp-monitor.py &

exec /usr/sbin/sshd -D -e
