FROM ubuntu:22.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      openssh-server sudo vim tmux htop curl net-tools iproute2 lsb-release \
      auditd python3 python3-pip && \
    pip3 install --no-cache-dir redis && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# SSH daemon run dir
RUN mkdir -p /run/sshd /var/run/sshd

# decoy user inside the container.
# The .sudo_as_admin_successful markers (in /etc/skel, copied by `useradd -m`,
# and in the attacker template home) suppress the Ubuntu "sudo_root" login
# banner for every provisioned user. provision-user.sh also touches it.
RUN useradd -ms /bin/bash attacker && \
    echo 'attacker:Password123!' | chpasswd && \
    usermod -aG sudo attacker && \
    touch /etc/skel/.sudo_as_admin_successful /home/attacker/.sudo_as_admin_successful && \
    chown attacker:attacker /home/attacker/.sudo_as_admin_successful

# simple fake dirs
RUN mkdir -p /opt/app /var/www/html /data && \
    touch /opt/app/README.txt /data/log1.log

# Harden sshd a bit
RUN sed -ri 's/#?PermitRootLogin .*/PermitRootLogin no/' /etc/ssh/sshd_config && \
    sed -ri 's/#?PasswordAuthentication .*/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    echo 'ClientAliveInterval 60' >> /etc/ssh/sshd_config && \
    echo 'ClientAliveCountMax 3' >> /etc/ssh/sshd_config

# TTY session recorder — ForceCommand wraps every login with script(1)
RUN mkdir -p /var/log/sessions && chmod 1777 /var/log/sessions
COPY record-session.sh /usr/local/sbin/record-session.sh
RUN chmod +x /usr/local/sbin/record-session.sh && \
    echo 'ForceCommand /usr/local/sbin/record-session.sh' >> /etc/ssh/sshd_config

# Dynamic user provisioner (creates mirrored attacker accounts at pivot time)
COPY provision-user.sh /usr/local/sbin/provision-user.sh
RUN chmod +x /usr/local/sbin/provision-user.sh

# In-container event forwarder (auditd → central Redis)
COPY hp-monitor.py /usr/local/sbin/hp-monitor.py
RUN chmod +x /usr/local/sbin/hp-monitor.py

# startup script that ensures /run/sshd exists and mimics host hostname (via env)
COPY start-sshd.sh /usr/local/sbin/start-sshd.sh
RUN chmod +x /usr/local/sbin/start-sshd.sh

EXPOSE 22
CMD ["/usr/local/sbin/start-sshd.sh"]
