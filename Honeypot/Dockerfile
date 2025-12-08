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
