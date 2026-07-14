# KernelTrap
KernelTrap is a distributed Host-based Intrusion Prevention System (HIPS) that uses eBPF-powered kernel telemetry with machine learning anomaly detection models to catch attackers in real time.
![Logo](https://github.com/H0R4T1U/KernelTrap/blob/main/KernelTrap.png)

# Abstract
Host-based Intrusion Detection Systems (HIDS) and honeypots are two complementary approaches in cyber defense, yet when used in isolation they suffer from significant limitations: the former generate alerts without an automatic response mechanism,
while the latter are passive and an attacker who has already compromised a system has
no incentive to migrate voluntarily into a trap. This paper presents KernelTrap, a distributed system that integrates both approaches into a single automated pipeline. The
system monitors system calls produced by active SSH sessions via eBPF technology,
evaluates each event using an Isolation Forest model trained on the BETH dataset, and
when a sliding window signals a critical level of anomalies, transparently redirects the
attacker’s session to a Docker honeypot container. Experimental results on real-world
attack scenarios confirm the feasibility of the approach, with CPU overhead below 2.5
