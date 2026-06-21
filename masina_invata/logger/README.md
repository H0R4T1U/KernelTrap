# KernelTrap Syscall Logger

Collects and transforms Linux syscall events for the Isolation Forest anomaly detection model, with optional real-time scoring and honeypot integration.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  Tracee (eBPF)  │────▶│  syscall_logger  │────▶│  CSV (BETH format) │
│  or auditd      │     │                  │     └────────────────────┘
└─────────────────┘     │  ┌────────────┐  │     ┌────────────────────┐
                        │  │ Transform  │  │────▶│  Isolation Forest  │
                        │  └────────────┘  │     │  (real-time score) │
                        │                  │     └────────────────────┘
                        │  ┌────────────┐  │            │
                        │  │  Scoring   │◀─┘            ▼
                        │  └────────────┘       ┌────────────────────┐
                        └──────────────────┘    │  Honeypot Pivot    │
                                                │  (hp_pivot_user)   │
                                                └────────────────────┘
```

## Quick Start

### 1. Install Tracee (eBPF syscall collector)

```bash
# Recommended: Docker installation
sudo ./install_tracee.sh docker

# Or binary download
sudo ./install_tracee.sh binary

# Or build from source
sudo ./install_tracee.sh source
```

### 2. Collect syscall events to CSV

```bash
# Basic collection (for training/evaluation)
sudo tracee --output json | python3 syscall_logger.py -s tracee -o events.csv
```

### 3. Stream to the central server (scoring + auto-pivot)

```bash
# Forward events to the central analysis server, which scores them and sends
# pivot commands back when a user crosses the threshold.
sudo tracee --output json | python3 syscall_logger.py \
    -s tracee --redis-host <central-server-ip>
```

> Anomaly scoring and the honeypot pivot decision live in `central_server/`,
> not in the agent. The standalone `honeypot_integration.py` from the original
> single-host design has been retired to `older_versions/logger/`.

## Components

### syscall_logger.py

Main logger module that:
- Parses Tracee JSON or auditd logs
- Transforms events to BETH dataset format
- Outputs CSV compatible with Isolation Forest
- Optionally scores events in real-time

**Usage:**

```bash
# From Tracee (recommended) — local CSV
tracee --output json | python3 syscall_logger.py --source tracee --output events.csv

# From auditd
python3 syscall_logger.py --source auditd --auditd-log /var/log/audit/audit.log -o events.csv

# Stream to the central server
tracee --output json | python3 syscall_logger.py -s tracee --redis-host <central-server-ip>

# Combined: write CSV backup + stream
tracee --output json | python3 syscall_logger.py -s tracee -o events.csv --redis-host <ip>
```

### Scoring & honeypot pivot

Anomaly scoring (Isolation Forest) and the pivot decision are handled by the
central server — see `central_server/`. The agent only collects, filters, and
forwards events, and executes a pivot command when the server sends one.

### install_tracee.sh

Automated Tracee installation script:
- Docker, binary, or source installation
- Creates systemd service for continuous logging
- Configures permissions

## Output Format

The logger outputs CSV in BETH dataset format with 16 columns:

| Column | Type | Description |
|--------|------|-------------|
| timestamp | float | Unix timestamp |
| processId | int | Process ID |
| threadId | int | Thread ID |
| parentProcessId | int | Parent process ID |
| userId | int | User ID |
| mountNamespace | int | Mount namespace ID |
| processName | str | Process name |
| hostName | str | Hostname |
| eventId | int | Syscall number |
| eventName | str | Syscall name |
| stackAddresses | json | Stack addresses |
| argsNum | int | Number of args |
| returnValue | int | Return value |
| args | json | Syscall arguments |
| sus | int | Suspicious label (0/1) |
| evil | int | Malicious label (0/1) |

## Feature Extraction

The Isolation Forest model uses 7 numeric features:

```python
[processId, parentProcessId, userId, mountNamespace, eventId, argsNum, returnValue]
```

## Configuration

The agent is configured entirely via command-line flags — run
`python syscall_logger.py --help` for the full list. Key options:

```
--source        tracee | auditd        (default: tracee)
--output        local CSV backup path  (optional)
--redis-host    central server host    (enables streaming mode)
--redis-port    central server port    (default: 6379)
--buffer-size   events per Redis flush (default: 100)
--pivot-script  honeypot pivot script  (default: /usr/local/sbin/hp_pivot_user)
```

Anomaly scoring and the pivot decision now live in `central_server/`, not the
agent. The agent only collects, filters, and forwards events.

## Systemd Service

After installation, enable continuous logging:

```bash
sudo systemctl enable kerneltrap-logger
sudo systemctl start kerneltrap-logger

# Check status
sudo systemctl status kerneltrap-logger

# View logs
sudo journalctl -u kerneltrap-logger -f
```

## Requirements

- Python 3.8+
- Linux kernel 5.4+ (for Tracee)
- Docker (if using Docker installation)
- joblib, numpy, scikit-learn (for scoring)

Install Python dependencies:

```bash
pip install redis
```

## Integration with Isolation Forest

### Training a new model

```bash
# 1. Collect benign activity
sudo tracee --output json | python3 syscall_logger.py -s tracee -o training_data.csv

# 2. Train the model
python3 ../isolation_forest/beth_iforest.py \
    --train training_data.csv \
    --val validation_data.csv \
    --test testing_data.csv \
    --model-dir ./my_model
```

### Scoring new events

```bash
# The agent forwards events to the central server, which scores them with the
# trained BETH model (configured server-side via MODEL_DIR — see central_server/).
sudo tracee --output json | python3 syscall_logger.py \
    -s tracee --redis-host <central-server-ip>
```

## Alert Severity Levels

| Severity | Description | Default Threshold |
|----------|-------------|-------------------|
| 0 | Normal | Above low percentile |
| 1 | Low | Below 2.0 percentile |
| 2 | High | Below 0.2 percentile |

## Troubleshooting

### Tracee not producing output

```bash
# Check if eBPF is supported
sudo bpftool prog list

# Run Tracee with verbose output
sudo tracee --debug
```

### Permission errors

```bash
# Ensure proper capabilities
sudo setcap cap_sys_admin,cap_sys_ptrace+ep $(which tracee)
```

### Model loading errors

```bash
# Verify model files exist
ls -la /path/to/model/
# Should contain: scaler.joblib, iforest.joblib, meta.json
```

## License

Part of the KernelTrap project.
