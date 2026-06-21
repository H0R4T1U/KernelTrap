#!/bin/bash
#
# KernelTrap - Tracee Installation Script
#
# Installs Aqua Security's Tracee eBPF-based runtime security tool
# for collecting syscall events.
#
# Requirements:
#   - Linux kernel 5.4+ (for full eBPF support)
#   - Docker (for containerized installation) OR
#   - Build tools for native installation
#
# Usage:
#   sudo ./install_tracee.sh [docker|binary|source]
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root"
        exit 1
    fi
}

check_kernel_version() {
    kernel_version=$(uname -r | cut -d. -f1-2)
    kernel_major=$(echo "$kernel_version" | cut -d. -f1)
    kernel_minor=$(echo "$kernel_version" | cut -d. -f2)

    if [[ $kernel_major -lt 5 ]] || [[ $kernel_major -eq 5 && $kernel_minor -lt 4 ]]; then
        log_warn "Kernel version $kernel_version detected. Tracee works best with kernel 5.4+"
        log_warn "Some features may be limited."
    else
        log_info "Kernel version $kernel_version - compatible with Tracee"
    fi
}

install_docker() {
    log_info "Installing Tracee via Docker..."

    if ! command -v docker &> /dev/null; then
        log_info "Docker not found, installing..."
        apt-get update
        apt-get install -y docker.io
        systemctl start docker
        systemctl enable docker
    fi

    # Pull the latest Tracee image
    log_info "Pulling Tracee Docker image..."
    docker pull aquasec/tracee:latest

    # Create wrapper script
    cat > /usr/local/bin/tracee << 'EOF'
#!/bin/bash
# Tracee Docker wrapper
docker run --name tracee --rm -it \
    --pid=host \
    --cgroupns=host \
    --privileged \
    -v /etc/os-release:/etc/os-release-host:ro \
    -v /var/run:/var/run:ro \
    aquasec/tracee:latest "$@"
EOF
    chmod +x /usr/local/bin/tracee

    log_info "Tracee installed successfully via Docker"
    log_info "Run with: tracee --output json"
}

install_binary() {
    log_info "Installing Tracee from binary release..."

    # Detect architecture
    ARCH=$(uname -m)
    case $ARCH in
        x86_64) ARCH="amd64" ;;
        aarch64) ARCH="arm64" ;;
        *) log_error "Unsupported architecture: $ARCH"; exit 1 ;;
    esac

    # Get latest release version
    log_info "Fetching latest Tracee release..."
    LATEST_VERSION=$(curl -s https://api.github.com/repos/aquasecurity/tracee/releases/latest | grep '"tag_name"' | cut -d'"' -f4)

    if [[ -z "$LATEST_VERSION" ]]; then
        log_error "Could not fetch latest version. Using v0.20.0"
        LATEST_VERSION="v0.20.0"
    fi

    log_info "Downloading Tracee $LATEST_VERSION for $ARCH..."

    DOWNLOAD_URL="https://github.com/aquasecurity/tracee/releases/download/${LATEST_VERSION}/tracee-${LATEST_VERSION#v}-linux-${ARCH}.tar.gz"

    cd /tmp
    curl -L -o tracee.tar.gz "$DOWNLOAD_URL"

    tar -xzf tracee.tar.gz
    mv tracee /usr/local/bin/
    chmod +x /usr/local/bin/tracee

    rm -f tracee.tar.gz

    log_info "Tracee binary installed to /usr/local/bin/tracee"
}

install_source() {
    log_info "Building Tracee from source..."

    # Install build dependencies
    apt-get update
    apt-get install -y \
        build-essential \
        pkgconf \
        libelf-dev \
        llvm \
        clang \
        golang-go \
        libbpf-dev \
        linux-headers-$(uname -r)

    # Clone repository
    cd /tmp
    rm -rf tracee
    git clone --depth 1 https://github.com/aquasecurity/tracee.git
    cd tracee

    # Build
    make

    # Install
    cp dist/tracee /usr/local/bin/
    chmod +x /usr/local/bin/tracee

    # Cleanup
    cd /
    rm -rf /tmp/tracee

    log_info "Tracee built and installed to /usr/local/bin/tracee"
}

setup_systemd_service() {
    log_info "Creating systemd service for Tracee..."

    # Get the script directory (where syscall_logger.py is)
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    cat > /etc/systemd/system/kerneltrap-logger.service << EOF
[Unit]
Description=KernelTrap Syscall Logger
Documentation=https://github.com/your-repo/KernelTrap
After=network.target docker.service

[Service]
Type=simple
ExecStart=/bin/bash -c 'set -o pipefail; tracee --output json 2>/dev/null | python3 ${SCRIPT_DIR}/syscall_logger.py --source tracee --output /var/log/kerneltrap/events.csv'
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=kerneltrap

# Security hardening
NoNewPrivileges=false
ProtectSystem=full
ProtectHome=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF

    # Create log directory
    mkdir -p /var/log/kerneltrap

    # Reload systemd
    systemctl daemon-reload

    log_info "Systemd service created: kerneltrap-logger.service"
    log_info "Enable with: systemctl enable kerneltrap-logger"
    log_info "Start with: systemctl start kerneltrap-logger"
}

show_usage() {
    echo "Usage: $0 [docker|binary|source]"
    echo ""
    echo "Installation methods:"
    echo "  docker  - Install Tracee via Docker (recommended, easiest)"
    echo "  binary  - Download pre-built binary from GitHub releases"
    echo "  source  - Build from source (requires development tools)"
    echo ""
    echo "After installation, run:"
    echo "  tracee --output json | python3 syscall_logger.py --source tracee --output events.csv"
}

main() {
    check_root
    check_kernel_version

    case "${1:-docker}" in
        docker)
            install_docker
            ;;
        binary)
            install_binary
            ;;
        source)
            install_source
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            log_error "Unknown installation method: $1"
            show_usage
            exit 1
            ;;
    esac

    setup_systemd_service

    echo ""
    log_info "Installation complete!"
    echo ""
    echo "Quick test:"
    echo "  sudo tracee --output json | head -10"
    echo ""
    echo "Collect events for Isolation Forest:"
    echo "  sudo tracee --output json | python3 syscall_logger.py -s tracee -o events.csv"
    echo ""
    echo "Stream to the central server (scoring + auto-pivot happen there):"
    echo "  sudo tracee --output json | python3 syscall_logger.py -s tracee --redis-host <central-server-ip>"
}

main "$@"
