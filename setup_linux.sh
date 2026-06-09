#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# setup_linux.sh – One-shot setup & build for Ubuntu 22.04 / 24.04
# Run inside your UTM Ubuntu VM in a terminal.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

GREEN='\033[0;32m'; BLUE='\033[0;34m'; NC='\033[0m'
info() { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()   { echo -e "${GREEN}[OK]${NC}    $*"; }

echo "╔══════════════════════════════════════════════╗"
echo "║  HAR Project – Linux (Ubuntu) Setup Script   ║"
echo "╚══════════════════════════════════════════════╝"

# ── 1. System packages ────────────────────────────────────────────────────────
info "Installing build tools and OpenCV..."
sudo apt-get update -qq
sudo apt-get install -y \
    build-essential \
    cmake \
    libopencv-dev \
    libopencv-contrib-dev \
    pkg-config

ok "Packages installed."
echo "  OpenCV version: $(pkg-config --modversion opencv4 2>/dev/null || pkg-config --modversion opencv 2>/dev/null)"

# ── 2. Build ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

info "Building project (Release mode)..."
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -Wno-dev
make -j"$(nproc)"
cd "$SCRIPT_DIR"

ok "Build successful!  Binary: ./har"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Ready!  Example commands:                   ║"
echo "║                                              ║"
echo "║  ./har /path/to/dataset --loocv              ║"
echo "║  ./har /path/to/dataset --loocv --visualize  ║"
echo "╚══════════════════════════════════════════════╝"
