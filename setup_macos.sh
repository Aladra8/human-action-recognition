#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# setup_macos.sh – One-shot setup & build for the HAR project on macOS
# Tested: macOS Sonoma / Sequoia, Apple Silicon & Intel, Homebrew
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; BLUE='\033[0;34m'; NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

echo "╔══════════════════════════════════════════════╗"
echo "║  HAR Project – macOS Setup Script            ║"
echo "╚══════════════════════════════════════════════╝"

# ── 1. Homebrew ───────────────────────────────────────────────────────────────
if ! command -v brew &>/dev/null; then
    info "Homebrew not found – installing..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    # Add brew to PATH for Apple Silicon Macs
    if [[ -f /opt/homebrew/bin/brew ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
else
    ok "Homebrew found: $(brew --version | head -1)"
fi

# ── 2. CMake ──────────────────────────────────────────────────────────────────
if ! command -v cmake &>/dev/null; then
    info "Installing cmake..."
    brew install cmake
else
    ok "cmake found: $(cmake --version | head -1)"
fi

# ── 3. OpenCV ─────────────────────────────────────────────────────────────────
if ! brew list opencv &>/dev/null 2>&1; then
    info "Installing OpenCV (this may take a few minutes)..."
    brew install opencv
else
    ok "OpenCV found: $(brew info --json=v2 opencv 2>/dev/null | \
        python3 -c 'import sys,json;d=json.load(sys.stdin);print(d["formulae"][0]["versions"]["stable"])' \
        2>/dev/null || echo 'installed')"
fi

# ── 4. Build ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

info "Building project (Release mode)..."
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -Wno-dev
make -j"$(sysctl -n hw.ncpu)"
cd "$SCRIPT_DIR"

if [[ -f "./har" ]]; then
    ok "Build successful!  Binary: ./har"
else
    error "Build failed – binary not found."
fi

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Ready!  Quick-start commands:               ║"
echo "╠══════════════════════════════════════════════╣"
echo "║                                              ║"
echo "║  Run LOOCV evaluation (recommended):         ║"
echo "║    ./har /path/to/dataset --loocv            ║"
echo "║                                              ║"
echo "║  Save annotated frames + CSV report:         ║"
echo "║    ./har /path/to/dataset --loocv \\          ║"
echo "║          --visualize --report results.csv    ║"
echo "║                                              ║"
echo "║  Train & save model, then run later:         ║"
echo "║    ./har /path/to/dataset --loocv \\          ║"
echo "║          --save-model model.svm              ║"
echo "║    ./har /path/to/dataset \\                  ║"
echo "║          --load-model model.svm --visualize  ║"
echo "║                                              ║"
echo "╚══════════════════════════════════════════════╝"
