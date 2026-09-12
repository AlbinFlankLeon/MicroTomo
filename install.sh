#!/usr/bin/env bash
# MicroTomo — one-shot installer.
#
# Works from a fresh source checkout OR an extracted release tarball:
#   1. creates the Python venv and installs dependencies
#   2. builds the demo dataset (first run only)
#   3. registers the "MicroTomo" app in the system menu
#
# Usage:
#   bash install.sh            # install
#   bash install.sh --check    # verify an existing install (no changes)
#   bash install.sh --remove   # uninstall the app-menu entry only
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
VERSION="$(cat VERSION 2>/dev/null || echo unknown)"

echo "== MicroTomo $VERSION installer =="

if [[ "${1:-}" == "--remove" ]]; then
    bash scripts/install_app.sh --remove
    exit 0
fi

if [[ "${1:-}" == "--check" ]]; then
    .venv/bin/python scripts/install_check.py || true
    bash scripts/install_app.sh --check 2>/dev/null || true
    echo "install check complete."
    exit 0
fi

if [[ ! -f .venv/bin/python ]]; then
    echo ">> creating virtual environment (.venv)…"
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip >/dev/null
fi

echo ">> installing dependencies (this can take a few minutes)…"
.venv/bin/pip install -r requirements.txt

echo ">> verifying install…"
.venv/bin/python scripts/install_check.py

if [[ ! -f datasets/demo/manifest.json ]]; then
    echo ">> demo package missing — building it now (one-time, a few minutes)…"
    .venv/bin/python scripts/make_demo.py
fi

echo ">> registering in the app menu…"
bash scripts/install_app.sh

echo
echo "MicroTomo $VERSION installed."
echo "  launch:        ./launch_gui.sh   or the \"MicroTomo\" menu entry"
echo "  uninstall:     bash install.sh --remove"