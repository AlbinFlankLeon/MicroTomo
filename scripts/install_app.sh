#!/usr/bin/env bash
# MicroTomo — install as an OS app (XDG desktop entry + icon).
#
# Registers a "MicroTomo" entry in the system app menu (CachyOS / Plasma /
# GNOME / any freedesktop launcher) that opens the interactive editor.
#
# Usage:
#   ./scripts/install_app.sh          # install
#   ./scripts/install_app.sh --remove # uninstall
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

APPS_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
ICONS_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/256x256/apps"
DESKTOP_FILE="$APPS_DIR/microtomo.desktop"
ICON_FILE="$ICONS_DIR/microtomo.png"

if [[ "${1:-}" == "--remove" ]]; then
    rm -f "$DESKTOP_FILE" "$ICON_FILE"
    echo "MicroTomo removed from the app menu."
    exit 0
fi

if [[ ! -f "$ROOT/datasets/demo/manifest.json" ]]; then
    echo "demo package missing — building it first (one-time, ~4 min)..."
    "$ROOT/.venv/bin/python" "$ROOT/scripts/make_demo.py"
fi

mkdir -p "$APPS_DIR" "$ICONS_DIR"
install -m 0644 "$ROOT/datasets/demo/icon.png" "$ICON_FILE"

cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=MicroTomo
Comment=60 GHz surface-contour simulator — edit scenes, place transceivers, run + visualise
Exec=$ROOT/launch_gui.sh
Icon=$ICON_FILE
Terminal=false
Categories=Science;
Keywords=Microwaves;Simulation;Tomography;Visualization;
StartupNotify=true
EOF

chmod +x "$ROOT/launch_gui.sh"
echo "installed: $DESKTOP_FILE"
echo "  icon     : $ICON_FILE"
echo "MicroTomo is now in the app menu (CachyOS: Apps / search \"MicroTomo\")."
echo "To uninstall: $0 --remove"