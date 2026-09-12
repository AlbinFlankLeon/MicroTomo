#!/usr/bin/env bash
# MicroTomo GUI launcher — simple one-liner entry point.
#
# Usage:
#   ./launch_gui.sh            # open the GUI window
#   ./launch_gui.sh --smoke    # headless render check (CI / quick verify)
set -euo pipefail
cd "$(dirname "$0")"
exec .venv/bin/python gui/main.py "$@"