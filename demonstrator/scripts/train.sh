#!/bin/bash
# Wrapper: detects environment and delegates to train_wsl.sh inside WSL.

echo "============================================================"
echo "  TenSafe Finance Demonstrator - Training Pipeline"
echo "  Crash-resilient: re-run to resume from where you left off."
echo "============================================================"
echo ""

if grep -qi microsoft /proc/version 2>/dev/null; then
    # Already inside WSL — run directly
    exec bash "$(dirname "${BASH_SOURCE[0]}")/train_wsl.sh"
else
    # Windows (Git Bash / MSYS2) — delegate to WSL
    WIN_PATH="$(cd "$(dirname "$0")/../.." && pwd -W 2>/dev/null || pwd)"
    WSL_PATH=$(wsl -e wslpath "$WIN_PATH")
    echo "Delegating to WSL..."
    echo "Project: ${WSL_PATH}"
    echo ""
    MSYS_NO_PATHCONV=1 wsl bash -c "${WSL_PATH}/demonstrator/scripts/train_wsl.sh"
fi
