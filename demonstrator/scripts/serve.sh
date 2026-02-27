#!/bin/bash
set -euo pipefail

echo "============================================================"
echo "  TenSafe Finance Demonstrator - Inference Server"
echo "  HE: CKKS (CuKKS/OpenFHE) | PQC: liboqs (Kyber+Dilithium)"
echo "============================================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ROOT_DIR="$(dirname "$PROJECT_DIR")"

export PYTHONPATH="${ROOT_DIR}"
export MOE_CONFIG_PATH="${PROJECT_DIR}/adapters/tgsp/moe_config.json"
export DEVICE="${DEVICE:-cuda}"

# Check config exists
if [ ! -f "${MOE_CONFIG_PATH}" ]; then
    echo "ERROR: MoE config not found at ${MOE_CONFIG_PATH}"
    echo "Run ./scripts/train.sh first to train and package adapters."
    exit 1
fi

# Get WSL IP for iPhone access
if command -v hostname &>/dev/null; then
    WSL_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "localhost")
else
    WSL_IP="localhost"
fi

echo "Server: http://${WSL_IP}:8000"
echo "Open this URL on your iPhone 15 Pro in Safari."
echo ""
echo "Endpoints:"
echo "  /                        Web interface"
echo "  /api/v1/chat/stream      WebSocket streaming chat"
echo "  /api/v1/chat/compare     Base vs LoRA comparison"
echo "  /api/v1/metrics          System metrics"
echo "  /health                  Health check"
echo "  /docs                    API documentation"
echo ""

python3 -m uvicorn demonstrator.server.app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level info
