#!/usr/bin/env bash
# This script runs inside WSL with the cukks_env virtualenv.
# Called by train.sh or run directly from WSL.
set -euo pipefail

echo "============================================================"
echo "  TenSafe Finance Demonstrator - Training Pipeline (WSL)"
echo "  Model: Qwen/Qwen2.5-1.5B | Rank: 32 | Alpha: 64.0"
echo "  Crash-resilient: per-adapter processes, auto-resume."
echo "============================================================"
echo ""

# Activate virtualenv
source /root/cukks_env/bin/activate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ROOT_DIR="$(dirname "$PROJECT_DIR")"

export PYTHONPATH="${ROOT_DIR}"

cd "$ROOT_DIR"

SFT_ARGS=(
    --output-dir "${PROJECT_DIR}/adapters"
    --max-steps 2000
    --rank 32
    --alpha 64.0
    --batch-size 1
    --grad-accum 8
)

# --- Phase 1: SFT Training ---
echo "--- Phase 1: SFT Training ---"
echo ""

for adapter in banking_expert investment_expert shared_attention; do
    echo ">>> SFT: ${adapter}"
    python3 -m demonstrator.training.train_sft \
        --adapter "${adapter}" \
        "${SFT_ARGS[@]}" \
    || {
        echo "!!! ${adapter} SFT failed (exit $?). Re-run to resume."
        exit 1
    }
    echo ""
done

# --- Phase 2: RL Training ---
echo "--- Phase 2: RL Training (REINFORCE) ---"
echo ""

for adapter in banking_expert investment_expert shared_attention; do
    echo ">>> RL: ${adapter}"
    python3 -m demonstrator.training.train_rl \
        --adapter "${adapter}" \
        --sft-dir "${PROJECT_DIR}/adapters" \
        --output-dir "${PROJECT_DIR}/adapters" \
        --rl-steps 500 \
        --rank 32 \
        --alpha 64.0 \
    || {
        echo "!!! ${adapter} RL failed (exit $?). Re-run to resume."
        exit 1
    }
    echo ""
done

# --- Phase 3: MoE Assembly + TGSP ---
echo "--- Phase 3: Gated MoE Assembly + TGSP Packaging ---"
echo ""
python3 -m demonstrator.training.assemble_moe \
    --adapters-dir "${PROJECT_DIR}/adapters" \
    --output-dir "${PROJECT_DIR}/adapters/tgsp"

echo ""
echo "============================================================"
echo "  Training Pipeline Complete"
echo "  Adapters: ${PROJECT_DIR}/adapters/tgsp/"
echo "  Run: ./scripts/serve.sh to start the inference server"
echo "============================================================"
