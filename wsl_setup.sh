#!/bin/bash
set -e
export DEBIAN_FRONTEND=noninteractive

echo "=== Step 1: Update packages ==="
apt-get update -qq

echo "=== Step 2: Install build tools + Python ==="
apt-get install -y -qq python3.11 python3.11-venv python3-pip git curl

echo "=== Step 3: Upgrade pip ==="
python3.11 -m pip install --upgrade pip

echo "=== Step 4: Install PyTorch with CUDA 12.8 ==="
python3.11 -m pip install torch --index-url https://download.pytorch.org/whl/cu128

echo "=== Step 5: Install cukks (auto-detects CUDA and installs cukks-cu128) ==="
python3.11 -m pip install cukks

echo "=== Step 6: Verify cukks backend ==="
python3.11 -c "import cukks; info = cukks.get_backend_info(); print('Backend:', info)"

echo "=== Done! ==="
