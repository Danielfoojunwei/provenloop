# syntax=docker/dockerfile:1
# TenSafe Finance Demonstrator
# GPU-enabled container for WSL with NVIDIA A2000
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    build-essential cmake git curl \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

# ---- Python dependencies ----
COPY demonstrator/requirements.txt /app/demonstrator/requirements.txt
RUN pip install --no-cache-dir -r /app/demonstrator/requirements.txt

# ---- TenSafe platform module (privacy budget, split inference) ----
COPY tensafe_platform/ /app/tensafe_platform/

# ---- Demonstrator code ----
COPY demonstrator/ /app/demonstrator/

# ---- Legacy project sources (external repos, optional) ----
# TenSafe_Extracted/, Adapter-Safety-Tensafe/, and
# TenSafe-Homormorphically-Encrypted-LoRA-Adaptation/ are separate git
# repos.  When present at the Docker build context root they are copied
# in; when absent, the build still succeeds — the inference engine falls
# back to its built-in CKKS emulator.
#
# To include them, clone the repos next to this repo before building:
#   git clone <url> ../TenSafe_Extracted
#   git clone <url> ../Adapter-Safety-Tensafe
#   git clone <url> ../TenSafe-Homormorphically-Encrypted-LoRA-Adaptation
#   docker compose build
RUN --mount=type=bind,source=.,target=/build_ctx \
    cp -r /build_ctx/TenSafe_Extracted/src /app/TenSafe_Extracted/src 2>/dev/null || true && \
    cp -r /build_ctx/Adapter-Safety-Tensafe/src /app/Adapter-Safety-Tensafe/src 2>/dev/null || true && \
    cp -r /build_ctx/TenSafe-Homormorphically-Encrypted-LoRA-Adaptation/src /app/TenSafe-HE-LoRA/src 2>/dev/null || true

# ---- Adapter storage ----
RUN mkdir -p /app/demonstrator/adapters/tgsp /app/demonstrator/checkpoints

# ---- Environment ----
# /app is on PYTHONPATH so both `demonstrator.*` and `tensafe_platform.*` are importable.
# Legacy source paths are added only when the external repos were present at build time.
ENV PYTHONPATH=/app
ENV MOE_CONFIG_PATH=/app/demonstrator/adapters/tgsp/moe_config.json
ENV DEVICE=cuda
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Default: serve inference. Override with training scripts as needed.
CMD ["python3", "-m", "uvicorn", "demonstrator.server.app:app", \
     "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
