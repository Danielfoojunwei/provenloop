# syntax=docker/dockerfile:1
# TenSafe Finance Demonstrator — Multi-stage build
# GPU-enabled container for WSL with NVIDIA A2000
#
# Build:   cd provenloop && docker compose -f demonstrator/docker-compose.yml build
# Run:     docker compose -f demonstrator/docker-compose.yml up
# Phone:   python relay.py  (separate terminal, exposes port 9095 to LAN)

# ======================================================================
# Stage 1: Build tensafe-he Rust CKKS library (PyO3 bindings)
# This stage is cached independently — only rebuilds when Rust source changes.
# Use Ubuntu 22.04 base to match runtime stage glibc (2.35).
# The previous rust:1.80-slim (Debian bookworm, glibc 2.36) produced
# wheels incompatible with the runtime stage's Ubuntu 22.04.
# ======================================================================
FROM ubuntu:22.04 AS rust-builder

# H7: Install Python 3.11 in builder to match runtime stage (cp311 wheels).
# Ubuntu 22.04 defaults to Python 3.10; maturin wheels target the build Python.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential software-properties-common pkg-config \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends python3.11 python3.11-dev python3-pip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.80.0
ENV PATH="/root/.cargo/bin:${PATH}"
RUN pip install --no-cache-dir --break-system-packages maturin>=1.0

COPY tensafe-he/ /build/tensafe-he/
RUN cd /build/tensafe-he/crates/tensafe-he-python \
    && maturin build --release \
    && ls /build/tensafe-he/target/wheels/

# ======================================================================
# Stage 2: Runtime (no Rust toolchain — saves ~1.5GB)
# Use 'devel' instead of 'runtime' — the tensafe-he-cuda kernels use
# NVRTC for runtime kernel compilation, which requires nvrtc headers
# and libcuda stubs only present in the devel image.  Flash Attention
# and AutoAWQ also need nvcc / CUDA headers for JIT compilation.
# ======================================================================
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    build-essential cmake git curl \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

# ---- Python dependencies ----
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ---- Install pre-built tensafe-he wheel from builder stage ----
COPY --from=rust-builder /build/tensafe-he/target/wheels/tensafe_he-*.whl /tmp/
RUN pip install --no-cache-dir /tmp/tensafe_he-*.whl && rm -f /tmp/tensafe_he-*.whl

# ---- TenSafe platform module (privacy budget, split inference) ----
COPY tensafe_platform/ /app/tensafe_platform/

# ---- Demonstrator code ----
COPY demonstrator/ /app/demonstrator/

# ---- Legacy project sources (external repos, optional) ----
RUN --mount=type=bind,source=.,target=/build_ctx \
    cp -r /build_ctx/TenSafe_Extracted/src /app/TenSafe_Extracted/src 2>/dev/null || true && \
    cp -r /build_ctx/Adapter-Safety-Tensafe/src /app/Adapter-Safety-Tensafe/src 2>/dev/null || true && \
    cp -r /build_ctx/TenSafe-Homormorphically-Encrypted-LoRA-Adaptation/src /app/TenSafe-HE-LoRA/src 2>/dev/null || true

# ---- Adapter storage ----
RUN mkdir -p /app/demonstrator/adapters/tgsp /app/demonstrator/checkpoints

# ---- L10: Non-root user for production security ----
RUN groupadd -r tensafe && useradd -r -g tensafe -d /app tensafe \
    && chown -R tensafe:tensafe /app
USER tensafe

# ---- Environment ----
ENV PYTHONPATH=/app
ENV MOE_CONFIG_PATH=/app/demonstrator/adapters/tgsp/moe_config.json
ENV DEVICE=cuda
ENV PYTHONUNBUFFERED=1

# Server on 8000 (API + frontend)
EXPOSE 8000

# Health check — model loading can take 30+ seconds so give a long start period
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: serve inference with GateLink split enabled
CMD ["python3", "-m", "uvicorn", "demonstrator.server.app:app", \
     "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
