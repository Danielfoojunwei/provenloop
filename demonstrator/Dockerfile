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

# ---- TenSafe project sources ----
COPY TenSafe_Extracted/src /app/TenSafe_Extracted/src
COPY TenSafe_Extracted/pyproject.toml /app/TenSafe_Extracted/pyproject.toml
COPY TenSafe_Extracted/setup.cfg /app/TenSafe_Extracted/setup.cfg 2>/dev/null || true

COPY Adapter-Safety-Tensafe/src /app/Adapter-Safety-Tensafe/src
COPY Adapter-Safety-Tensafe/pyproject.toml /app/Adapter-Safety-Tensafe/pyproject.toml 2>/dev/null || true

COPY TenSafe-Homormorphically-Encrypted-LoRA-Adaptation/src /app/TenSafe-HE-LoRA/src

# ---- Demonstrator code ----
COPY demonstrator/ /app/demonstrator/

# ---- Adapter storage ----
RUN mkdir -p /app/demonstrator/adapters/tgsp /app/demonstrator/checkpoints

# ---- Environment ----
ENV PYTHONPATH=/app/TenSafe_Extracted/src:/app/Adapter-Safety-Tensafe/src:/app/TenSafe-HE-LoRA/src:/app
ENV MOE_CONFIG_PATH=/app/demonstrator/adapters/tgsp/moe_config.json
ENV DEVICE=cuda
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Default: serve inference. Override with training scripts as needed.
CMD ["python3", "-m", "uvicorn", "demonstrator.server.app:app", \
     "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
