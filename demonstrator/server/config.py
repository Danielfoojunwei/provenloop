"""
Centralized configuration for the TenSafe Finance Demonstrator.

All environment variables are namespaced with TENSAFE_ and validated
at startup via Pydantic. Import ``settings`` from this module instead
of scattering ``os.getenv()`` throughout the codebase.

Usage:
    from demonstrator.server.config import settings
    print(settings.device)  # "cuda" or "cpu"
"""

import os

import torch
from pydantic import Field
from pydantic_settings import BaseSettings


class DemoConfig(BaseSettings):
    """Validated, type-safe configuration with env-var + .env support."""

    moe_config_path: str = Field(
        default="demonstrator/adapters/tgsp/moe_config.json",
        description="Path to the MoE config JSON (expert definitions + HE params).",
    )
    device: str = Field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        description="PyTorch device for inference ('cuda' or 'cpu').",
    )
    cors_origins: str = Field(
        default="http://localhost:8000,http://localhost:3000",
        description="Comma-separated list of allowed CORS origins.",
    )
    rate_limit_rpm: int = Field(
        default=60,
        ge=1,
        le=10000,
        description="Max requests per minute per IP.",
    )
    max_concurrent_generations: int = Field(
        default=2,
        ge=1,
        le=32,
        description="Max simultaneous GPU-bound generation tasks.",
    )
    tg_environment: str = Field(
        default="development",
        description="Environment name: 'development' or 'production'.",
    )
    log_level: str = Field(
        default="INFO",
        description="Python logging level.",
    )

    model_config = {
        "env_prefix": "TENSAFE_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# Singleton — import this from other modules
settings = DemoConfig()
