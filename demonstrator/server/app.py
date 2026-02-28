"""
TenSafe Finance Demonstrator — FastAPI server.

Endpoints:
  WS   /api/v1/chat/stream      — streaming chat with per-token HE metrics
  WS   /api/v1/split/stream     — persistent WebSocket for split inference
  POST /api/v1/chat/compare      — base model vs LoRA-adapted comparison
  POST /api/v1/split/forward     — GateLink-Split server-side forward pass
  GET  /api/v1/metrics           — live system metrics (HE, DP, adapters)
  GET  /api/v1/split/config      — split inference config for WASM client
  GET  /health                   — health check
  POST /api/v1/adapters/{name}/swap — hot-swap a TGSP adapter
  GET  /api/v1/adapters           — list loaded adapters
  /                              — static web frontend
"""

import asyncio
import base64
import json
import logging
import os
import time as _time
import uuid as _uuid
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .inference_engine import FinanceInferenceEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MOE_CONFIG = os.getenv(
    "MOE_CONFIG_PATH", "demonstrator/adapters/tgsp/moe_config.json",
)
TGSP_DIR = os.getenv(
    "TGSP_DIR", "demonstrator/adapters/tgsp",
)
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(
    title="TenSafe Finance Demonstrator",
    version="2.0.0",
    docs_url="/docs" if os.getenv("TG_ENVIRONMENT") != "production" else None,
)

CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS", "*"  # Allow all origins: phone accesses via LAN IP (e.g. 192.168.x.x:9090)
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# ======================================================================
# Security headers
# ======================================================================

@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

    # Prevent caching of HTML/JS/CSS files — critical for development.
    # Without this, phone browsers serve stale JS with old (broken) code.
    # Weight files (.bin) are large and safe to cache (versioned via cache key).
    path = request.url.path
    if path.endswith((".html", ".js", ".css")) or path == "/":
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"

    return response


# ======================================================================
# Rate limiting (simple per-IP token bucket)
# ======================================================================

_RATE_WINDOW = 60.0  # seconds
_RATE_MAX = int(os.getenv("RATE_LIMIT_RPM", "600"))  # requests per minute (split mode sends 1 req/token)
_rate_buckets: dict[str, list[float]] = defaultdict(list)


def _check_rate_limit(client_ip: str) -> bool:
    """Return True if request is allowed, False if rate-limited."""
    now = _time.time()
    bucket = _rate_buckets[client_ip]
    _rate_buckets[client_ip] = [t for t in bucket if now - t < _RATE_WINDOW]
    if len(_rate_buckets[client_ip]) >= _RATE_MAX:
        return False
    _rate_buckets[client_ip].append(now)
    return True


# ======================================================================
# Request latency tracking
# ======================================================================

_request_stats: dict = {"total": 0, "errors": 0, "latencies_ms": []}


engine: FinanceInferenceEngine | None = None
marketplace = None  # AdapterMarketplace instance (initialized at startup)

# Concurrency gate: limit simultaneous GPU-bound generations
_MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT_GENERATIONS", "2"))
_gen_semaphore = asyncio.Semaphore(_MAX_CONCURRENT)


# ======================================================================
# Lifecycle
# ======================================================================

@app.on_event("startup")
async def startup():
    global engine, marketplace
    logger.info("Starting inference engine ...")
    engine = FinanceInferenceEngine(moe_config_path=MOE_CONFIG, device=DEVICE)
    engine.initialize()

    # Initialize marketplace registry (scans TGSP directory)
    from .marketplace import AdapterMarketplace
    try:
        marketplace = AdapterMarketplace(TGSP_DIR)
        logger.info(
            f"Marketplace ready: {len(marketplace.list_adapters())} adapters indexed"
        )
    except Exception as e:
        logger.warning(f"Marketplace init failed (non-fatal): {e}")
        marketplace = None
    logger.info("Engine ready.")


@app.on_event("shutdown")
async def shutdown():
    logger.info("Shutting down — clearing KV cache and GPU memory...")
    if engine:
        engine._kv_cache_store.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    logger.info("Shutdown complete.")


# ======================================================================
# Health
# ======================================================================

@app.get("/health")
async def health():
    engine_ok = engine is not None and engine._initialized
    gpu_ok = True
    if torch.cuda.is_available():
        try:
            mem_used = torch.cuda.memory_allocated(0) / 1048576
            props = torch.cuda.get_device_properties(0)
            total = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / 1048576
            gpu_ok = total == 0 or (mem_used / total) < 0.95
        except Exception:
            gpu_ok = False

    status = "ok" if engine_ok and gpu_ok else "degraded"
    return {
        "status": status,
        "engine_ready": engine_ok,
        "gpu_healthy": gpu_ok,
        "adapters_loaded": len(engine.adapters) if engine else 0,
        "he_backend": type(engine._cukks).__name__ if engine and engine._cukks else "none",
        "kv_cache_sessions": len(engine._kv_cache_store._store) if engine else 0,
    }


# ======================================================================
# WebSocket streaming chat
# ======================================================================

@app.websocket("/api/v1/chat/stream")
async def chat_stream(ws: WebSocket):
    await ws.accept()
    logger.info("WS client connected")
    try:
        while True:
            raw = await ws.receive_text()
            req = json.loads(raw)

            query = (req.get("query", "") or "")[:10000]
            max_tok = min(max(int(req.get("max_tokens", 256)), 1), 2048)
            temp = min(max(float(req.get("temperature", 0.7)), 0.0), 2.0)
            top_p = min(max(float(req.get("top_p", 0.9)), 0.0), 1.0)
            top_k = min(max(int(req.get("top_k", 50)), 0), 200)
            use_he = bool(req.get("use_he", True))
            session_id = str(req.get("session_id", "default"))[:128]

            logger.info(f"Chat: {query[:80]}...")

            async with _gen_semaphore:
                for chunk in engine.generate_stream(
                    query=query,
                    max_tokens=max_tok,
                    temperature=temp,
                    top_p=top_p,
                    top_k=top_k,
                    use_he=use_he,
                    session_id=session_id,
                ):
                    await ws.send_text(json.dumps(chunk, default=str))
                    await asyncio.sleep(0)
    except WebSocketDisconnect:
        logger.info("WS client disconnected")
    except Exception as exc:
        logger.exception(f"WS error: {exc}")
        try:
            # Send error JSON so frontend can display it (app.js:172 handles type=error)
            await ws.send_text(json.dumps({
                "type": "error",
                "message": str(exc)[:500],
            }))
            await ws.close(code=1011, reason=str(exc)[:120])
        except Exception:
            pass


# ======================================================================
# WebSocket split inference (persistent connection, binary-capable)
# ======================================================================

@app.websocket("/api/v1/split/stream")
async def split_stream(ws: WebSocket):
    """WebSocket endpoint for split inference.

    Eliminates per-token HTTP overhead by keeping a persistent connection.
    Supports BOTH text (JSON+base64) and binary protocols:

    Text protocol (backward compatible):
      Client -> Server:  {"type": "forward", "hidden_states_b64": "...", ...}
      Server -> Client:  {"type": "result", "pre_activations_b64": "...", ...}

    Binary protocol (faster, no base64 overhead):
      Client -> Server:  [4B json_len][JSON metadata][raw float32 hidden_states]
      Server -> Client:  [4B json_len][JSON metadata][raw float32 pre_activations]
                         [raw float32 logits_top_k][raw int32 logits_top_ids]
      JSON metadata contains type, metrics, dimensions.

      Client -> Server:  {"type": "ping"}
      Server -> Client:  {"type": "pong"}
    """
    await ws.accept()
    logger.info("Split-WS client connected")
    try:
        while True:
            # Accept BOTH text and binary frames
            ws_msg = await ws.receive()
            if "text" in ws_msg and ws_msg["text"]:
                raw = ws_msg["text"]
                msg = json.loads(raw)
                msg_type = msg.get("type", "forward")
                binary_mode = False
            elif "bytes" in ws_msg and ws_msg["bytes"]:
                # Binary protocol: [4B json_len][JSON][float32 payload]
                raw_bytes = ws_msg["bytes"]
                json_len = int.from_bytes(raw_bytes[:4], "little")
                msg = json.loads(raw_bytes[4 : 4 + json_len])
                binary_payload = raw_bytes[4 + json_len :]
                msg_type = msg.get("type", "forward")
                binary_mode = True
            else:
                continue

            if msg_type == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
                continue

            if msg_type == "config":
                cfg = {
                    "type": "config",
                    "model": engine.moe_config.get("model", ""),
                    "hidden_dim": engine.model.config.hidden_size,
                    "vocab_size": engine.tokenizer.vocab_size,
                    "num_layers": len(engine.model.model.layers),
                    "client_layers": 1,
                    "dp_epsilon": engine._dp_epsilon,
                    "dp_delta": engine._dp_delta,
                    "dp_sigma": round(engine._dp_sigma, 6),
                    "dp_sensitivity": engine._dp_sensitivity,
                    "experts": list(engine.adapters.keys()),
                    "expert_keywords": {
                        n: engine.moe_config["experts"][n].get("gate_keywords", [])
                        for n in engine.adapters
                        if n in engine.moe_config.get("experts", {})
                    },
                    "simd_slots": engine.simd_slots,
                    "he_active": engine.he_ctx is not None,
                }
                await ws.send_text(json.dumps(cfg))
                continue

            if msg_type == "tokenize":
                text = str(msg.get("text", ""))[:10000]
                ids = engine.tokenizer.encode(text)
                await ws.send_text(json.dumps({
                    "type": "token_ids",
                    "token_ids": ids,
                    "length": len(ids),
                }))
                continue

            if msg_type != "forward":
                await ws.send_text(json.dumps({
                    "type": "error",
                    "message": "Unknown message type: " + str(msg_type),
                }))
                continue

            # ---- forward pass (same logic as HTTP split/forward) ----
            client_ip = ws.client.host if ws.client else "unknown"
            if not _check_rate_limit(client_ip):
                await ws.send_text(json.dumps({
                    "type": "error",
                    "message": "Rate limit exceeded",
                }))
                continue

            try:
                seq_len = min(max(int(msg.get("seq_len", 1)), 1), 4096)
                hidden_dim = int(msg.get("hidden_dim", engine.model.config.hidden_size))
                expert_name = str(msg.get("expert_name", "shared_attention"))[:128]
                use_he = bool(msg.get("use_he", True))
                session_id = str(msg.get("session_id", "default"))[:128]
                incremental = bool(msg.get("incremental", False))

                # Decode hidden states: binary payload or base64 JSON field
                if binary_mode:
                    h_np = np.frombuffer(binary_payload, dtype=np.float32).reshape(seq_len, hidden_dim)
                else:
                    h_b64 = msg.get("hidden_states_b64", "")
                    h_bytes = base64.b64decode(h_b64)
                    h_np = np.frombuffer(h_bytes, dtype=np.float32).reshape(seq_len, hidden_dim)

                async with _gen_semaphore:
                    result = engine.split_forward(
                        hidden_states_np=h_np,
                        expert_name=expert_name,
                        use_he=use_he,
                        session_id=session_id,
                        incremental=incremental,
                    )

                pre_np = result["pre_activations"].astype(np.float32)

                # Build response metadata (no base64 in binary mode)
                resp = {
                    "type": "result",
                    "seq_len": pre_np.shape[0] if pre_np.ndim > 1 else 1,
                    "hidden_dim": hidden_dim,
                    "he_active": result.get("he_active", False),
                    "encrypt_ms": result.get("encrypt_ms", 0),
                    "compute_ms": result.get("compute_ms", 0),
                    "decrypt_ms": result.get("decrypt_ms", 0),
                    "he_operations": result.get("he_operations", 0),
                    "dp_sigma": result.get("dp_sigma", 0),
                    "total_ms": result.get("total_ms", 0),
                    "incremental": result.get("incremental", False),
                    "cached_seq_len": result.get("cached_seq_len", 0),
                    "layers_computed": result.get("layers_computed", 0),
                }

                has_logits = result.get("logits_top_k") is not None
                if has_logits:
                    resp["lm_head_ms"] = result.get("lm_head_ms", 0)
                    resp["logits_k"] = len(result["logits_top_k"])

                if binary_mode:
                    # Binary response: [4B json_len][JSON][float32 pre_act]
                    #   [float32 logits_top_k][int32 logits_top_ids]
                    json_bytes = json.dumps(resp, default=str).encode()
                    parts = [
                        len(json_bytes).to_bytes(4, "little"),
                        json_bytes,
                        pre_np.tobytes(),
                    ]
                    if has_logits:
                        parts.append(np.array(result["logits_top_k"], dtype=np.float32).tobytes())
                        parts.append(np.array(result["logits_top_ids"], dtype=np.int32).tobytes())
                    await ws.send_bytes(b"".join(parts))
                else:
                    # Text response (backward compatible)
                    resp["pre_activations_b64"] = base64.b64encode(
                        pre_np.tobytes()
                    ).decode()
                    if has_logits:
                        resp["logits_top_k"] = result["logits_top_k"]
                        resp["logits_top_ids"] = result["logits_top_ids"]
                    await ws.send_text(json.dumps(resp, default=str))

            except Exception as exc:
                logger.exception("Split-WS forward error: %s" % exc)
                await ws.send_text(json.dumps({
                    "type": "error",
                    "message": str(exc)[:500],
                }))

    except WebSocketDisconnect:
        logger.info("Split-WS client disconnected")
    except Exception as exc:
        logger.exception("Split-WS error: %s" % exc)
        try:
            await ws.send_text(json.dumps({
                "type": "error",
                "message": str(exc)[:500],
            }))
            await ws.close(code=1011, reason=str(exc)[:120])
        except Exception:
            pass


# ======================================================================
# Comparison
# ======================================================================

class CompareRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000)
    max_tokens: int = Field(128, ge=1, le=2048)
    temperature: float = Field(0.7, ge=0.0, le=2.0)


@app.post("/api/v1/chat/compare")
async def compare(req: CompareRequest, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_ip):
        return JSONResponse(status_code=429, content={"error": "Rate limit exceeded"})
    _request_stats["total"] += 1
    t0 = _time.time()
    try:
        async with _gen_semaphore:
            result = engine.generate_comparison(
                query=req.query,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
            )
        _request_stats["latencies_ms"].append((_time.time() - t0) * 1000)
        _request_stats["latencies_ms"] = _request_stats["latencies_ms"][-200:]
        return JSONResponse(content=result)
    except Exception as exc:
        _request_stats["errors"] += 1
        logger.exception(f"Compare error: {exc}")
        return JSONResponse(
            status_code=500,
            content={"error": str(exc)[:500]},
        )


# ======================================================================
# GateLink-Split: server-side forward pass endpoint
# ======================================================================

class SplitForwardRequest(BaseModel):
    """Client sends DP-noised hidden states after running K client layers."""
    hidden_states_b64: str  # base64-encoded float32 numpy array
    seq_len: int = Field(..., ge=1, le=4096)
    hidden_dim: int = Field(1536, ge=1, le=8192)
    expert_name: str = Field("shared_attention", max_length=128)
    use_he: bool = True
    session_id: str = Field("default", max_length=128)
    incremental: bool = False  # True = use cached KV from previous call


@app.post("/api/v1/split/forward")
async def split_forward(req: SplitForwardRequest, request: Request):
    """GateLink-Split server endpoint.

    Client sends DP-noised hidden states (after embedding + K client layers).
    Server runs remaining N-K layers with HE-encrypted LoRA, returns
    pre-LM-head hidden states for client-side logit projection.
    """
    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_ip):
        return JSONResponse(status_code=429, content={"error": "Rate limit exceeded"})
    _request_stats["total"] += 1
    t0 = _time.time()
    try:
        # Decode hidden states from base64
        raw = base64.b64decode(req.hidden_states_b64)

        # Validate payload size matches declared dimensions
        expected_bytes = req.seq_len * req.hidden_dim * 4  # float32 = 4 bytes
        if len(raw) != expected_bytes:
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"Payload size mismatch: got {len(raw)} bytes, "
                    f"expected {expected_bytes} (seq_len={req.seq_len} × "
                    f"hidden_dim={req.hidden_dim} × 4)",
                },
            )

        hidden_np = np.frombuffer(raw, dtype=np.float32).reshape(
            req.seq_len, req.hidden_dim
        )

        # Diagnostic: log input stats for debugging phone issues
        h_norm = float(np.linalg.norm(hidden_np[-1]))
        h_nan = int(np.isnan(hidden_np).sum())
        h_inf = int(np.isinf(hidden_np).sum())
        logger.info(
            f"split_forward: seq={req.seq_len} expert={req.expert_name} "
            f"he={req.use_he} incr={req.incremental} sid={req.session_id[:20]} "
            f"last_h_norm={h_norm:.2f} nan={h_nan} inf={h_inf}"
        )

        async with _gen_semaphore:
            result = engine.split_forward(
                hidden_states_np=hidden_np,
                expert_name=req.expert_name,
                use_he=req.use_he,
                session_id=req.session_id,
                incremental=req.incremental,
            )

            # Encode output pre-activations as base64
            pre_act = result.pop("pre_activations")
            result["pre_activations_b64"] = base64.b64encode(
                pre_act.astype(np.float32).tobytes()
            ).decode("ascii")
            result["output_shape"] = list(pre_act.shape)

            _request_stats["latencies_ms"].append((_time.time() - t0) * 1000)
            _request_stats["latencies_ms"] = _request_stats["latencies_ms"][-200:]
            return JSONResponse(content=result)
    except Exception as exc:
        _request_stats["errors"] += 1
        logger.exception(f"Split forward error: {exc}")
        return JSONResponse(
            status_code=500,
            content={"error": str(exc)[:500]},
        )


# ======================================================================
# Split inference config (for WASM client bootstrap)
# ======================================================================

@app.get("/api/v1/split/config")
async def split_config():
    """Return configuration needed by the split inference client layer."""
    glc = engine.moe_config.get("gatelink_config", {}) if engine else {}

    # Build expert keyword map for client-side routing
    expert_keywords = {}
    if engine:
        for name, adp in engine.adapters.items():
            expert_keywords[name] = list(adp.get("gate_keywords", set()))

    return {
        "model": "Qwen/Qwen2.5-1.5B",
        "hidden_dim": 1536,
        "vocab_size": 151936,
        "num_layers": 28,
        "client_layers": glc.get("client_layers", 1),
        # Split-mode DP: noise injected SERVER-SIDE on post-transformer
        # hidden states (norm ~165-190), NOT on phone-side raw embeddings
        # (norm ~0.8-1.2). client_dp_sigma=0 tells the phone to skip noisify().
        "dp_epsilon": engine._dp_epsilon if engine else 0.0,
        "dp_delta": engine._dp_delta if engine else 1e-5,
        "dp_sigma": round(engine._dp_sigma, 6) if engine else 0.0,
        "client_dp_sigma": 0.0,  # Phone must NOT noise raw embeddings (SNR=0.005 → hallucination)
        "dp_sensitivity": engine._dp_sensitivity if engine else 1.0,
        "max_epsilon": engine._max_epsilon if engine else 10.0,
        "experts": list(engine.adapters.keys()) if engine else [],
        "expert_keywords": expert_keywords,
        "simd_slots": engine.simd_slots if engine else 0,
        "he_active": engine.he_ctx is not None if engine else False,
        "lora_rank": _get_effective_lora_rank(engine),
        "target_lora_rank": glc.get("target_lora_rank", 32),
    }


def _get_effective_lora_rank(eng) -> int:
    """Return the actual LoRA rank after SVD truncation."""
    if not eng or not eng.adapters:
        return 0
    for adp in eng.adapters.values():
        w = adp.get("weights", {})
        a = w.get("lora_A")
        if a is not None:
            return a.shape[0] if hasattr(a, "shape") else 0
    return 0


# ======================================================================
# Server-side tokenization (reliable fallback for JS BPE mismatch)
# ======================================================================

class TokenizeRequest(BaseModel):
    text: str = Field(..., max_length=4096)

@app.post("/api/v1/split/tokenize")
async def split_tokenize(req: TokenizeRequest):
    """Tokenize text using the model's HuggingFace tokenizer.

    The client's JS BPE tokenizer can produce wrong token IDs due to
    Unicode regex differences across browser engines.  This endpoint
    provides a reliable fallback using the exact same tokenizer that
    the model was trained with.
    """
    if not engine or not engine._initialized:
        return JSONResponse(status_code=503, content={"error": "Engine not ready"})
    ids = engine.tokenizer.encode(req.text)
    return {"token_ids": ids, "length": len(ids)}


# ======================================================================
# E2E integrity verification (client calls during init to verify weights)
# ======================================================================

class VerifyRequest(BaseModel):
    probe_text: str = Field("hello", max_length=256)

@app.post("/api/v1/split/verify")
async def split_verify(req: VerifyRequest):
    """Verify client can produce correct embeddings.

    Returns token IDs and an embedding hash for the probe text.
    Client compares its local embedding hash to detect weight corruption.
    """
    if not engine or not engine._initialized:
        return JSONResponse(status_code=503, content={"error": "Engine not ready"})

    ids = engine.tokenizer.encode(req.probe_text)
    if not ids:
        return {"token_ids": [], "embed_hash": 0, "status": "empty"}

    # Compute expected embedding hash from the SAME embed_tokens.bin file
    # that the JS client uses (not from the model, which might differ in precision).
    # Hash: sum of round(float16_to_float32(emb[d]) * 10000) for d in 0..7
    bin_path = _FRONTEND / "weights" / "embed_tokens.bin"
    hidden_dim = engine.model.config.hidden_size  # 1536
    if bin_path.exists():
        raw = np.fromfile(str(bin_path), dtype=np.float16)
        off = ids[0] * hidden_dim
        first_emb_f32 = raw[off : off + hidden_dim].astype(np.float32)
    else:
        # Fallback: use model weights
        first_emb_f32 = engine.model.model.embed_tokens.weight[ids[0]].float().cpu().numpy()

    embed_hash = 0
    for d in range(min(8, hidden_dim)):
        embed_hash += round(float(first_emb_f32[d]) * 10000)

    return {
        "token_ids": ids,
        "embed_hash": embed_hash,
        "vocab_size": engine.tokenizer.vocab_size,
        "hidden_dim": hidden_dim,
        "status": "ok",
    }


# ======================================================================
# Self-test: definitive server-side E2E pipeline check
# ======================================================================

@app.get("/api/v1/selftest")
async def selftest(query: str = "What is a savings account?"):
    """Run the FULL generate_stream pipeline server-side and return text.

    Open this URL in the phone browser to verify the server produces correct
    output THROUGH the TCP relay.  If this returns garbage, the server has a
    problem (possibly CuKKS CUDA memory corruption).  If it returns clean
    English, the server is fine and the issue is in the WebSocket/JS path.

    Usage: GET /api/v1/selftest?query=What+is+a+savings+account

    Returns JSON with generated text, HE status, model weight hash, and
    CuKKS backend info for debugging.
    """
    if not engine or not engine._initialized:
        return JSONResponse(status_code=503, content={"error": "Engine not ready"})

    # Run generation with GREEDY sampling (temperature=0) for determinism
    tokens = []
    agg_data = None
    try:
        async with _gen_semaphore:
            for chunk in engine.generate_stream(
                query=query,
                max_tokens=32,
                temperature=0.0,  # GREEDY — deterministic, no randomness
                top_p=1.0,
                top_k=0,
                use_he=True,
                session_id="selftest_" + str(int(_time.time())),
            ):
                if chunk.get("type") == "token":
                    tokens.append(chunk["token"])
                elif chunk.get("type") == "done":
                    agg_data = chunk.get("aggregate", {})
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "error": str(e)[:500],
            "tokens_before_error": tokens,
        })

    generated_text = "".join(tokens)

    # Model weight integrity check: hash first 16 values of LM head weight
    lm_head_hash = 0
    try:
        w = engine.model.lm_head.weight[:4, :4].float().cpu().detach().numpy()
        for i in range(4):
            for j in range(4):
                lm_head_hash += round(float(w[i, j]) * 10000)
    except Exception:
        lm_head_hash = -1

    # Also test with HE=False for comparison
    tokens_nohe = []
    try:
        async with _gen_semaphore:
            for chunk in engine.generate_stream(
                query=query,
                max_tokens=32,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                use_he=False,
                session_id="selftest_nohe_" + str(int(_time.time())),
            ):
                if chunk.get("type") == "token":
                    tokens_nohe.append(chunk["token"])
    except Exception:
        tokens_nohe = ["(error)"]

    return {
        "query": query,
        "generated_text_HE": generated_text,
        "generated_text_noHE": "".join(tokens_nohe),
        "tokens_HE": len(tokens),
        "tokens_noHE": len(tokens_nohe),
        "he_match": generated_text == "".join(tokens_nohe),
        "lm_head_weight_hash": lm_head_hash,
        "he_backend": type(engine._cukks).__name__ if engine._cukks else "none",
        "he_status": engine._cukks.status() if hasattr(engine._cukks, 'status') else {},
        "simd_slots": engine.simd_slots,
        "dp_sigma": round(engine._dp_sigma, 4),
        "aggregate": agg_data,
        "verdict": "PASS — server generates clean output" if (
            generated_text and
            all(ord(c) < 0x4E00 or ord(c) > 0x9FFF for c in generated_text)
        ) else "FAIL — output contains CJK characters (possible corruption)",
    }


# ======================================================================
# Split Mode E2E Self-Test (simulates the phone's JS client server-side)
# ======================================================================

@app.get("/api/v1/split/selftest")
async def split_selftest(query: str = "What is a savings account?"):
    """Simulate the ENTIRE split inference pipeline server-side.

    This does EXACTLY what the phone's split_client.js does:
    1. Tokenize with server tokenizer (same as /api/v1/split/tokenize)
    2. Look up embeddings from embed_tokens.bin (same file phone downloads)
    3. Send through split_forward (transformer + HE LoRA)
    4. Project through LM head using same float16 weights
    5. Greedy sample and return result

    If this produces clean output, the split pipeline is correct.
    If garbage, something is wrong in the server split path.
    """
    import struct

    if not engine or not engine._initialized:
        return JSONResponse(status_code=503, content={"error": "Engine not ready"})

    # Load the SAME embed_tokens.bin the client downloads
    bin_path = _FRONTEND / "weights" / "embed_tokens.bin"
    if not bin_path.exists():
        return JSONResponse(status_code=500, content={
            "error": "embed_tokens.bin not found at " + str(bin_path)
        })

    HD = engine.model.config.hidden_size  # 1536

    # Read as float16 (same as JS Uint16Array)
    raw_f16 = np.fromfile(str(bin_path), dtype=np.float16)
    # Derive VS from actual file size (includes special tokens beyond tokenizer.vocab_size)
    VS = raw_f16.shape[0] // HD
    if raw_f16.shape[0] != VS * HD:
        return JSONResponse(status_code=500, content={
            "error": f"Weight size not divisible by HD: {raw_f16.shape[0]} / {HD} = {raw_f16.shape[0]/HD:.2f}"
        })

    # 1. Tokenize (same prompt format as split_client.js)
    chat_prompt = (
        "### System:\nYou are a helpful financial assistant.\n\n"
        f"### Instruction:\n{query}\n\n### Response:\n"
    )
    token_ids = engine.tokenizer.encode(chat_prompt)

    # 2. Embed using float16 weights → float32 (same as JS F16_TABLE lookup)
    def embed_from_bin(tid):
        off = tid * HD
        return raw_f16[off:off + HD].astype(np.float32)

    # 3. LM head projection using same weights (same as JS project())
    def project_from_bin(hidden_f32):
        # hidden: (HD,) float32.  weights: (VS, HD) float16
        # dot product per vocab token → logits
        logits = np.zeros(VS, dtype=np.float32)
        for v in range(VS):
            off = v * HD
            row = raw_f16[off:off + HD].astype(np.float32)
            logits[v] = np.dot(hidden_f32, row)
        return logits

    # Faster vectorized version of project
    def project_from_bin_fast(hidden_f32):
        W = raw_f16.reshape(VS, HD).astype(np.float32)
        return W @ hidden_f32

    generated_tokens = []
    generated_text = []
    session_id = "split_selftest_" + str(int(_time.time()))
    diag_steps = []

    try:
        cur_ids = list(token_ids)
        async with _gen_semaphore:
            for step in range(64):  # max 64 tokens (enough for full sentences)
                incremental = step > 0

                if incremental:
                    # Only embed last token
                    emb = embed_from_bin(cur_ids[-1])
                    flat = emb  # (HD,) float32
                    seq_len = 1
                else:
                    # Embed full sequence
                    flat = np.stack([embed_from_bin(tid) for tid in cur_ids])
                    # flat: (seq_len, HD) float32
                    seq_len = len(cur_ids)

                if flat.ndim == 1:
                    flat = flat.reshape(1, HD)

                # Send through split_forward (same as POST /api/v1/split/forward)
                result = engine.split_forward(
                    hidden_states_np=flat,
                    expert_name=engine.route_expert(query),
                    use_he=True,
                    session_id=session_id,
                    incremental=incremental,
                )

                # Prefer server-side top-256 logits (GPU, ~5ms) over
                # numpy LM head projection (~500ms per token)
                if result.get("logits_top_k") is not None:
                    top_k = np.array(result["logits_top_k"], dtype=np.float32)
                    top_ids = np.array(result["logits_top_ids"], dtype=np.int32)
                    next_id = int(top_ids[np.argmax(top_k)])
                    # For diagnostics, build sparse logits view
                    logits_sparse = (top_k, top_ids)
                else:
                    # Fallback: numpy LM head projection (same as JS project())
                    pre_act = result["pre_activations"]
                    if result.get("incremental"):
                        last_h = pre_act.flatten()[:HD]
                    else:
                        last_h = pre_act.reshape(-1, HD)[-1]
                    logits_full = project_from_bin_fast(last_h.astype(np.float32))
                    next_id = int(np.argmax(logits_full))
                    logits_sparse = None

                # Diagnostic for first 5 tokens
                if step < 5:
                    if logits_sparse is not None:
                        top_k_v, top_k_i = logits_sparse
                        sort_idx = np.argsort(top_k_v)[-5:][::-1]
                        top5_info = [
                            {"id": int(top_k_i[i]), "logit": f"{top_k_v[i]:.2f}",
                             "tok": engine.tokenizer.decode([int(top_k_i[i])])}
                            for i in sort_idx
                        ]
                    else:
                        top5_ids = np.argsort(logits_full)[-5:][::-1]
                        top5_info = [
                            {"id": int(i), "logit": f"{logits_full[i]:.2f}",
                             "tok": engine.tokenizer.decode([int(i)])}
                            for i in top5_ids
                        ]
                    # Compute h_norm from pre_activations
                    pre_act = result["pre_activations"]
                    if result.get("incremental"):
                        h_for_norm = pre_act.flatten()[:HD]
                    else:
                        h_for_norm = pre_act.reshape(-1, HD)[-1]
                    diag_steps.append({
                        "step": step,
                        "next_id": next_id,
                        "token": engine.tokenizer.decode([next_id]),
                        "top5": top5_info,
                        "h_norm": f"{np.linalg.norm(h_for_norm):.2f}",
                        "he_active": result.get("he_active", False),
                        "dp_sigma": result.get("dp_sigma", 0.0),
                    })

                # EOS check
                tok_text = engine.tokenizer.decode([next_id])
                eos_ids = {
                    engine.tokenizer.eos_token_id,
                    engine.tokenizer.convert_tokens_to_ids("<|im_end|>"),
                }
                if next_id in eos_ids:
                    break

                # Stop on section boundary
                if len(generated_text) > 2:
                    tail = "".join(generated_text[-3:]) + tok_text
                    if "\n###" in tail or "\n\n###" in tail:
                        break

                cur_ids.append(next_id)
                generated_tokens.append(next_id)
                generated_text.append(tok_text)

    except Exception as e:
        import traceback
        return JSONResponse(status_code=500, content={
            "error": str(e)[:500],
            "traceback": traceback.format_exc()[-1000:],
            "tokens_before_error": generated_text,
            "diag_steps": diag_steps,
        })

    full_text = "".join(generated_text)
    has_cjk = any(0x4E00 <= ord(c) <= 0x9FFF for c in full_text)

    return {
        "query": query,
        "generated_text": full_text,
        "total_tokens": len(generated_tokens),
        "input_tokens": len(token_ids),
        "he_backend": type(engine._cukks).__name__ if engine._cukks else "none",
        "simd_slots": engine.simd_slots,
        "dp_sigma": round(engine._dp_sigma, 4),
        "dp_epsilon": engine._dp_epsilon,
        "diag_steps": diag_steps,
        "verdict": (
            "PASS — split pipeline generates clean output with DP noise"
            if (full_text and not has_cjk)
            else "FAIL — split output corrupted"
        ),
    }


# ======================================================================
# Adapter management (Phase 6 — marketplace hot-swap)
# ======================================================================

@app.get("/api/v1/adapters")
async def list_adapters():
    """List all loaded adapters with their config and marketplace metadata."""
    if not engine or not engine._initialized:
        return JSONResponse(
            status_code=503,
            content={"error": "Engine not initialized"},
        )
    result = {}
    for name, adp in engine.adapters.items():
        cfg = adp.get("config", {})
        w = adp.get("weights", {})
        result[name] = {
            "adapter_id": cfg.get("adapter_id", ""),
            "format_version": cfg.get("format_version", "1.0"),
            "license": cfg.get("license", "unknown"),
            "rank": cfg.get("rank", 0),
            "alpha": cfg.get("alpha", 0),
            "always_active": adp.get("always_active", False),
            "gate_keywords": sorted(adp.get("gate_keywords", set())),
            "lora_a_shape": list(w["lora_A"].shape) if "lora_A" in w else None,
            "lora_b_shape": list(w["lora_B"].shape) if "lora_B" in w else None,
        }
    return {"adapters": result, "count": len(result)}


@app.post("/api/v1/adapters/{expert_name}/swap")
async def swap_adapter(expert_name: str, tgsp_file: UploadFile):
    """Hot-swap a TGSP adapter without restarting the engine.

    Upload a .tgsp file and the engine will parse, validate, and
    atomically replace the named adapter slot.
    """
    if not engine or not engine._initialized:
        return JSONResponse(
            status_code=503,
            content={"error": "Engine not initialized"},
        )
    try:
        tgsp_data = await tgsp_file.read()
        engine.swap_adapter(expert_name, tgsp_data)
        return {
            "status": "swapped",
            "expert": expert_name,
            "adapter_id": engine.adapters[expert_name]["config"].get(
                "adapter_id", ""
            ),
        }
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        logger.exception(f"Adapter swap failed for {expert_name}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/v1/marketplace")
async def marketplace_browse(
    domain: Optional[str] = None,
    tag: Optional[str] = None,
):
    """Browse the adapter marketplace — list available TGSP adapters."""
    if marketplace is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Marketplace not initialized"},
        )
    if domain or tag:
        tags = [tag] if tag else None
        results = marketplace.search(domain=domain, tags=tags)
        return {
            "adapters": [
                {
                    "adapter_id": a.adapter_id,
                    "name": a.name,
                    "domain": a.domain,
                    "description": a.description,
                    "license": a.license,
                    "price_per_1k_tokens": a.price_per_1k_tokens,
                    "tags": a.tags,
                }
                for a in results
            ],
            "count": len(results),
        }
    return marketplace.to_dict()


# ======================================================================
# Metrics
# ======================================================================

@app.get("/api/v1/metrics")
async def metrics():
    gpu = {}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_mem = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "mem_used_mb": round(torch.cuda.memory_allocated(0) / 1048576, 1),
            "mem_total_mb": round(total_mem / 1048576, 1),
        }

    # Privacy state
    dp_info = {}
    if engine and engine._privacy_tracker:
        pstate = engine._privacy_tracker.get_state("default")
        dp_info = {
            "dp_epsilon_per_request": engine._dp_epsilon,
            "dp_sigma": round(engine._dp_sigma, 4),
            "dp_total_epsilon_spent": round(pstate.total_epsilon, 4),
            "dp_total_requests": pstate.total_requests,
            "dp_max_epsilon": engine._max_epsilon,
            "dp_budget_remaining": round(
                engine._max_epsilon - pstate.total_epsilon, 4
            ),
        }

    return {
        "engine_ready": engine is not None and engine._initialized,
        "model": "Qwen/Qwen2.5-1.5B",
        "he_active": engine.he_ctx is not None if engine else False,
        "simd_slots": engine.simd_slots if engine else 0,
        "adapters": list(engine.adapters) if engine else [],
        "gpu": gpu,
        "device_profile": "phone",
        "gatelink": {
            "client_layers": 1,
            "dp_epsilon": engine._dp_epsilon if engine else 1.0,
            "max_lora_rank": 30,
            "split_forward_endpoint": "/api/v1/split/forward",
        },
        "differential_privacy": dp_info,
        "request_stats": _get_request_stats(),
    }


def _get_request_stats() -> dict:
    """Compute p50/p95 latency from recent requests."""
    latencies = _request_stats["latencies_ms"][-200:]
    n = len(latencies)
    if n == 0:
        return {"total_requests": 0, "error_count": 0, "p50_ms": 0, "p95_ms": 0}
    s = sorted(latencies)
    return {
        "total_requests": _request_stats["total"],
        "error_count": _request_stats["errors"],
        "p50_ms": round(s[n // 2], 1),
        "p95_ms": round(s[min(int(n * 0.95), n - 1)], 1),
    }


# ======================================================================
# Privacy budget reset
# ======================================================================

@app.post("/api/v1/privacy/reset")
async def reset_privacy_budget(request: Request):
    """Reset the DP privacy budget so the demo can continue."""
    client_ip = request.client.host if request.client else "unknown"
    if engine and engine._privacy_tracker:
        # Clear session state
        logger.warning(
            f"PRIVACY_RESET by {client_ip} — clearing all DP session budgets"
        )
        engine._privacy_tracker._states.clear()
        pstate = engine._privacy_tracker.get_state("default")
        return {
            "ok": True,
            "epsilon_spent": 0.0,
            "max_epsilon": engine._max_epsilon,
            "budget_remaining": engine._max_epsilon,
        }
    return {"ok": True, "message": "No privacy tracker active"}


# ======================================================================
# Static frontend (mounted last so API routes take precedence)
# ======================================================================

_FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
if _FRONTEND.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND), html=True), name="frontend")


# ======================================================================
# Dev entry
# ======================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8095")))
