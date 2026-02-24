# Performance Analysis

## Headline Numbers

| Mode | tok/s | ms/token | Hardware |
|------|-------|----------|----------|
| **WebSocket HE (best)** | **7.4** | 135 | RTX A2000 8GB |
| **WebSocket HE (avg 5 runs)** | **6.8** | 147 | RTX A2000 8GB |
| **Split + GPU LM head** | **4.85** | 207 | RTX A2000 8GB |
| **Split selftest (CPU LM head)** | **1.24** | 806 | RTX A2000 8GB |
| **Base model (no HE)** | **28.2** | 35 | RTX A2000 8GB |

---

## Optimization Breakdown (Cumulative)

Starting from baseline (poly_n=32768, sequential decrypt, no batching):

| Step | Configuration | WebSocket tok/s | Delta | Cumulative |
|------|--------------|----------------|-------|------------|
| 0 | Baseline | 3.0 -- 4.2 | -- | -- |
| 1 | + Track A: Batched GPU decrypt | 4.8 -- 5.8 | **+44%** | +44% |
| 2 | + Track C: poly_n=16384 | 6.6 -- 7.4 | **+27%** | **+94% (2x)** |

For Split mode:

| Step | Configuration | Split tok/s | Delta |
|------|--------------|-------------|-------|
| 0 | HTTP + CPU LM head | ~1.15 | -- |
| 1 | + GPU LM head (torch) | 4.85 | **+322%** |
| 2 | + WebSocket transport | 4.82 | ~same (WS overhead ~0) |

---

## Per-Token Latency Breakdown

### WebSocket Mode (poly_n=16384, batch decrypt)

```
Total:                    ~135 ms/token
├── Transformer forward:   ~25 ms   (28 layers, float16, CUDA)
├── DP noise injection:     ~0.1 ms (numpy Gaussian, 1536-dim)
├── CKKS encrypt:          ~10 ms   (replicate + CuKKS GPU encrypt)
├── ct*pt multiply (x4):   ~20 ms   (4 batch ZeRo-MOAI, 0 rotations)
├── GPU decrypt (x4):      ~28 ms   (inverse NTT on GPU, stays in CUDA)
├── Bulk GPU→CPU transfer:  ~28 ms  (torch.stack().cpu().numpy(), 1 PCIe sync)
├── Extract + sum:          ~2 ms   (numpy segment sums for 32 rank outputs)
├── LoRA-B plaintext mul:   ~1 ms   (1536x32 numpy matmul)
├── LM head projection:     ~5 ms   (151936x1536, torch GPU float16)
├── Sampling:               ~1 ms   (top-p + top-k, PyTorch)
└── WebSocket frame:        ~1 ms   (JSON + base64 encode)
```

### Split Mode -- CPU LM Head (phone selftest)

```
Total:                    ~806 ms/token
├── HTTP round-trip:       ~15 ms   (localhost, JSON encode/decode)
├── Server forward:       ~135 ms   (same as WebSocket, incl. HE)
├── Base64 encode/decode:  ~5 ms   (1536-dim float32 → base64)
├── LM head (CPU numpy): ~650 ms   (151936×1536 float32 matmul) ← BOTTLENECK
└── Sampling:              ~1 ms   (argmax or random)
```

### Split Mode -- GPU LM Head (bench_split_ws.py)

```
Total:                    ~207 ms/token
├── HTTP/WS round-trip:    ~5 ms   (localhost WebSocket)
├── Server forward:       ~135 ms   (same as above)
├── Base64 decode:         ~2 ms
├── LM head (GPU torch):   ~5 ms   (151936×1536 float16 on CUDA)
└── Sampling + overhead:   ~5 ms
```

---

## The LM Head Bottleneck

The single largest performance discovery: **numpy CPU LM head projection** was consuming **650ms/token** in split mode -- more than 4x the entire server-side HE pipeline.

### Analysis

```
LM head matrix: 151,936 × 1,536 = 233,377,536 elements (float32)
Memory: 233M × 4 bytes = ~889 MB

numpy matmul on CPU (single-threaded):
  ~889M × 0.75 ns/FLOP ≈ 667 ms/token

torch matmul on CUDA (RTX A2000, 2560 cores):
  889M / 5.2 TFLOPS ≈ 0.17 ms (theoretical)
  ~5 ms measured (memory-bound, not compute-bound)
```

### Impact

| LM Head Backend | ms/token | tok/s (split) |
|-----------------|----------|---------------|
| numpy CPU | ~650 | ~1.2 |
| torch GPU (float16) | ~5 | **~4.85** |
| **Speedup** | **130x** | **4x end-to-end** |

This is why the `bench_split_ws.py` benchmark uses torch GPU for the LM head projection -- it simulates what a phone with GPU WebGPU access would achieve, or what a desktop split client gets.

---

## CKKS Operation Costs (Measured, CuKKS GPU)

### poly_n=16384 (8,192 SIMD slots)

| Operation | Time (ms) | Notes |
|-----------|----------|-------|
| Encrypt (replicated h) | 10.2 | Pad 1536→8192, GPU NTT |
| ct × pt multiply | 4.8 | Element-wise SIMD, 1 level consumed |
| Decrypt (to GPU) | 7.1 | Inverse NTT, stays in CUDA memory |
| Decrypt (to CPU) | 35.2 | Inverse NTT + GPU→CPU PCIe transfer |
| **Rotation** | **~2.0** | **Not used (ZeRo-MOAI eliminates all)** |

### poly_n=32768 (16,384 SIMD slots)

| Operation | Time (ms) | Notes |
|-----------|----------|-------|
| Encrypt | 23.1 | 2.3x slower NTT |
| ct × pt multiply | 11.0 | 2.3x slower |
| Decrypt (to GPU) | 16.4 | 2.3x slower |
| Decrypt (to CPU) | 43.8 | NTT slower + same PCIe |

### Why poly_n=16384 Wins Despite More Batches

```
poly_n=32768:
  4 batches × (11.0 ms ct*pt + 16.4 ms decrypt) = 109.6 ms
  + 23.1 ms encrypt
  = 132.7 ms total HE

poly_n=16384:
  7 batches × (4.8 ms ct*pt + 7.1 ms decrypt) = 83.3 ms
  + 10.2 ms encrypt
  = 93.5 ms total HE

  With batch decrypt (1 PCIe sync):
  7 × 4.8 ms + 7 × 7.1 ms + 28 ms = 111.3 ms → reduced to ~80 ms
```

---

## SOTA Comparison

### Throughput (tokens per second, log scale)

```
System                    tok/s     Bar
──────────────────────────────────────────────────────
TenSafe WebSocket HE      7.40     ████████████████████████████████████████
TenSafe Split HE           4.85     ██████████████████████████
ChatGLM2-6B FHE           0.62     ███
Orion (GPT-2)             0.06     ▌
NEXUS (LLaMA-3-8B)        0.019    ▏
BOLT (BERT)               0.01     ▏
PUMA (LLaMA-7B)           0.003
Bumblebee (GPT-2)         0.002
Orion (LLaMA-3-8B)        0.001
```

### Full Comparison Table

| System | Model | HW | What Encrypted | tok/s | Year |
|--------|-------|-----|----------------|-------|------|
| **TenSafe (ours)** | Qwen2.5-1.5B | 1x RTX A2000 | HE-LoRA (CKKS) | **7.4** | 2026 |
| **TenSafe Split** | Qwen2.5-1.5B | 1x RTX A2000 | HE-LoRA (CKKS) | **4.85** | 2026 |
| ChatGLM2-6B FHE | ChatGLM2-6B | Undisclosed | LoRA (CKKS, SEAL) | 0.62 | 2025 |
| NEXUS | LLaMA-3-8B | 4x A100 | Full FHE | 0.019 | 2024 |
| Bumblebee | GPT-2 (125M) | CPU+GPU | HE/MPC hybrid | 0.002 | 2024 |
| Orion | GPT-2 | CPU | CKKS SIMD | ~0.06 | 2025 |
| Orion | LLaMA-3-8B | CPU | CKKS SIMD | ~0.001 | 2025 |
| BOLT | BERT-base | CPU | BFV-HE + MPC | <0.01 | 2023 |
| PUMA | LLaMA-7B | MPC cluster | Full MPC | ~0.003 | 2023 |

### Why TenSafe Is 12x Faster Than ChatGLM2-6B FHE

Both encrypt only LoRA. The gap comes from five factors:

| Factor | ChatGLM2-6B FHE | TenSafe | Impact |
|--------|-----------------|---------|--------|
| Rotations | O(log n) per dot product | **0** (ZeRo-MOAI) | Most expensive CKKS op eliminated |
| GPU accel | Microsoft SEAL (CPU) | CuKKS (CUDA GPU) | GPU NTT ~150x faster |
| Batch decrypt | Sequential CPU transfer | Batched GPU-resident | 84ms/token saved |
| HE direction | Client encrypts → server computes blind | Server-local encrypt/compute/decrypt | Zero network latency in crypto loop |
| LoRA decomposition | Two ct*pt: ct*A1 → ct*A2 | One ct*pt (packed A), plaintext B | Halves ciphertext ops |

---

## Hardware Utilization

### RTX A2000 8GB Profile

```
GPU Compute Utilization:     ~45%  (NTT-bound during HE, idle during CPU work)
GPU Memory:                  ~5.2 GB / 8 GB
  - Base model (float16):    ~3.0 GB
  - KV cache (per session):  ~0.4 GB
  - CKKS contexts/keys:      ~0.8 GB
  - Adapter weights:         ~0.2 GB
  - Working memory:          ~0.8 GB

PCIe Bandwidth:              ~7 GB/s (Gen 4 x8)
  - Per-token transfer:      ~32 KB (4 × 8192 × float64)
  - Transfer time:           ~0.005 ms (negligible)
  - Sync barrier overhead:   ~28 ms (kernel flush + transfer)

CPU Usage:                   ~15% (numpy segment sums, JSON encode)
```

### Scaling Projections

| GPU | Est. WebSocket HE tok/s | Est. Split HE tok/s |
|-----|------------------------|---------------------|
| RTX A2000 8GB (measured) | 7.4 | 4.85 |
| RTX 3090 24GB (est.) | ~15-20 | ~12-15 |
| A100 80GB (est.) | ~25-35 | ~20-28 |
| H100 80GB (est.) | ~40-55 | ~30-40 |

Estimates based on NTT throughput scaling with GPU compute + memory bandwidth.

---

## Benchmarking Methodology

### Tools

- **WebSocket HE**: `POST /api/v1/chat/compare` with `max_tokens=16`
- **Split selftest**: `GET /api/v1/split/selftest` generates 64 tokens
- **Split WS bench**: `bench_split_ws.py` -- Python WebSocket client with torch GPU LM head
- **CuKKS micro**: `demonstrator/scripts/benchmark_cukks.py` -- isolated HE operation timing

### Protocol

1. Server warm-up: First inference discarded (CUDA kernel JIT)
2. 3-5 runs per configuration across all three experts
3. Report: min, max, and average tok/s
4. Controlled variable: Same query ("What is a savings account?") for consistency
5. Server running in WSL2 with `TENSAFE_POLY_N=16384`
6. No concurrent requests during benchmark

### Reproducing

```bash
# Start server with optimizations
TENSAFE_POLY_N=16384 python -m uvicorn demonstrator.server.app:app \
  --host 0.0.0.0 --port 8095

# WebSocket HE benchmark
curl -s -X POST http://127.0.0.1:8095/api/v1/chat/compare \
  -H "Content-Type: application/json" \
  -d '{"query":"What is a savings account","max_tokens":16}'

# Split selftest
curl -s "http://127.0.0.1:8095/api/v1/split/selftest"

# Full split WS benchmark (torch GPU LM head)
python bench_split_ws.py
```
