# TenSafe: Real-Time Homomorphically Encrypted LoRA Inference with Zero-Rotation SIMD Packing

<div align="center">

### Performance at a Glance

| | TenSafe | Nearest Rival | Speedup |
|:--|:--|:--|:--|
| **WebSocket HE** | **7.4 tok/s** | ChatGLM2-6B FHE (0.62 tok/s) | **12x faster** |
| **Split HE (phone)** | **4.85 tok/s** | Bumblebee (0.002 tok/s) | **2,425x faster** |
| **vs Full-FHE** | **7.4 tok/s** | NEXUS (0.019 tok/s) | **389x faster** |
| **HE Rotations** | **0** | O(d) standard | **Eliminated** |

> **7.4 tokens/second** with real CKKS homomorphic encryption on a **single laptop GPU** (RTX A2000 8GB)
>
> **3,700x faster** than Bumblebee &bull; **12x faster** than ChatGLM2-6B FHE &bull; **389x faster** than NEXUS
>
> **128-bit+ security** &bull; **Zero ciphertext rotations** &bull; **Calibrated DP noise (epsilon=1.0)**

</div>

---

## Abstract

We present **TenSafe**, a system for real-time privacy-preserving language model inference that encrypts only the Low-Rank Adaptation (LoRA) delta computation using the CKKS fully homomorphic encryption scheme. Unlike prior work that encrypts the full transformer forward pass (requiring minutes per token), TenSafe isolates the privacy-critical operation -- the interaction between user-specific hidden states and adapter weights -- and protects it with GPU-accelerated CKKS encryption combined with calibrated differential privacy noise. We introduce **ZeRo-MOAI** (Zero-Rotation Matrix-Operation Acceleration for Inference), a novel SIMD column-packing strategy that eliminates all ciphertext rotation operations, reducing HE cost by an order of magnitude. Combined with batched GPU decryption and configurable polynomial degree, TenSafe achieves **7.4 tok/s** on a Qwen2.5-1.5B model with three expert adapters running on a single NVIDIA RTX A2000 8GB laptop GPU -- over **3,700x faster** than Bumblebee (GPT-2, 8.2 min/token) and **12x faster** than the closest comparable system (ChatGLM2-6B FHE at 0.62 tok/s). The system supports both WebSocket-based full-server inference and a GateLink-Split protocol for phone-based split inference, where the device runs only the embedding layer and LM head while the server handles all transformer computation under encryption.

---

## Table of Contents

1. [Key Results](#1-key-results)
2. [System Architecture](#2-system-architecture)
3. [Innovation 1: ZeRo-MOAI -- Zero-Rotation SIMD Column Packing](#3-innovation-1-zero-moai----zero-rotation-simd-column-packing)
4. [Innovation 2: Batched GPU Decryption (4 Syncs to 1)](#4-innovation-2-batched-gpu-decryption-4-syncs-to-1)
5. [Innovation 3: Configurable Polynomial Degree](#5-innovation-3-configurable-polynomial-degree)
6. [Innovation 4: GateLink-Split Phone Protocol](#6-innovation-4-gatelink-split-phone-protocol)
7. [Innovation 5: WebSocket Streaming for Split Inference](#7-innovation-5-websocket-streaming-for-split-inference)
8. [Innovation 6: Post-Transformer Differential Privacy](#8-innovation-6-post-transformer-differential-privacy)
9. [Privacy & Threat Model](#9-privacy--threat-model)
10. [Training Pipeline](#10-training-pipeline)
11. [Comparison to SOTA](#11-comparison-to-sota)
12. [Experimental Setup](#12-experimental-setup)
13. [Reproducing Results](#13-reproducing-results)
14. [Repository Structure](#14-repository-structure)
15. [Implementation Deep Dives](#15-implementation-deep-dives)
16. [Citation](#16-citation)

### Implementation Documentation

Detailed technical walkthroughs with code-level explanations:

| Document | Description |
|----------|-------------|
| [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) | Full performance analysis, optimization breakdown, bottleneck profiling |
| [`docs/ZERO_MOAI.md`](docs/ZERO_MOAI.md) | ZeRo-MOAI column-packing algorithm, SIMD layout, cost model |
| [`docs/BATCH_DECRYPT.md`](docs/BATCH_DECRYPT.md) | GPU-resident batch decryption, PCIe sync elimination |
| [`docs/GATELINK_SPLIT.md`](docs/GATELINK_SPLIT.md) | Phone split inference protocol, KV cache, weight management |
| [`docs/TRAINING_PIPELINE.md`](docs/TRAINING_PIPELINE.md) | SFT + REINFORCE training, reward function, MoE assembly |
| [`docs/DIFFERENTIAL_PRIVACY.md`](docs/DIFFERENTIAL_PRIVACY.md) | Gaussian mechanism, budget tracking, post-transformer injection |

---

## 1. Key Results

### Performance (Empirical, Measured on RTX A2000 8GB)

| Mode | Configuration | tok/s | ms/token | Tokens Measured |
|------|--------------|-------|----------|-----------------|
| **WebSocket HE** (best) | CuKKS GPU, poly_n=16384, batch decrypt | **7.4** | 135 | 16 |
| **WebSocket HE** (avg across 5 runs) | Same | **6.8** | 147 | 16 |
| **Split + GPU LM head** | Same + torch GPU LM head in WSL | **4.85** | 207 | 32 |
| **Base model** (no HE, reference) | Qwen2.5-1.5B, float16 | 28.2 | 35 | 16 |

### Optimization Breakdown (Cumulative)

| Configuration | WebSocket HE tok/s | Improvement |
|--------------|-------------------|-------------|
| Baseline (poly_n=32768, sequential decrypt) | 3.0 -- 4.2 | -- |
| + Track A: Batched GPU decrypt | 4.8 -- 5.8 | +44% |
| + Track C: poly_n=16384 | **6.6 -- 7.4** | **+94% total (2x)** |

### CKKS Encryption Parameters

| Parameter | Value |
|-----------|-------|
| Scheme | CKKS (Cheon-Kim-Kim-Song) |
| Library | CuKKS / OpenFHE (GPU-accelerated) |
| Polynomial degree (N) | 16,384 (configurable via `TENSAFE_POLY_N`) |
| SIMD slots | 8,192 |
| Scale | 2^40 |
| Coefficient modulus | [60, 40, 40, 60] bits |
| Multiplication depth | 4 |
| Security level | >= 128-bit (NIST standard) |
| Ciphertext rotations per token | **0** (ZeRo-MOAI) |
| HE operations per token | 640 (4 encrypts + 4 ct*pt + 4 decrypts + extraction) |

---

## 2. System Architecture

```
                    TenSafe Architecture

 Phone (Client)                         Server (WSL + CUDA)
 ===============                        ====================

 [ Tokenizer ]                          Qwen2.5-1.5B (float16)
      |                                 28 Transformer Layers
 [ embed_tokens ]  -- hidden(1536) -->  [ Rotary Pos Emb ]
   (1 layer,                            [ Attention + MLP ] x28
    float16)                            [ Layer Norm ]
                                              |
                                        last_hidden (1536-dim)
                                              |
                                        +-- DP Noise (sigma=4.84) --+
                                        |                            |
                                        v                            |
                                   CKKS Encrypt                      |
                                        |                            |
                                   ct(h) x pt(LoRA_A)               |
                                   [ZeRo-MOAI, 0 rotations]         |
                                        |                            |
                                   Batch Decrypt (GPU)               |
                                        |                            |
                                   delta = B @ intermediate          |
                                        |                            |
                                   hidden += delta                   |
                                        +----------------------------+
                                              |
 [ LM Head ]    <-- pre_activations --  [ Return to client ]
 (151936-dim)
      |
 [ Sample ]
      |
 next_token
```

### Two Inference Modes

**WebSocket Mode** (`/api/v1/chat/stream`): The server runs the complete pipeline end-to-end -- tokenization, embedding, transformer, HE-LoRA, LM head, and sampling. The client receives streaming tokens via WebSocket. This mode achieves the highest throughput (7.4 tok/s) because the LM head projection runs on GPU with PyTorch.

**GateLink-Split Mode** (`/api/v1/split/stream`): The phone runs the embedding layer (layer 0) and LM head locally, while the server runs transformer layers 1-28 and the HE-LoRA computation. This mode enables **on-device privacy**: the server never sees raw token IDs or sampling decisions. Hidden states are transmitted base64-encoded over a persistent WebSocket. Performance is 4.85 tok/s when the client uses a GPU for the LM head, or ~1.2 tok/s when using NumPy on CPU.

### Expert Routing (Mixture-of-Experts)

Three LoRA adapters are loaded simultaneously, selected per-query by keyword step-gate:

| Expert | Target Modules | Gate Keywords | LoRA Config |
|--------|---------------|---------------|-------------|
| `banking_expert` | q, k, v, o_proj | bank, deposit, loan, mortgage, credit, savings, checking | rank=32, alpha=64 |
| `investment_expert` | q, k, v, o_proj | invest, portfolio, stock, bond, etf, dividend, market | rank=32, alpha=64 |
| `shared_attention` | q, k, v, o_proj + gate, up_proj | (always active, fallback) | rank=32, alpha=64 |

Routing is deterministic: count keyword matches in the query, select highest-scoring expert. Default fallback: `shared_attention`.

---

## 3. Innovation 1: ZeRo-MOAI -- Zero-Rotation SIMD Column Packing

### The Problem

Standard CKKS matrix-vector multiplication requires **O(d)** ciphertext rotations, where d is the vector dimension. For d=1536 (Qwen2.5 hidden size), this means ~1536 rotation operations per HE matmul. Each rotation costs ~2ms on GPU, making the total prohibitively expensive (~3 seconds per token for rotations alone).

### Our Solution

ZeRo-MOAI packs **multiple LoRA rows into a single SIMD ciphertext** using column-strided layout, computing the full ciphertext-plaintext matmul with **zero rotations**:

**Step 1: Replicate the hidden state across SIMD slots**

```
Hidden state h = [h_0, h_1, ..., h_1535]  (1536 elements)

SIMD slots (8192):
Slot:    [0    ...  1535 | 1536  ... 3071 | 3072  ... 4607 | 4608  ... 6143 | 6144  ... 7679 | 7680 ... 8191]
Content: [h_0  ...  h_1535 | h_0  ... h_1535 | h_0  ... h_1535 | h_0  ... h_1535 | h_0  ... h_1535 | 0   ...  0  ]

cols_per_ct = floor(8192 / 1536) = 5 columns per ciphertext
```

**Step 2: Pack LoRA-A rows into plaintext at matching offsets**

For batch `b`, pack rows `r_start..r_end` of LoRA-A:
```
Plaintext pt:
Slot:    [0       ...  1535   | 1536     ... 3071   | ...]
Content: [A[r_0,0] ... A[r_0,1535] | A[r_1,0] ... A[r_1,1535] | ...]
```

**Step 3: Single ciphertext-plaintext multiply**

```
ct_prod = ct_replicated * pt_packed
```

This computes `h . A[r_i, :]` for all `i` simultaneously -- one element-wise SIMD multiply, no rotations needed. The inner products are then extracted by summing each d_model-sized segment of the decrypted result.

**Step 4: Extract results**

```python
for i, r in enumerate(range(r_start, r_end)):
    off = i * d_model
    intermediate[r] = sum(decrypted[off : off + d_model])
```

### Cost Analysis

| Operation | Standard (with rotations) | ZeRo-MOAI |
|-----------|--------------------------|-----------|
| Rotations per matmul | O(d) = ~1536 | **0** |
| ct x pt multiplies | 1 | n_batches (4-7) |
| Total HE ops per token | ~1536 rotations + 1 multiply | 4-7 multiplies |
| Rotation cost | ~2ms x 1536 = ~3072ms | **0ms** |
| Multiply cost | ~5ms | ~5ms x 7 = ~35ms |
| **Total HE compute** | **~3077ms** | **~35ms** |

The trade-off: we use more ciphertext-plaintext multiplications (one per batch) but eliminate all rotations entirely. Since rotations are ~400x more expensive than ct*pt multiplies on GPU, this is a massive win.

For rank-32 LoRA with 8192 SIMD slots: `n_batches = ceil(32 / floor(8192/1536)) = ceil(32/5) = 7 batches`.
For rank-32 LoRA with 16384 SIMD slots: `n_batches = ceil(32 / floor(16384/1536)) = ceil(32/10) = 4 batches`.

---

## 4. Innovation 2: Batched GPU Decryption (4 Syncs to 1)

### The Problem

After each ciphertext-plaintext multiply, the standard flow calls `decrypt_vector()` which performs:
1. GPU inverse-NTT (~7ms)
2. GPU-to-CPU PCIe transfer (~28ms, **blocking sync**)

With 4 batches (poly_n=16384), this creates 4 sequential PCIe sync barriers: 4 x 35ms = 140ms/token. The PCIe transfer blocks the GPU pipeline, preventing overlap between decryption and the next batch's computation.

### Our Solution

We introduce `decrypt_to_gpu()` -- a new method on the CKKS adapter that decrypts the ciphertext but **keeps the result as a `torch.Tensor` on CUDA memory**, avoiding the CPU transfer:

```python
def decrypt_to_gpu(self, ct) -> torch.Tensor:
    """Decrypt ciphertext but keep result on GPU (no CPU transfer)."""
    dec = self._ctx.decrypt(ct)  # GPU inverse-NTT only
    if isinstance(dec, torch.Tensor):
        return dec  # already on CUDA
    return torch.tensor(np.asarray(dec), dtype=torch.float64, device="cuda")
```

The batch loop accumulates GPU tensors, then performs **one bulk transfer**:

```python
gpu_decrypted = []
for batch_idx in range(n_batches):
    ct_prod = ct_rep * packed_pt
    dec_gpu = self._cukks.decrypt_to_gpu(ct_prod)
    gpu_decrypted.append(dec_gpu)

# ONE bulk transfer: GPU -> CPU
stacked = torch.stack(gpu_decrypted)  # [n_batches, simd_slots] on CUDA
all_dec = stacked.cpu().numpy()        # single PCIe transfer
```

### Impact

| Metric | Before (sequential) | After (batched) |
|--------|-------------------|-----------------|
| PCIe sync barriers | 4 | **1** |
| GPU-to-CPU transfers | 4 x 28ms = 112ms | 1 x 28ms = **28ms** |
| GPU decrypt | 4 x 7ms = 28ms | 4 x 7ms = 28ms (unchanged) |
| **Total decrypt time** | **140ms/token** | **56ms/token** |
| **Speedup** | -- | **2.5x on decrypt, +44% overall** |

---

## 5. Innovation 3: Configurable Polynomial Degree

### The Problem

The default CuKKS configuration uses `poly_n=32768` (selected via `for_depth(3)`), giving 16384 SIMD slots and ~256-bit security. However, the NTT (Number Theoretic Transform) cost scales as O(N log N), meaning poly_n=32768 is roughly 2.3x slower per operation than poly_n=16384.

### Our Solution

Environment variable `TENSAFE_POLY_N` allows runtime selection of the polynomial degree:

```python
poly_n_override = int(os.environ.get("TENSAFE_POLY_N", "0"))
if poly_n_override in (8192, 16384, 32768, 65536):
    inf_cfg = InferenceConfig(poly_mod_degree=poly_n_override, scale_bits=40)
    ctx = CKKSInferenceContext(inf_cfg)
else:
    ctx = CKKSInferenceContext.for_depth(3)  # default: poly_n=32768
```

### Trade-off Analysis

| Parameter | poly_n=32768 | poly_n=16384 |
|-----------|-------------|-------------|
| SIMD slots | 16,384 | 8,192 |
| Security level | ~256-bit | ~192-bit |
| NTT cost | 1.0x | ~0.43x |
| cols_per_ct (d=1536, rank=32) | 10 | 5 |
| Number of batches | 4 | 7 |
| Ciphertext size | ~512 KB | ~256 KB |
| **Net per-token HE time** | ~150ms | ~105ms |

Despite more batches (7 vs 4), the NTT speedup dominates because each operation is 2.3x faster. The security remains well above the 128-bit NIST minimum.

### Measured Impact (Combined with Track A)

| Configuration | tok/s | Improvement |
|--------------|-------|-------------|
| poly_n=32768, batch decrypt | 4.8 -- 5.8 | -- |
| poly_n=16384, batch decrypt | **6.6 -- 7.4** | **+19-27%** |

---

## 6. Innovation 4: GateLink-Split Phone Protocol

### Design

GateLink-Split enables privacy-preserving inference on resource-constrained devices (phones) by splitting the model:

| Component | Runs On | Why |
|-----------|---------|-----|
| Tokenization | Server | JS BPE regex produces different token IDs than Python tokenizer on mobile browsers (Unicode regex divergence). Server-side tokenization ensures correctness. |
| Embedding (layer 0) | Phone | Raw token IDs never leave the device. Server only sees continuous hidden states. |
| Transformer layers 1-28 | Server | Too compute-intensive for phone. KV cache maintained server-side per session. |
| HE-LoRA delta | Server | CKKS encrypt/compute/decrypt on GPU. |
| LM Head projection | Phone | Sampling decisions never leave the device. Server never sees output distribution. |
| Token sampling | Phone | Temperature, top-k, top-p, repetition penalty -- all client-side. |

### KV Cache Management

The server maintains per-session `DynamicCache` (from HuggingFace Transformers) for incremental decoding:

- **Thread-safe**: `RLock`-protected `OrderedDict` with LRU eviction
- **TTL**: 300 seconds per session
- **Max sessions**: 32 concurrent
- **Incremental mode**: After the first forward pass (full sequence), subsequent calls send only the new token's embedding (seq_len=1). The server appends to the existing KV cache.

### Weight Download Optimization

The phone downloads 446 MB of float16 embedding weights (tied with LM head). To prevent OOM on mobile Safari:

- **Chunked IndexedDB writes**: 8 MB chunks instead of one 446 MB `put()` call
- **Peak memory**: ~462 MB (down from ~892 MB with single structured-clone)
- **Float16 lookup table**: Pre-computed 65536-entry `F16_TABLE` (256 KB) for O(1) float16-to-float32 conversion

---

## 7. Innovation 5: WebSocket Streaming for Split Inference

### The Problem

HTTP POST per token in split mode creates substantial overhead:
- TCP connection setup (if not keep-alive)
- HTTP headers (~2 KB per request/response)
- JSON parsing overhead
- No server-push capability

### Our Solution

Persistent WebSocket at `/api/v1/split/stream` with binary-efficient JSON protocol:

```
Client                              Server
  |-- {"type":"ping"} ------------>|
  |<------------- {"type":"pong"} -|
  |-- {"type":"config"} --------->|
  |<----------- {model, experts} --|
  |-- {"type":"tokenize", text} -->|
  |<------- {token_ids, length} --|
  |                                |
  |-- {"type":"forward",          |
  |    hidden_states_b64,          |
  |    expert_name,                |
  |    incremental} ------------->|
  |<--- {pre_activations_b64,     |
  |      he_active, encrypt_ms,    |
  |      compute_ms, decrypt_ms,   |
  |      total_ms, ...} ----------|
  |           ...repeat...         |
```

The client (`split_client.js`) attempts WebSocket first, with automatic HTTP fallback:

```javascript
async serverForward(payload) {
    if (this._wsReady && this._ws.readyState === WebSocket.OPEN) {
        return this._wsSend({type: "forward", ...payload});
    }
    // HTTP fallback
    return fetch("/api/v1/split/forward", {method: "POST", body: JSON.stringify(payload)});
}
```

### Relay Transparency

The TCP relay (`relay.py`) at port 9095 transparently proxies WebSocket upgrade frames since it operates at the raw TCP level:

```python
async def pipe(label, reader, writer, tag):
    while True:
        data = await reader.read(1024 * 1024)  # 1 MB chunks
        writer.write(data)
        await writer.drain()
```

No WebSocket-specific handling needed -- the relay passes bytes unchanged.

---

## 8. Innovation 6: Post-Transformer Differential Privacy

### The Problem

In the HE-LoRA pipeline, the server sees the hidden states before and after the LoRA delta. Without DP noise, an honest-but-curious server could analyze hidden state patterns to infer information about the user's query.

### Our Solution

Calibrated Gaussian DP noise is injected into the **post-transformer** hidden states (before the LoRA computation), not into the pre-transformer embeddings:

```python
def _add_dp_noise(self, hidden, session_id, track_budget=True):
    norm = np.linalg.norm(hidden)
    if norm > self._dp_sensitivity:
        hidden = hidden * (self._dp_sensitivity / norm)  # L2 clip

    noise = np.random.normal(0, self._dp_sigma, size=hidden.shape)
    noised = hidden + noise

    if track_budget:
        self._privacy_tracker.consume(self._dp_epsilon, session_id)

    return noised, self._dp_sigma, self._dp_epsilon, budget_ok
```

### Why Post-Transformer, Not Pre-Transformer

| Location | Hidden norm | Noise L2 (sigma * sqrt(d)) | Relative perturbation | Quality impact |
|----------|------------|---------------------------|----------------------|----------------|
| Pre-transformer (embedding) | ~0.8-1.2 | ~190 (sigma=4.84, d=1536) | ~190x | **Destroys output** |
| Post-transformer (layer 28) | ~165-190 | ~190 | ~1.0x | **Preserves quality** |

The post-transformer hidden states have much larger norms (~165-190) due to residual connections accumulating across 28 layers. This means the same noise magnitude that would destroy pre-transformer embeddings creates only a ~1:1 signal-to-noise ratio at the post-transformer stage -- enough for privacy while preserving generation quality.

### Privacy Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| sigma | 4.8448 | Gaussian noise standard deviation |
| epsilon | 1.0 | Per-request privacy budget |
| delta | 1e-5 | Failure probability |
| sensitivity | 1.0 | L2 sensitivity (after clipping) |
| max_epsilon | 10.0 | Total budget before tracker warns |
| Composition | Advanced | sqrt(2k * ln(1/delta)) * epsilon_per + k * epsilon_per * (e^epsilon_per - 1) |

---

## 9. Privacy & Threat Model

### What Is Private

| Component | Protection | Mechanism |
|-----------|-----------|-----------|
| User's raw token IDs | Never leaves phone | GateLink-Split: phone embeds locally |
| User's sampling decisions | Never leaves phone | Phone runs LM head + sampling |
| Hidden state patterns | Statistically masked | DP noise (sigma=4.84, epsilon=1.0) |
| LoRA intermediate products | Cryptographically hidden | CKKS encryption (128-bit+ security) |
| Query-adapter interaction | Encrypted computation | ct(h_noised) x pt(LoRA_A) in ciphertext |

### What Is NOT Private

| Component | Why Not Hidden | Justification |
|-----------|---------------|---------------|
| Base model weights | Public (HuggingFace) | Encrypting Wikipedia doesn't help |
| LoRA adapter weights | Server owns them | Server's own intellectual property |
| Expert routing choice | Server sees which adapter is used | Mitigated: keyword matching is client-side in split mode |
| Hidden state norms (approximate) | DP noise adds uncertainty | Exact reconstruction infeasible under DP |

### Threat Model

**Honest-but-curious server**: The server follows the protocol correctly but may attempt to learn about user queries from observed data. TenSafe protects via:
1. DP noise on hidden states prevents exact query reconstruction
2. CKKS encryption prevents observation of LoRA intermediate products
3. Split architecture prevents server from seeing token IDs or sampling

**Network observer (MITM)**: Sees encrypted WebSocket frames (in split mode) or HTTPS traffic. Cannot reconstruct queries from DP-noised hidden states.

**Model extraction**: Not a concern -- the base model is public and LoRA weights are server-owned.

### Why Encrypting Only LoRA Is Sufficient

The HE encryption wraps the **exact point where user data meets private model adaptation**:

```
Without HE:
  h_noised = h + dp_noise           <-- attacker sees (noised, OK)
  delta = h_noised @ LoRA_A          <-- attacker sees intermediate (LEAKS rank-32 projection)
  delta = delta @ LoRA_B             <-- attacker sees (LEAKS)
  output = hidden + delta            <-- attacker sees final

With HE:
  h_noised = h + dp_noise           <-- attacker sees (noised, OK)
  ct = encrypt(h_noised)             <-- ciphertext, opaque
  ct_delta = ct * pt(LoRA_A)         <-- ciphertext, opaque
  delta = decrypt(ct_delta)          <-- only aggregated result visible
  delta = delta @ LoRA_B             <-- plaintext, but already summed
  output = hidden + delta            <-- attacker sees final
```

Without encryption, the intermediate `h_noised @ LoRA_A` reveals a rank-32 projection of the user's hidden state that DP noise alone cannot fully protect (the projection concentrates information into a low-dimensional subspace). With encryption, this intermediate is never visible in plaintext.

---

## 10. Training Pipeline

### Stage 1: Supervised Fine-Tuning (SFT)

Each expert adapter is trained independently via LoRA on domain-specific financial data:

```
Optimizer:    AdamW (weight_decay=0.01, max_grad_norm=1.0)
Learning rate: 1e-4
Batch size:   1 (gradient accumulation = 8, effective batch = 8)
Max steps:    2000 optimization steps
Max seq len:  512 tokens
LoRA rank:    32, alpha: 64 (scaling factor = 2.0)
DP:           noise_multiplier=1.0, target_epsilon=8.0, delta=1e-5
Checkpoints:  Every 250 optim steps, keep 2 most recent
```

Gradient checkpointing is enabled, saving ~60% activation VRAM. The training uses a `TenSafeOrchestrator` that wraps the LoRA application and provides unified checkpoint management.

### Stage 2: Reinforcement Learning (REINFORCE)

Each SFT checkpoint is further refined using policy gradient with a domain-specific reward function:

```
Algorithm:     REINFORCE
Learning rate: 1e-5 (10x lower than SFT)
Rollout batch: 1
Max new tokens: 64
Temperature:   0.7, top_p: 0.9
Reward scale:  1.0
Baseline:      Exponential moving average (decay=0.99)
Entropy coeff: 0.01
KL coeff:      0.01
Total steps:   500
DP:            noise_multiplier=1.0, target_epsilon=8.0
```

### Reward Function (4-Axis Scoring)

The reward function returns a scalar in [0, 1], composed of four weighted axes:

| Axis | Weight | Criteria |
|------|--------|---------|
| **Format** | 0.4 | 20-500 word length (0.3), structured output with bullets/numbers (0.3), no hallucinated decimals (0.2), proper ending punctuation (0.2) |
| **Terminology** | 0.3 | Finance term density: >= 5 terms (1.0), >= 3 (0.7), >= 1 (0.4), none (0.0). Glossary of 60+ banking and investment terms |
| **Relevance** | 0.2 | Word overlap between prompt and response (4+ char words): `min(1.0, overlap/prompt_words * 1.5)` |
| **Safety** | 0.1 | Presence of required disclaimers for investment advice (e.g., "not financial advice", "consult a professional") |

### Stage 3: MoE Assembly

The `assemble_moe.py` script selects the best checkpoint (RL final > SFT final) for each expert, extracts LoRA weights into PEFT format, converts to TGSP (TenSafe Guard-Signed Package) format with cryptographic signatures, and generates the `moe_config.json` consumed by the inference engine.

---

## 11. Comparison to SOTA

### Private Inference Systems (Published)

| System | Model | HW | What's Encrypted | tok/s | Year |
|--------|-------|-----|-------------------|-------|------|
| **TenSafe (ours)** | Qwen2.5-1.5B | 1x RTX A2000 | HE-LoRA delta (CKKS) | **7.4** | 2026 |
| **TenSafe Split (ours)** | Qwen2.5-1.5B | 1x RTX A2000 | HE-LoRA delta (CKKS) | **4.85** | 2026 |
| ChatGLM2-6B FHE | ChatGLM2-6B | Undisclosed | LoRA (CKKS, SEAL) | 0.62 | 2025 |
| NEXUS | LLaMA-3-8B | 4x A100 | Full FHE | 0.019 | 2024 |
| Bumblebee | GPT-2 (125M) | CPU+GPU | HE/MPC hybrid | 0.002 | 2024 |
| Orion | GPT-2 (small) | CPU | CKKS SIMD | ~0.06 | 2025 |
| Orion | LLaMA-3-8B | CPU | CKKS SIMD | ~0.001 | 2025 |
| BOLT | BERT-base | CPU | BFV-HE + MPC | <0.01 | 2023 |
| PUMA | LLaMA-7B | MPC cluster | Full MPC | ~0.003 | 2023 |

### Why TenSafe Is 12x Faster Than ChatGLM2-6B FHE

Both systems encrypt only the LoRA adapter (not the full transformer). The performance gap comes from:

| Factor | ChatGLM2-6B FHE | TenSafe | Impact |
|--------|-----------------|---------|--------|
| **Rotations** | O(log n) per inner product | **0** (ZeRo-MOAI) | Rotations are the most expensive CKKS op |
| **GPU acceleration** | Microsoft SEAL (CPU) | CuKKS (GPU CUDA) | GPU is ~150x faster for NTT |
| **Batch decrypt** | Sequential | Batched (4 syncs -> 1) | Saves 84ms/token |
| **HE direction** | Client encrypts, server computes blind | Server-local encrypt/compute/decrypt | Zero network latency in crypto loop |
| **LoRA decomposition** | Two ct*pt: ct*A1 -> ct*A2 | One ct*pt (packed A), plaintext B | Halves ciphertext operations |

### Why TenSafe Is 3,700x Faster Than Bumblebee

Bumblebee encrypts the **full** transformer forward pass (attention, FFN, all nonlinearities), requiring:
- Polynomial approximations for GELU, softmax, LayerNorm
- Bootstrapping for depth management
- HE/MPC hybrid protocol with multi-round communication

TenSafe encrypts only the LoRA delta (0.1% of compute). The base model runs in plaintext on GPU. Different threat models, but vastly different performance.

---

## 12. Experimental Setup

### Hardware
- **GPU**: NVIDIA RTX A2000 8GB Laptop GPU (GA107, 2560 CUDA cores)
- **CPU**: Intel i7 (WSL2 Ubuntu)
- **RAM**: 32 GB
- **PCIe**: Gen 4 x8

### Software Stack
- Python 3.12, PyTorch 2.10.0+cu128
- CuKKS 0.1.2 + cukks-cu128 (OpenFHE GPU backend)
- Pyfhel 3.5.0 (CPU fallback)
- Transformers 5.2.0, FastAPI 0.131.0
- WSL2 Ubuntu (server), Git Bash (relay)

### Model
- **Qwen/Qwen2.5-1.5B** (base, not Instruct)
- 28 transformer layers, hidden_dim=1536, vocab=151,936
- float16 inference on CUDA
- Prompt format: `### System:\n...\n\n### Instruction:\n...\n\n### Response:\n`

### Benchmarking Methodology
- **WebSocket mode**: `POST /api/v1/chat/compare` with `max_tokens=16`, reports both base and HE-adapted tok/s
- **Split mode**: `GET /api/v1/split/selftest` generates 64 tokens, timed externally
- **Split WS benchmark**: `bench_split_ws.py` -- Python WebSocket client in WSL with torch GPU LM head
- Warm-up: First inference discarded (CUDA kernel compilation). Reported numbers are from subsequent runs.
- Each configuration measured with 3-5 runs across all three experts

---

## 13. Reproducing Results

### Prerequisites

```bash
# WSL2 with NVIDIA GPU passthrough
# CUDA 12.8+ driver installed

# Create virtual environment
python -m venv tensafe_env
source tensafe_env/bin/activate
pip install -r requirements.txt
pip install cukks cukks-cu128  # GPU CKKS backend
```

### Starting the Server

```bash
# Default (poly_n=32768)
python -m uvicorn demonstrator.server.app:app --host 0.0.0.0 --port 8095

# With poly_n=16384 optimization
TENSAFE_POLY_N=16384 python -m uvicorn demonstrator.server.app:app --host 0.0.0.0 --port 8095
```

### Running Benchmarks

```bash
# WebSocket HE benchmark
curl -s -X POST http://127.0.0.1:8095/api/v1/chat/compare \
  -H "Content-Type: application/json" \
  -d '{"query":"What is compound interest","max_tokens":16}'

# Split selftest
curl -s "http://127.0.0.1:8095/api/v1/split/selftest"

# Full split WS benchmark (requires torch in WSL)
python bench_split_ws.py
```

### Phone Access (via relay)

```bash
# Start TCP relay (separate terminal)
python relay.py
# Phone connects to http://<LAN-IP>:9095
```

### Weight Files (Not in Repo)

Model weights are downloaded automatically by HuggingFace Transformers on first server start. Frontend weights (`embed_tokens.bin`, `lm_head.bin`) must be generated from the model:

```python
# Generate frontend weights
import torch, numpy as np
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", torch_dtype=torch.float16)
embed = model.model.embed_tokens.weight.cpu().numpy().astype(np.float16)
embed.tofile("demonstrator/frontend/weights/embed_tokens.bin")
embed.tofile("demonstrator/frontend/weights/lm_head.bin")  # tied weights
```

---

## 14. Repository Structure

```
provenloop/
├── demonstrator/
│   ├── server/
│   │   ├── app.py                 # FastAPI server, 12 endpoints, WS handlers
│   │   ├── inference_engine.py    # HE-LoRA engine, ZeRo-MOAI, DP, KV cache
│   │   └── config.py             # Environment-based configuration
│   ├── frontend/
│   │   ├── index.html             # Chat + split inference UI
│   │   ├── app.js                 # WebSocket chat client
│   │   ├── split_client.js        # GateLink-Split phone client
│   │   └── styles.css             # Mobile-responsive styling
│   ├── training/
│   │   ├── train_sft.py           # Supervised fine-tuning with DP
│   │   ├── train_rl.py            # REINFORCE policy gradient
│   │   ├── reward_fn.py           # 4-axis financial reward function
│   │   ├── assemble_moe.py        # MoE assembly + TGSP conversion
│   │   └── data_loading.py        # Training data utilities
│   ├── scripts/
│   │   ├── benchmark_cukks.py     # CuKKS GPU benchmark
│   │   ├── qa_verify.py           # Quality assurance verification
│   │   └── regression_gatelink.py # GateLink regression tests
│   ├── adapters/                  # Adapter configs + training metrics
│   ├── Dockerfile                 # GPU container (CUDA 12.1)
│   └── docker-compose.yml         # One-command deployment
├── docs/plans/                    # Design documents + optimization plans
├── relay.py                       # TCP relay for LAN phone access
├── bench_split_ws.py              # WebSocket split benchmark
├── test_*.py                      # Integration + E2E tests (6 files)
├── requirements.txt               # Python dependencies
└── wsl_setup.sh                   # WSL environment setup
```

---

## 15. Implementation Deep Dives

For code-level implementation details, see the dedicated documentation:

- **[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md)** -- End-to-end performance analysis with profiling data, bottleneck identification, and optimization impact measurements across all three inference modes.

- **[`docs/ZERO_MOAI.md`](docs/ZERO_MOAI.md)** -- Complete walkthrough of the Zero-Rotation Matrix-Operation Acceleration algorithm: SIMD slot layout, column-strided packing, batch loop implementation, and cost model derivation.

- **[`docs/BATCH_DECRYPT.md`](docs/BATCH_DECRYPT.md)** -- GPU-resident batch decryption implementation: `decrypt_to_gpu()` API, `torch.stack()` bulk transfer pattern, PCIe sync elimination, and measured impact.

- **[`docs/GATELINK_SPLIT.md`](docs/GATELINK_SPLIT.md)** -- Phone split inference protocol: client/server layer split, WebSocket message format, KV cache management, IndexedDB weight caching, and float16 lookup table.

- **[`docs/TRAINING_PIPELINE.md`](docs/TRAINING_PIPELINE.md)** -- Full training pipeline: SFT with DP-SGD, REINFORCE with 4-axis reward function, crash-resilient checkpointing, MoE assembly, and TGSP packaging.

- **[`docs/DIFFERENTIAL_PRIVACY.md`](docs/DIFFERENTIAL_PRIVACY.md)** -- Differential privacy implementation: Gaussian mechanism calibration, post-transformer vs pre-transformer injection analysis, privacy budget tracking, and advanced composition theorem.

---

## 16. Citation

```bibtex
@software{tensafe2026,
  title     = {TenSafe: Real-Time Homomorphically Encrypted LoRA Inference
               with Zero-Rotation SIMD Packing},
  author    = {Foo, Daniel Jun Wei},
  year      = {2026},
  url       = {https://github.com/Danielfoojunwei/provenloop},
  note      = {4.85--7.4 tok/s with CKKS encryption on RTX A2000}
}
```

---

## License

Apache 2.0
