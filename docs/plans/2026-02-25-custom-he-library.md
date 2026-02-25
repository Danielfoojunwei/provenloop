# Custom HE Library: TenSafe-HE

## Why Build Our Own

TenSafe uses **~15% of what general CKKS libraries implement**. ZeRo-MOAI eliminates rotations (the most expensive operation), and we never do ct×ct multiply or bootstrapping. Every existing library — OpenFHE, SEAL, Pyfhel, Lattigo — carries the weight of features we don't use. That abstraction overhead costs real milliseconds per token.

### What We Use vs What Libraries Ship

| Operation | TenSafe Needs | OpenFHE Ships | SEAL Ships |
|-----------|:---:|:---:|:---:|
| CKKS encode (float64 → polynomial) | Yes | Yes | Yes |
| CKKS decode (polynomial → float64) | Yes | Yes | Yes |
| Encrypt (RLWE sampling) | Yes | Yes | Yes |
| Decrypt | Yes | Yes | Yes |
| ct × pt multiply (element-wise SIMD) | Yes | Yes | Yes |
| NTT / iNTT | Yes | Yes | Yes |
| **Rotation (Galois automorphism)** | **No** | Yes | Yes |
| **ct × ct multiply** | **No** | Yes | Yes |
| **Relinearization** | **No** | Yes | Yes |
| **Key switching** | **No** | Yes | Yes |
| **Bootstrapping** | **No** | Yes | Yes |
| **Galois key generation** | **No** | Yes | Yes |
| BFV scheme | No | Yes | Yes |
| BGV scheme | No | Yes | No |
| TFHE/FHEW scheme | No | Yes | No |
| Multi-party HE | No | Yes | No |
| Serialization (full) | Minimal | Yes | Yes |

**We need 6 operations. Libraries ship 15+.** Every unnecessary abstraction layer adds indirection, cache misses, and memory allocations.

### Performance Opportunity

Current CuKKS (wrapping OpenFHE) per-token HE breakdown on RTX A2000:

```
CKKS encrypt:          ~10 ms  (encode + NTT + RLWE sample)
ct×pt multiply ×7:     ~20 ms  (7 batches, polynomial multiply in NTT domain)
GPU decrypt ×7:        ~28 ms  (7× inverse NTT + decode)
Bulk GPU→CPU transfer: ~28 ms  (PCIe sync)
────────────────────────────────
HE total:              ~86 ms  (63.7% of per-token latency)
```

**Estimated savings with a custom fused library:**

| Optimization | Estimated Savings | How |
|---|---|---|
| Fused encode+encrypt kernel | -3 ms | Eliminate intermediate polynomial allocation |
| Fused decrypt+decode kernel | -5 ms | Skip intermediate buffer, decode directly |
| Specialized NTT (fixed poly_n) | -4 ms | No branching for poly_n, unrolled butterfly |
| Eliminate abstraction dispatch | -2 ms | No virtual calls, no runtime scheme checks |
| Pinned memory for PCIe transfer | -8 ms | cudaHostAlloc pinned buffers |
| Fused batch ct×pt pipeline | -3 ms | Stream-overlap multiply+decrypt across batches |
| **Total estimated savings** | **~25 ms** | **86 → ~61 ms (29% faster HE pipeline)** |

On A2000: **135 → ~110 ms/token → 9.1 tok/s** (up from 7.4)
On H100 (6x scaling): **~14 → ~10 ms HE → ~13 ms total → 77 tok/s** (up from 37)

---

## Existing Library Landscape

### Libraries with GPU CKKS Support

| Library | Language | GPU | CKKS | License | Status | Notes |
|---------|----------|:---:|:---:|---------|--------|-------|
| **OpenFHE** | C++ | No (CPU HAL) | Yes | BSD-2 | Active (v1.4.2, Oct 2025) | Our current backend via CuKKS wrapper. CPU-only; HAL makes GPU integration hard |
| **FIDESlib** | C++/CUDA | **Yes** | Yes | Apache-2.0 | New (Jul 2025) | First open-source GPU CKKS with bootstrapping. OpenFHE interop. **Best candidate to learn from** |
| **HEonGPU** | C++/CUDA | **Yes** | Yes | MIT | Active (2024) | GPU-native BFV+CKKS. Object-oriented C++ API |
| **Phantom-FHE** | C++/CUDA | **Yes** | Yes | BSD-3 | Active | GPU-accelerated CKKS, BFV, BGV |
| **Microsoft SEAL** | C++ | No (Intel HEXL AVX) | Yes | MIT | Active (v4.1) | Most widely used. CPU-only, Intel HEXL NTT acceleration |
| **TenSEAL** | Python/C++ | No | Yes | Apache-2.0 | Maintenance | Python wrapper for SEAL. No GPU |
| **Lattigo** | Go | No | Yes | Apache-2.0 | Active (v6.x) | Pure Go. Good for distributed/WASM. No GPU |
| **HElib** | C++ | No | Yes (limited) | Apache-2.0 | Slow updates | IBM. BGV-focused, CKKS basic |
| **HEAAN** | C++ | No | Yes | MIT-like | Research | Original CKKS authors. Reference implementation |

### Libraries Without CKKS (Not Suitable)

| Library | Scheme | Why Not |
|---------|--------|---------|
| TFHE-rs (Zama) | TFHE | Exact integer/boolean only, no approximate float arithmetic |
| Concrete (Zama) | TFHE | TFHE compiler, no CKKS |
| cuFHE | TFHE | CUDA TFHE, no CKKS |
| cuHE | BFV only | No CKKS |

### GPU NTT Implementations (Reference for Custom Kernels)

| Source | Language | Performance | Notes |
|--------|----------|------------|-------|
| **FIDESlib** NTT kernels | CUDA | 70x over AVX-OpenFHE | Best open-source GPU CKKS, OpenFHE interop (Apache-2.0) |
| **HEonGPU** NTT | CUDA | 380x over SEAL, bootstrap <170ms | Multi-stream CUDA, BFV+CKKS (MIT) |
| **Phantom-FHE** NTT | CUDA | 380x over SEAL | Kernel fusion, hybrid key-switch (GPLv3) |
| **Cheddar** | CUDA | 2.9-25.6x over prior GPU | 32-bit RNS for GPU-native datapath — **key for Groq-native NTT research** |
| **Neo** (ISCA 2025) | CUDA | 3.28x over TensorFHE | Uses Tensor Cores for FHE — relevant if Groq adds tensor ops |
| **HI-CKKS** | CUDA | 692 kop/s NTT on 4090 | High-throughput batch NTT optimization |
| Troy | CUDA | — | GPU BFV/CKKS/BGV implementation |

---

## Architecture: TenSafe-HE

### Design Principles

1. **CKKS-only.** No BFV, BGV, TFHE. Zero scheme-selection overhead.
2. **ZeRo-MOAI-native.** No rotation support. No Galois keys. No key switching.
3. **Groq-first.** Optimized for Groq LPU + GPU co-processor architecture. Minimize Groq↔GPU data transfer.
4. **GPU-accelerated HE.** All polynomial operations in CUDA (H100/B200). CPU path for testing only.
5. **Fused kernels.** Encode+encrypt in one kernel. Decrypt+decode in one kernel.
6. **Fixed polynomial degrees.** Specialize NTT for poly_n ∈ {8192, 16384, 32768}. No generic N.
7. **Rust core + CUDA kernels.** Memory safety for parameter management, raw CUDA for compute.
8. **Python bindings.** PyO3 for seamless integration with existing TenSafe inference engine.
9. **Zero-copy GPU interop.** Direct torch.Tensor ↔ ciphertext via DLPack/CUDA pointers.
10. **Groq data path optimization.** Compact serialization for Groq LPU → GPU hidden state transfer.

### Module Layout

```
tensafe-he/
├── Cargo.toml                    # Rust workspace
├── crates/
│   ├── tensafe-he-core/          # Pure Rust: parameters, RNS, error sampling
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── params.rs         # CKKS parameter sets (fixed poly_n options)
│   │   │   ├── rns.rs            # Residue Number System arithmetic
│   │   │   ├── sampling.rs       # RLWE error sampling (discrete Gaussian)
│   │   │   ├── encoding.rs       # CKKS encode/decode (FFT-based)
│   │   │   └── ciphertext.rs     # Ciphertext struct (two polynomials)
│   │   └── Cargo.toml
│   │
│   ├── tensafe-he-cuda/          # CUDA kernels (NTT, multiply, fused ops)
│   │   ├── src/
│   │   │   ├── lib.rs            # Rust FFI to CUDA
│   │   │   └── bindings.rs       # cuda_ntt(), cuda_encrypt(), etc.
│   │   ├── kernels/
│   │   │   ├── ntt.cu            # Forward/inverse NTT (butterfly, fixed N)
│   │   │   ├── encrypt.cu        # Fused encode+encrypt kernel
│   │   │   ├── decrypt.cu        # Fused decrypt+decode kernel
│   │   │   ├── ct_pt_mul.cu      # Ciphertext × plaintext multiply
│   │   │   └── batch_pipeline.cu # Fused batch ct×pt + decrypt pipeline
│   │   └── Cargo.toml
│   │
│   ├── tensafe-he-cpu/           # CPU fallback (for testing, no CUDA needed)
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── ntt_cpu.rs        # CPU NTT (AVX2/AVX-512 intrinsics)
│   │   │   └── ops_cpu.rs        # CPU encrypt/decrypt/multiply
│   │   └── Cargo.toml
│   │
│   └── tensafe-he-python/        # Python bindings (PyO3)
│       ├── src/
│       │   ├── lib.rs            # PyO3 module definition
│       │   ├── context.rs        # TenSafeHEContext Python class
│       │   ├── ciphertext.rs     # Ciphertext Python wrapper (__mul__ etc.)
│       │   └── torch_interop.rs  # torch.Tensor ↔ ciphertext (DLPack)
│       └── Cargo.toml
│
├── python/
│   └── tensafe_he/
│       ├── __init__.py           # from tensafe_he._native import *
│       └── adapter.py            # Drop-in _TenSafeHEAdapter for inference_engine.py
│
├── benches/
│   ├── ntt_bench.rs              # NTT microbenchmark
│   ├── encrypt_bench.rs          # Encrypt microbenchmark
│   └── full_pipeline_bench.rs    # Full ct×pt pipeline benchmark
│
└── tests/
    ├── test_correctness.rs       # Encrypt → ct×pt → decrypt == plaintext mul
    ├── test_parameters.rs        # Security parameter validation
    └── test_python.py            # Python binding tests
```

### Core API (Python-facing)

```python
import tensafe_he

# Initialize context (fixed parameters, no runtime scheme selection)
ctx = tensafe_he.Context(
    poly_n=16384,       # Only 8192, 16384, 32768 supported
    scale_bits=40,
    device="cuda:0",    # or "cpu" for testing
)

# Encrypt: float64 array → GPU-resident ciphertext
ct = ctx.encrypt(data: np.ndarray) -> tensafe_he.Ciphertext

# Ciphertext × plaintext multiply (element-wise SIMD)
ct_prod = ct * plaintext_array  # __mul__ overload

# Decrypt to GPU tensor (NO CPU transfer — stays in CUDA memory)
gpu_tensor = ctx.decrypt_to_gpu(ct) -> torch.Tensor

# Decrypt to numpy (with CPU transfer)
np_array = ctx.decrypt(ct) -> np.ndarray

# Batch pipeline: encrypt once, multiply N plaintexts, decrypt all to GPU
# This is the ZeRo-MOAI hot path — fused into a single GPU stream
results = ctx.batch_ct_pt_decrypt(
    ct,                           # Single encrypted hidden state
    plaintexts: list[np.ndarray], # N packed LoRA-A rows
) -> list[torch.Tensor]           # N GPU-resident decrypted results

# Status/info
ctx.status() -> dict
ctx.num_slots -> int
```

### Key Innovation: Fused Batch Pipeline

The highest-impact optimization is fusing the entire ZeRo-MOAI hot path into a single GPU operation:

```
Current (CuKKS, 7 batches):
  for each batch:
    ct_prod = ct * pt          # GPU kernel launch
    dec_gpu = decrypt(ct_prod) # GPU kernel launch
  stacked = torch.stack(decs)  # GPU memory allocation
  result = stacked.cpu()       # PCIe transfer

Custom (TenSafe-HE, fused):
  results = batch_ct_pt_decrypt(ct, [pt0, pt1, ..., pt6])
  # ONE kernel launch:
  #   - Stream-overlaps multiply and decrypt across batches
  #   - Pre-allocated output buffer (no torch.stack allocation)
  #   - Uses pinned memory for PCIe (async transfer)
  result = results.cpu()  # Single async PCIe transfer
```

**Savings:** Eliminate 13 kernel launch overheads (7 multiply + 7 decrypt → 1 fused), eliminate torch.stack allocation, use pinned memory.

### Fused Encode+Encrypt CUDA Kernel

Currently CuKKS does encode and encrypt as separate operations:

```
Standard CKKS encrypt:
  1. Encode: FFT(float64[]) → polynomial a(X) in Z_q[X]/(X^N+1)    // CPU or GPU
  2. NTT(a) → â                                                      // GPU
  3. Sample secret key s, error e ~ χ                                 // GPU
  4. ct₀ = â + e, ct₁ = â·s̃ + e'                                    // GPU
```

Fused kernel does steps 1-4 in a single launch:

```
tensafe_encrypt_fused<<<blocks, threads>>>(
    float64_input,    // Raw float64 array
    ct_out,           // Two polynomials in NTT domain
    secret_key_ntt,   // Pre-computed NTT of secret key
    prng_state        // For error sampling
);
```

**Saves:** 1 kernel launch overhead + 1 intermediate polynomial allocation (~128 KB for poly_n=16384).

### Fused Decrypt+Decode CUDA Kernel

```
Standard CKKS decrypt:
  1. ct₀ - ct₁·s → polynomial in NTT domain     // GPU
  2. iNTT → polynomial in coefficient domain      // GPU
  3. Decode: iFFT → float64[]                     // CPU (currently!)
```

Fused kernel:

```
tensafe_decrypt_fused<<<blocks, threads>>>(
    ct_in,            // Two polynomials
    secret_key_ntt,   // Pre-computed
    float64_output    // Direct float64 result on GPU
);
```

**Saves:** 1 kernel launch + eliminates CPU decode (the iFFT step moves to GPU).

### Specialized NTT Kernels

General libraries support arbitrary N with runtime branching. We only need 3 values:

```cuda
// ntt.cu — three specialized kernels, no branching

__global__ void ntt_8192(uint64_t* poly, const uint64_t* twiddles) {
    // 13 butterfly stages, fully unrolled for N=8192
    // Uses shared memory for first 10 stages, registers for last 3
}

__global__ void ntt_16384(uint64_t* poly, const uint64_t* twiddles) {
    // 14 butterfly stages, optimized for N=16384
    // Block-level sync pattern tuned for this exact N
}

__global__ void ntt_32768(uint64_t* poly, const uint64_t* twiddles) {
    // 15 butterfly stages
}
```

**Savings:** No runtime size checks, unrolled loops, optimal shared memory tiling per N.

---

## Implementation Plan

### Phase 1: Rust Core + CPU Backend (Weeks 1-3)

**Goal:** Correct CKKS implementation, CPU-only, passing all correctness tests.

```
Week 1:
  - params.rs: Define parameter sets for poly_n ∈ {8192, 16384, 32768}
  - rns.rs: RNS (Residue Number System) arithmetic for multi-limb modular ops
  - encoding.rs: CKKS encode/decode (canonical embedding via FFT)
  - sampling.rs: Discrete Gaussian error sampling (constant-time)

Week 2:
  - ntt_cpu.rs: CPU NTT (iterative Cooley-Tukey butterfly)
  - ops_cpu.rs: encrypt(), decrypt(), ct_pt_multiply()
  - ciphertext.rs: Ciphertext struct (two RNS polynomials)
  - Unit tests: encrypt-decrypt roundtrip, ct×pt correctness

Week 3:
  - Cross-validation against OpenFHE/Pyfhel (bit-exact where possible)
  - Security parameter validation (128-bit minimum)
  - CPU benchmark: compare against Pyfhel on same operations
  - Edge cases: zero vectors, max-range values, all-same values
```

**Correctness criterion:** For any float64 input x and plaintext p:
`decrypt(encrypt(x) * p) ≈ x * p` within CKKS approximation error (< 2^-30).

### Phase 2: CUDA Kernels (Weeks 4-6)

**Goal:** GPU-accelerated NTT, encrypt, decrypt, ct×pt multiply.

```
Week 4:
  - ntt.cu: Forward NTT for poly_n=16384 (primary target)
  - ntt.cu: Inverse NTT for poly_n=16384
  - Benchmark NTT vs CuKKS NTT (target: ≥ parity)

Week 5:
  - encrypt.cu: Fused encode+encrypt kernel
  - decrypt.cu: Fused decrypt+decode kernel
  - ct_pt_mul.cu: Ciphertext × plaintext multiply
  - Correctness: GPU encrypt → GPU decrypt == CPU encrypt → CPU decrypt

Week 6:
  - batch_pipeline.cu: Fused batch ct×pt + decrypt (ZeRo-MOAI hot path)
  - Pinned memory allocation for PCIe transfers
  - CUDA stream management for overlap
  - NTT kernels for poly_n=8192 and poly_n=32768
```

### Phase 3: Python Bindings + Integration (Weeks 7-8)

**Goal:** Drop-in replacement for CuKKS in inference_engine.py.

```
Week 7:
  - PyO3 bindings: Context, Ciphertext, encrypt/decrypt/multiply
  - torch.Tensor interop via DLPack (zero-copy GPU↔GPU)
  - __mul__ operator overload for Ciphertext × np.ndarray
  - Python tests: verify API compatibility with _CuKKSAdapter

Week 8:
  - tensafe_he.adapter: Drop-in _TenSafeHEAdapter class
  - Integration test: run benchmark_cukks.py with new backend
  - A/B benchmark: CuKKS vs TenSafe-HE on identical workloads
  - Quality parity test: HE vs no-HE token match rate
```

**Integration point:** In `_init_ckks()`, add as Tier 0 before CuKKS:

```python
# --- Try 0: TenSafe-HE (custom, optimized) ---
try:
    import tensafe_he
    ctx = tensafe_he.Context(poly_n=poly_n, scale_bits=40, device="cuda:0")
    self._cukks = tensafe_he.adapter.TenSafeHEAdapter(ctx)
    self.he_ctx = self._cukks
    self.simd_slots = ctx.num_slots
    logger.info(f"TenSafe-HE ready: poly_n={poly_n} slots={self.simd_slots}")
    return
except (ImportError, Exception) as e:
    logger.warning(f"TenSafe-HE unavailable: {e}")

# --- Try 1: CuKKS GPU (fallback to OpenFHE) ---
...
```

### Phase 4: Optimization + Benchmarking (Weeks 9-10)

**Goal:** Beat CuKKS by ≥25% on end-to-end tok/s.

```
Week 9:
  - Profile with nsys/ncu: find remaining bottlenecks
  - Tune NTT block sizes and shared memory usage per GPU arch
  - Optimize RNS operations (Barrett reduction, lazy reduction)
  - Test on multiple GPUs: RTX A2000, RTX 4090, A100, H100

Week 10:
  - Final A/B benchmark: TenSafe-HE vs CuKKS
  - Publish results in docs/PERFORMANCE.md
  - CI/CD: automated correctness + performance regression tests
  - Package: pip install tensafe-he (manylinux wheel with CUDA)
```

### Phase 5: Groq LPU Integration (Weeks 11-14)

**Goal:** Optimize TenSafe-HE specifically for Groq LPU + GPU co-processor architecture.

```
Week 11:
  - Groq API integration: call Groq for transformer forward pass
  - Define compact serialization format for hidden state transfer
  - Measure Groq→GPU transfer latency baseline (target: <1 ms for 6 KB)
  - Profile end-to-end: Groq transformer + GPU HE pipeline

Week 12:
  - Optimize Groq↔GPU data path:
    - Pre-allocate pinned receive buffer for Groq output
    - Overlap Groq API call with previous token's HE pipeline
    - Pipeline hidden state arrival with encrypt kernel launch
  - Implement async double-buffering (Groq produces while GPU computes)

Week 13:
  - Multi-model support: adapt pipeline for 8B and 70B hidden dimensions
    - 8B (hidden=4096): cols_per_ct=2, 16 batches
    - 70B (hidden=8192): cols_per_ct=1 at poly_n=16384 → use poly_n=32768
  - Benchmark all model sizes on Groq + H100

Week 14:
  - End-to-end agentic benchmark: 10-step and 30-step agent workflows
  - Stress test: concurrent sessions on single Groq+H100 instance
  - Production readiness: error handling, reconnection, monitoring
  - Target performance:
    - 1.5B on Groq + H100: ~77 tok/s (vs 37 with CuKKS)
    - 8B on Groq + H100: ~50 tok/s (vs 24 with CuKKS)
    - 70B on Groq + H100: ~25 tok/s (vs 14 with CuKKS)
```

### Phase 6: Next-Gen GPU + Groq-Native NTT Research (Weeks 15-20)

**Goal:** Prepare for B200 and explore running NTT on Groq LPU natively.

```
Weeks 15-16:
  - Port and tune kernels for NVIDIA B200 (Blackwell architecture)
  - B200 has ~2x memory bandwidth over H100 → ~2x NTT speedup
  - Target: ~5 ms HE pipeline on B200 → ~125 tok/s with Groq

Weeks 17-18:
  - Research: can NTT butterfly operations run on Groq's deterministic
    datapath? Groq has 80 TB/s internal bandwidth — ideal for NTT.
  - Key question: does Groq support 64-bit modular arithmetic?
  - If yes: prototype NTT as structured matrix ops on Groq LPU
  - If no: explore reduced-precision NTT (32-bit RNS, cf. Cheddar paper)

Weeks 19-20:
  - If Groq-native NTT viable: prototype single-chip encrypted inference
  - Target: eliminate co-processor entirely → ~3 ms total → ~330 tok/s
  - If not viable: document findings, continue GPU co-processor path
```

---

## Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| NTT implementation bugs (wrong results, silent errors) | Medium | Critical | Cross-validate against OpenFHE for 10K+ random inputs |
| CUDA kernel correctness across GPU architectures | Medium | High | Test on Ampere (A2000, A100), Ada (4090), Hopper (H100) |
| Security parameter misconfiguration | Low | Critical | Hardcode only NIST-approved parameter sets, no user-configurable security |
| Performance doesn't beat CuKKS | Low | Medium | CuKKS wraps generic OpenFHE; specialized kernels should win |
| Groq API latency variability | Medium | Medium | Double-buffer + async overlap; fall back to local GPU if Groq unavailable |
| Groq↔GPU transfer becomes bottleneck | Medium | High | Compact serialization, pinned memory, PCIe Gen5; hidden state is only 6-32 KB |
| Groq doesn't support 64-bit modular arithmetic for native NTT | High | Low | Continue GPU co-processor path; this is a research bet, not a dependency |
| PyO3 binding memory leaks (GPU buffers) | Medium | Medium | Rust RAII + explicit CUDA free in Drop trait |
| Fused kernel register pressure | Medium | Low | Fall back to separate kernels if fused version hits occupancy limits |

---

## Dependencies

### Build Requirements
- Rust 1.75+ (for stable CUDA FFI patterns)
- CUDA Toolkit 12.0+ (for cooperative groups, DLPack)
- Python 3.10+ (for PyO3 bindings)
- maturin (Python-Rust build tool)

### Reference Implementations to Study
1. **FIDESlib** — best open-source GPU CKKS NTT kernels (Apache-2.0)
2. **HEonGPU** — GPU-native CKKS object model (MIT)
3. **Phantom-FHE** — alternative GPU CKKS approach (BSD-3)
4. **SEAL** — reference CKKS correctness (MIT, for cross-validation)
5. **OpenFHE** — current backend, reference for parameter generation (BSD-2)

### Key Papers
- [CKKS original](https://eprint.iacr.org/2016/421) — Cheon, Kim, Kim, Song (2017)
- [Full-RNS CKKS](https://eprint.iacr.org/2018/931) — RNS variant we implement
- [GPU NTT for FHE](https://eprint.iacr.org/2024/1543) — HEonGPU paper
- [FIDESlib](https://arxiv.org/abs/2507.04775) — GPU CKKS optimization techniques (Jul 2025)
- [Cheddar](https://arxiv.org/abs/2407.13055) — 32-bit RNS for GPU-native datapath (ASPLOS 2026, 2.9-25.6x over prior GPU)
- [Neo](https://dl.acm.org/doi/10.1145/3695053.3731408) — Tensor Core acceleration for all CKKS kernels (ISCA 2025, 3.28x over TensorFHE)
- [HI-CKKS](https://eprint.iacr.org/2024/1976) — High-throughput GPU NTT: 692 kop/s on RTX 4090
- [GPU NTT optimal parameters](https://eprint.iacr.org/2023/1410) — Ozcan et al., state-of-art CUDA NTT for all power-of-2 degrees

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Correctness** | 100% token match vs CuKKS (greedy) | benchmark_cukks.py quality parity test |
| **HE pipeline latency (A2000)** | ≤61 ms (vs 86 ms CuKKS) | Per-token HE breakdown |
| **HE pipeline latency (H100)** | ≤10 ms (vs ~14 ms CuKKS) | Per-token HE breakdown |
| **tok/s (A2000 standalone)** | ≥9.1 (vs 7.4 CuKKS) | benchmark_cukks.py |
| **tok/s (Groq + H100, 8B)** | ≥50 (vs ~24 with CuKKS) | End-to-end Groq benchmark |
| **tok/s (Groq + B200, 8B)** | ≥91 | End-to-end Groq benchmark |
| **10-step agent (Groq + H100)** | ≤40s | Agentic workflow benchmark |
| **Security** | ≥128-bit for all parameter sets | NIST parameter validation |
| **Binary size** | <50 MB wheel | pip package size |
| **Build time** | <5 min from clean | CI build time |
| **API compatibility** | Drop-in for inference_engine.py | Integration test passes |

---

## Groq-Optimized Performance Projections

### Why No FPGA

FPGA was originally proposed as a dedicated HE accelerator. We're dropping it because:

1. **Cost.** FPGA engineers cost $200-400K/yr and are scarce. A single hire costs more than the Groq partnership.
2. **Time.** FPGA development is 6-12 months before first working silicon. Custom CUDA kernels ship in weeks.
3. **GPU keeps getting faster.** H100 → B200 doubles memory bandwidth → doubles NTT speed. NVIDIA solves hardware scaling for free.
4. **Groq may absorb HE natively.** 80 TB/s internal bandwidth is ideal for NTT butterflies. If Groq adds 64-bit integer support, the LPU itself becomes the HE accelerator.
5. **Simpler architecture.** "Groq rack + GPU card" is one procurement. Adding "custom FPGA board" makes enterprise deployment harder.

### Projected Performance: TenSafe-HE vs CuKKS on Groq

| Configuration | CuKKS (current) | TenSafe-HE (custom) | Improvement |
|---|---|---|---|
| A2000 standalone, 1.5B | 7.4 tok/s | ~9.1 tok/s | +23% |
| Groq + H100, 1.5B | ~37 tok/s | **~77 tok/s** | **+108%** |
| Groq + H100, 8B | ~24 tok/s | **~50 tok/s** | **+108%** |
| Groq + H100, 70B | ~14 tok/s | **~25 tok/s** | **+79%** |
| Groq + B200, 1.5B | — | **~125 tok/s** | — |
| Groq + B200, 8B | — | **~91 tok/s** | — |
| Groq + B200, 70B | — | **~45 tok/s** | — |
| Groq-native NTT (research) | — | **~330 tok/s** | — |

### Agentic AI Impact (10-step agent, 200 tokens/step)

| Configuration | CuKKS | TenSafe-HE | Viable? |
|---|---|---|---|
| A2000 standalone | 4.5 min | 3.7 min | No |
| Groq + H100, 8B | 83s | **40s** | **Yes (background)** |
| Groq + B200, 8B | — | **22s** | **Yes (interactive)** |
| Groq + B200, 70B | — | **44s** | **Yes (background, GPT-4 quality)** |

---

## Strategic Value

Building TenSafe-HE gives us:

1. **Performance control.** We optimize for our exact operation profile. No library update can break us.
2. **Groq-native path.** Custom library means custom data formats optimized for Groq LPU ↔ GPU data path, and a research path toward running NTT on Groq's deterministic architecture directly.
3. **GPU generation scaling.** Same CUDA kernels benefit automatically from H100 → B200 → next-gen bandwidth improvements.
4. **IP moat.** A custom HE library optimized for zero-rotation inference is defensible technology.
5. **Supply chain security.** No dependency on OpenFHE's release cycle or breaking changes.
6. **Packaging simplicity.** One pip wheel instead of OpenFHE C++ + CUDA runtime + Python bindings chain.
