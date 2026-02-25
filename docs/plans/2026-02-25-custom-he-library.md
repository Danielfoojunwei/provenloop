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

### GPU NTT Implementations (Can Reference)

| Source | Language | Notes |
|--------|----------|-------|
| FIDESlib NTT kernels | CUDA | Heavily optimized for CKKS, open-source |
| HEonGPU NTT | CUDA | Multi-limb RNS NTT |
| Phantom-FHE NTT | CUDA | Butterfly-style CUDA NTT |
| cuNTT (standalone) | CUDA | Standalone NTT library |
| Troy (GPU HE) | CUDA | Another GPU CKKS with NTT focus |

### FPGA HE Accelerators (Future Reference)

| Accelerator | Paper | Speedup | Notes |
|---|---|---|---|
| FAB | HPCA 2023 | 9.5x over GPU for bootstrapping | FPGA-based, NTT + automorphism engines |
| CraterLake | ISCA 2022 | 4,600x over CPU | ASIC design, 512 NTT lanes |
| F1 | MICRO 2021 | 5,400x over CPU | MIT, programmable FHE accelerator |
| BTS | ISCA 2022 | Bootstrapping-focused | Targets the bootstrapping bottleneck |
| REED | arXiv 2023 | Chiplet-based scaling | Multi-die FHE, good for large polynomials |

---

## Architecture: TenSafe-HE

### Design Principles

1. **CKKS-only.** No BFV, BGV, TFHE. Zero scheme-selection overhead.
2. **ZeRo-MOAI-native.** No rotation support. No Galois keys. No key switching.
3. **GPU-first.** All polynomial operations in CUDA. CPU path for testing only.
4. **Fused kernels.** Encode+encrypt in one kernel. Decrypt+decode in one kernel.
5. **Fixed polynomial degrees.** Specialize NTT for poly_n ∈ {8192, 16384, 32768}. No generic N.
6. **Rust core + CUDA kernels.** Memory safety for parameter management, raw CUDA for compute.
7. **Python bindings.** PyO3 for seamless integration with existing TenSafe inference engine.
8. **Zero-copy GPU interop.** Direct torch.Tensor ↔ ciphertext via DLPack/CUDA pointers.

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

### Phase 5: FPGA Backend (Weeks 11-16, parallel with Phase 4)

**Goal:** FPGA NTT accelerator for maximum throughput.

```
Weeks 11-12:
  - FPGA NTT engine design (Verilog/SystemVerilog or HLS)
  - Target: Xilinx Alveo U250 or Intel Agilex
  - NTT butterfly pipeline with configurable poly_n

Weeks 13-14:
  - Integrate FPGA NTT with Rust core via PCIe DMA
  - Benchmark FPGA NTT vs CUDA NTT

Weeks 15-16:
  - Full pipeline on FPGA: encrypt + ct×pt + decrypt
  - Integration with TenSafe inference engine
  - Benchmark: target 30x speedup over RTX A2000
```

---

## Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| NTT implementation bugs (wrong results, silent errors) | Medium | Critical | Cross-validate against OpenFHE for 10K+ random inputs |
| CUDA kernel correctness across GPU architectures | Medium | High | Test on Ampere (A2000, A100), Ada (4090), Hopper (H100) |
| Security parameter misconfiguration | Low | Critical | Hardcode only NIST-approved parameter sets, no user-configurable security |
| Performance doesn't beat CuKKS | Low | Medium | CuKKS wraps generic OpenFHE; specialized kernels should win. If not, still valuable for FPGA path |
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
- [FIDESlib](https://arxiv.org/abs/2507.04775) — GPU CKKS optimization techniques
- [FAB FPGA accelerator](https://bu-icsg.github.io/publications/2023/fhe_accelerator_fpga_hpca2023.pdf) — FPGA NTT reference

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Correctness** | 100% token match vs CuKKS (greedy) | benchmark_cukks.py quality parity test |
| **HE pipeline latency** | ≤61 ms on RTX A2000 (vs 86 ms CuKKS) | Per-token HE breakdown |
| **End-to-end tok/s** | ≥9.1 on A2000 (vs 7.4 CuKKS) | benchmark_cukks.py |
| **Security** | ≥128-bit for all parameter sets | NIST parameter validation |
| **Binary size** | <50 MB wheel | pip package size |
| **Build time** | <5 min from clean | CI build time |
| **API compatibility** | Drop-in for inference_engine.py | Integration test passes |

---

## Strategic Value

Building TenSafe-HE gives us:

1. **Performance control.** We optimize for our exact operation profile. No library update can break us.
2. **FPGA path.** Same Rust core + FPGA backend gives us the 30x acceleration path without depending on anyone.
3. **Groq integration.** Custom library means custom data formats for Groq LPU ↔ HE co-processor.
4. **IP moat.** A custom HE library optimized for zero-rotation inference is defensible technology.
5. **Supply chain security.** No dependency on OpenFHE's release cycle or breaking changes.
6. **Packaging simplicity.** One pip wheel instead of OpenFHE C++ + CUDA runtime + Python bindings chain.
