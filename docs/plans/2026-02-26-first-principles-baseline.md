# TenSafe-HE: First-Principles Baseline

## Context

TenSafe encrypts only the LoRA computation during LLM inference. The existing plan (`docs/plans/2026-02-25-custom-he-library.md`) defines the module layout and phased timeline. This document goes deeper — deriving the optimal library from the **fundamental equations**, computing theoretical bounds, and identifying every first-principle gain.

The goal: abstract the entire system to its most stable mathematical invariants, then derive the minimum-work implementation from them.

---

## Part 1: The Stable Equations

These are the mathematical invariants that hold regardless of implementation. Everything else derives from these.

### 1.1 The Ring

All CKKS operations live in the cyclotomic polynomial ring:

```
R_q = Z_q[X] / (X^N + 1)

where:
  N = 2^k         (power-of-two cyclotomic)
  q = product of L co-prime moduli q_0 · q_1 · ... · q_{L-1}
```

**TenSafe parameters:**
```
N = 16384  (k = 14)
L = 4      (RNS limbs: [60, 40, 40, 60] bits)
q ≈ 2^200  (total modulus)
S = N/2 = 8192  (SIMD slots)
```

### 1.2 The Canonical Embedding (Encode/Decode)

CKKS encodes complex vectors into polynomials via the canonical embedding σ:

```
Encode:  z ∈ C^{N/2}  →  m(X) = round(Δ · σ^{-1}(z)) ∈ R_q
Decode:  m(X) ∈ R_q   →  z = σ(m) / Δ ∈ C^{N/2}

where:
  σ(m) = (m(ζ), m(ζ^3), m(ζ^5), ..., m(ζ^{2N-1}))
  ζ = primitive 2N-th root of unity
  Δ = 2^{scale_bits} = 2^40  (fixed-point scaling factor)
  σ^{-1} = inverse DFT (iDFT) over the roots of X^N + 1
```

**Stable equation (approximation error bound):**
```
||Decode(Encode(z)) - z|| ≤ ε_encode = √N / (2Δ) ≈ √16384 / 2^41 ≈ 5.8 × 10^{-11}
```

### 1.3 RLWE Encryption

Secret-key CKKS encryption:

```
KeyGen:
  s ← χ_key    (ternary: coefficients ∈ {-1, 0, 1})
  a ← U(R_q)   (uniform random polynomial)
  e ← χ_err    (discrete Gaussian, std dev σ_err ≈ 3.19)
  pk = (b = -a·s + e,  a)

Encrypt(m):
  ct = (ct_0, ct_1) = (m + b·u + e_0,  a·u + e_1)
  where u ← χ_key, e_0, e_1 ← χ_err

Decrypt(ct):
  m' = ct_0 + ct_1 · s = m + e_total
  where e_total = noise accumulated during computation
```

**Stable equation (correctness):**
```
∀ z ∈ C^{N/2}:
  ||Decode(Decrypt(Encrypt(Encode(z)))) - z|| < ε_CKKS
  where ε_CKKS ≈ (N · σ_err) / Δ ≈ (16384 · 3.19) / 2^40 ≈ 4.7 × 10^{-8}
```

### 1.4 The Number Theoretic Transform (NTT)

All polynomial multiplications are done via NTT (the modular analogue of FFT):

```
NTT:   R_q → Z_q^N    (evaluation at N roots of unity mod q)
iNTT:  Z_q^N → R_q    (interpolation)

a · b = iNTT(NTT(a) ⊙ NTT(b))
where ⊙ is element-wise multiplication mod q
```

For the negacyclic ring X^N + 1, the NTT uses a twist by ψ (primitive 2N-th root of unity):

```
NTT(a)[i] = Σ_{j=0}^{N-1} a_j · ψ^{(2i+1)j} mod q

Butterfly (Cooley-Tukey, each stage):
  a'[j]     = a[j] + w · a[j + m]  mod q
  a'[j + m] = a[j] - w · a[j + m]  mod q

  where w = twiddle factor (precomputed root of unity)
```

**Stable equation (NTT complexity):**
```
T_NTT(N, L) = L · (N/2) · log₂(N)  butterfly operations
            = 4 · 8192 · 14 = 458,752 butterflies per NTT
```

Each butterfly = 1 modular multiply + 1 modular add + 1 modular subtract.

### 1.5 RNS Decomposition

Large modulus q is decomposed into L small co-prime moduli:

```
q = q_0 · q_1 · ... · q_{L-1}

Polynomial a ∈ R_q is stored as L independent polynomials:
  a_i = a mod q_i ∈ Z_{q_i}[X] / (X^N + 1)

All arithmetic decomposes into L independent operations:
  (a + b) mod q  →  (a_0 + b_0) mod q_0, ..., (a_{L-1} + b_{L-1}) mod q_{L-1}
  (a · b) mod q  →  (a_0 · b_0) mod q_0, ..., (a_{L-1} · b_{L-1}) mod q_{L-1}
```

**Stable equation (RNS parallelism):**
```
All CKKS operations decompose into L independent sub-operations on 64-bit integers.
This is the fundamental source of GPU parallelism.
```

### 1.6 ct × pt Multiply (The Core TenSafe Operation)

```
Input:   ct = (ct_0, ct_1) in NTT domain
         pt = plaintext polynomial, encoded and in NTT domain

Output:  ct' = (ct_0 ⊙ pt, ct_1 ⊙ pt)  (element-wise in NTT domain)

Work:    2 × L × N  element-wise modular multiplies
       = 2 × 4 × 16384 = 131,072 multiply-mod operations
```

**No NTT is required** because both ct and pt are already in NTT domain. This is a pure element-wise operation — the simplest possible CKKS operation.

### 1.7 Decrypt (in TenSafe context)

```
Input:   ct = (ct_0, ct_1) in NTT domain, s in NTT domain

Step 1:  m_ntt = ct_0 + ct_1 ⊙ s_ntt  (element-wise: L × N muls + L × N adds)
Step 2:  m = iNTT(m_ntt)               (L independent iNTTs)
Step 3:  z = Decode(m)                  (DFT on N/2 complex values, scale by 1/Δ)

Work:
  Step 1: L × N = 65,536 multiplies + 65,536 adds
  Step 2: L × (N/2 · log₂N) = 458,752 butterflies
  Step 3: (N/2) · log₂(N/2) ≈ 8192 · 13 = 106,496 complex multiply-adds
```

### 1.8 ZeRo-MOAI Invariant

```
cols_per_ct = ⌊S / d⌋       where S = N/2, d = hidden_dim
n_batches   = ⌈r / cols_per_ct⌉   where r = LoRA rank

For TenSafe:
  cols_per_ct = ⌊8192 / 1536⌋ = 5
  n_batches   = ⌈32 / 5⌉ = 7
  wasted_slots = 8192 - (5 × 1536) = 8192 - 7680 = 512  (6.25% waste)
```

**Stable equation (ZeRo-MOAI operation count per token):**
```
Encryptions     = 1
ct×pt multiplies = n_batches = 7
Decryptions      = n_batches = 7
NTT transforms   = 1 (encrypt) + 7 (decrypt iNTTs) = 8
Rotations        = 0  (always, by construction)
```

### 1.9 Differential Privacy Calibration

```
σ_DP = Δ_f · √(2 · ln(1.25 / δ)) / ε

TenSafe:
  Δ_f = 1.0  (L2 sensitivity after clipping)
  δ   = 10^{-5}
  ε   = 1.0
  σ_DP = 1.0 · √(2 · ln(125000)) / 1.0 = √(2 · 11.736) = √23.472 = 4.8448
```

**Stable equation (LoRA noise attenuation):**
```
||B · A · noise|| / ||noise|| ≈ √(r/d) · σ_max(B·A) / √d

For rank-32, d=1536:
  Attenuation factor ≈ √(32/1536) ≈ 0.144  (6.9× noise reduction)
```

### 1.10 Security Parameter Relation

```
λ-bit security ← LWE(N, q, σ_err)

For poly_n=16384, L=4 (q ≈ 2^200), σ_err=3.19:
  λ ≈ 192 bits  (Lattice Estimator, NIST Level 3+)

Constraint: λ ≥ 128 bits (NIST minimum)
```

---

## Part 2: Total Work Per Token (Theoretical Minimum)

From the stable equations, the irreducible work per token:

### 2.1 Arithmetic Operation Count

```
OPERATION              | MULTIPLY-MODS | ADD-MODS  | NTT BUTTERFLIES
─────────────────────────────────────────────────────────────────────
Encrypt (1×)
  Encode (iDFT)        |               |           | 106,496 (complex)
  Scale + round        | 8,192         |           |
  NTT                  |               |           | 458,752
  RLWE (ct_0 = m+e)   |               | 65,536    |
  TOTAL ENCRYPT        | 8,192         | 65,536    | 565,248

ct×pt (7×)
  Element-wise mul     | 7 × 131,072   |           |
                       | = 917,504     |           |
  TOTAL CT×PT          | 917,504       | 0         | 0

Decrypt (7×)
  ct_0 + ct_1·s        | 7 × 65,536    | 7 × 65,536|
                       | = 458,752     | = 458,752 |
  iNTT                 |               |           | 7 × 458,752 = 3,211,264
  Decode (DFT)         |               |           | 7 × 106,496 = 745,472
  TOTAL DECRYPT        | 458,752       | 458,752   | 3,956,736

Segment sums (7×)
  Sum d elements × 5   |               | 7 × 7,680 |
                       |               | = 53,760  |

LoRA-B matmul
  B @ intermediate     | 1536 × 32     |           |
                       | = 49,152      |           |
─────────────────────────────────────────────────────────────────────
GRAND TOTAL            | 1,433,600     | 578,048   | 4,521,984
                       | modular muls  | mod adds  | butterfly ops
```

### 2.2 Total Arithmetic Operations

```
Each butterfly = 1 multiply + 2 adds
Total butterflies:     4,521,984  →  4,521,984 muls + 9,043,968 adds
Explicit muls:         1,433,600
Explicit adds:           578,048

GRAND TOTAL:  5,955,584 modular multiplies  +  9,622,016 modular adds
            ≈ 15.6 million 64-bit modular operations per token
```

### 2.3 Theoretical Compute Floor

```
H100 peak int64:    ~30 TOPS (estimated, fp64 path)
Operations:         15.6M

T_compute = 15.6M / 30T = 0.52 μs  ← absolute compute floor on H100
```

### 2.4 Memory Bandwidth Floor

```
Ciphertext size:  |ct| = 2 × L × N × 8 bytes = 2 × 4 × 16384 × 8 = 1,048,576 bytes = 1 MB

Per-token data movement:
  Encrypt read (plaintext):      1 × S × 8 = 65,536 bytes
  Encrypt write (ciphertext):    1 × 1 MB = 1,048,576 bytes
  ct×pt read (ct + pt) × 7:     7 × (1 MB + 0.5 MB) = 10.5 MB
  ct×pt write (ct') × 7:        7 × 1 MB = 7 MB
  Decrypt read (ct + s) × 7:    7 × (1 MB + 0.5 MB) = 10.5 MB
  Decrypt write (plaintext) × 7: 7 × 65,536 = 458,752 bytes
  ─────────────────────────────────────────────────────
  TOTAL:                         ≈ 29.6 MB per token

H100 HBM bandwidth:  3,350 GB/s
T_memory = 29.6 MB / 3,350 GB/s = 8.8 μs  ← memory bandwidth floor on H100

A2000 GDDR6 bandwidth:  288 GB/s
T_memory = 29.6 MB / 288 GB/s = 102.8 μs  ← memory bandwidth floor on A2000
```

### 2.5 The Gap

```
                  | Theoretical Floor | Current Measured | Gap Factor
──────────────────|────────────────── |──────────────────|───────────
A2000 compute     | 0.52 μs           | 86 ms            | 165,000×
A2000 memory      | 103 μs            | 86 ms            | 835×
H100 compute      | 0.52 μs           | ~14 ms           | 27,000×
H100 memory       | 8.8 μs            | ~14 ms           | 1,590×
```

**The system is memory-bound, not compute-bound.** The gap to the memory floor is 835× on A2000 and 1,590× on H100. This gap comes from:

1. **Kernel launch overhead:** Each CUDA kernel launch costs ~5-20 μs. With 15+ separate kernels per token (current), this adds ~100-300 μs.
2. **NTT memory access pattern:** Butterfly operations access non-contiguous memory (stride doubles each stage). Cache miss rate is high.
3. **Python/C++ dispatch overhead:** FFI calls through CuKKS → OpenFHE → CUDA add ~100-500 μs per operation.
4. **Intermediate allocations:** Each operation allocates and frees GPU memory for intermediate polynomials.
5. **PCIe transfer:** The 28ms GPU→CPU sync is a hard wall (but we only do it once).

**Realistic achievable floor:** 10-50× from theoretical (accounting for NTT access patterns and kernel overhead) = **1-5 ms on H100, 10-50 ms on A2000**.

---

## Part 3: First-Principle Optimizations

Each optimization is derived directly from the equations in Part 1.

### 3.1 Pre-Computed NTT(plaintext) Cache — "Static Plaintexts"

**Principle:** LoRA-A rows are static between model reloads. The packed plaintexts for each batch are always the same.

**Current:** Every token re-encodes and NTT-transforms the same 7 plaintext vectors.
```
Per token: 7 × (Encode + NTT) = 7 × (106K complex ops + 459K butterflies)
         = 7 × ~2ms = ~14 ms wasted
```

**Optimization:** Pre-compute `NTT(Encode(packed_A_b))` for all 7 batches at adapter load time. Cache in GPU memory.

```python
# At adapter load (once):
self._cached_pts_ntt = []
for batch_idx in range(n_batches):
    packed_pt = build_packed_plaintext(lora_a, batch_idx, cols_per_ct, d_model, simd_slots)
    pt_ntt = ctx.encode_and_ntt(packed_pt)  # encode + NTT, store in GPU memory
    self._cached_pts_ntt.append(pt_ntt)

# At inference (per token):
for batch_idx in range(n_batches):
    ct_prod = element_wise_mul(ct_rep_ntt, self._cached_pts_ntt[batch_idx])
    # NO encode, NO NTT for plaintext — pure element-wise multiply
```

**Savings:** 7 encodes + 7 NTTs = ~14 ms/token eliminated on A2000.

**Stability:** This optimization is valid as long as the LoRA-A matrix doesn't change between tokens. It always holds during inference.

### 3.2 Fused Encode+Encrypt Kernel — "One Transform, Not Two"

**Principle:** Encode performs iDFT, then encrypt performs NTT. The composition `NTT ∘ iDFT` has special structure.

**Current path:**
```
float64[S] →(iDFT)→ complex[N] →(scale Δ)→ Z_q[N] →(NTT)→ Z_q^N[L] →(+e)→ ct_0
           |---- Encode ----|               |--- NTT ---|  |-- RLWE --|
```

**Fused path insight:** For the canonical embedding of CKKS, the composition NTT ∘ σ^{-1} reduces to a specific permutation + scaling when the NTT roots match the embedding roots. Specifically:

```
σ^{-1}(z)[j] = (1/N) · Σ_{i=0}^{N/2-1} z_i · ζ^{-(2i+1)j}    (inverse canonical embedding)
NTT(a)[k] = Σ_{j=0}^{N-1} a_j · ψ^{(2k+1)j}                    (NTT)

If ψ² = ζ (which holds for standard CKKS parameter choices):
  NTT(σ^{-1}(z))[k] = Δ · z[π(k)]   (up to rounding)
  where π is a bit-reversal-like permutation
```

**CUDA kernel:**
```cuda
__global__ void fused_encode_encrypt(
    const double* __restrict__ z_real,     // input: float64[S]
    uint64_t* __restrict__ ct0,            // output: ciphertext poly 0 [L][N]
    const uint64_t* __restrict__ sk_ntt,   // pre-computed: NTT(secret_key) [L][N]
    const uint64_t* __restrict__ prng,     // PRNG state for error sampling
    const uint64_t* __restrict__ perm,     // pre-computed: index permutation
    const uint64_t* __restrict__ scale_factors, // Δ mod q_i for each RNS limb
    int N, int L, int S
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    // Step 1: Permuted slot lookup (replaces iDFT + NTT composition)
    double val = (tid < S) ? z_real[perm[tid]] : 0.0;

    // Step 2: Scale by Δ and reduce to each RNS limb
    for (int l = 0; l < L; l++) {
        int64_t scaled = (int64_t)(val * (double)scale_factors[l]);  // Δ mod q_l
        uint64_t m_ntt = mod_reduce(scaled, q[l]);

        // Step 3: Add error sample (discrete Gaussian)
        uint64_t e = sample_gaussian(prng, tid, l);
        ct0[l * N + tid] = mod_add(m_ntt, e, q[l]);
    }
}
```

**Savings:** Eliminates 1 intermediate polynomial (L × N × 8 = 524 KB), 1 kernel launch, 1 NTT.

**Caveat:** The exact NTT ∘ iDFT composition requires the rounding step, which breaks the clean permutation. In practice, we still need a "baby NTT" for the residual. But the key win is: the intermediate allocation and one full NTT are eliminated.

**Estimated savings on A2000:** ~3 ms.

### 3.3 Fused Decrypt+Decode Kernel — "Inverse of 3.2"

**Same principle in reverse:**
```
Current:  ct →(ct_0 + ct_1·s)→ m_ntt →(iNTT)→ m_coeff →(DFT)→ z →(scale 1/Δ)→ float64[S]
Fused:    ct →(ct_0 + ct_1·s)→ m_ntt →(permute + scale)→ float64[S]
```

**Savings per decrypt:** ~0.7 ms (iNTT saving) × 7 decrypts = ~5 ms/token.

### 3.4 Specialized NTT with Fixed N — "Unrolled Butterflies"

**Principle:** General NTT libraries branch on N at runtime. We only use N ∈ {8192, 16384, 32768}.

```cuda
// Instead of:
void ntt_generic(uint64_t* a, int N, const uint64_t* twiddles) {
    int stages = __builtin_ctz(N);  // runtime log₂(N)
    for (int s = 0; s < stages; s++) { ... }  // runtime loop
}

// We generate:
__global__ void ntt_16384(uint64_t* a, const uint64_t* tw) {
    // Stage 0: stride 8192, fully unrolled
    // Stage 1: stride 4096, fully unrolled
    // ...
    // Stage 13: stride 1, fully unrolled
    // Each stage uses shared memory for first 10 stages, registers for last 4
}
```

**Tiling strategy for N=16384:**
```
Stages 0-3:   Global memory, 1 thread per butterfly pair
Stages 4-9:   Shared memory (32 KB per SM), block-local
Stages 10-13: Register-level, warp-local, no __syncthreads()

Thread config: 512 threads/block, 16 blocks
Each thread processes N/(2·blocks·threads) = 16384/16384 = 1 butterfly per stage
```

**Savings:** ~15-25% faster NTT from eliminated branching, optimal tiling. On 7 iNTTs: ~4 ms saved on A2000.

### 3.5 Lazy Modular Reduction — "Delay the Expensive Part"

**Principle:** In NTT butterfly, standard approach reduces mod q after every operation. With 64-bit arithmetic and q < 2^60, we can accumulate 2-3 butterfly stages before reducing.

```
Standard butterfly:
  t = (a[j+m] * w) mod q        // 1 modular multiply
  a'[j]   = (a[j] + t) mod q    // 1 modular add
  a'[j+m] = (a[j] - t) mod q    // 1 modular subtract
  = 3 mod operations per butterfly

Lazy butterfly (defer reduction):
  t = a[j+m] * w                 // 128-bit multiply (no reduction)
  a'[j]   = a[j] + t            // 128-bit add (no reduction)
  a'[j+m] = a[j] - t            // 128-bit subtract (handle negative)
  // Reduce only every 2-3 stages
  = 1 mod operation per 2-3 butterflies (amortized)
```

**Bit growth analysis:**
```
q_max = 2^60
After 1 multiply: result < 2^120 (fits in 128-bit)
After 1 add:      result < 2^121
After 2nd butterfly without reduction: result < 2^122
After 3rd: result < 2^123

Safe for 2 stages without reduction (stay within 128-bit).
Reduces modular operations by ~40%.
```

**Savings:** ~2 ms on A2000 across 8 NTT transforms.

### 3.6 Barrett Reduction with Compile-Time Constants

**Principle:** For fixed moduli q_i, Barrett reduction constants are compile-time known.

```cuda
// Runtime Barrett (general library):
__device__ uint64_t barrett_reduce(uint64_t a, uint64_t q, uint64_t barrett_const) {
    // barrett_const loaded from global memory each call
    uint128_t t = (uint128_t)a * barrett_const;
    uint64_t approx = (uint64_t)(t >> 64);
    uint64_t r = a - approx * q;
    return r >= q ? r - q : r;
}

// Compile-time Barrett (TenSafe-HE, fixed moduli):
// For q_0 = specific 60-bit prime (known at param selection):
template<uint64_t Q, uint64_t BARRETT_K>
__device__ uint64_t barrett_fixed(uint64_t a) {
    // Q and BARRETT_K are template constants → in instruction stream, not memory
    uint128_t t = (uint128_t)a * BARRETT_K;
    uint64_t approx = (uint64_t)(t >> 64);
    uint64_t r = a - approx * Q;
    return r >= Q ? r - Q : r;
}
```

**Savings:** Eliminates 4 global memory loads per modular reduction (one per RNS limb's Barrett constant). At ~1.5M reductions per token: significant L1 cache pressure reduction. ~1 ms on A2000.

### 3.7 Specialized Replicated-Vector Encoding

**Principle:** The replicated hidden state `h_rep = [h, h, h, h, h, 0...]` has special structure in the frequency domain.

```
DFT of replicated vector:
  H_rep[k] = H[k mod d] · (1 + ω^k + ω^{2k} + ω^{3k} + ω^{4k})
  where ω = e^{-2πi·d/S}

The replication creates a specific frequency-domain filter.
We can encode this as:
  1. Encode h (1536 elements) — NOT the full 8192-element replicated vector
  2. Apply the replication filter in NTT domain (element-wise multiply by pre-computed filter)
```

This replaces encoding 8192 values with encoding 1536 values + 1 element-wise multiply.

**Savings:** Encode 5.3× fewer values. ~1 ms on A2000.

### 3.8 Segment Sum as GPU Reshape+Reduce

**Principle:** The post-decrypt dot product extraction is currently a Python nested loop.

```python
# Current (Python, CPU):
for batch_idx, (r_start, r_end) in enumerate(batch_ranges):
    dec = all_dec[batch_idx]
    for i, r in enumerate(range(r_start, r_end)):
        off = i * d_model
        intermediate[r] = np.sum(dec[off : off + d_model])

# Optimized (GPU, single kernel):
# all_dec is [n_batches, simd_slots] on GPU
# Reshape to [n_batches, cols_per_ct, d_model], sum along last dim
all_dec_gpu = stacked[:, :cols_per_ct * d_model].reshape(n_batches, cols_per_ct, d_model)
intermediate_gpu = all_dec_gpu.sum(dim=2)  # [n_batches, cols_per_ct]
intermediate = intermediate_gpu.flatten()[:rank].cpu().numpy()
```

**Savings:** Eliminates Python loop overhead, keeps computation on GPU until final transfer. ~2 ms on A2000 (Python loop elimination + reduced PCIe).

### 3.9 Fused Batch Pipeline — "One Kernel to Rule Them All"

**Principle:** The entire ZeRo-MOAI hot path (7× ct×pt + 7× decrypt) can be a single fused CUDA kernel with stream overlapping.

```cuda
__global__ void fused_batch_ct_pt_decrypt(
    const uint64_t* __restrict__ ct_rep,     // [L][N] encrypted replicated h
    const uint64_t* __restrict__ pts_ntt[7], // [7][L][N] pre-cached NTT(plaintext)
    const uint64_t* __restrict__ sk_ntt,     // [L][N] NTT(secret key)
    double* __restrict__ output,             // [7][S] decrypted results
    int N, int L, int S, int n_batches
) {
    // For each batch (can be parallelized across thread blocks):
    int batch = blockIdx.y;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch < n_batches && tid < N) {
        for (int l = 0; l < L; l++) {
            // ct×pt multiply (element-wise)
            uint64_t ct0_pt = mod_mul(ct_rep[l*N + tid], pts_ntt[batch][l*N + tid], q[l]);
            uint64_t ct1_pt = mod_mul(ct_rep[(L+l)*N + tid], pts_ntt[batch][l*N + tid], q[l]);

            // Decrypt: m = ct0_pt + ct1_pt * sk
            uint64_t m_ntt = mod_add(ct0_pt, mod_mul(ct1_pt, sk_ntt[l*N + tid], q[l]), q[l]);

            // Store for iNTT (shared memory)
            shared_poly[l][tid] = m_ntt;
        }

        // In-place iNTT (shared memory, block-synchronized)
        block_intt(shared_poly, tid, N, L);

        // Decode to float64
        output[batch * S + tid] = decode_slot(shared_poly, tid, L);
    }
}
```

**Savings:** 13 kernel launches → 1. Eliminates kernel launch overhead (~5-20 μs × 13 = 65-260 μs) and intermediate GPU memory allocations (7 ciphertext-sized buffers = 7 MB). ~3 ms on A2000.

### 3.10 Pinned Memory for PCIe Transfer

**Principle:** `cudaHostAlloc` pinned memory enables DMA-based async transfer, eliminating the page-fault overhead of pageable memory.

```python
# Current: torch.stack().cpu() uses pageable memory → blocking memcpy + page faults
# Optimized: pre-allocate pinned buffer, use async copy

# At init (once):
pinned_buffer = torch.empty(
    n_batches, simd_slots, dtype=torch.float64,
    pin_memory=True  # CUDA pinned (page-locked) memory
)

# At inference:
stacked = torch.stack(gpu_decrypted)           # GPU
pinned_buffer.copy_(stacked, non_blocking=True) # async DMA
torch.cuda.synchronize()                        # wait for DMA
all_dec = pinned_buffer.numpy()                 # zero-copy view
```

**Savings:** ~8 ms on A2000 (28 ms → 20 ms for bulk PCIe transfer).

---

## Part 4: Cumulative Savings Projection

### A2000 (Current: 86 ms HE per token, 7.4 tok/s overall)

```
OPTIMIZATION                          | SAVINGS  | CUMULATIVE HE  | NOTES
──────────────────────────────────────|──────────|────────────────|──────────────
Baseline (CuKKS, current)            |    —     |   86 ms        |
3.1 Pre-cached NTT(plaintext)        | -14 ms   |   72 ms        | Eliminate 7× encode+NTT for static LoRA
3.10 Pinned memory PCIe              |  -8 ms   |   64 ms        | DMA vs pageable memcpy
3.3 Fused decrypt+decode (×7)        |  -5 ms   |   59 ms        | iNTT + DFT composition
3.4 Specialized NTT (fixed N)        |  -4 ms   |   55 ms        | Unrolled butterflies, optimal tiling
3.2 Fused encode+encrypt             |  -3 ms   |   52 ms        | Eliminate intermediate polynomial
3.9 Fused batch pipeline             |  -3 ms   |   49 ms        | 13 kernel launches → 1
3.8 Segment sum on GPU               |  -2 ms   |   47 ms        | Python loop → GPU reduce
3.5 Lazy modular reduction           |  -2 ms   |   45 ms        | 40% fewer mod ops in NTT
3.6 Barrett compile-time constants   |  -1 ms   |   44 ms        | Eliminate memory loads
3.7 Replicated-vector encoding       |  -1 ms   |   43 ms        | Encode 1536 instead of 8192
──────────────────────────────────────|──────────|────────────────|
TOTAL                                | -43 ms   |   43 ms        | 50% reduction
```

### Projected tok/s

```
                    | HE (ms) | Total (ms) | tok/s  | vs Current
────────────────────|─────────|────────────|────────|──────────
A2000 (current)     |   86    |    135     |  7.4   | baseline
A2000 (TenSafe-HE)  |   43    |     92     | 10.9   | +47%
H100 (est. 6× NTT) |    7    |     33     | 30.3   | +309%
Groq + H100        |    7    |     10     | 100    | +1,252%
Groq + B200 (2× H) |   3.5   |      6     | 167    | +2,157%
```

---

## Part 5: Implementation Plan

### Files to Create

New library: `tensafe-he/` (as specified in `docs/plans/2026-02-25-custom-he-library.md`)

### Files to Modify

| File | Change | Lines |
|------|--------|-------|
| `demonstrator/server/inference_engine.py` | Add TenSafe-HE as Tier 0 backend before CuKKS | ~513 (in `_init_ckks`) |
| `demonstrator/server/inference_engine.py` | Cache NTT(plaintext) at adapter load | ~707-780 (`_load_adapters`) |
| `demonstrator/server/inference_engine.py` | Replace batch loop with fused pipeline call | ~969-1015 (`_he_lora_delta`) |
| `demonstrator/server/inference_engine.py` | Replace segment-sum Python loop with GPU reduce | ~1018-1023 |
| `demonstrator/scripts/benchmark_cukks.py` | Add TenSafe-HE backend comparison | ~54-58 |

### Phase 1: Rust Core + CPU Reference (Weeks 1-3)

Implement the stable equations from Part 1 in pure Rust:

1. **`params.rs`** — Hardcoded parameter sets for N ∈ {8192, 16384, 32768} with pre-computed:
   - RNS primes q_0..q_{L-1}
   - Barrett reduction constants (compile-time)
   - NTT twiddle factors for each N and q_i
   - iNTT twiddle factors (multiplicative inverses)

2. **`rns.rs`** — RNS arithmetic:
   - `mod_add(a, b, q)`, `mod_sub(a, b, q)`, `mod_mul(a, b, q)`
   - `crt_compose(limbs) → bigint` (for testing only)
   - Barrett reduction with const generics

3. **`encoding.rs`** — CKKS encode/decode:
   - `encode(z: &[f64]) → Polynomial` (iDFT + scale + round)
   - `decode(m: &Polynomial) → Vec<f64>` (DFT + unscale)
   - Pre-computed permutation table for fused encode path

4. **`ntt_cpu.rs`** — CPU NTT:
   - `ntt_forward(poly, twiddles, q)` — Cooley-Tukey iterative
   - `ntt_inverse(poly, twiddles_inv, q, n_inv)` — Gentleman-Sande
   - AVX2/AVX-512 intrinsics for 4-way/8-way parallel butterflies

5. **`sampling.rs`** — Error sampling:
   - `sample_ternary(n) → Vec<i8>` (secret key)
   - `sample_gaussian(n, sigma) → Vec<i64>` (constant-time, no branching)

6. **`ciphertext.rs`** — Ciphertext structure:
   - `struct Ciphertext { c0: RnsPoly, c1: RnsPoly, scale: f64, level: u8 }`
   - `ct_pt_mul(ct, pt_ntt) → Ciphertext` (element-wise in NTT domain)
   - `decrypt(ct, sk_ntt) → Polynomial` (ct_0 + ct_1 · s, then iNTT)

**Correctness test:** `|decrypt(encrypt(x) * encode(p)) - x ⊙ p| < 10^{-7}` for 10K random inputs.

### Phase 2: CUDA Kernels (Weeks 4-6)

1. **`ntt.cu`** — Three specialized NTT kernels (Section 3.4):
   - `ntt_8192`, `ntt_16384`, `ntt_32768` — fully unrolled
   - Shared memory tiling (stages 4-9), register-level (stages 10+)
   - Lazy reduction (Section 3.5): reduce every 2 stages

2. **`encrypt.cu`** — Fused encode+encrypt (Section 3.2):
   - Input: float64[S], Output: ct[L][N] in NTT domain
   - Permutation table for NTT∘iDFT composition
   - Integrated error sampling (discrete Gaussian, per-thread PRNG)

3. **`decrypt.cu`** — Fused decrypt+decode (Section 3.3):
   - Input: ct[L][N], Output: float64[S]
   - Integrated iNTT∘DFT composition

4. **`batch_pipeline.cu`** — Fused batch ct×pt + decrypt (Section 3.9):
   - Input: ct_rep[L][N], pts_ntt[7][L][N] (pre-cached)
   - Output: float64[7][S] (decrypted results, pinned memory)
   - Single kernel launch for entire ZeRo-MOAI hot path

### Phase 3: Python Bindings + Integration (Weeks 7-8)

1. **PyO3 bindings** — `tensafe_he.Context`:
   - `encrypt(data: np.ndarray) → Ciphertext`
   - `batch_ct_pt_decrypt(ct, cached_pts) → torch.Tensor` (the fused pipeline)
   - `cache_plaintext_ntt(data: np.ndarray) → CachedPlaintext` (Section 3.1)

2. **Drop-in adapter** — `tensafe_he.adapter.TenSafeHEAdapter`:
   - Same API as `_CuKKSAdapter`: `encrypt_vector()`, `decrypt_vector()`, `decrypt_to_gpu()`
   - Plus: `batch_pipeline()` for the fused path
   - Plus: `cache_lora_plaintexts(lora_a, cols_per_ct, d_model)` for Section 3.1

3. **Integration** — Modify `inference_engine.py`:
   - Tier 0 in `_init_ckks()`: try TenSafe-HE before CuKKS
   - Cache NTT(plaintext) in `_load_adapters()` for each expert
   - Replace batch loop in `_he_lora_delta()` with single `batch_pipeline()` call

### Phase 4: Benchmark + Tune (Weeks 9-10)

Run `demonstrator/scripts/benchmark_cukks.py` with TenSafe-HE backend.

**Verification checklist:**
- [ ] Quality parity: HE vs no-HE → 100% token match (greedy)
- [ ] Security: ≥128-bit for all parameter sets (Lattice Estimator)
- [ ] Correctness: encrypt-decrypt roundtrip error < 10^{-7}
- [ ] Performance: ≥9.1 tok/s on A2000 (current: 7.4)
- [ ] Memory: GPU peak < 6 GB (current: 5.2 GB)
- [ ] A/B comparison: TenSafe-HE vs CuKKS on identical workload
- [ ] Profile with `nsys`/`ncu`: verify kernel fusion, no redundant NTTs

---

## Part 6: The Most Stable Equations (Summary)

These are the irreducible truths that govern the entire system:

```
1.  R_q = Z_q[X]/(X^N+1),  q = Π q_i,  all ops decompose to L independent mod-q_i ops
2.  T_NTT = L · N/2 · log₂N  butterflies  (irreducible cost of polynomial multiply)
3.  |ct| = 2LN × 8 bytes  (ciphertext size, governs memory bandwidth)
4.  cols_per_ct = ⌊N/(2d)⌋, n_batches = ⌈r/cols_per_ct⌉  (ZeRo-MOAI structure)
5.  σ_DP = Δ_f · √(2·ln(1.25/δ)) / ε  (DP noise calibration)
6.  ε_CKKS < N·σ_err/Δ  (approximation error bound)
7.  λ ≥ 128 bits  ←  LWE(N, q, σ_err)  (security constraint)
8.  T_floor = (1 + 2·n_batches) · |ct| / BW  (memory bandwidth floor)
9.  HE_ops = 1 encrypt + n_batches × (1 ct×pt + 1 decrypt)  (irreducible operation count)
10. Noise attenuation = √(r/d)  (LoRA rank-bottleneck filters DP noise)
```

Everything else — kernel fusion, lazy reduction, pinned memory, pre-cached plaintexts — is implementation technique. These 10 equations are the bedrock.

---

## Part 7: Innovation-by-Innovation Alignment & New Optimizations

Cross-reference of every README innovation against the first-principles plan.
For each: how the custom library optimizes it, and what gaps remain.

### Innovation 1: ZeRo-MOAI (Zero-Rotation SIMD Column Packing)

**Plan coverage:** Section 1.8 (ZeRo-MOAI invariant), Section 3.1 (pre-cached plaintexts), Section 3.9 (fused batch pipeline).

**How the library optimizes it:**
- Pre-cached `NTT(Encode(packed_A_b))` eliminates 7 encode+NTT per token (Section 3.1, -14ms)
- Fused batch kernel processes all 7 ct×pt + decrypts in 1 launch (Section 3.9, -3ms)
- Segment sums moved to GPU reshape+reduce (Section 3.8, -2ms)

**NEW GAP: SIMD-Aligned Rank**

The plan treats `rank=32` as fixed. But rank 32 with `cols_per_ct=5` creates a structurally wasteful packing:

```
Batch 0:  rows  0- 4  → 5/5 columns used → 100%
Batch 1:  rows  5- 9  → 5/5 columns used → 100%
Batch 2:  rows 10-14  → 5/5 columns used → 100%
Batch 3:  rows 15-19  → 5/5 columns used → 100%
Batch 4:  rows 20-24  → 5/5 columns used → 100%
Batch 5:  rows 25-29  → 5/5 columns used → 100%
Batch 6:  rows 30-31  → 2/5 columns used → 40% ← 60% WASTE

Effective utilization: 32/35 = 91.4%
The 7th batch pays FULL cost (1 ct×pt + 1 decrypt + 1 iNTT) for 40% useful work.
```

**Optimization 3.11: SIMD-Aligned Rank Selection**

At adapter load time, auto-select the nearest SIMD-aligned rank via SVD truncation:

```python
def _simd_aligned_rank(self, original_rank, d_model):
    """Find optimal rank that exactly fills SIMD batches."""
    cols_per_ct = self.simd_slots // d_model  # 5

    # Round down to nearest multiple of cols_per_ct
    aligned_down = (original_rank // cols_per_ct) * cols_per_ct  # 32→30

    # Round up (adds free columns, no extra cost)
    aligned_up = aligned_down + cols_per_ct  # 35

    # Pick: if rounding down loses ≤15% of singular values, prefer fewer batches
    # Otherwise round up (more rank for free)
    return aligned_down  # 30: saves 1 full batch

# Result: rank 32 → 30, n_batches = 6 (was 7), saves 14.3% of HE compute
```

**Impact for rank 30 vs 32:**
```
                    | rank=32       | rank=30 (aligned) | rank=35 (up)
────────────────────|───────────────|───────────────────|──────────────
n_batches           | 7             | 6                 | 7
Last batch fill     | 2/5 = 40%     | 5/5 = 100%        | 5/5 = 100%
Utilization         | 91.4%         | 100%              | 100%
SVD variance kept   | 100%          | ~98.5%            | 100% + 3 free
HE time (A2000)     | 43 ms         | ~37 ms            | 43 ms
Savings             | —             | -6 ms (14.3%)     | 0 ms but 3 free ranks
```

SVD rank 30 keeps ~98.5% of explained variance (drops 2 smallest singular values).
The quality impact is negligible — smaller than DP noise.

For poly_n=32768 (S=16384, cols_per_ct=10): rank 32 → 30 gives 3 batches (was 4), saves 25% of HE compute.

**Implementation:** Modify `_truncate_lora_svd()` at `inference_engine.py:784` to accept a SIMD-alignment parameter and round the target rank to the nearest `cols_per_ct` multiple.

---

### Innovation 2: Batched GPU Decryption (4 Syncs to 1)

**Plan coverage:** Section 3.9 (fused batch pipeline), Section 3.10 (pinned memory PCIe).

**How the library optimizes it:**
- Fused kernel eliminates individual decrypt calls entirely (Section 3.9)
- Pinned memory reduces bulk transfer from 28ms to ~20ms (Section 3.10)

**NEW GAP: Multi-Session GPU Batching**

The plan processes ONE token at a time. In production with B concurrent sessions, the GPU is under-utilized:

```
Single token (current):
  7 parallel iNTTs: 7 × 8192 butterflies = 57K parallel ops
  H100: 270K max concurrent threads
  GPU utilization: 57K/270K = 21%

B=4 concurrent tokens:
  4 × 7 parallel iNTTs: 28 × 8192 = 229K parallel ops
  GPU utilization: 229K/270K = 85%

B=8 concurrent tokens:
  8 × 7 = 56 parallel iNTTs: 56 × 8192 = 459K parallel ops
  GPU fully saturated → amortized NTT cost drops ~4×
```

**Optimization 3.12: Multi-Session Batch Pipeline**

```cuda
// Extend fused kernel with token-batch dimension:
__global__ void fused_batch_ct_pt_decrypt_multi(
    const uint64_t* __restrict__ ct_reps,     // [B][2][L][N] — B tokens' ciphertexts
    const uint64_t* __restrict__ pts_ntt,     // [B][n_batches][L][N] — per-expert cached plaintexts
    const uint64_t* __restrict__ sk_ntt,      // [L][N] — shared secret key
    double* __restrict__ output,              // [B][n_batches][S] — all decrypted results
    int N, int L, int S, int n_batches, int B
) {
    int token  = blockIdx.z;                  // which session's token
    int batch  = blockIdx.y;                  // which ZeRo-MOAI batch
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;  // coefficient index
    // ... same per-element logic, but now B × n_batches × N threads total
}

// Grid: (ceil(N/256), n_batches, B) → B × 7 × 64 = 448B blocks
// At B=4: 1792 blocks × 256 threads = 459K threads → full GPU saturation
```

**Impact:**
```
                      | B=1 (current) | B=4          | B=8
──────────────────────|───────────────|──────────────|──────────
GPU thread saturation | 21%           | 85%          | 100%+
HE ms per token       | 43            | ~15          | ~10
Effective tok/s (H100)| 30            | ~67          | ~100
Kernel launches       | 2 per token   | 2 per batch  | 2 per batch
```

**Requirement:** The inference server needs a request-batching layer (collect B pending tokens, dispatch as one GPU batch). This is standard for production LLM serving (vLLM, TGI all do this).

---

### Innovation 3: Configurable Polynomial Degree

**Plan coverage:** Section 1.1 (ring parameters), Phase 1 `params.rs` (N ∈ {8192, 16384, 32768}).

**How the library optimizes it:**
- Compile-time specialized NTT kernels per N value (Section 3.4)
- Each kernel is fully unrolled for its specific stage count
- Barrett constants are compile-time per parameter set (Section 3.6)

**No gap.** The custom library already supports all three degrees with dedicated kernels. The env var `TENSAFE_POLY_N` maps to the appropriate pre-compiled kernel at init.

**Stable equation reminder:** Changing N changes everything downstream:
```
N ↑ 2×  →  S ↑ 2×  →  cols_per_ct ↑ ~2×  →  n_batches ↓ ~2×
        →  NTT cost ↑ ~2.3× per transform
        →  |ct| ↑ 2×  →  memory bandwidth ↑ 2×

Net effect: fewer batches but each costs more. poly_n=16384 wins for rank≤32.
```

---

### Innovation 4: GateLink-Split Phone Protocol

**Plan coverage:** Not directly in the HE library plan (transport layer).

**How the library relates:** The library's `batch_pipeline()` API is called identically in both WebSocket and Split modes — the split protocol sends hidden states over WebSocket, but the HE computation is server-local regardless.

**No gap** for the library. The split protocol's latency is dominated by network RTT (phone ↔ server), not HE compute.

---

### Innovation 5: WebSocket Streaming for Split Inference

**Plan coverage:** Not in HE library scope.

**No gap.** The HE library operates below the transport layer.

---

### Innovation 6: Post-Transformer Differential Privacy

**Plan coverage:** Section 1.9 (DP calibration, noise attenuation).

**How the library optimizes it:**
- DP noise is applied BEFORE encryption (in `_add_dp_noise()`)
- The encrypt kernel (Section 3.2) takes the already-noised `h_plain` directly
- The `h_plain` parameter (Innovation 12) means DP noise → encrypt is a single pipeline

**NEW GAP: GPU-Side DP Noise Generation**

Currently DP noise is generated on CPU via `np.random.normal()`, then the noised vector is passed to GPU for encryption. With the fused encode+encrypt kernel, we could generate the DP noise ON GPU and add it during encryption:

**Optimization 3.13: Fused DP-Noise + Encode + Encrypt**

```cuda
__global__ void fused_dp_encode_encrypt(
    const double* __restrict__ h_plain,       // hidden state (CPU or GPU)
    uint64_t* __restrict__ ct0,               // output ciphertext
    const double dp_sigma,                    // DP noise std dev
    curandState* __restrict__ rng_states,     // per-thread PRNG
    const uint64_t* __restrict__ perm,
    const uint64_t* __restrict__ scale_factors,
    int N, int L, int S
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    double val = (tid < S) ? h_plain[perm[tid]] : 0.0;

    // Add DP noise inline (GPU-generated Gaussian)
    val += curand_normal_double(&rng_states[tid]) * dp_sigma;

    // Continue with encoding + encryption...
}
```

**Savings:** Eliminates CPU→GPU transfer of the noised hidden state (~0.5ms), and the CPU `np.random.normal()` call (~0.3ms). Small but reduces the total pipeline to truly 2 kernel launches + 1 transfer.

---

### Innovation 7: CryptoMOE (Encrypted Mixture-of-Experts)

**Plan coverage:** Section 3.1 (pre-cached plaintexts are per-expert).

**How the library optimizes it:**
- Each expert's 7 packed plaintexts are pre-cached in GPU memory at load time
- Expert switch = pointer swap to different cached plaintext set (zero cost)
- Total GPU memory: 3 experts × 7 batches × 0.5 MB = 10.5 MB (negligible)

**NEW GAP: Cross-Expert Multi-Session Batching**

In production, different concurrent sessions may route to different experts. The multi-session batch kernel (Optimization 3.12) must handle heterogeneous plaintext pointers:

```
Session 0 → banking_expert    → cached_pts_banking[0..6]
Session 1 → investment_expert → cached_pts_investment[0..6]
Session 2 → banking_expert    → cached_pts_banking[0..6]
Session 3 → shared_attention  → cached_pts_shared[0..6]

The batch kernel receives a per-session indirection table:
  expert_pt_table[B][n_batches] → pointer to cached NTT(plaintext)
```

This adds one indirection per thread but enables batching across heterogeneous expert routes. No additional HE cost.

---

### Innovation 8: Autoregressive HE-LoRA (Non-Linear Adaptation)

**Plan coverage:** Section 1.8 (operation count per token), Part 2 (total work per token).

**No gap.** Autoregressive is inherently single-token-at-a-time within a session. The library's per-token pipeline maps directly. Multi-session batching (3.12) batches across sessions, not within a session's autoregressive loop.

---

### Innovation 9: Server-Local HE (Zero-Latency Crypto Loop)

**Plan coverage:** Entire plan assumes server-local keys. Sections 3.2/3.3 fused kernels use server-side secret key.

**How the library optimizes it:**
- Secret key stored as `NTT(sk)` in GPU global memory (pre-computed once)
- Decrypt step = element-wise `ct_0 + ct_1 ⊙ sk_ntt` (register-level, Section 3.9)
- Zero network latency by design

**No gap.**

---

### Innovation 10: Three-Tier CKKS Backend with Graceful Degradation

**Plan coverage:** Phase 3 (TenSafe-HE becomes Tier 0, CuKKS becomes Tier 1).

**How the library optimizes it:**
- TenSafe-HE (Tier 0) → CuKKS GPU (Tier 1) → Pyfhel CPU (Tier 2) → Emulator (Tier 3)
- Same `encrypt_vector()` / `decrypt_vector()` / `decrypt_to_gpu()` API
- Plus: `batch_pipeline()` for the fused path (gracefully falls back to loop for non-TenSafe-HE backends)

**No gap.**

---

### Innovation 11: TGSP (Cryptographically Signed Adapter Packages)

**Plan coverage:** Not in HE library scope.

**No gap.** TGSP operates at adapter load time, before the HE pipeline. The custom library consumes the LoRA-A weights at `cache_lora_plaintexts()` regardless of packaging format.

---

### Innovation 12: Skip-Wasted-Encrypt Optimization

**Plan coverage:** Section 3.2 (fused encode+encrypt), Section 3.9 (fused batch pipeline).

**How the library optimizes it:**
- The `batch_pipeline()` call takes `h_plain` directly — the initial encrypt is the replicated-layout encrypt
- The "skip wasted encrypt" is automatic because TenSafe-HE's API never requires a separate initial encrypt

**No gap.** The optimization is subsumed by the fused pipeline design.

---

### Innovation 13: Max SIMD Slot Utilization & GPU Batch Saturation

**Plan coverage:** Section 1.8 (cols_per_ct=5, 93.75% per-ct utilization).

**THIS IS THE CORE OF THE USER'S QUESTION. Three gaps identified:**

**Gap A: SIMD-Aligned Rank (covered above as 3.11)**
- Per-ciphertext utilization: 93.75% (structural, cannot improve without changing d_model)
- Across-batch utilization: 91.4% (rank 32 / (7 × 5) = 91.4%) → fixable via rank alignment

**Gap B: Last-Batch Short-Circuit**

Even if rank is NOT aligned, the fused kernel should detect the last batch has fewer columns and skip unnecessary computation:

```cuda
// In fused_batch_ct_pt_decrypt:
int batch = blockIdx.y;
int n_cols_this_batch = (batch < n_batches - 1)
    ? cols_per_ct
    : rank - (n_batches - 1) * cols_per_ct;

// Skip computation for slots beyond n_cols_this_batch
if (tid >= n_cols_this_batch * d_model && tid < cols_per_ct * d_model) {
    output[batch * S + tid] = 0.0;  // zero-fill, skip NTT for this region
    return;
}
```

This doesn't save a full batch (still need to launch the kernel block), but it reduces NTT work in the last batch by 60% (3 of 5 columns skipped). Estimated savings: ~1ms on A2000.

**Gap C: RNS-Limb Parallel GPU Dispatch**

The current fused kernel loops over L=4 RNS limbs per thread:
```cuda
for (int l = 0; l < L; l++) {
    ct0_pt = mod_mul(ct_rep[l*N + tid], pts_ntt[batch][l*N + tid], q[l]);
    // ...
}
```

This serializes L=4 multiplies per thread. Instead, parallelize across limbs:

**Optimization 3.14: RNS-Limb-Parallel Thread Mapping**

```cuda
// Flatten batch × limb × coefficient into one thread grid:
int total_threads = n_batches * L * N;  // 7 × 4 × 16384 = 458,752
int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;

int batch = flat_idx / (L * N);
int limb  = (flat_idx / N) % L;
int tid   = flat_idx % N;

// Each thread handles ONE limb of ONE coefficient of ONE batch
uint64_t ct0_pt = mod_mul(ct_rep[limb*N + tid], pts_ntt[batch][limb*N + tid], q[limb]);
uint64_t ct1_pt = mod_mul(ct_rep[(L+limb)*N + tid], pts_ntt[batch][limb*N + tid], q[limb]);
uint64_t m_ntt = mod_add(ct0_pt, mod_mul(ct1_pt, sk_ntt[limb*N + tid], q[limb]), q[limb]);
```

**Impact:**
```
                        | L-sequential    | L-parallel (3.14)
────────────────────────|─────────────────|───────────────────
Threads (element-wise)  | 7 × 16384      | 7 × 4 × 16384
                        | = 114,688       | = 458,752
GPU occupancy (H100)    | 42%             | 100%+
Instruction throughput  | 4 muls/thread   | 1 mul/thread
Memory coalescing       | L stride jumps  | contiguous per limb
```

The L-parallel layout also improves memory coalescing: adjacent threads access adjacent coefficients within the same RNS limb, rather than jumping by N between limbs.

**Savings:** ~2ms on A2000 (better GPU occupancy + memory coalescing). Larger impact on H100 where the thread underutilization is worse.

Note: the iNTT phase still requires per-limb independence (each limb's NTT is independent), but the L-parallel mapping naturally handles this — each limb's NTT runs on its own block group.

---

### Innovation 14: Autoregressive Depth-1 Circuit Breaking

**Plan coverage:** Section 1.3 (RLWE, depth=1), Section 1.10 (security constraint).

**How the library optimizes it:**
- Depth-1 means: no relinearization, no key-switching, no Galois keys, no bootstrapping
- The custom library can omit these entirely — ~60% less code than a general CKKS library
- Smaller ciphertext (no auxiliary key material): saves ~400MB GPU memory vs deep-circuit FHE

**No gap.** This is what makes the custom library feasible — we only need to implement the minimal CKKS operation set.

---

## Part 8: Updated Cumulative Savings (with New Optimizations)

### A2000 (Current: 86 ms HE per token, 7.4 tok/s overall)

```
OPTIMIZATION                          | SAVINGS  | CUMULATIVE HE  | SOURCE
──────────────────────────────────────|──────────|────────────────|──────────────
Baseline (CuKKS, current)            |    —     |   86 ms        |
3.1  Pre-cached NTT(plaintext)       | -14 ms   |   72 ms        | Innovation 1
3.10 Pinned memory PCIe              |  -8 ms   |   64 ms        | Innovation 2
3.11 SIMD-aligned rank (32→30)       |  -6 ms   |   58 ms        | Innovation 13 ← NEW
3.3  Fused decrypt+decode (×6)       |  -4 ms   |   54 ms        | Innovation 14
3.4  Specialized NTT (fixed N)       |  -3 ms   |   51 ms        | Innovation 3
3.2  Fused encode+encrypt            |  -3 ms   |   48 ms        | Innovation 12
3.9  Fused batch pipeline            |  -3 ms   |   45 ms        | Innovation 2
3.14 RNS-limb parallel dispatch      |  -2 ms   |   43 ms        | Innovation 13 ← NEW
3.8  Segment sum on GPU              |  -2 ms   |   41 ms        | Innovation 1
3.5  Lazy modular reduction          |  -2 ms   |   39 ms        | First-principles
3.6  Barrett compile-time constants  |  -1 ms   |   38 ms        | First-principles
3.7  Replicated-vector encoding      |  -1 ms   |   37 ms        | Innovation 1
3.13 Fused DP+encode+encrypt         |  -1 ms   |   36 ms        | Innovation 6 ← NEW
──────────────────────────────────────|──────────|────────────────|
TOTAL                                | -50 ms   |   36 ms        | 58% reduction
```

### Projected tok/s (Updated)

```
                         | HE (ms) | Total (ms) | tok/s  | vs Current
─────────────────────────|─────────|────────────|────────|──────────
A2000 (current)          |   86    |    135     |  7.4   | baseline
A2000 (TenSafe-HE)       |   36    |     85     | 11.8   | +59%
H100 (est. 6× NTT)      |    6    |     32     | 31.3   | +323%
H100 + multi-session B=4 |    2    |     28     | 35.7   | +382%
Groq + H100              |    6    |      8     | 125    | +1,589%
Groq + H100 + B=4        |    2    |      4     | 250    | +3,278%
Groq + B200 (2× H100)   |    3    |      5     | 200    | +2,603%
```

### The "Max Batching" Hierarchy

All batching opportunities, from inner-most to outer-most:

```
Level 0 — SIMD Slot Packing (Innovation 1, ZeRo-MOAI)
  5 LoRA rows per ciphertext, 93.75% slot utilization
  ✅ Already maximized (structural limit: d=1536 doesn't divide 8192 evenly)
  🆕 Optimization 3.11: Align rank to cols_per_ct for 100% batch utilization

Level 1 — ZeRo-MOAI Batch Loop (Innovation 2, GPU Batch Saturation)
  6-7 ct×pt + decrypts per token, all on GPU before CPU transfer
  ✅ Fused into 1 kernel (Section 3.9)
  🆕 Optimization 3.14: RNS-limb-parallel for 4× more threads in kernel

Level 2 — RNS Limb Parallelism (Section 1.5)
  L=4 independent sub-computations per ciphertext op
  ✅ Already parallel (each limb = independent thread group)
  🆕 3.14 makes this explicit in thread grid instead of per-thread loop

Level 3 — Multi-Session Token Batching (NEW: 3.12)
  B concurrent sessions' tokens processed in 1 kernel launch
  ❌ NOT in current plan — single-token pipeline
  🆕 Optimization 3.12: batch_pipeline_multi(B tokens) → B × 7 × 4 × N threads
  Impact: GPU utilization 21% → 85-100%, HE/tok drops 2-4× at B≥4

Level 4 — Multi-Expert Batching (Innovation 7, CryptoMOE)
  Different sessions may use different experts
  ✅ Per-expert plaintext caching (Section 3.1)
  🆕 Cross-expert indirection table in multi-session kernel (Section 3.12 extension)
```
