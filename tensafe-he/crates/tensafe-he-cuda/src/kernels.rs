//! CUDA kernel source for TenSafe-HE.
//!
//! All GPU kernels are embedded as a single CUDA C string, compiled at runtime
//! via NVRTC (NVIDIA Runtime Compilation). This avoids requiring nvcc at build time.
//!
//! Kernel categories:
//! 1. Modular arithmetic (element-wise polynomial ops)
//! 2. NTT butterfly stages (Cooley-Tukey forward, Gentleman-Sande inverse)
//! 3. Negacyclic twist/untwist
//! 4. Fused encrypt/decrypt composite kernels

/// CUDA module name used when loading the compiled PTX.
pub const MODULE: &str = "tensafe_he";

/// All kernel function names — must match the `extern "C"` declarations below.
pub const KERNEL_NAMES: &[&str] = &[
    "poly_add",
    "poly_sub",
    "poly_hadamard",
    "poly_negate",
    "ntt_fwd_stage",
    "ntt_inv_stage",
    "ntt_fwd_fused",
    "ntt_inv_fused",
    "poly_scale",
    "fwd_twist",
    "inv_twist",
    "encrypt_fused",
    "decrypt_fused",
    // 32-bit NTT kernels (signed Montgomery reduction, ~2× faster on GPU)
    "ntt_fwd_stage_32",
    "ntt_inv_stage_32",
    "ntt_fwd_fused_32",
    "ntt_inv_fused_32",
    "poly_hadamard_32",
    "poly_scale_32",
];

/// CUDA C source for all TenSafe-HE kernels.
///
/// Barrett reduction on GPU:
///   For moduli q < 2^60 and inputs a,b < q, the product a*b < 2^120.
///   Barrett constant = floor(2^128 / q), stored as (bh, bl) = (hi64, lo64).
///   Quotient estimate = hi * bh + __umul64hi(hi, bl) + __umul64hi(lo, bh),
///   where hi:lo = a * b as 128 bits.
///   Remainder = lo - quot * q, with at most 2 corrections.
///
///   Proof that no 64-bit overflow occurs in the remainder:
///   Since approx_quot ≤ true_quot, the true remainder r = a*b - approx_quot*q ≥ 0.
///   And r < 3q < 2^62, so the high 64 bits of (a*b - approx_quot*q) are zero,
///   meaning product_hi = __umul64hi(approx_quot, q) + any carry from low part.
///   The 64-bit subtraction product_lo - (approx_quot * q) yields the correct result.
pub const KERNEL_SOURCE: &str = r#"
extern "C" {

// =====================================================================
// Device helpers: 64-bit modular arithmetic with Barrett reduction
// =====================================================================

__device__ __forceinline__ unsigned long long
gpu_mod_add(unsigned long long a, unsigned long long b, unsigned long long q) {
    unsigned long long r = a + b;
    return (r >= q) ? (r - q) : r;
}

__device__ __forceinline__ unsigned long long
gpu_mod_sub(unsigned long long a, unsigned long long b, unsigned long long q) {
    return (a >= b) ? (a - b) : (q - b + a);
}

// Barrett modular multiply: (a * b) mod q
// Requires: a, b < q < 2^60, bh:bl = floor(2^128 / q) as (hi64, lo64).
__device__ __forceinline__ unsigned long long
gpu_mod_mul(unsigned long long a, unsigned long long b,
            unsigned long long q, unsigned long long bh,
            unsigned long long bl) {
    unsigned long long lo = a * b;
    unsigned long long hi = __umul64hi(a, b);

    // Full 128-bit Barrett: quot ≈ (hi:lo * bh:bl) >> 128
    // = hi*bh + __umul64hi(hi, bl) + __umul64hi(lo, bh)
    // (lower-order terms cannot affect the top 64 bits enough to matter)
    unsigned long long quot = hi * bh + __umul64hi(hi, bl) + __umul64hi(lo, bh);

    // Remainder: lo - quot * q  (no overflow — see module doc)
    unsigned long long r = lo - quot * q;

    // At most 2 corrections
    if (r >= q) r -= q;
    if (r >= q) r -= q;
    return r;
}

// =====================================================================
// Element-wise polynomial arithmetic kernels
// =====================================================================

// out[i] = (a[i] + b[i]) mod q
__global__ void poly_add(
    unsigned long long* __restrict__ out,
    const unsigned long long* __restrict__ a,
    const unsigned long long* __restrict__ b,
    unsigned long long q,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = gpu_mod_add(a[i], b[i], q);
    }
}

// out[i] = (a[i] - b[i]) mod q
__global__ void poly_sub(
    unsigned long long* __restrict__ out,
    const unsigned long long* __restrict__ a,
    const unsigned long long* __restrict__ b,
    unsigned long long q,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = gpu_mod_sub(a[i], b[i], q);
    }
}

// out[i] = (a[i] * b[i]) mod q   (Hadamard / element-wise product)
__global__ void poly_hadamard(
    unsigned long long* __restrict__ out,
    const unsigned long long* __restrict__ a,
    const unsigned long long* __restrict__ b,
    unsigned long long q,
    unsigned long long bh,
    unsigned long long bl,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = gpu_mod_mul(a[i], b[i], q, bh, bl);
    }
}

// out[i] = (q - a[i]) mod q
__global__ void poly_negate(
    unsigned long long* __restrict__ out,
    const unsigned long long* __restrict__ a,
    unsigned long long q,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned long long v = a[i];
        out[i] = (v == 0) ? 0 : (q - v);
    }
}

// =====================================================================
// NTT butterfly kernels — one stage per launch, N/2 threads
// =====================================================================

// Forward NTT stage (Cooley-Tukey butterfly).
//
// For stage s: t = 1 << s, twiddle_offset = 1 + N - (N >> s).
// Each thread handles one butterfly pair.
//
// Butterfly:
//   u = data[idx0]
//   v = data[idx1] * w mod q
//   data[idx0] = u + v mod q
//   data[idx1] = u - v mod q
__global__ void ntt_fwd_stage(
    unsigned long long* __restrict__ data,
    const unsigned long long* __restrict__ twiddles,
    unsigned long long q,
    unsigned long long bh,
    unsigned long long bl,
    unsigned int t,
    unsigned int tw_offset
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // N/2 total threads — caller ensures grid covers this

    unsigned int group_idx = tid / t;
    unsigned int inner_idx = tid % t;
    unsigned int idx0 = group_idx * 2 * t + inner_idx;
    unsigned int idx1 = idx0 + t;

    unsigned long long w = twiddles[tw_offset + group_idx];
    unsigned long long u = data[idx0];
    unsigned long long v = gpu_mod_mul(data[idx1], w, q, bh, bl);

    data[idx0] = gpu_mod_add(u, v, q);
    data[idx1] = gpu_mod_sub(u, v, q);
}

// Inverse NTT stage (Gentleman-Sande butterfly).
//
// For stage s: t = N >> (s+1), twiddle_offset = 1 << s.
// Each thread handles one butterfly pair.
//
// Butterfly:
//   u = data[idx0]
//   v = data[idx1]
//   data[idx0] = u + v mod q
//   data[idx1] = (u - v) * w mod q
__global__ void ntt_inv_stage(
    unsigned long long* __restrict__ data,
    const unsigned long long* __restrict__ twiddles,
    unsigned long long q,
    unsigned long long bh,
    unsigned long long bl,
    unsigned int t,
    unsigned int tw_offset
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int group_idx = tid / t;
    unsigned int inner_idx = tid % t;
    unsigned int idx0 = group_idx * 2 * t + inner_idx;
    unsigned int idx1 = idx0 + t;

    unsigned long long w = twiddles[tw_offset + group_idx];
    unsigned long long u = data[idx0];
    unsigned long long v = data[idx1];

    data[idx0] = gpu_mod_add(u, v, q);
    data[idx1] = gpu_mod_mul(gpu_mod_sub(u, v, q), w, q, bh, bl);
}

// =====================================================================
// Fused NTT kernels — multiple stages in shared memory
// =====================================================================

// Forward NTT fused: processes stages [first_stage, log_n) in shared memory.
// Block size = B threads, segment size = 2*B elements.
// Each block loads a contiguous 2B-element segment, runs butterfly stages,
// then writes back. Reduces kernel launches from log_n to ~5-6.
__global__ void ntt_fwd_fused(
    unsigned long long* __restrict__ data,
    const unsigned long long* __restrict__ twiddles,
    unsigned long long q, unsigned long long bh,
    unsigned long long bl,
    unsigned int log_n, unsigned int first_stage
) {
    extern __shared__ unsigned long long smem[];

    unsigned int B = blockDim.x;
    unsigned int segment_start = blockIdx.x * 2 * B;
    unsigned int tid = threadIdx.x;

    // Load segment into shared memory
    smem[tid] = data[segment_start + tid];
    smem[tid + B] = data[segment_start + tid + B];
    __syncthreads();

    unsigned int global_m = 1u << first_stage;

    for (unsigned int s = first_stage; s < log_n; s++) {
        unsigned int global_t = 1u << (log_n - s - 1);
        unsigned int global_tid = blockIdx.x * B + tid;

        unsigned int group_idx = global_tid / global_t;
        unsigned int inner_idx = global_tid % global_t;
        unsigned int global_idx0 = group_idx * 2 * global_t + inner_idx;
        unsigned int local_idx0 = global_idx0 - segment_start;
        unsigned int local_idx1 = local_idx0 + global_t;

        unsigned long long w = twiddles[global_m + group_idx];
        unsigned long long u = smem[local_idx0];
        unsigned long long v_raw = smem[local_idx1];
        __syncthreads();

        unsigned long long v = gpu_mod_mul(v_raw, w, q, bh, bl);
        smem[local_idx0] = gpu_mod_add(u, v, q);
        smem[local_idx1] = gpu_mod_sub(u, v, q);
        __syncthreads();

        global_m <<= 1;
    }

    // Write back to global memory
    data[segment_start + tid] = smem[tid];
    data[segment_start + tid + B] = smem[tid + B];
}

// Inverse NTT fused: processes stages [0, num_fused_stages) in shared memory.
// Uses Gentleman-Sande butterfly.
__global__ void ntt_inv_fused(
    unsigned long long* __restrict__ data,
    const unsigned long long* __restrict__ twiddles,
    unsigned long long q, unsigned long long bh,
    unsigned long long bl,
    unsigned int num_fused_stages
) {
    extern __shared__ unsigned long long smem[];

    unsigned int B = blockDim.x;
    unsigned int segment_start = blockIdx.x * 2 * B;
    unsigned int tid = threadIdx.x;

    // Load segment into shared memory
    smem[tid] = data[segment_start + tid];
    smem[tid + B] = data[segment_start + tid + B];
    __syncthreads();

    unsigned int global_m_init = B;  // m starts at N/2, but locally it's B for first fused stage

    for (unsigned int s = 0; s < num_fused_stages; s++) {
        unsigned int global_t = 1u << s;
        unsigned int global_m = (blockIdx.x * 2 * B) >> (s + 1);  // N >> (s+1) globally, but need group count
        unsigned int global_tid = blockIdx.x * B + tid;

        unsigned int group_idx = global_tid / global_t;
        unsigned int inner_idx = global_tid % global_t;
        unsigned int global_idx0 = group_idx * 2 * global_t + inner_idx;
        unsigned int local_idx0 = global_idx0 - segment_start;
        unsigned int local_idx1 = local_idx0 + global_t;

        // Twiddle index: in inverse NTT, tw_offset = m = N >> (s+1)
        // The actual m for stage s is half the total data size >> s
        // For the full polynomial: m = N >> (s+1)
        // We pass the total N through the segment calculation
        unsigned int half_n = (blockIdx.x * 2 * B + 2 * B);  // This is a hack; need N
        // Actually, we need N. Let's pass it or compute from gridDim.
        // N = gridDim.x * 2 * B
        unsigned int N = gridDim.x * 2 * B;
        unsigned int inv_m = N >> (s + 1);

        unsigned long long w = twiddles[inv_m + group_idx];
        unsigned long long u = smem[local_idx0];
        unsigned long long v = smem[local_idx1];
        __syncthreads();

        smem[local_idx0] = gpu_mod_add(u, v, q);
        smem[local_idx1] = gpu_mod_mul(gpu_mod_sub(u, v, q), w, q, bh, bl);
        __syncthreads();
    }

    // Write back to global memory
    data[segment_start + tid] = smem[tid];
    data[segment_start + tid + B] = smem[tid + B];
}

// Scale every element by a constant: data[i] = data[i] * scalar mod q
// Used for iNTT normalization (scalar = N^{-1} mod q).
__global__ void poly_scale(
    unsigned long long* __restrict__ data,
    unsigned long long scalar,
    unsigned long long q,
    unsigned long long bh,
    unsigned long long bl,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = gpu_mod_mul(data[i], scalar, q, bh, bl);
    }
}

// =====================================================================
// Negacyclic twist kernels
// =====================================================================

// Forward twist: data[i] *= psi_powers[i] mod q
// Converts standard NTT to negacyclic: Z_q[X]/(X^N-1) → Z_q[X]/(X^N+1).
__global__ void fwd_twist(
    unsigned long long* __restrict__ data,
    const unsigned long long* __restrict__ psi_powers,
    unsigned long long q,
    unsigned long long bh,
    unsigned long long bl,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = gpu_mod_mul(data[i], psi_powers[i], q, bh, bl);
    }
}

// Inverse twist: data[i] *= inv_psi_powers[i] mod q
// Undoes the negacyclic twist after inverse NTT.
__global__ void inv_twist(
    unsigned long long* __restrict__ data,
    const unsigned long long* __restrict__ inv_psi_powers,
    unsigned long long q,
    unsigned long long bh,
    unsigned long long bl,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = gpu_mod_mul(data[i], inv_psi_powers[i], q, bh, bl);
    }
}

// =====================================================================
// Fused composite kernels for encrypt/decrypt
// =====================================================================

// RLWE encrypt: c0[i] = -a[i]*s[i] + m[i] + e[i]  (mod q)
// Fuses 4 modular ops into 1 kernel launch.
__global__ void encrypt_fused(
    unsigned long long* __restrict__ c0,
    const unsigned long long* __restrict__ a,
    const unsigned long long* __restrict__ s,
    const unsigned long long* __restrict__ m,
    const unsigned long long* __restrict__ e,
    unsigned long long q,
    unsigned long long bh,
    unsigned long long bl,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned long long as_val = gpu_mod_mul(a[i], s[i], q, bh, bl);
        unsigned long long neg_as = (as_val == 0) ? 0 : (q - as_val);
        unsigned long long m_plus_e = gpu_mod_add(m[i], e[i], q);
        c0[i] = gpu_mod_add(neg_as, m_plus_e, q);
    }
}

// RLWE decrypt: m[i] = c0[i] + c1[i]*s[i]  (mod q)
// Fuses 2 modular ops into 1 kernel launch.
__global__ void decrypt_fused(
    unsigned long long* __restrict__ m_out,
    const unsigned long long* __restrict__ c0,
    const unsigned long long* __restrict__ c1,
    const unsigned long long* __restrict__ s,
    unsigned long long q,
    unsigned long long bh,
    unsigned long long bl,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned long long c1s = gpu_mod_mul(c1[i], s[i], q, bh, bl);
        m_out[i] = gpu_mod_add(c0[i], c1s, q);
    }
}

// =====================================================================
// 32-bit NTT kernels — signed Montgomery reduction (Cheddar ASPLOS '26)
//
// ~2× faster on GPUs: native 32-bit int throughput, halved shmem usage.
// Signed Montgomery: fewer integer ops than Barrett, no 128-bit products.
//
// Montgomery form: a_mont = a * R mod q, where R = 2^32.
// Multiply: mont_mul(a, b) = a * b * R^{-1} mod q.
// =====================================================================

// Signed Montgomery reduction.
// Given a product p = a * b (up to 62 bits for 30-bit a, b):
//   t = (unsigned int)(p * q_inv)       [low 32 bits]
//   u = (p - (long long)t * q) >> 32    [high word]
//   if u < 0: u += q
// Result is in [0, q).
__device__ __forceinline__ unsigned int
gpu_mont_reduce(long long p, unsigned int q, unsigned int q_inv) {
    unsigned int t = (unsigned int)((unsigned long long)p * (unsigned long long)q_inv);
    int u = (int)((p - (long long)t * (long long)q) >> 32);
    return (u < 0) ? (unsigned int)(u + (int)q) : (unsigned int)u;
}

// Montgomery modular multiply: (a * b) mod q in Montgomery form.
__device__ __forceinline__ unsigned int
gpu_mont_mul(unsigned int a, unsigned int b, unsigned int q, unsigned int q_inv) {
    long long p = (long long)a * (long long)b;
    return gpu_mont_reduce(p, q, q_inv);
}

__device__ __forceinline__ unsigned int
gpu_mod_add_32(unsigned int a, unsigned int b, unsigned int q) {
    unsigned int r = a + b;
    return (r >= q) ? (r - q) : r;
}

__device__ __forceinline__ unsigned int
gpu_mod_sub_32(unsigned int a, unsigned int b, unsigned int q) {
    return (a >= b) ? (a - b) : (q - b + a);
}

// 32-bit forward NTT butterfly stage.
__global__ void ntt_fwd_stage_32(
    unsigned int* __restrict__ data,
    const unsigned int* __restrict__ twiddles,
    unsigned int q, unsigned int q_inv,
    unsigned int t, unsigned int tw_offset
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int group_idx = tid / t;
    unsigned int inner_idx = tid % t;
    unsigned int idx0 = group_idx * 2 * t + inner_idx;
    unsigned int idx1 = idx0 + t;

    unsigned int w = twiddles[tw_offset + group_idx];
    unsigned int u = data[idx0];
    unsigned int v = gpu_mont_mul(data[idx1], w, q, q_inv);

    data[idx0] = gpu_mod_add_32(u, v, q);
    data[idx1] = gpu_mod_sub_32(u, v, q);
}

// 32-bit inverse NTT butterfly stage.
__global__ void ntt_inv_stage_32(
    unsigned int* __restrict__ data,
    const unsigned int* __restrict__ twiddles,
    unsigned int q, unsigned int q_inv,
    unsigned int t, unsigned int tw_offset
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int group_idx = tid / t;
    unsigned int inner_idx = tid % t;
    unsigned int idx0 = group_idx * 2 * t + inner_idx;
    unsigned int idx1 = idx0 + t;

    unsigned int w = twiddles[tw_offset + group_idx];
    unsigned int u = data[idx0];
    unsigned int v = data[idx1];

    data[idx0] = gpu_mod_add_32(u, v, q);
    data[idx1] = gpu_mont_mul(gpu_mod_sub_32(u, v, q), w, q, q_inv);
}

// 32-bit fused forward NTT (shared memory).
// Shared memory per block: 2*B * 4 bytes (vs 2*B * 8 bytes for 64-bit).
__global__ void ntt_fwd_fused_32(
    unsigned int* __restrict__ data,
    const unsigned int* __restrict__ twiddles,
    unsigned int q, unsigned int q_inv,
    unsigned int log_n, unsigned int first_stage
) {
    extern __shared__ unsigned int smem32[];

    unsigned int B = blockDim.x;
    unsigned int segment_start = blockIdx.x * 2 * B;
    unsigned int tid = threadIdx.x;

    smem32[tid] = data[segment_start + tid];
    smem32[tid + B] = data[segment_start + tid + B];
    __syncthreads();

    unsigned int global_m = 1u << first_stage;

    for (unsigned int s = first_stage; s < log_n; s++) {
        unsigned int global_t = 1u << (log_n - s - 1);
        unsigned int global_tid = blockIdx.x * B + tid;

        unsigned int group_idx = global_tid / global_t;
        unsigned int inner_idx = global_tid % global_t;
        unsigned int global_idx0 = group_idx * 2 * global_t + inner_idx;
        unsigned int local_idx0 = global_idx0 - segment_start;
        unsigned int local_idx1 = local_idx0 + global_t;

        unsigned int w = twiddles[global_m + group_idx];
        unsigned int u = smem32[local_idx0];
        unsigned int v_raw = smem32[local_idx1];
        __syncthreads();

        unsigned int v = gpu_mont_mul(v_raw, w, q, q_inv);
        smem32[local_idx0] = gpu_mod_add_32(u, v, q);
        smem32[local_idx1] = gpu_mod_sub_32(u, v, q);
        __syncthreads();

        global_m <<= 1;
    }

    data[segment_start + tid] = smem32[tid];
    data[segment_start + tid + B] = smem32[tid + B];
}

// 32-bit fused inverse NTT (shared memory).
__global__ void ntt_inv_fused_32(
    unsigned int* __restrict__ data,
    const unsigned int* __restrict__ twiddles,
    unsigned int q, unsigned int q_inv,
    unsigned int num_fused_stages
) {
    extern __shared__ unsigned int smem32[];

    unsigned int B = blockDim.x;
    unsigned int segment_start = blockIdx.x * 2 * B;
    unsigned int tid = threadIdx.x;

    smem32[tid] = data[segment_start + tid];
    smem32[tid + B] = data[segment_start + tid + B];
    __syncthreads();

    for (unsigned int s = 0; s < num_fused_stages; s++) {
        unsigned int global_t = 1u << s;
        unsigned int global_tid = blockIdx.x * B + tid;

        unsigned int group_idx = global_tid / global_t;
        unsigned int inner_idx = global_tid % global_t;
        unsigned int global_idx0 = group_idx * 2 * global_t + inner_idx;
        unsigned int local_idx0 = global_idx0 - segment_start;
        unsigned int local_idx1 = local_idx0 + global_t;

        unsigned int N = gridDim.x * 2 * B;
        unsigned int inv_m = N >> (s + 1);

        unsigned int w = twiddles[inv_m + group_idx];
        unsigned int u = smem32[local_idx0];
        unsigned int v = smem32[local_idx1];
        __syncthreads();

        smem32[local_idx0] = gpu_mod_add_32(u, v, q);
        smem32[local_idx1] = gpu_mont_mul(gpu_mod_sub_32(u, v, q), w, q, q_inv);
        __syncthreads();
    }

    data[segment_start + tid] = smem32[tid];
    data[segment_start + tid + B] = smem32[tid + B];
}

// 32-bit Hadamard product: out[i] = mont_mul(a[i], b[i]).
__global__ void poly_hadamard_32(
    unsigned int* __restrict__ out,
    const unsigned int* __restrict__ a,
    const unsigned int* __restrict__ b,
    unsigned int q, unsigned int q_inv,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = gpu_mont_mul(a[i], b[i], q, q_inv);
    }
}

// 32-bit scale: data[i] = mont_mul(data[i], scalar) for iNTT normalization.
__global__ void poly_scale_32(
    unsigned int* __restrict__ data,
    unsigned int scalar,
    unsigned int q, unsigned int q_inv,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = gpu_mont_mul(data[i], scalar, q, q_inv);
    }
}

} // extern "C"
"#;
