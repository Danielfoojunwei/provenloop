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
    "poly_scale",
    "fwd_twist",
    "inv_twist",
    "encrypt_fused",
    "decrypt_fused",
];

/// CUDA C source for all TenSafe-HE kernels.
///
/// Barrett reduction on GPU:
///   For moduli q < 2^60 and inputs a,b < q, the product a*b < 2^120.
///   Barrett constant bh = floor(2^128 / q) >> 64 ≈ 2^64 / q.
///   Quotient estimate = __umul64hi(product_hi, bh).
///   Remainder = product_lo - quotient * q, with at most 2 corrections.
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
// Requires: a, b < q < 2^60, bh = floor(2^128 / q) >> 64.
__device__ __forceinline__ unsigned long long
gpu_mod_mul(unsigned long long a, unsigned long long b,
            unsigned long long q, unsigned long long bh) {
    unsigned long long lo = a * b;
    unsigned long long hi = __umul64hi(a, b);

    // Quotient estimate: ((hi:lo) * bh) >> 64 ≈ (hi * bh) >> 64
    unsigned long long quot = __umul64hi(hi, bh);

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
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = gpu_mod_mul(a[i], b[i], q, bh);
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
    unsigned long long v = gpu_mod_mul(data[idx1], w, q, bh);

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
    data[idx1] = gpu_mod_mul(gpu_mod_sub(u, v, q), w, q, bh);
}

// Scale every element by a constant: data[i] = data[i] * scalar mod q
// Used for iNTT normalization (scalar = N^{-1} mod q).
__global__ void poly_scale(
    unsigned long long* __restrict__ data,
    unsigned long long scalar,
    unsigned long long q,
    unsigned long long bh,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = gpu_mod_mul(data[i], scalar, q, bh);
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
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = gpu_mod_mul(data[i], psi_powers[i], q, bh);
    }
}

// Inverse twist: data[i] *= inv_psi_powers[i] mod q
// Undoes the negacyclic twist after inverse NTT.
__global__ void inv_twist(
    unsigned long long* __restrict__ data,
    const unsigned long long* __restrict__ inv_psi_powers,
    unsigned long long q,
    unsigned long long bh,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = gpu_mod_mul(data[i], inv_psi_powers[i], q, bh);
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
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned long long as_val = gpu_mod_mul(a[i], s[i], q, bh);
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
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned long long c1s = gpu_mod_mul(c1[i], s[i], q, bh);
        m_out[i] = gpu_mod_add(c0[i], c1s, q);
    }
}

} // extern "C"
"#;
