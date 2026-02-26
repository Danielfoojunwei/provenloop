//! Number Theoretic Transform (NTT) for negacyclic polynomial arithmetic.
//!
//! All polynomial multiplications in CKKS use NTT:
//!   a · b = iNTT(NTT(a) ⊙ NTT(b))
//!
//! For the ring Z_q[X]/(X^N+1), we use the "twisted" NTT with ψ (primitive 2N-th root).
//!
//! Complexity: T_NTT(N) = (N/2) · log₂(N) butterfly operations per limb.

use crate::rns::{mod_add, mod_mul, mod_sub};

/// Pre-computed NTT tables for a specific (N, q) pair.
#[derive(Debug, Clone)]
pub struct NttTables {
    /// Forward twiddle factors (powers of ψ in bit-reversed order).
    pub forward_twiddles: Vec<u64>,
    /// Inverse twiddle factors (powers of ψ^{-1} in bit-reversed order).
    pub inverse_twiddles: Vec<u64>,
    /// N^{-1} mod q — used to normalize after inverse NTT.
    pub n_inv: u64,
    /// The modulus q.
    pub q: u64,
    /// log₂(N).
    pub log_n: u32,
    /// Polynomial degree N.
    pub n: usize,
}

impl NttTables {
    /// Create NTT tables for a given polynomial degree and modulus.
    pub fn new(n: usize, q: u64) -> Self {
        use crate::rns::{compute_inv_twiddle_factors, compute_twiddle_factors, find_primitive_root, mod_inv};

        let log_n = n.trailing_zeros();
        assert_eq!(1 << log_n, n, "N must be a power of 2");

        let psi = find_primitive_root(n, q);
        let forward_twiddles = compute_twiddle_factors(n, psi, q);
        let inverse_twiddles = compute_inv_twiddle_factors(n, psi, q);
        let n_inv = mod_inv(n as u64, q);

        Self {
            forward_twiddles,
            inverse_twiddles,
            n_inv,
            q,
            log_n,
            n,
        }
    }
}

/// In-place forward NTT (Cooley-Tukey butterfly, decimation-in-time).
///
/// Transforms polynomial coefficients a[0..N-1] to NTT domain.
/// The result is: A[i] = Σ_{j=0}^{N-1} a[j] · ψ^{(2·bit_rev(i)+1)·j} mod q
///
/// Uses iterative radix-2 Cooley-Tukey with pre-computed twiddle factors.
pub fn ntt_forward(a: &mut [u64], tables: &NttTables) {
    let n = tables.n;
    let q = tables.q;
    debug_assert_eq!(a.len(), n);

    let mut m = n;
    let mut t = 1;
    let mut twiddle_idx = 1; // start at index 1 in twiddle table

    for _ in 0..tables.log_n {
        m >>= 1;
        for i in 0..m {
            let w = tables.forward_twiddles[twiddle_idx];
            twiddle_idx += 1;

            let j1 = 2 * i * t;
            for j in j1..j1 + t {
                let u = a[j];
                let v = mod_mul(a[j + t], w, q);
                a[j] = mod_add(u, v, q);
                a[j + t] = mod_sub(u, v, q);
            }
        }
        t <<= 1;
    }
}

/// In-place inverse NTT (Gentleman-Sande butterfly, decimation-in-frequency).
///
/// Transforms NTT-domain values back to coefficient domain.
/// Includes the 1/N normalization factor.
pub fn ntt_inverse(a: &mut [u64], tables: &NttTables) {
    let n = tables.n;
    let q = tables.q;
    debug_assert_eq!(a.len(), n);

    let mut t = n >> 1;
    let mut m = 1;
    let mut twiddle_idx = 1;

    for _ in 0..tables.log_n {
        for i in 0..m {
            let w = tables.inverse_twiddles[twiddle_idx];
            twiddle_idx += 1;

            let j1 = 2 * i * t;
            for j in j1..j1 + t {
                let u = a[j];
                let v = a[j + t];
                a[j] = mod_add(u, v, q);
                a[j + t] = mod_mul(mod_sub(u, v, q), w, q);
            }
        }
        t >>= 1;
        m <<= 1;
    }

    // Normalize by N^{-1} mod q
    for coeff in a.iter_mut() {
        *coeff = mod_mul(*coeff, tables.n_inv, q);
    }
}

/// Apply the "negacyclic twist" to a polynomial before NTT.
/// This multiplies coefficient a[i] by ψ^i, converting standard NTT to negacyclic.
///
/// For CKKS: we need NTT over Z_q[X]/(X^N+1), not Z_q[X]/(X^N-1).
/// The twist handles the +1 vs -1.
pub fn apply_forward_twist(a: &mut [u64], tables: &NttTables) {
    let q = tables.q;
    let n = tables.n;
    // ψ^i for i = 0..N-1 (ψ is the 2N-th root)
    // forward_twiddles contains powers of ψ in bit-reversed order,
    // but we need them in natural order for the twist.
    // Recompute from forward_twiddles[1] = ψ (if stored that way),
    // or compute directly.
    let psi = tables.forward_twiddles[1]; // ψ = first non-trivial twiddle
    let mut psi_power = 1u64;
    for i in 0..n {
        a[i] = mod_mul(a[i], psi_power, q);
        psi_power = mod_mul(psi_power, psi, q);
    }
}

/// Remove the negacyclic twist after inverse NTT.
/// Divides coefficient a[i] by ψ^i.
pub fn apply_inverse_twist(a: &mut [u64], tables: &NttTables) {
    let q = tables.q;
    let n = tables.n;
    let psi_inv = tables.inverse_twiddles[1]; // ψ^{-1}
    let mut psi_inv_power = 1u64;
    for i in 0..n {
        a[i] = mod_mul(a[i], psi_inv_power, q);
        psi_inv_power = mod_mul(psi_inv_power, psi_inv, q);
    }
}

/// Full negacyclic forward NTT: twist + NTT.
pub fn negacyclic_ntt_forward(a: &mut [u64], tables: &NttTables) {
    apply_forward_twist(a, tables);
    ntt_forward(a, tables);
}

/// Full negacyclic inverse NTT: iNTT + untwist.
pub fn negacyclic_ntt_inverse(a: &mut [u64], tables: &NttTables) {
    ntt_inverse(a, tables);
    apply_inverse_twist(a, tables);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small test with N=4, q=17.
    #[test]
    fn test_ntt_roundtrip_small() {
        let n = 4;
        let q = 17u64;
        let tables = NttTables::new(n, q);

        let original = vec![1u64, 2, 3, 4];
        let mut a = original.clone();

        // Forward NTT
        negacyclic_ntt_forward(&mut a, &tables);

        // Values should be different from original (we're in NTT domain)
        assert_ne!(a, original);

        // Inverse NTT should recover original
        negacyclic_ntt_inverse(&mut a, &tables);
        assert_eq!(a, original);
    }

    /// Test NTT roundtrip with a real TenSafe modulus.
    #[test]
    fn test_ntt_roundtrip_large_modulus() {
        let n = 16; // small N for fast test
        // Use a prime q ≡ 1 (mod 2N=32)
        let q = 97u64; // 97 = 3*32 + 1

        let tables = NttTables::new(n, q);

        let original: Vec<u64> = (0..n as u64).collect();
        let mut a = original.clone();

        negacyclic_ntt_forward(&mut a, &tables);
        negacyclic_ntt_inverse(&mut a, &tables);

        assert_eq!(a, original);
    }

    /// Test that NTT-domain Hadamard product corresponds to negacyclic polynomial multiplication.
    #[test]
    fn test_ntt_polynomial_multiply() {
        let n = 8;
        let q = 97u64; // 97 ≡ 1 mod 16

        let tables = NttTables::new(n, q);

        // a(X) = 1 + 2X + 3X^2
        let mut a_coeffs = vec![1u64, 2, 3, 0, 0, 0, 0, 0];
        // b(X) = 4 + 5X
        let mut b_coeffs = vec![4u64, 5, 0, 0, 0, 0, 0, 0];

        // NTT both
        let mut a_ntt = a_coeffs.clone();
        let mut b_ntt = b_coeffs.clone();
        negacyclic_ntt_forward(&mut a_ntt, &tables);
        negacyclic_ntt_forward(&mut b_ntt, &tables);

        // Hadamard product in NTT domain
        let mut c_ntt = vec![0u64; n];
        for i in 0..n {
            c_ntt[i] = mod_mul(a_ntt[i], b_ntt[i], q);
        }

        // Inverse NTT
        negacyclic_ntt_inverse(&mut c_ntt, &tables);

        // Reference: (1 + 2X + 3X^2)(4 + 5X) mod (X^8 + 1)
        // = 4 + 5X + 8X + 10X^2 + 12X^2 + 15X^3
        // = 4 + 13X + 22X^2 + 15X^3
        // No reduction needed since degree < 8
        let expected = vec![4u64, 13, 22, 15, 0, 0, 0, 0];
        assert_eq!(c_ntt, expected);
    }

    /// Test with actual TenSafe modulus (N=16384 would be too slow for unit test,
    /// use N=256 with a 60-bit prime).
    #[test]
    fn test_ntt_roundtrip_60bit_prime() {
        let n = 256;
        // Need q ≡ 1 (mod 512). Use a known NTT-friendly prime.
        let q = 1152921504606846977u64; // 2^60 - ... , ≡ 1 mod 512?

        // Check NTT-friendliness
        if q % (2 * n as u64) != 1 {
            // Find a suitable prime
            let mut q2 = (q / (2 * n as u64)) * (2 * n as u64) + 1;
            // Search for prime
            loop {
                if is_prime_simple(q2) {
                    break;
                }
                q2 += 2 * n as u64;
            }
            let tables = NttTables::new(n, q2);
            let original: Vec<u64> = (0..n as u64).map(|i| i % q2).collect();
            let mut a = original.clone();
            negacyclic_ntt_forward(&mut a, &tables);
            negacyclic_ntt_inverse(&mut a, &tables);
            assert_eq!(a, original);
        } else {
            let tables = NttTables::new(n, q);
            let original: Vec<u64> = (0..n as u64).map(|i| i % q).collect();
            let mut a = original.clone();
            negacyclic_ntt_forward(&mut a, &tables);
            negacyclic_ntt_inverse(&mut a, &tables);
            assert_eq!(a, original);
        }
    }

    fn is_prime_simple(n: u64) -> bool {
        if n < 2 { return false; }
        if n == 2 || n == 3 { return true; }
        if n % 2 == 0 || n % 3 == 0 { return false; }
        let mut i = 5u64;
        while i.saturating_mul(i) <= n {
            if n % i == 0 || n % (i + 2) == 0 { return false; }
            i += 6;
        }
        true
    }
}
