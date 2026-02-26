//! RNS (Residue Number System) modular arithmetic.
//!
//! All CKKS operations decompose into L independent sub-operations on 64-bit integers.
//! This module provides the fundamental modular arithmetic primitives.

use crate::params::Modulus;

/// Modular addition: (a + b) mod q.
/// Assumes a, b < q.
#[inline(always)]
pub fn mod_add(a: u64, b: u64, q: u64) -> u64 {
    let sum = a + b;
    if sum >= q {
        sum - q
    } else {
        sum
    }
}

/// Modular subtraction: (a - b) mod q.
/// Assumes a, b < q.
#[inline(always)]
pub fn mod_sub(a: u64, b: u64, q: u64) -> u64 {
    if a >= b {
        a - b
    } else {
        q - b + a
    }
}

/// Modular multiplication: (a * b) mod q using 128-bit intermediate.
/// Assumes a, b < q < 2^63.
#[inline(always)]
pub fn mod_mul(a: u64, b: u64, q: u64) -> u64 {
    ((a as u128 * b as u128) % q as u128) as u64
}

/// Barrett reduction: reduce a value < q^2 to [0, q).
/// Uses pre-computed Barrett constant from Modulus.
/// For a < q^2: result = a mod q.
#[inline(always)]
pub fn barrett_reduce(a: u128, m: &Modulus) -> u64 {
    let q = m.value as u128;
    // Approximate quotient: (a * barrett) >> 128
    // barrett ≈ 2^128 / q, stored as barrett_hi = floor(2^128 / q) >> 64
    let approx_quot = ((a >> 64) * m.barrett_hi as u128) >> 64;
    let mut r = (a - approx_quot * q) as u64;
    // At most 2 corrections needed
    if r >= m.value {
        r -= m.value;
    }
    if r >= m.value {
        r -= m.value;
    }
    r
}

/// Modular multiplication using Barrett reduction.
/// More efficient than mod_mul when the modulus is known at compile time.
#[inline(always)]
pub fn mod_mul_barrett(a: u64, b: u64, m: &Modulus) -> u64 {
    let product = a as u128 * b as u128;
    barrett_reduce(product, m)
}

/// Modular exponentiation: a^exp mod q.
/// Used for computing primitive roots and inverses.
pub fn mod_pow(mut base: u64, mut exp: u64, q: u64) -> u64 {
    let mut result: u64 = 1;
    base %= q;
    while exp > 0 {
        if exp & 1 == 1 {
            result = mod_mul(result, base, q);
        }
        exp >>= 1;
        base = mod_mul(base, base, q);
    }
    result
}

/// Modular inverse: a^{-1} mod q using Fermat's little theorem.
/// Requires q to be prime: a^{-1} = a^{q-2} mod q.
pub fn mod_inv(a: u64, q: u64) -> u64 {
    assert!(a != 0, "Cannot invert zero");
    mod_pow(a, q - 2, q)
}

/// Find a primitive 2N-th root of unity modulo q.
/// For NTT on the ring Z_q[X]/(X^N+1), we need ψ such that ψ^{2N} ≡ 1 (mod q)
/// and ψ^N ≡ -1 (mod q).
///
/// Requires: q ≡ 1 (mod 2N).
pub fn find_primitive_root(n: usize, q: u64) -> u64 {
    let two_n = (2 * n) as u64;
    assert_eq!(
        q % two_n,
        1,
        "q={q} is not NTT-friendly: q mod 2N = {} (expected 1)",
        q % two_n
    );

    // q-1 = 2N * k for some k. A generator g of Z_q* has order q-1.
    // ψ = g^((q-1)/(2N)) is a primitive 2N-th root of unity.
    let exponent = (q - 1) / two_n;

    // Try small generators until we find one that works
    for g in 2..q {
        let psi = mod_pow(g, exponent, q);

        // Verify: ψ^N ≡ -1 (mod q) (ensures it's a *primitive* 2N-th root)
        let psi_n = mod_pow(psi, n as u64, q);
        if psi_n == q - 1 {
            return psi;
        }
    }
    panic!("No primitive 2N-th root of unity found for q={q}, N={n}");
}

/// Compute NTT twiddle factors (powers of ψ in bit-reversed order).
///
/// For the negacyclic NTT on Z_q[X]/(X^N+1):
/// - Forward twiddle[i] = ψ^{bit_reverse(i)} for Cooley-Tukey butterfly
/// - These are pre-computed once per parameter set
pub fn compute_twiddle_factors(n: usize, psi: u64, q: u64) -> Vec<u64> {
    let mut twiddles = vec![0u64; n];

    // Compute powers of ψ in bit-reversed order
    let log_n = (n as f64).log2() as u32;
    for i in 0..n {
        let rev = bit_reverse(i as u32, log_n) as usize;
        twiddles[rev] = mod_pow(psi, i as u64, q);
    }
    twiddles
}

/// Compute inverse NTT twiddle factors.
/// inv_twiddle[i] = ψ^{-bit_reverse(i)} mod q.
pub fn compute_inv_twiddle_factors(n: usize, psi: u64, q: u64) -> Vec<u64> {
    let psi_inv = mod_inv(psi, q);
    let mut twiddles = vec![0u64; n];

    let log_n = (n as f64).log2() as u32;
    for i in 0..n {
        let rev = bit_reverse(i as u32, log_n) as usize;
        twiddles[rev] = mod_pow(psi_inv, i as u64, q);
    }
    twiddles
}

/// Bit-reverse an integer of given bit width.
#[inline]
pub fn bit_reverse(mut x: u32, bits: u32) -> u32 {
    let mut result = 0u32;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

/// An RNS polynomial: L independent coefficient vectors, one per modulus.
#[derive(Debug, Clone)]
pub struct RnsPoly {
    /// Coefficients for each RNS limb: limbs[l][i] = coefficient i mod q_l.
    pub limbs: Vec<Vec<u64>>,
    /// Polynomial degree N.
    pub n: usize,
}

impl RnsPoly {
    /// Create a zero polynomial with L limbs of N coefficients each.
    pub fn zero(n: usize, num_limbs: usize) -> Self {
        Self {
            limbs: vec![vec![0u64; n]; num_limbs],
            n,
        }
    }

    /// Element-wise addition of two RNS polynomials.
    pub fn add(&self, other: &Self, moduli: &[Modulus]) -> Self {
        assert_eq!(self.n, other.n);
        let mut result = Self::zero(self.n, self.limbs.len());
        for l in 0..self.limbs.len() {
            let q = moduli[l].value;
            for i in 0..self.n {
                result.limbs[l][i] = mod_add(self.limbs[l][i], other.limbs[l][i], q);
            }
        }
        result
    }

    /// Element-wise subtraction of two RNS polynomials.
    pub fn sub(&self, other: &Self, moduli: &[Modulus]) -> Self {
        assert_eq!(self.n, other.n);
        let mut result = Self::zero(self.n, self.limbs.len());
        for l in 0..self.limbs.len() {
            let q = moduli[l].value;
            for i in 0..self.n {
                result.limbs[l][i] = mod_sub(self.limbs[l][i], other.limbs[l][i], q);
            }
        }
        result
    }

    /// Element-wise (Hadamard) multiplication — used for NTT-domain products.
    pub fn hadamard_mul(&self, other: &Self, moduli: &[Modulus]) -> Self {
        assert_eq!(self.n, other.n);
        let mut result = Self::zero(self.n, self.limbs.len());
        for l in 0..self.limbs.len() {
            let q = moduli[l].value;
            for i in 0..self.n {
                result.limbs[l][i] = mod_mul(self.limbs[l][i], other.limbs[l][i], q);
            }
        }
        result
    }

    /// Negate all coefficients: result[i] = -self[i] mod q.
    pub fn negate(&self, moduli: &[Modulus]) -> Self {
        let mut result = Self::zero(self.n, self.limbs.len());
        for l in 0..self.limbs.len() {
            let q = moduli[l].value;
            for i in 0..self.n {
                if self.limbs[l][i] == 0 {
                    result.limbs[l][i] = 0;
                } else {
                    result.limbs[l][i] = q - self.limbs[l][i];
                }
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mod_add() {
        let q = 17u64;
        assert_eq!(mod_add(5, 7, q), 12);
        assert_eq!(mod_add(10, 10, q), 3); // 20 mod 17
        assert_eq!(mod_add(0, 0, q), 0);
        assert_eq!(mod_add(16, 1, q), 0); // 17 mod 17
    }

    #[test]
    fn test_mod_sub() {
        let q = 17u64;
        assert_eq!(mod_sub(10, 3, q), 7);
        assert_eq!(mod_sub(3, 10, q), 10); // -7 mod 17 = 10
        assert_eq!(mod_sub(0, 0, q), 0);
    }

    #[test]
    fn test_mod_mul() {
        let q = 17u64;
        assert_eq!(mod_mul(5, 3, q), 15);
        assert_eq!(mod_mul(5, 4, q), 3); // 20 mod 17
        assert_eq!(mod_mul(0, 10, q), 0);
    }

    #[test]
    fn test_mod_pow() {
        let q = 17u64;
        assert_eq!(mod_pow(2, 0, q), 1);
        assert_eq!(mod_pow(2, 4, q), 16); // 16 mod 17
        assert_eq!(mod_pow(3, 16, q), 1); // Fermat's little theorem
    }

    #[test]
    fn test_mod_inv() {
        let q = 17u64;
        for a in 1..q {
            let inv = mod_inv(a, q);
            assert_eq!(mod_mul(a, inv, q), 1, "Inverse of {a} mod {q} failed");
        }
    }

    #[test]
    fn test_bit_reverse() {
        assert_eq!(bit_reverse(0b0000, 4), 0b0000);
        assert_eq!(bit_reverse(0b0001, 4), 0b1000);
        assert_eq!(bit_reverse(0b0110, 4), 0b0110);
        assert_eq!(bit_reverse(0b1010, 4), 0b0101);
    }

    #[test]
    fn test_primitive_root() {
        // Small test: N=4, q=17 (17 ≡ 1 mod 8)
        let n = 4;
        let q = 17u64;
        let psi = find_primitive_root(n, q);

        // ψ^{2N} = ψ^8 ≡ 1 (mod 17)
        assert_eq!(mod_pow(psi, 8, q), 1);
        // ψ^N = ψ^4 ≡ -1 (mod 17)
        assert_eq!(mod_pow(psi, 4, q), q - 1);
    }

    #[test]
    fn test_primitive_root_large() {
        // Test with an actual TenSafe modulus
        use crate::params::CkksParams;
        let params = CkksParams::n16384();
        let q = params.moduli[0].value;
        let n = params.poly_degree;
        let psi = find_primitive_root(n, q);

        // ψ^{2N} ≡ 1 (mod q)
        assert_eq!(mod_pow(psi, (2 * n) as u64, q), 1);
        // ψ^N ≡ -1 (mod q)
        assert_eq!(mod_pow(psi, n as u64, q), q - 1);
    }

    #[test]
    fn test_rns_poly_add() {
        use crate::params::CkksParams;
        let params = CkksParams::n16384();
        let n = 4; // small test
        let moduli = &params.moduli[..2]; // just use 2 limbs

        let mut a = RnsPoly::zero(n, 2);
        let mut b = RnsPoly::zero(n, 2);
        a.limbs[0] = vec![1, 2, 3, 4];
        b.limbs[0] = vec![5, 6, 7, 8];
        a.limbs[1] = vec![10, 20, 30, 40];
        b.limbs[1] = vec![50, 60, 70, 80];

        let c = a.add(&b, moduli);
        assert_eq!(c.limbs[0], vec![6, 8, 10, 12]);
        assert_eq!(c.limbs[1], vec![60, 80, 100, 120]);
    }
}
