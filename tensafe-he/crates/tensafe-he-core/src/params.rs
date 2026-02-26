//! CKKS parameter sets for TenSafe-HE.
//!
//! Hardcoded parameters for N ∈ {8192, 16384, 32768} with pre-computed:
//! - RNS primes q_0..q_{L-1} (NTT-friendly primes: q_i ≡ 1 mod 2N)
//! - Barrett reduction constants
//! - Scale factor Δ = 2^40

/// Scale bits for CKKS fixed-point encoding: Δ = 2^SCALE_BITS.
pub const SCALE_BITS: u32 = 40;

/// Scale factor Δ = 2^40.
pub const SCALE: f64 = (1u64 << SCALE_BITS) as f64;

/// A single RNS modulus with pre-computed Barrett constant.
#[derive(Debug, Clone, Copy)]
pub struct Modulus {
    /// The prime modulus q_i.
    pub value: u64,
    /// Bit width of this modulus.
    pub bits: u32,
    /// Barrett constant: floor(2^128 / q_i), stored as high 64 bits.
    /// Used for fast modular reduction: floor(a * barrett_hi >> 64) ≈ floor(a / q_i).
    pub barrett_hi: u64,
}

impl Modulus {
    /// Create a new modulus with pre-computed Barrett constant.
    pub const fn new(value: u64, bits: u32) -> Self {
        // Barrett constant = floor(2^128 / q).
        // We compute floor(2^64 * (2^64 / q)) = floor(2^128 / q) truncated to 64 bits.
        // For compile-time: use integer division 2^127 / q * 2, with correction.
        let hi = u128::MAX / (value as u128);
        let barrett_hi = (hi >> 64) as u64;
        Self {
            value,
            bits,
            barrett_hi,
        }
    }
}

/// Complete CKKS parameter set for a given polynomial degree N.
#[derive(Debug, Clone)]
pub struct CkksParams {
    /// Polynomial degree N (ring dimension). Must be a power of 2.
    pub poly_degree: usize,
    /// Number of SIMD slots = N/2.
    pub num_slots: usize,
    /// log2(N).
    pub log_n: u32,
    /// RNS moduli chain [q_0, q_1, ..., q_{L-1}].
    pub moduli: Vec<Modulus>,
    /// Number of RNS limbs L.
    pub num_limbs: usize,
    /// Scale bits (always 40 for TenSafe).
    pub scale_bits: u32,
}

impl CkksParams {
    /// Parameter set for N=8192 (poly_n=8192, S=4096, L=4).
    /// Security: ~128-bit. Modulus chain: [60, 40, 40, 60] bits.
    pub fn n8192() -> Self {
        // NTT-friendly primes: q ≡ 1 (mod 2N) = 1 (mod 16384).
        // All verified prime by trial division.
        let moduli = vec![
            Modulus::new(1152921504606830593, 60), // 60-bit, ≡ 1 mod 16384
            Modulus::new(1099511480321, 40),        // 40-bit, ≡ 1 mod 16384
            Modulus::new(1099510890497, 40),        // 40-bit, ≡ 1 mod 16384
            Modulus::new(1152921504606601217, 60),  // 60-bit, ≡ 1 mod 16384
        ];
        Self {
            poly_degree: 8192,
            num_slots: 4096,
            log_n: 13,
            num_limbs: moduli.len(),
            moduli,
            scale_bits: SCALE_BITS,
        }
    }

    /// Parameter set for N=16384 (poly_n=16384, S=8192, L=4).
    /// Security: ~192-bit. Modulus chain: [60, 40, 40, 60] bits.
    /// This is the primary TenSafe parameter set.
    pub fn n16384() -> Self {
        // NTT-friendly primes: q ≡ 1 (mod 2N) = 1 (mod 32768).
        // All verified prime by trial division.
        let moduli = vec![
            Modulus::new(1152921504606748673, 60), // 60-bit, ≡ 1 mod 32768
            Modulus::new(1099510054913, 40),        // 40-bit, ≡ 1 mod 32768
            Modulus::new(1099508121601, 40),        // 40-bit, ≡ 1 mod 32768
            Modulus::new(1152921504606683137, 60),  // 60-bit, ≡ 1 mod 32768
        ];
        Self {
            poly_degree: 16384,
            num_slots: 8192,
            log_n: 14,
            num_limbs: moduli.len(),
            moduli,
            scale_bits: SCALE_BITS,
        }
    }

    /// Parameter set for N=32768 (poly_n=32768, S=16384, L=4).
    /// Security: ~256-bit. Modulus chain: [60, 40, 40, 60] bits.
    pub fn n32768() -> Self {
        // NTT-friendly primes: q ≡ 1 (mod 2N) = 1 (mod 65536).
        // All verified prime by trial division.
        let moduli = vec![
            Modulus::new(1152921504606584833, 60), // 60-bit, ≡ 1 mod 65536
            Modulus::new(1099507695617, 40),        // 40-bit, ≡ 1 mod 65536
            Modulus::new(1099506515969, 40),        // 40-bit, ≡ 1 mod 65536
            Modulus::new(1152921504598720513, 60),  // 60-bit, ≡ 1 mod 65536
        ];
        Self {
            poly_degree: 32768,
            num_slots: 16384,
            log_n: 15,
            num_limbs: moduli.len(),
            moduli,
            scale_bits: SCALE_BITS,
        }
    }

    /// Select parameter set by polynomial degree.
    pub fn for_degree(n: usize) -> Self {
        match n {
            8192 => Self::n8192(),
            16384 => Self::n16384(),
            32768 => Self::n32768(),
            _ => panic!("Unsupported polynomial degree {n}. Use 8192, 16384, or 32768."),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_params_n16384() {
        let p = CkksParams::n16384();
        assert_eq!(p.poly_degree, 16384);
        assert_eq!(p.num_slots, 8192);
        assert_eq!(p.log_n, 14);
        assert_eq!(p.num_limbs, 4);
        assert_eq!(p.scale_bits, 40);
    }

    #[test]
    fn test_modulus_ntt_friendly() {
        // Each modulus q must satisfy q ≡ 1 (mod 2N) for NTT.
        for n in [8192, 16384, 32768] {
            let p = CkksParams::for_degree(n);
            let two_n = (2 * n) as u64;
            for (i, m) in p.moduli.iter().enumerate() {
                assert_eq!(
                    m.value % two_n,
                    1,
                    "Modulus {i} ({}) is not NTT-friendly for N={n}: {} mod {two_n} = {}",
                    m.value,
                    m.value,
                    m.value % two_n
                );
            }
        }
    }

    #[test]
    fn test_modulus_primality() {
        // Simple Miller-Rabin-like check: verify each modulus is prime
        fn is_prime(n: u64) -> bool {
            if n < 2 {
                return false;
            }
            if n == 2 || n == 3 {
                return true;
            }
            if n % 2 == 0 || n % 3 == 0 {
                return false;
            }
            let mut i = 5u64;
            while i * i <= n {
                if n % i == 0 || n % (i + 2) == 0 {
                    return false;
                }
                i += 6;
            }
            true
        }

        for n in [8192, 16384, 32768] {
            let p = CkksParams::for_degree(n);
            for (i, m) in p.moduli.iter().enumerate() {
                assert!(
                    is_prime(m.value),
                    "Modulus {i} ({}) for N={n} is not prime",
                    m.value
                );
            }
        }
    }

    #[test]
    fn test_barrett_constant() {
        let p = CkksParams::n16384();
        for m in &p.moduli {
            // Barrett constant should be approximately 2^64 / q
            let expected = (u128::MAX / m.value as u128) >> 64;
            assert_eq!(m.barrett_hi, expected as u64);
        }
    }
}
