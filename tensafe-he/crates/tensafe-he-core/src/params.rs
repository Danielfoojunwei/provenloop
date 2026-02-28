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
    /// Barrett constant: floor(2^128 / q_i), stored as two 64-bit words.
    /// barrett_hi is the high word, barrett_lo is the low word.
    /// Together they represent the full ~68-bit constant needed for 60-bit moduli.
    pub barrett_hi: u64,
    /// Low 64 bits of floor(2^128 / q_i).
    pub barrett_lo: u64,
}

impl Modulus {
    /// Create a new modulus with pre-computed Barrett constant.
    pub const fn new(value: u64, bits: u32) -> Self {
        // Barrett constant = floor(2^128 / q), stored as (hi, lo) pair.
        // For q ≈ 2^60, this is about 2^68 which needs both words.
        let full = u128::MAX / (value as u128);
        let barrett_hi = (full >> 64) as u64;
        let barrett_lo = full as u64;
        Self {
            value,
            bits,
            barrett_hi,
            barrett_lo,
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

    /// Build a custom parameter set with given modulus bit widths.
    ///
    /// Automatically finds NTT-friendly primes for each width.
    /// This allows runtime selection of depth/security trade-offs.
    pub fn custom(poly_degree: usize, modulus_bits: &[u32], scale_bits: u32) -> Self {
        assert!(poly_degree.is_power_of_two(), "poly_degree must be a power of 2");
        let two_n = (2 * poly_degree) as u64;

        let moduli: Vec<Modulus> = modulus_bits
            .iter()
            .map(|&bits| {
                let q = find_ntt_friendly_prime(bits, two_n);
                Modulus::new(q, bits)
            })
            .collect();

        Self {
            poly_degree,
            num_slots: poly_degree / 2,
            log_n: poly_degree.trailing_zeros(),
            num_limbs: moduli.len(),
            moduli,
            scale_bits,
        }
    }
}

/// Find a prime q with `bits` bit-width such that q ≡ 1 (mod two_n).
///
/// Searches downward from the largest `bits`-bit number for primes
/// that are NTT-friendly (q - 1 divisible by 2N).
fn find_ntt_friendly_prime(bits: u32, two_n: u64) -> u64 {
    let upper = if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };
    let lower = 1u64 << (bits - 1);

    // Start from the largest candidate ≡ 1 mod two_n
    let mut candidate = upper - (upper % two_n) + 1;
    if candidate > upper {
        candidate -= two_n;
    }

    while candidate >= lower {
        if is_prime_u64(candidate) {
            return candidate;
        }
        candidate = candidate.wrapping_sub(two_n);
        if candidate < lower {
            break;
        }
    }
    panic!("No {bits}-bit NTT-friendly prime found for 2N={two_n}");
}

/// 30-bit RNS modulus for 32-bit NTT (Cheddar-style rational rescaling).
///
/// 32-bit NTT with signed Montgomery reduction offers ~2× throughput on
/// GPUs (native 32-bit integer units, halved shared memory usage).
///
/// Uses more primes (25-30 × 30-bit ≈ 750-900 bits total, vs 4 × 50-bit
/// ≈ 200 bits for 64-bit). The extra limbs are compensated by ~2× faster
/// NTT per limb.
///
/// Montgomery constant: R = 2^32, q_inv = -q^{-1} mod R.
#[derive(Debug, Clone, Copy)]
pub struct Modulus32 {
    /// The prime modulus q (≤ 30 bits).
    pub value: u32,
    /// Montgomery constant: q_inv = -q^{-1} mod 2^32.
    pub q_inv: u32,
    /// R^2 mod q = (2^32)^2 mod q, for converting to Montgomery form.
    pub r2: u32,
}

impl Modulus32 {
    /// Create a 32-bit modulus with pre-computed Montgomery constants.
    pub const fn new(value: u32) -> Self {
        // Compute q_inv = -q^{-1} mod 2^32 via Newton's method.
        // q^{-1} mod 2^k: start with q (since q*q ≡ q^2 ≡ 1 mod 2 for odd q)
        // Lift: inv *= 2 - q * inv, doubling precision each step.
        let q = value as u64;
        let mut inv = value as u64; // q^{-1} mod 2
        // 5 iterations: 1 → 2 → 4 → 8 → 16 → 32 bits of precision
        inv = inv.wrapping_mul(2u64.wrapping_sub(q.wrapping_mul(inv)));
        inv = inv.wrapping_mul(2u64.wrapping_sub(q.wrapping_mul(inv)));
        inv = inv.wrapping_mul(2u64.wrapping_sub(q.wrapping_mul(inv)));
        inv = inv.wrapping_mul(2u64.wrapping_sub(q.wrapping_mul(inv)));
        inv = inv.wrapping_mul(2u64.wrapping_sub(q.wrapping_mul(inv)));
        let q_inv = (0u64.wrapping_sub(inv)) as u32; // -q^{-1} mod 2^32

        // R^2 mod q = (2^32)^2 mod q = 2^64 mod q
        let r2 = ((1u128 << 64) % value as u128) as u32;

        Self { value, q_inv, r2 }
    }
}

/// 32-bit RNS parameter set for fast GPU NTT (Cheddar-inspired).
///
/// Uses 25-30 30-bit NTT-friendly primes with signed Montgomery reduction.
/// Each prime q satisfies q ≡ 1 (mod 2N) and q < 2^30.
#[derive(Debug, Clone)]
pub struct CkksParams32 {
    /// Polynomial degree N.
    pub poly_degree: usize,
    /// Number of SIMD slots = N/2.
    pub num_slots: usize,
    /// log2(N).
    pub log_n: u32,
    /// 30-bit RNS moduli chain.
    pub moduli: Vec<Modulus32>,
    /// Number of RNS limbs.
    pub num_limbs: usize,
    /// Scale bits.
    pub scale_bits: u32,
}

impl CkksParams32 {
    /// 30-bit parameter set for N=16384 (25 primes × 30 bits ≈ 750-bit modulus).
    ///
    /// NTT-friendly 30-bit primes: q ≡ 1 (mod 32768), q < 2^30.
    pub fn n16384() -> Self {
        // Verified primes < 2^30 with q ≡ 1 (mod 32768).
        // Generated by searching downward from 2^30 and testing primality.
        // NOTE: The original list (simple arithmetic decrement) contained 21
        // composite values out of 25. These are individually verified primes.
        let primes: Vec<u32> = vec![
            1073643521, // verified prime, ≡ 1 mod 32768
            1073479681,
            1073184769,
            1073053697,
            1072857089,
            1072496641,
            1071513601,
            1071415297,
            1071087617,
            1070727169,
            1070432257,
            1069219841,
            1068564481,
            1068466177,
            1068433409,
            1068236801,
            1067876353,
            1067548673,
            1066893313,
            1066172417,
            1065811969,
            1065779201,
            1065484289,
            1064697857,
            1064599553,
        ];

        // Validate every candidate: must be prime AND NTT-friendly (q ≡ 1 mod 2N).
        let two_n: u32 = 2 * 16384; // 32768
        let mut validated_primes = Vec::with_capacity(primes.len());
        for &p in &primes {
            assert!(
                is_prime_u32(p),
                "CkksParams32: value {} is NOT prime — remove or replace it",
                p
            );
            assert_eq!(
                p % two_n, 1,
                "CkksParams32: prime {} is NOT NTT-friendly ({} mod {} = {}, expected 1)",
                p, p, two_n, p % two_n
            );
            assert!(
                p < (1u32 << 30),
                "CkksParams32: prime {} exceeds 30-bit limit (max {})",
                p, (1u32 << 30) - 1
            );
            validated_primes.push(p);
        }

        let moduli: Vec<Modulus32> = validated_primes.into_iter().map(Modulus32::new).collect();
        let num_limbs = moduli.len();

        Self {
            poly_degree: 16384,
            num_slots: 8192,
            log_n: 14,
            moduli,
            num_limbs,
            scale_bits: SCALE_BITS,
        }
    }
}

/// Simple primality test for 32-bit values (used by Modulus32 validation).
pub fn is_prime_u32(n: u32) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 || n == 3 {
        return true;
    }
    if n % 2 == 0 || n % 3 == 0 {
        return false;
    }
    let mut i = 5u32;
    while i.saturating_mul(i) <= n {
        if n % i == 0 || n % (i + 2) == 0 {
            return false;
        }
        i += 6;
    }
    true
}

/// Simple primality test sufficient for 60-bit moduli.
fn is_prime_u64(n: u64) -> bool {
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
    while i.saturating_mul(i) <= n {
        if n % i == 0 || n % (i + 2) == 0 {
            return false;
        }
        i += 6;
    }
    true
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

    #[test]
    fn test_custom_params() {
        let p = CkksParams::custom(16384, &[60, 40, 40, 60], 40);
        assert_eq!(p.poly_degree, 16384);
        assert_eq!(p.num_slots, 8192);
        assert_eq!(p.num_limbs, 4);
        assert_eq!(p.log_n, 14);

        // Verify all found primes are NTT-friendly
        let two_n = (2 * p.poly_degree) as u64;
        for (i, m) in p.moduli.iter().enumerate() {
            assert_eq!(
                m.value % two_n, 1,
                "Custom modulus {i} is not NTT-friendly"
            );
        }
    }

    // ==========================================================
    // CkksParams32 / Modulus32 tests
    // ==========================================================

    #[test]
    fn test_params32_n16384_construction() {
        // This will panic if ANY of the hard-coded primes fail
        // the primality or NTT-friendliness checks added to n16384().
        let p = CkksParams32::n16384();
        assert_eq!(p.poly_degree, 16384);
        assert_eq!(p.num_slots, 8192);
        assert_eq!(p.log_n, 14);
        assert_eq!(p.scale_bits, SCALE_BITS);
        assert!(p.num_limbs >= 20, "Expected at least 20 primes, got {}", p.num_limbs);
    }

    #[test]
    fn test_params32_all_primes_prime() {
        let p = CkksParams32::n16384();
        for (i, m) in p.moduli.iter().enumerate() {
            assert!(
                is_prime_u32(m.value),
                "Modulus32[{i}] value {} is NOT prime",
                m.value
            );
        }
    }

    #[test]
    fn test_params32_all_ntt_friendly() {
        let p = CkksParams32::n16384();
        let two_n: u32 = 2 * 16384;
        for (i, m) in p.moduli.iter().enumerate() {
            assert_eq!(
                m.value % two_n, 1,
                "Modulus32[{i}] ({}) is not NTT-friendly: {} mod {} = {}",
                m.value, m.value, two_n, m.value % two_n
            );
        }
    }

    #[test]
    fn test_params32_all_under_30_bits() {
        let p = CkksParams32::n16384();
        let limit = 1u32 << 30;
        for (i, m) in p.moduli.iter().enumerate() {
            assert!(
                m.value < limit,
                "Modulus32[{i}] ({}) exceeds 30-bit limit ({})",
                m.value, limit
            );
        }
    }

    #[test]
    fn test_modulus32_montgomery_constants() {
        let p = CkksParams32::n16384();
        for (i, m) in p.moduli.iter().enumerate() {
            let q = m.value as u64;
            // Verify q_inv: q * (-q_inv) ≡ -1 (mod 2^32)
            // Equivalently: q * q_inv ≡ 2^32 - 1 ... no.
            // Actually: -q^{-1} mod 2^32 means q * q_inv ≡ -1 (mod 2^32)
            // i.e. (q as u32).wrapping_mul(q_inv) should == u32::MAX (which is -1 mod 2^32)
            let product = (m.value).wrapping_mul(m.q_inv);
            assert_eq!(
                product,
                u32::MAX,
                "Modulus32[{i}]: q*q_inv mod 2^32 = {}, expected {} (-1 mod 2^32). q={}, q_inv={}",
                product, u32::MAX, m.value, m.q_inv
            );

            // Verify r2: R^2 mod q = 2^64 mod q
            let expected_r2 = ((1u128 << 64) % q as u128) as u32;
            assert_eq!(
                m.r2, expected_r2,
                "Modulus32[{i}]: r2={}, expected {} (2^64 mod {})",
                m.r2, expected_r2, m.value
            );
        }
    }

    #[test]
    fn test_params32_primes_are_distinct() {
        let p = CkksParams32::n16384();
        let mut values: Vec<u32> = p.moduli.iter().map(|m| m.value).collect();
        let orig_len = values.len();
        values.sort();
        values.dedup();
        assert_eq!(
            values.len(), orig_len,
            "CkksParams32 has duplicate primes"
        );
    }

    #[test]
    fn test_modulus32_odd_value_required() {
        // Montgomery form requires odd modulus. All NTT-friendly primes > 2 are odd.
        let p = CkksParams32::n16384();
        for (i, m) in p.moduli.iter().enumerate() {
            assert!(
                m.value % 2 == 1,
                "Modulus32[{i}] ({}) is even — Montgomery form requires odd modulus",
                m.value
            );
        }
    }
}
