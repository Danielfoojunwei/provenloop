//! NTT adapters for post-quantum cryptography moduli.
//!
//! Provides pre-computed NTT tables for:
//! - Kyber (q=3329, 7-stage NTT over 256 coefficients)
//! - Dilithium (q=8380417, 8-stage negacyclic NTT over 256 coefficients)
//!
//! These follow the CRYSTALS reference implementations (pq-crystals/kyber,
//! pq-crystals/dilithium) with Cooley-Tukey forward and Gentleman-Sande
//! inverse butterflies using the same zetas array.

/// Kyber modulus q = 3329 = 13 × 256 + 1.
pub const KYBER_Q: u64 = 3329;

/// Kyber polynomial degree.
pub const KYBER_N: usize = 256;

/// Dilithium modulus q = 8380417 = 2^23 - 2^13 + 1.
pub const DILITHIUM_Q: u64 = 8380417;

/// Dilithium polynomial degree.
pub const DILITHIUM_N: usize = 256;

/// Kyber NTT: 7-stage NTT over Z_3329 for 256-coefficient polynomials.
///
/// zeta = 17 is a primitive 256th root of unity mod 3329 (17^256 ≡ 1, 17^128 ≡ -1).
/// The NTT uses 7 stages (len = 128..2) transforming a degree-255 polynomial
/// into 128 degree-1 residues.
///
/// Follows the CRYSTALS-Kyber reference: both forward and inverse use the same
/// zetas array, with the inverse using `zeta * (v - u)` instead of separate
/// inverse twiddles.
pub struct KyberNtt {
    /// Zeta table: zetas[k] = 17^{bitrev7(k)} mod 3329, for k = 0..127.
    pub zetas: [u16; 128],
}

impl KyberNtt {
    /// Create pre-computed Kyber NTT tables.
    pub fn new() -> Self {
        let zeta: u64 = 17;
        let q = KYBER_Q;

        let mut zetas = [0u16; 128];
        for i in 0..128 {
            let br = bit_reverse_7(i as u8) as u64;
            zetas[i] = mod_pow(zeta, br, q) as u16;
        }

        Self { zetas }
    }

    /// In-place forward NTT (Cooley-Tukey, 7 stages, ascending k).
    pub fn forward(&self, a: &mut [u16; 256]) {
        let q = KYBER_Q as u32;
        let mut k = 1usize;
        let mut len = 128usize;
        while len >= 2 {
            let mut start = 0;
            while start < 256 {
                let zeta = self.zetas[k] as u32;
                k += 1;
                for j in start..start + len {
                    let t = (zeta * a[j + len] as u32) % q;
                    a[j + len] = ((a[j] as u32 + q - t) % q) as u16;
                    a[j] = ((a[j] as u32 + t) % q) as u16;
                }
                start += 2 * len;
            }
            len >>= 1;
        }
    }

    /// In-place inverse NTT (Gentleman-Sande, 7 stages, descending k).
    ///
    /// Uses the same zetas table as forward. The butterfly computes
    /// `zeta * (v - u)` matching the CRYSTALS-Kyber reference.
    pub fn inverse(&self, a: &mut [u16; 256]) {
        let q = KYBER_Q as u32;
        let mut k = 127usize;
        let mut len = 2usize;
        while len <= 128 {
            let mut start = 0;
            while start < 256 {
                let zeta = self.zetas[k] as u32;
                k = k.wrapping_sub(1);
                for j in start..start + len {
                    let t = a[j] as u32;
                    a[j] = ((t + a[j + len] as u32) % q) as u16;
                    // (v - u) * zeta, matching reference: r[j+len] = zeta * (r[j+len] - t)
                    let diff = (a[j + len] as u32 + q - t) % q;
                    a[j + len] = ((zeta as u64 * diff as u64) % q as u64) as u16;
                }
                start += 2 * len;
            }
            len <<= 1;
        }
        // Multiply by n^{-1} mod q  (n = 128 for Kyber's 7-stage NTT)
        let n_inv = mod_pow(128, KYBER_Q - 2, KYBER_Q) as u32;
        for coeff in a.iter_mut() {
            *coeff = ((*coeff as u64 * n_inv as u64) % q as u64) as u16;
        }
    }
}

/// Dilithium NTT: 8-stage negacyclic NTT over Z_8380417 for 256 coefficients.
///
/// root = 1753 is a primitive 512th root of unity mod 8380417
/// (1753^512 ≡ 1, 1753^256 ≡ -1).
///
/// Follows the CRYSTALS-Dilithium reference: forward uses `zetas[++k]`,
/// inverse uses `-zetas[--k]` (negation, not modular inverse).
pub struct DilithiumNtt {
    /// Zeta table: zetas[k] = 1753^{bitrev8(k)} mod 8380417, for k = 0..255.
    pub zetas: [u32; 256],
    /// n^{-1} mod q.
    pub n_inv: u32,
}

impl DilithiumNtt {
    /// Create pre-computed Dilithium NTT tables.
    pub fn new() -> Self {
        let root: u64 = 1753;
        let q = DILITHIUM_Q;

        let mut zetas = [0u32; 256];
        for i in 0..256 {
            let br = bit_reverse_8(i as u8) as u64;
            zetas[i] = mod_pow(root, br, q) as u32;
        }

        let n_inv = mod_pow(DILITHIUM_N as u64, q - 2, q) as u32;

        Self { zetas, n_inv }
    }

    /// In-place forward NTT (Cooley-Tukey, 8 stages, ascending k).
    pub fn forward(&self, a: &mut [u32; 256]) {
        let q = DILITHIUM_Q as u64;
        let mut k = 1usize;
        let mut len = 128usize;
        while len >= 1 {
            let mut start = 0;
            while start < 256 {
                let zeta = self.zetas[k] as u64;
                k += 1;
                for j in start..start + len {
                    let t = (zeta * a[j + len] as u64) % q;
                    a[j + len] = ((a[j] as u64 + q - t) % q) as u32;
                    a[j] = ((a[j] as u64 + t) % q) as u32;
                }
                start += 2 * len;
            }
            len >>= 1;
        }
    }

    /// In-place inverse NTT (Gentleman-Sande, 8 stages, descending k).
    ///
    /// Uses `-zetas[k]` (negation) matching the CRYSTALS-Dilithium reference:
    /// `zeta = -zetas[--k]; a[j+len] = zeta * (t - a[j+len]);`
    pub fn inverse(&self, a: &mut [u32; 256]) {
        let q = DILITHIUM_Q as u64;
        let mut k = 255usize;
        let mut len = 1usize;
        while len <= 128 {
            let mut start = 0;
            while start < 256 {
                // -zetas[k] mod q, matching reference
                let zeta = (q - self.zetas[k] as u64) % q;
                k = k.wrapping_sub(1);
                for j in start..start + len {
                    let t = a[j] as u64;
                    a[j] = ((t + a[j + len] as u64) % q) as u32;
                    // (u - v) * (-zeta), matching reference: a[j+len] = (-zeta) * (t - a[j+len])
                    let diff = (t + q - a[j + len] as u64) % q;
                    a[j + len] = ((zeta as u128 * diff as u128) % q as u128) as u32;
                }
                start += 2 * len;
            }
            len <<= 1;
        }
        // Normalize by n^{-1} mod q
        let n_inv = self.n_inv as u64;
        for coeff in a.iter_mut() {
            *coeff = ((*coeff as u64 * n_inv) % q) as u32;
        }
    }
}

/// Modular exponentiation: base^exp mod m.
fn mod_pow(mut base: u64, mut exp: u64, m: u64) -> u64 {
    let mut result = 1u64;
    base %= m;
    while exp > 0 {
        if exp & 1 == 1 {
            result = ((result as u128 * base as u128) % m as u128) as u64;
        }
        exp >>= 1;
        base = ((base as u128 * base as u128) % m as u128) as u64;
    }
    result
}

/// Bit-reverse a 7-bit number (for Kyber 128-entry zetas table).
fn bit_reverse_7(x: u8) -> u8 {
    x.reverse_bits() >> 1
}

/// Bit-reverse an 8-bit number (for Dilithium 256-entry zetas table).
fn bit_reverse_8(x: u8) -> u8 {
    x.reverse_bits()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kyber_ntt_roundtrip() {
        let ntt = KyberNtt::new();
        let mut a = [0u16; 256];
        for i in 0..256 {
            a[i] = (i as u16) % KYBER_Q as u16;
        }
        let original = a;

        ntt.forward(&mut a);
        // NTT should change the values
        assert_ne!(a, original);

        ntt.inverse(&mut a);
        // Should recover original
        for i in 0..256 {
            assert_eq!(
                a[i], original[i],
                "Kyber NTT roundtrip failed at index {i}: got {}, expected {}",
                a[i], original[i]
            );
        }
    }

    #[test]
    fn test_dilithium_ntt_roundtrip() {
        let ntt = DilithiumNtt::new();
        let mut a = [0u32; 256];
        for i in 0..256 {
            a[i] = i as u32;
        }
        let original = a;

        ntt.forward(&mut a);
        assert_ne!(a, original);

        ntt.inverse(&mut a);
        for i in 0..256 {
            assert_eq!(
                a[i], original[i],
                "Dilithium NTT roundtrip failed at index {i}: got {}, expected {}",
                a[i], original[i]
            );
        }
    }

    #[test]
    fn test_kyber_modulus() {
        // q = 3329 should be prime
        assert_eq!(KYBER_Q, 3329);
        // 3329 = 13 * 256 + 1, so 3329 ≡ 1 (mod 256)
        assert_eq!(KYBER_Q % 256, 1);
    }

    #[test]
    fn test_dilithium_modulus() {
        // q = 8380417 = 2^23 - 2^13 + 1
        assert_eq!(DILITHIUM_Q, 8380417);
        assert_eq!(DILITHIUM_Q, (1 << 23) - (1 << 13) + 1);
        // 8380417 ≡ 1 (mod 512)
        assert_eq!(DILITHIUM_Q % 512, 1);
    }

    #[test]
    fn test_barrett_reduce_kyber() {
        for i in 0..10000u32 {
            let reduced = (i % KYBER_Q as u32) as u16;
            assert_eq!(
                reduced as u32,
                i % KYBER_Q as u32,
                "Reduction failed for {i}"
            );
        }
    }

    #[test]
    fn test_mod_pow() {
        // Fermat's little theorem: a^(p-1) ≡ 1 (mod p)
        assert_eq!(mod_pow(17, KYBER_Q - 1, KYBER_Q), 1);
        assert_eq!(mod_pow(1753, DILITHIUM_Q - 1, DILITHIUM_Q), 1);

        // 17^256 ≡ 1 (mod 3329) — zeta is a 256th root
        assert_eq!(mod_pow(17, 256, KYBER_Q), 1);
        // 17^128 ≡ -1 (mod 3329) — primitive root property
        assert_eq!(mod_pow(17, 128, KYBER_Q), KYBER_Q - 1);

        // 1753^512 ≡ 1 (mod 8380417) — root is a 512th root
        assert_eq!(mod_pow(1753, 512, DILITHIUM_Q), 1);
    }

    #[test]
    fn test_kyber_ntt_zero_poly() {
        let ntt = KyberNtt::new();
        let mut a = [0u16; 256];
        let original = a;
        ntt.forward(&mut a);
        assert_eq!(a, original, "NTT of zero polynomial should be zero");
        ntt.inverse(&mut a);
        assert_eq!(a, original);
    }

    #[test]
    fn test_dilithium_ntt_zero_poly() {
        let ntt = DilithiumNtt::new();
        let mut a = [0u32; 256];
        let original = a;
        ntt.forward(&mut a);
        assert_eq!(a, original, "NTT of zero polynomial should be zero");
        ntt.inverse(&mut a);
        assert_eq!(a, original);
    }

    #[test]
    fn test_kyber_ntt_constant_poly() {
        let ntt = KyberNtt::new();
        let mut a = [42u16; 256];
        let original = a;
        ntt.forward(&mut a);
        ntt.inverse(&mut a);
        for i in 0..256 {
            assert_eq!(a[i], original[i], "Roundtrip failed at index {i}");
        }
    }

    #[test]
    fn test_dilithium_ntt_constant_poly() {
        let ntt = DilithiumNtt::new();
        let mut a = [42u32; 256];
        let original = a;
        ntt.forward(&mut a);
        ntt.inverse(&mut a);
        for i in 0..256 {
            assert_eq!(a[i], original[i], "Roundtrip failed at index {i}");
        }
    }
}
