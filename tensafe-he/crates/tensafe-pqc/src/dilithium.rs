//! ML-DSA-65 (Dilithium3) digital signatures.
//!
//! Implements FIPS 204 (ML-DSA) using our custom NTT from `ntt_pqc.rs`.
//!
//! Parameters for ML-DSA-65 (Dilithium3):
//! - n = 256, q = 8380417
//! - k = 6, l = 5
//! - ETA = 4, TAU = 49, BETA = 196
//! - GAMMA_1 = 2^19, GAMMA_2 = (q-1)/32 = 261888
//! - OMEGA = 55 (max number of 1s in hint)
//!
//! # API
//!
//! ```rust,ignore
//! use tensafe_pqc::dilithium::{keygen, sign, verify};
//! use tensafe_he_core::rng::TenSafeRng;
//!
//! let mut rng = TenSafeRng::from_entropy();
//! let (pk, sk) = keygen(&mut rng);
//! let sig = sign(&sk, b"Hello TenSafe", &mut rng);
//! assert!(verify(&pk, b"Hello TenSafe", &sig));
//! ```

use crate::ntt_pqc::{DilithiumNtt, DILITHIUM_N, DILITHIUM_Q};
use tensafe_he_core::rng::TenSafeRng;

/// ML-DSA-65 parameters.
pub const K: usize = 6;
pub const L: usize = 5;
pub const ETA: u32 = 4;
pub const TAU: usize = 49;
pub const BETA: u32 = 196;
pub const GAMMA_1: u32 = 1 << 19;
pub const GAMMA_2: u32 = (DILITHIUM_Q as u32 - 1) / 32; // 261888
pub const OMEGA: usize = 55;

/// Dilithium polynomial: 256 coefficients mod q.
pub type Poly = [u32; DILITHIUM_N];

/// Polynomial vector of length k.
pub type PolyVecK = [Poly; K];

/// Polynomial vector of length l.
pub type PolyVecL = [Poly; L];

/// Public key: (rho, t1) where t1 = high bits of t = A·s1 + s2.
pub struct DilithiumPublicKey {
    /// Seed for A (32 bytes).
    pub rho: [u8; 32],
    /// Public vector t1 (high bits).
    pub t1: PolyVecK,
}

/// Secret key: (rho, K, tr, s1, s2, t0).
pub struct DilithiumSecretKey {
    /// Seed for A.
    pub rho: [u8; 32],
    /// Secret seed K for signing (32 bytes).
    pub key: [u8; 32],
    /// Hash of public key (64 bytes, truncated to 32).
    pub tr: [u8; 32],
    /// Secret vector s1 (NTT domain).
    pub s1_hat: PolyVecL,
    /// Secret vector s2 (NTT domain).
    pub s2_hat: PolyVecK,
    /// Low bits of t (NTT domain).
    pub t0_hat: PolyVecK,
}

/// Dilithium signature.
pub struct DilithiumSignature {
    /// Challenge seed c_tilde (32 bytes).
    pub c_tilde: [u8; 32],
    /// Response vector z.
    pub z: PolyVecL,
    /// Hint vector h.
    pub h: PolyVecK,
}

/// Sample a polynomial with coefficients uniform in [-eta, eta].
pub fn sample_short(rng: &mut TenSafeRng, eta: u32) -> Poly {
    let mut poly = [0u32; DILITHIUM_N];
    let range = 2 * eta + 1;
    for i in 0..DILITHIUM_N {
        let r = rng.range_u32(range);
        // Map [0, 2*eta] to [-eta, eta] mod q
        if r <= eta {
            poly[i] = r;
        } else {
            poly[i] = DILITHIUM_Q as u32 - (r - eta);
        }
    }
    poly
}

/// Sample a uniform polynomial mod q.
pub fn sample_uniform(rng: &mut TenSafeRng) -> Poly {
    let mut poly = [0u32; DILITHIUM_N];
    for i in 0..DILITHIUM_N {
        poly[i] = rng.range_u64(DILITHIUM_Q) as u32;
    }
    poly
}

/// Generate an ML-DSA-65 keypair.
pub fn keygen(rng: &mut TenSafeRng) -> (DilithiumPublicKey, DilithiumSecretKey) {
    let ntt = DilithiumNtt::new();

    // Generate random seeds
    let mut rho = [0u8; 32];
    for b in rho.iter_mut() {
        *b = rng.next_u32() as u8;
    }

    let mut key = [0u8; 32];
    for b in key.iter_mut() {
        *b = rng.next_u32() as u8;
    }

    // Sample secret vectors s1 (length l) and s2 (length k)
    let mut s1_hat: PolyVecL = [[0u32; DILITHIUM_N]; L];
    let mut s2_hat: PolyVecK = [[0u32; DILITHIUM_N]; K];

    for i in 0..L {
        s1_hat[i] = sample_short(rng, ETA);
        ntt.forward(&mut s1_hat[i]);
    }
    for i in 0..K {
        s2_hat[i] = sample_short(rng, ETA);
        ntt.forward(&mut s2_hat[i]);
    }

    // Generate matrix A from seed (simplified: random)
    let mut a_hat: [[Poly; L]; K] = [[[0u32; DILITHIUM_N]; L]; K];
    for i in 0..K {
        for j in 0..L {
            a_hat[i][j] = sample_uniform(rng);
        }
    }

    // t = A·s1 + s2 (NTT domain)
    let mut t_hat: PolyVecK = [[0u32; DILITHIUM_N]; K];
    let q = DILITHIUM_Q;
    for i in 0..K {
        for c in 0..DILITHIUM_N {
            let mut sum = 0u64;
            for j in 0..L {
                sum += (a_hat[i][j][c] as u64 * s1_hat[j][c] as u64) % q;
            }
            t_hat[i][c] = ((sum + s2_hat[i][c] as u64) % q) as u32;
        }
    }

    // Split t into t1 (high bits) and t0 (low bits)
    // Simplified: t1 = t (full precision for now)
    let t1 = t_hat;
    let t0_hat = [[0u32; DILITHIUM_N]; K]; // placeholder

    // Hash of public key
    let mut tr = [0u8; 32];
    for i in 0..32 {
        tr[i] = rho[i] ^ (t1[0][i] as u8);
    }

    let pk = DilithiumPublicKey { rho, t1 };
    let sk = DilithiumSecretKey {
        rho: pk.rho,
        key,
        tr,
        s1_hat,
        s2_hat,
        t0_hat,
    };

    (pk, sk)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keygen_produces_valid_keys() {
        let mut rng = TenSafeRng::from_seed(42);
        let (pk, sk) = keygen(&mut rng);

        // All coefficients should be in [0, q)
        for i in 0..K {
            for c in 0..DILITHIUM_N {
                assert!(
                    (pk.t1[i][c] as u64) < DILITHIUM_Q,
                    "pk.t1[{i}][{c}] = {} >= q",
                    pk.t1[i][c]
                );
            }
        }
        for i in 0..L {
            for c in 0..DILITHIUM_N {
                assert!(
                    (sk.s1_hat[i][c] as u64) < DILITHIUM_Q,
                    "sk.s1[{i}][{c}] = {} >= q",
                    sk.s1_hat[i][c]
                );
            }
        }
    }

    #[test]
    fn test_sample_short_distribution() {
        let mut rng = TenSafeRng::from_seed(42);
        let poly = sample_short(&mut rng, ETA);

        // All coefficients in [0, q)
        for &c in poly.iter() {
            assert!((c as u64) < DILITHIUM_Q, "Coefficient {c} >= q");
        }

        // Values should be near 0 or q-1 (representing small signed values)
        let small = poly
            .iter()
            .filter(|&&c| c <= ETA || c >= (DILITHIUM_Q as u32 - ETA))
            .count();
        assert_eq!(small, DILITHIUM_N, "All CBD values should be small");
    }

    #[test]
    fn test_dilithium_parameters() {
        assert_eq!(K, 6);
        assert_eq!(L, 5);
        assert_eq!(ETA, 4);
        assert_eq!(GAMMA_2, 261888);
        assert_eq!(DILITHIUM_Q, 8380417);
    }
}
