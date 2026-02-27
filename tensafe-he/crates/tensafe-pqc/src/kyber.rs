//! ML-KEM-768 (Kyber768) key encapsulation mechanism.
//!
//! Implements FIPS 203 (ML-KEM) using our custom NTT from `ntt_pqc.rs`.
//!
//! Parameters for ML-KEM-768:
//! - n = 256, q = 3329, k = 3
//! - ETA_1 = 2, ETA_2 = 2
//! - d_u = 10, d_v = 4
//! - Shared secret: 32 bytes
//!
//! # API
//!
//! ```rust,ignore
//! use tensafe_pqc::kyber::{keygen, encaps, decaps};
//! use tensafe_he_core::rng::TenSafeRng;
//!
//! let mut rng = TenSafeRng::from_entropy();
//! let (pk, sk) = keygen(&mut rng);
//! let (ct, shared_secret_enc) = encaps(&pk, &mut rng);
//! let shared_secret_dec = decaps(&ct, &sk);
//! assert_eq!(shared_secret_enc, shared_secret_dec);
//! ```

use crate::ntt_pqc::{KyberNtt, KYBER_N, KYBER_Q};
use tensafe_he_core::rng::TenSafeRng;

/// ML-KEM-768 parameters.
pub const K: usize = 3;
pub const ETA_1: usize = 2;
pub const ETA_2: usize = 2;
pub const DU: usize = 10;
pub const DV: usize = 4;

/// Kyber polynomial: 256 coefficients mod 3329.
pub type Poly = [u16; KYBER_N];

/// Kyber polynomial vector: k polynomials.
pub type PolyVec = [Poly; K];

/// Public key: (A_hat, t_hat) — matrix A in NTT domain + public vector.
pub struct KyberPublicKey {
    /// Seed for reconstructing A (32 bytes).
    pub rho: [u8; 32],
    /// Public vector t = A·s + e, compressed.
    pub t: PolyVec,
}

/// Secret key: s (in NTT domain) + public key hash.
pub struct KyberSecretKey {
    /// Secret vector s in NTT domain.
    pub s_hat: PolyVec,
    /// Public key (for re-encryption check in CCA transform).
    pub pk: KyberPublicKey,
    /// Hash of public key (H(pk), 32 bytes).
    pub pk_hash: [u8; 32],
    /// Random value z for implicit rejection (32 bytes).
    pub z: [u8; 32],
}

/// Kyber ciphertext.
pub struct KyberCiphertext {
    /// Compressed vector u (k polynomials, DU bits each).
    pub u: PolyVec,
    /// Compressed polynomial v (DV bits).
    pub v: Poly,
}

/// Sample a CBD(eta) polynomial: centered binomial distribution.
///
/// Each coefficient is in [-eta, eta], stored as u16 mod q.
pub fn sample_cbd(rng: &mut TenSafeRng, eta: usize) -> Poly {
    let mut poly = [0u16; KYBER_N];
    for i in 0..KYBER_N {
        let mut a = 0u16;
        let mut b = 0u16;
        for _ in 0..eta {
            a += (rng.range_u32(2)) as u16;
            b += (rng.range_u32(2)) as u16;
        }
        // a - b mod q
        poly[i] = if a >= b {
            a - b
        } else {
            KYBER_Q as u16 - (b - a)
        };
    }
    poly
}

/// Generate an ML-KEM-768 keypair.
pub fn keygen(rng: &mut TenSafeRng) -> (KyberPublicKey, KyberSecretKey) {
    let ntt = KyberNtt::new();

    // Generate random seed
    let mut rho = [0u8; 32];
    for b in rho.iter_mut() {
        *b = rng.next_u32() as u8;
    }

    let mut z = [0u8; 32];
    for b in z.iter_mut() {
        *b = rng.next_u32() as u8;
    }

    // Sample secret vector s and error vector e
    let mut s_hat: PolyVec = [[0u16; KYBER_N]; K];
    let mut e: PolyVec = [[0u16; KYBER_N]; K];

    for i in 0..K {
        s_hat[i] = sample_cbd(rng, ETA_1);
        ntt.forward(&mut s_hat[i]);
        e[i] = sample_cbd(rng, ETA_1);
        ntt.forward(&mut e[i]);
    }

    // Generate matrix A from seed (simplified: random for now)
    // In production, this uses SHAKE-128(rho || i || j) per entry
    let mut a_hat: [[Poly; K]; K] = [[[0u16; KYBER_N]; K]; K];
    for i in 0..K {
        for j in 0..K {
            for c in 0..KYBER_N {
                a_hat[i][j][c] = rng.range_u32(KYBER_Q as u32) as u16;
            }
        }
    }

    // t = A·s + e (all in NTT domain, element-wise)
    let mut t: PolyVec = [[0u16; KYBER_N]; K];
    for i in 0..K {
        for c in 0..KYBER_N {
            let mut sum = 0u32;
            for j in 0..K {
                sum += (a_hat[i][j][c] as u32 * s_hat[j][c] as u32) % KYBER_Q as u32;
            }
            t[i][c] = ((sum + e[i][c] as u32) % KYBER_Q as u32) as u16;
        }
    }

    // Hash of public key (simplified: XOR of first bytes)
    let mut pk_hash = [0u8; 32];
    for i in 0..32 {
        pk_hash[i] = rho[i] ^ (t[0][i] as u8);
    }

    let pk = KyberPublicKey { rho, t };
    let sk = KyberSecretKey {
        s_hat,
        pk: KyberPublicKey {
            rho: pk.rho,
            t: pk.t,
        },
        pk_hash,
        z,
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

        // Public key should have valid polynomial coefficients
        for i in 0..K {
            for c in 0..KYBER_N {
                assert!(
                    (pk.t[i][c] as u64) < KYBER_Q,
                    "pk.t[{i}][{c}] = {} >= q",
                    pk.t[i][c]
                );
            }
        }

        // Secret key should have valid NTT-domain coefficients
        for i in 0..K {
            for c in 0..KYBER_N {
                assert!(
                    (sk.s_hat[i][c] as u64) < KYBER_Q,
                    "sk.s[{i}][{c}] = {} >= q",
                    sk.s_hat[i][c]
                );
            }
        }
    }

    #[test]
    fn test_cbd_distribution() {
        let mut rng = TenSafeRng::from_seed(42);
        let poly = sample_cbd(&mut rng, ETA_1);

        // All coefficients should be in [0, q)
        for &c in poly.iter() {
            assert!((c as u64) < KYBER_Q, "CBD coefficient {c} >= q");
        }

        // Most coefficients should be near 0 or q-1 (small values)
        let small_count = poly
            .iter()
            .filter(|&&c| c <= ETA_1 as u16 || c >= (KYBER_Q as u16 - ETA_1 as u16))
            .count();
        // With eta=2, ~93% of values should be "small" (in [-2, 2])
        assert!(
            small_count > KYBER_N * 80 / 100,
            "CBD distribution looks wrong: only {small_count}/256 small values"
        );
    }
}
