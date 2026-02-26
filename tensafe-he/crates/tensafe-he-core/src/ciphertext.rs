//! CKKS ciphertext structure and operations.
//!
//! A CKKS ciphertext is a pair (c0, c1) of RNS polynomials in the NTT domain.
//!
//! Operations:
//! - encrypt(plaintext, sk) → ct
//! - decrypt(ct, sk) → plaintext
//! - ct × pt (element-wise in NTT domain)

use crate::encoding::CkksEncoder;
use crate::ntt::{negacyclic_ntt_forward, negacyclic_ntt_inverse, NttTables};
use crate::params::CkksParams;
use crate::rns::{mod_add, mod_mul, mod_sub, RnsPoly};
use crate::sampling::{sample_gaussian, sample_ternary, sample_uniform, ERROR_STD_DEV};
use rand::Rng;

/// A CKKS ciphertext: pair of RNS polynomials (c0, c1) in NTT domain.
#[derive(Debug, Clone)]
pub struct Ciphertext {
    /// First component c0 (in NTT domain).
    pub c0: RnsPoly,
    /// Second component c1 (in NTT domain).
    pub c1: RnsPoly,
    /// Current scale factor (Δ).
    pub scale: f64,
}

/// Secret key: ternary polynomial s ∈ {-1, 0, 1}^N, stored in NTT domain.
#[derive(Debug, Clone)]
pub struct SecretKey {
    /// Secret key polynomial in NTT domain, one vector per RNS limb.
    pub s_ntt: RnsPoly,
}

/// Pre-computed NTT tables for all RNS limbs.
#[derive(Debug, Clone)]
pub struct CkksContext {
    /// Parameter set.
    pub params: CkksParams,
    /// NTT tables, one per RNS limb.
    pub ntt_tables: Vec<NttTables>,
    /// Encoder.
    pub encoder: CkksEncoder,
}

impl CkksContext {
    /// Create a new CKKS context with pre-computed NTT tables.
    pub fn new(params: CkksParams) -> Self {
        let ntt_tables: Vec<NttTables> = params
            .moduli
            .iter()
            .map(|m| NttTables::new(params.poly_degree, m.value))
            .collect();
        let encoder = CkksEncoder::new(&params);

        Self {
            params,
            ntt_tables,
            encoder,
        }
    }

    /// Generate a secret key.
    pub fn keygen<R: Rng>(&self, rng: &mut R) -> SecretKey {
        let n = self.params.poly_degree;
        let num_limbs = self.params.num_limbs;

        // Sample ternary secret key coefficients
        // We use the first modulus to represent {-1, 0, 1}
        let mut s_ntt = RnsPoly::zero(n, num_limbs);

        // Generate ternary coefficients and convert to each RNS limb
        let s_ternary = sample_ternary(rng, n, 3); // values in {0, 1, 2} representing {q-1, 0, 1}

        for l in 0..num_limbs {
            let q = self.params.moduli[l].value;
            for i in 0..n {
                s_ntt.limbs[l][i] = match s_ternary[i] {
                    0 => q - 1, // -1 mod q
                    1 => 0,     // 0
                    2 => 1,     // 1
                    _ => unreachable!(),
                };
            }
            // Transform to NTT domain
            negacyclic_ntt_forward(&mut s_ntt.limbs[l], &self.ntt_tables[l]);
        }

        SecretKey { s_ntt }
    }

    /// Encrypt a real-valued vector.
    ///
    /// RLWE encryption:
    ///   a ← uniform(R_q)
    ///   e ← discrete_gaussian(σ)
    ///   ct = (c0, c1) = (m + a·s + e, -a)   [all in NTT domain]
    ///
    /// Equivalently (standard form):
    ///   c0 = -a·s + m + e
    ///   c1 = a
    ///   Decrypt: c0 + c1·s = m + e
    pub fn encrypt<R: Rng>(
        &self,
        z: &[f64],
        sk: &SecretKey,
        rng: &mut R,
    ) -> Ciphertext {
        let n = self.params.poly_degree;
        let num_limbs = self.params.num_limbs;

        // Encode plaintext
        let mut m = self.encoder.encode(z, &self.params);

        // Transform plaintext to NTT domain
        for l in 0..num_limbs {
            negacyclic_ntt_forward(&mut m.limbs[l], &self.ntt_tables[l]);
        }

        // Sample random 'a' polynomial (in NTT domain)
        let mut a_ntt = RnsPoly::zero(n, num_limbs);
        for l in 0..num_limbs {
            let q = self.params.moduli[l].value;
            let a_coeffs = sample_uniform(rng, n, q);
            a_ntt.limbs[l] = a_coeffs;
            negacyclic_ntt_forward(&mut a_ntt.limbs[l], &self.ntt_tables[l]);
        }

        // Sample error polynomial
        let mut e_ntt = RnsPoly::zero(n, num_limbs);
        for l in 0..num_limbs {
            let q = self.params.moduli[l].value;
            e_ntt.limbs[l] = sample_gaussian(rng, n, q, ERROR_STD_DEV);
            negacyclic_ntt_forward(&mut e_ntt.limbs[l], &self.ntt_tables[l]);
        }

        // c0 = -a·s + m + e  (all in NTT domain = element-wise)
        // c1 = a
        let mut c0 = RnsPoly::zero(n, num_limbs);
        for l in 0..num_limbs {
            let q = self.params.moduli[l].value;
            for i in 0..n {
                // -a·s
                let neg_as = mod_sub(0, mod_mul(a_ntt.limbs[l][i], sk.s_ntt.limbs[l][i], q), q);
                // -a·s + m + e
                let val = mod_add(neg_as, mod_add(m.limbs[l][i], e_ntt.limbs[l][i], q), q);
                c0.limbs[l][i] = val;
            }
        }

        Ciphertext {
            c0,
            c1: a_ntt,
            scale: crate::params::SCALE,
        }
    }

    /// Decrypt a ciphertext to a real-valued vector.
    ///
    /// Decrypt: m = c0 + c1·s  (element-wise in NTT domain, then iNTT + decode)
    pub fn decrypt(&self, ct: &Ciphertext, sk: &SecretKey) -> Vec<f64> {
        let n = self.params.poly_degree;
        let num_limbs = self.params.num_limbs;

        // m_ntt = c0 + c1·s (element-wise in NTT domain)
        let mut m_ntt = RnsPoly::zero(n, num_limbs);
        for l in 0..num_limbs {
            let q = self.params.moduli[l].value;
            for i in 0..n {
                let c1_s = mod_mul(ct.c1.limbs[l][i], sk.s_ntt.limbs[l][i], q);
                m_ntt.limbs[l][i] = mod_add(ct.c0.limbs[l][i], c1_s, q);
            }
        }

        // Inverse NTT to get coefficient domain
        for l in 0..num_limbs {
            negacyclic_ntt_inverse(&mut m_ntt.limbs[l], &self.ntt_tables[l]);
        }

        // Decode
        self.encoder.decode(&m_ntt, &self.params)
    }

    /// Ciphertext × plaintext multiply (the core TenSafe operation).
    ///
    /// ct' = (c0 ⊙ pt_ntt, c1 ⊙ pt_ntt)  — element-wise in NTT domain.
    ///
    /// This is the ZeRo-MOAI inner product: after decrypt, each d_model-sized
    /// segment contains h[j] * A[r, j], and summing gives the dot product.
    pub fn ct_pt_mul(&self, ct: &Ciphertext, pt: &[f64]) -> Ciphertext {
        let n = self.params.poly_degree;
        let num_limbs = self.params.num_limbs;

        // Encode plaintext and transform to NTT domain
        let mut pt_ntt = self.encoder.encode(pt, &self.params);
        for l in 0..num_limbs {
            negacyclic_ntt_forward(&mut pt_ntt.limbs[l], &self.ntt_tables[l]);
        }

        self.ct_pt_mul_ntt(ct, &pt_ntt)
    }

    /// Ciphertext × plaintext multiply where plaintext is already in NTT domain.
    /// This is the fast path used with pre-cached NTT(plaintext) (Optimization 3.1).
    pub fn ct_pt_mul_ntt(&self, ct: &Ciphertext, pt_ntt: &RnsPoly) -> Ciphertext {
        let moduli = &self.params.moduli;
        Ciphertext {
            c0: ct.c0.hadamard_mul(pt_ntt, moduli),
            c1: ct.c1.hadamard_mul(pt_ntt, moduli),
            scale: ct.scale * ct.scale, // scale doubles after multiply
        }
    }

    /// Encode a plaintext vector and transform to NTT domain for caching.
    /// This is Optimization 3.1: pre-computed NTT(plaintext) cache.
    pub fn encode_to_ntt(&self, z: &[f64]) -> RnsPoly {
        let mut poly = self.encoder.encode(z, &self.params);
        for l in 0..self.params.num_limbs {
            negacyclic_ntt_forward(&mut poly.limbs[l], &self.ntt_tables[l]);
        }
        poly
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn make_ctx_and_key() -> (CkksContext, SecretKey) {
        let params = CkksParams::for_degree(8192);
        let ctx = CkksContext::new(params);
        let mut rng = StdRng::seed_from_u64(42);
        let sk = ctx.keygen(&mut rng);
        (ctx, sk)
    }

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let (ctx, sk) = make_ctx_and_key();
        let mut rng = StdRng::seed_from_u64(123);

        let z: Vec<f64> = (0..10).map(|i| i as f64 * 0.5).collect();
        let ct = ctx.encrypt(&z, &sk, &mut rng);
        let decoded = ctx.decrypt(&ct, &sk);

        for i in 0..z.len() {
            let err = (decoded[i] - z[i]).abs();
            assert!(
                err < 1e-4,
                "Slot {i}: decoded={}, expected={}, error={err}",
                decoded[i],
                z[i]
            );
        }
    }

    #[test]
    fn test_ct_pt_multiply() {
        let (ctx, sk) = make_ctx_and_key();
        let mut rng = StdRng::seed_from_u64(456);

        // x = [1.0, 2.0, 3.0, 4.0, 5.0]
        let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        // p = [0.5, 0.5, 0.5, 0.5, 0.5]
        let p: Vec<f64> = vec![0.5; 5];
        // Expected: x ⊙ p = [0.5, 1.0, 1.5, 2.0, 2.5]

        let ct = ctx.encrypt(&x, &sk, &mut rng);
        let ct_prod = ctx.ct_pt_mul(&ct, &p);
        let decoded = ctx.decrypt(&ct_prod, &sk);

        let expected: Vec<f64> = x.iter().zip(p.iter()).map(|(a, b)| a * b).collect();
        for i in 0..expected.len() {
            let err = (decoded[i] - expected[i]).abs();
            assert!(
                err < 0.1, // ct×pt introduces more noise due to scale multiplication
                "Slot {i}: decoded={}, expected={}, error={err}",
                decoded[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_ct_pt_mul_with_cached_ntt() {
        let (ctx, sk) = make_ctx_and_key();
        let mut rng = StdRng::seed_from_u64(789);

        let x: Vec<f64> = (0..20).map(|i| (i as f64) * 0.1).collect();
        let p: Vec<f64> = (0..20).map(|i| 1.0 - (i as f64) * 0.04).collect();

        // Pre-cache plaintext in NTT domain (Optimization 3.1)
        let pt_ntt = ctx.encode_to_ntt(&p);

        let ct = ctx.encrypt(&x, &sk, &mut rng);
        let ct_prod = ctx.ct_pt_mul_ntt(&ct, &pt_ntt);
        let decoded = ctx.decrypt(&ct_prod, &sk);

        let expected: Vec<f64> = x.iter().zip(p.iter()).map(|(a, b)| a * b).collect();
        for i in 0..expected.len() {
            let err = (decoded[i] - expected[i]).abs();
            assert!(
                err < 0.1,
                "Slot {i}: decoded={}, expected={}, error={err}",
                decoded[i],
                expected[i]
            );
        }
    }
}
