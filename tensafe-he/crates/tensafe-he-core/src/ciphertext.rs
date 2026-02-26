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
use crate::params::SCALE;
use crate::rns::{mod_add, mod_inv, mod_mul_barrett, mod_sub, RnsPoly};
use crate::sampling::{sample_gaussian_signed, sample_ternary, sample_uniform, ERROR_STD_DEV};
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

/// RLWE public key: pk = (b, a) where b = -a·s + e.
///
/// Anyone holding pk can encrypt messages; only the holder of the
/// corresponding SecretKey s can decrypt.
#[derive(Debug, Clone)]
pub struct PublicKey {
    /// b = -a·s + e  (NTT domain)
    pub b: RnsPoly,
    /// a (NTT domain)
    pub a: RnsPoly,
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

    /// Generate a public key from the secret key.
    ///
    /// pk = (b, a) where:
    ///   a ← uniform(R_q)
    ///   e ← Gaussian(σ)
    ///   b = -a·s + e
    ///
    /// The client generates (sk, pk), sends pk to the server.
    /// The server can encrypt using pk; only the client can decrypt with sk.
    pub fn keygen_public<R: Rng>(&self, sk: &SecretKey, rng: &mut R) -> PublicKey {
        let n = self.params.poly_degree;
        let num_limbs = self.params.num_limbs;

        // Sample random 'a' polynomial
        let mut a_ntt = RnsPoly::zero(n, num_limbs);
        for l in 0..num_limbs {
            let q = self.params.moduli[l].value;
            a_ntt.limbs[l] = sample_uniform(rng, n, q);
            negacyclic_ntt_forward(&mut a_ntt.limbs[l], &self.ntt_tables[l]);
        }

        // Sample error polynomial
        let e_signed = sample_gaussian_signed(rng, n, ERROR_STD_DEV);
        let mut e_ntt = RnsPoly::zero(n, num_limbs);
        for l in 0..num_limbs {
            let q = self.params.moduli[l].value;
            for i in 0..n {
                e_ntt.limbs[l][i] = if e_signed[i] >= 0 {
                    (e_signed[i] as u64) % q
                } else {
                    q - ((-e_signed[i]) as u64 % q)
                };
            }
            negacyclic_ntt_forward(&mut e_ntt.limbs[l], &self.ntt_tables[l]);
        }

        // b = -a·s + e
        let mut b = RnsPoly::zero(n, num_limbs);
        for l in 0..num_limbs {
            let q = self.params.moduli[l].value;
            let modulus = &self.params.moduli[l];
            for i in 0..n {
                let as_val = mod_mul_barrett(a_ntt.limbs[l][i], sk.s_ntt.limbs[l][i], modulus);
                let neg_as = if as_val == 0 { 0 } else { q - as_val };
                b.limbs[l][i] = mod_add(neg_as, e_ntt.limbs[l][i], q);
            }
        }

        PublicKey { b, a: a_ntt }
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

        // Sample error polynomial ONCE as signed integers, then reduce to each RNS limb.
        // This ensures cross-limb consistency (required for CRT-based decode after multiply).
        let e_signed = sample_gaussian_signed(rng, n, ERROR_STD_DEV);
        let mut e_ntt = RnsPoly::zero(n, num_limbs);
        for l in 0..num_limbs {
            let q = self.params.moduli[l].value;
            for i in 0..n {
                e_ntt.limbs[l][i] = if e_signed[i] >= 0 {
                    (e_signed[i] as u64) % q
                } else {
                    q - ((-e_signed[i]) as u64 % q)
                };
            }
            negacyclic_ntt_forward(&mut e_ntt.limbs[l], &self.ntt_tables[l]);
        }

        // c0 = -a·s + m + e  (all in NTT domain = element-wise)
        // c1 = a
        let mut c0 = RnsPoly::zero(n, num_limbs);
        for l in 0..num_limbs {
            let q = self.params.moduli[l].value;
            let modulus = &self.params.moduli[l];
            for i in 0..n {
                // -a·s (Barrett reduction)
                let neg_as = mod_sub(0, mod_mul_barrett(a_ntt.limbs[l][i], sk.s_ntt.limbs[l][i], modulus), q);
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

    /// Encrypt a real-valued vector using a public key (asymmetric RLWE).
    ///
    /// ct = (c0, c1) where:
    ///   u ← ternary (ephemeral randomness)
    ///   e0, e1 ← Gaussian(σ)
    ///   c0 = b·u + e0 + m
    ///   c1 = a·u + e1
    ///
    /// Only the holder of sk can decrypt (since b = -a·s + e_pk).
    pub fn encrypt_pk<R: Rng>(
        &self,
        z: &[f64],
        pk: &PublicKey,
        rng: &mut R,
    ) -> Ciphertext {
        let n = self.params.poly_degree;
        let num_limbs = self.params.num_limbs;

        // Encode plaintext
        let mut m = self.encoder.encode(z, &self.params);
        for l in 0..num_limbs {
            negacyclic_ntt_forward(&mut m.limbs[l], &self.ntt_tables[l]);
        }

        // Sample ephemeral ternary u
        let u_ternary = sample_ternary(rng, n, 3);
        let mut u_ntt = RnsPoly::zero(n, num_limbs);
        for l in 0..num_limbs {
            let q = self.params.moduli[l].value;
            for i in 0..n {
                u_ntt.limbs[l][i] = match u_ternary[i] {
                    0 => q - 1, // -1 mod q
                    1 => 0,     // 0
                    2 => 1,     // 1
                    _ => unreachable!(),
                };
            }
            negacyclic_ntt_forward(&mut u_ntt.limbs[l], &self.ntt_tables[l]);
        }

        // Sample two error polynomials e0, e1
        let e0_signed = sample_gaussian_signed(rng, n, ERROR_STD_DEV);
        let e1_signed = sample_gaussian_signed(rng, n, ERROR_STD_DEV);

        let signed_to_rns = |e_signed: &[i64], l: usize| -> Vec<u64> {
            let q = self.params.moduli[l].value;
            e_signed
                .iter()
                .map(|&e| {
                    if e >= 0 {
                        (e as u64) % q
                    } else {
                        q - ((-e) as u64 % q)
                    }
                })
                .collect()
        };

        let mut e0_ntt = RnsPoly::zero(n, num_limbs);
        let mut e1_ntt = RnsPoly::zero(n, num_limbs);
        for l in 0..num_limbs {
            e0_ntt.limbs[l] = signed_to_rns(&e0_signed, l);
            negacyclic_ntt_forward(&mut e0_ntt.limbs[l], &self.ntt_tables[l]);
            e1_ntt.limbs[l] = signed_to_rns(&e1_signed, l);
            negacyclic_ntt_forward(&mut e1_ntt.limbs[l], &self.ntt_tables[l]);
        }

        // c0 = b·u + e0 + m
        // c1 = a·u + e1
        let mut c0 = RnsPoly::zero(n, num_limbs);
        let mut c1 = RnsPoly::zero(n, num_limbs);
        for l in 0..num_limbs {
            let q = self.params.moduli[l].value;
            let modulus = &self.params.moduli[l];
            for i in 0..n {
                let bu = mod_mul_barrett(pk.b.limbs[l][i], u_ntt.limbs[l][i], modulus);
                c0.limbs[l][i] = mod_add(bu, mod_add(e0_ntt.limbs[l][i], m.limbs[l][i], q), q);

                let au = mod_mul_barrett(pk.a.limbs[l][i], u_ntt.limbs[l][i], modulus);
                c1.limbs[l][i] = mod_add(au, e1_ntt.limbs[l][i], q);
            }
        }

        Ciphertext {
            c0,
            c1,
            scale: SCALE,
        }
    }

    /// Decrypt a ciphertext to a real-valued vector.
    ///
    /// Handles both fresh ciphertexts (scale = Δ) and post-multiply
    /// ciphertexts (scale = Δ²) by automatically rescaling when needed.
    ///
    /// Decrypt: m = c0 + c1·s  (element-wise in NTT domain, then iNTT + rescale + decode)
    pub fn decrypt(&self, ct: &Ciphertext, sk: &SecretKey) -> Vec<f64> {
        let n = self.params.poly_degree;
        // Use the ciphertext's actual limb count (may be fewer after rescale)
        let num_limbs = ct.c0.limbs.len();

        // m_ntt = c0 + c1·s (element-wise in NTT domain)
        let mut m_ntt = RnsPoly::zero(n, num_limbs);
        for l in 0..num_limbs {
            let q = self.params.moduli[l].value;
            let modulus = &self.params.moduli[l];
            for i in 0..n {
                let c1_s = mod_mul_barrett(ct.c1.limbs[l][i], sk.s_ntt.limbs[l][i], modulus);
                m_ntt.limbs[l][i] = mod_add(ct.c0.limbs[l][i], c1_s, q);
            }
        }

        // Inverse NTT to get coefficient domain
        for l in 0..num_limbs {
            negacyclic_ntt_inverse(&mut m_ntt.limbs[l], &self.ntt_tables[l]);
        }

        // After ct×pt multiply, polynomial coefficients grow to ~Δ² magnitude,
        // which overflows individual RNS limbs. Use 2-limb CRT (q_0 × q_1 ≈ 2^100)
        // to reconstruct the true coefficient value before decoding.
        if ct.scale > SCALE * 1.5 && num_limbs >= 2 {
            let q0 = self.params.moduli[0].value;
            let q1 = self.params.moduli[1].value;
            let q0_inv_mod_q1 = mod_inv(q0 % q1, q1);
            let q_product: u128 = q0 as u128 * q1 as u128;
            let half_product = q_product / 2;

            let mut coeffs_f64 = vec![0.0f64; n];
            for i in 0..n {
                let m0 = m_ntt.limbs[0][i];
                let m1 = m_ntt.limbs[1][i];

                // CRT: M = m0 + q0 * k, where k = (m1 - m0 mod q1) * q0_inv mod q1
                let m0_mod_q1 = (m0 as u128 % q1 as u128) as u64;
                let diff = if m1 >= m0_mod_q1 {
                    m1 - m0_mod_q1
                } else {
                    q1 - m0_mod_q1 + m1
                };
                let k = ((diff as u128 * q0_inv_mod_q1 as u128) % q1 as u128) as u64;
                let m_unsigned: u128 = m0 as u128 + k as u128 * q0 as u128;

                // CRT overflow validation: m must be in [0, q0*q1)
                debug_assert!(
                    m_unsigned < q_product,
                    "CRT overflow at coefficient {i}: m={m_unsigned} >= q0*q1={q_product}"
                );
                // Fallback: if overflow occurs in release mode, reduce modularly
                let m_unsigned = if m_unsigned >= q_product {
                    m_unsigned % q_product
                } else {
                    m_unsigned
                };

                // Center: convert to signed representation
                let m_signed_f64 = if m_unsigned > half_product {
                    -((q_product - m_unsigned) as f64)
                } else {
                    m_unsigned as f64
                };

                coeffs_f64[i] = m_signed_f64 / ct.scale;
            }

            // Apply canonical embedding (DFT at roots of unity)
            self.encoder.decode_coefficients(&coeffs_f64)
        } else {
            self.encoder.decode(&m_ntt, &self.params, ct.scale)
        }
    }

    /// Ciphertext × plaintext multiply (the core TenSafe operation).
    ///
    /// ct' = (c0 ⊙ pt_ntt, c1 ⊙ pt_ntt)  — element-wise in NTT domain.
    ///
    /// This is the ZeRo-MOAI inner product: after decrypt, each d_model-sized
    /// segment contains h[j] * A[r, j], and summing gives the dot product.
    pub fn ct_pt_mul(&self, ct: &Ciphertext, pt: &[f64]) -> Ciphertext {
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

    /// Ciphertext + Ciphertext addition.
    ///
    /// ct' = (c0_a + c0_b, c1_a + c1_b). Both ciphertexts must have the same scale.
    /// Used for accumulation in multi-adapter inference and future relinearization.
    pub fn ct_add(&self, ct_a: &Ciphertext, ct_b: &Ciphertext) -> Ciphertext {
        assert!(
            (ct_a.scale - ct_b.scale).abs() < 1.0,
            "Scale mismatch in ct_add: {} vs {}",
            ct_a.scale,
            ct_b.scale
        );
        let moduli = &self.params.moduli;
        Ciphertext {
            c0: ct_a.c0.add(&ct_b.c0, moduli),
            c1: ct_a.c1.add(&ct_b.c1, moduli),
            scale: ct_a.scale,
        }
    }

    /// Ciphertext - Ciphertext subtraction.
    ///
    /// ct' = (c0_a - c0_b, c1_a - c1_b). Both ciphertexts must have the same scale.
    pub fn ct_sub(&self, ct_a: &Ciphertext, ct_b: &Ciphertext) -> Ciphertext {
        assert!(
            (ct_a.scale - ct_b.scale).abs() < 1.0,
            "Scale mismatch in ct_sub: {} vs {}",
            ct_a.scale,
            ct_b.scale
        );
        let moduli = &self.params.moduli;
        Ciphertext {
            c0: ct_a.c0.sub(&ct_b.c0, moduli),
            c1: ct_a.c1.sub(&ct_b.c1, moduli),
            scale: ct_a.scale,
        }
    }

    /// Rescale: divide ciphertext by the last modulus q_{L-1}, dropping that limb.
    ///
    /// After ct×pt multiply, the scale doubles (Δ → Δ²). Rescale divides by
    /// q_{L-1} ≈ Δ, bringing the scale back to ~Δ and reducing the number of
    /// RNS limbs by 1. This enables chaining multiple multiplications.
    ///
    /// Algorithm (per remaining limb l):
    ///   c'[l][i] = (c[l][i] - c[last][i] mod q_l) × q_last^{-1} mod q_l
    pub fn rescale(&self, ct: &Ciphertext) -> Ciphertext {
        let n = self.params.poly_degree;
        let num_limbs = ct.c0.limbs.len();
        assert!(num_limbs >= 2, "Cannot rescale: only 1 limb left");

        let last_l = num_limbs - 1;
        let last_q = self.params.moduli[last_l].value;

        // We need the last limb's coefficients in coefficient domain for reduction.
        // The ciphertext is in NTT domain, so first we need to convert the last limb.
        let mut c0_last = ct.c0.limbs[last_l].clone();
        let mut c1_last = ct.c1.limbs[last_l].clone();
        negacyclic_ntt_inverse(&mut c0_last, &self.ntt_tables[last_l]);
        negacyclic_ntt_inverse(&mut c1_last, &self.ntt_tables[last_l]);

        // Also convert remaining limbs to coefficient domain for the reduction
        let mut c0_new = RnsPoly::zero(n, num_limbs - 1);
        let mut c1_new = RnsPoly::zero(n, num_limbs - 1);

        for l in 0..(num_limbs - 1) {
            let q_l = self.params.moduli[l].value;
            let last_q_inv = mod_inv(last_q % q_l, q_l);

            // Convert this limb to coefficient domain
            let mut c0_l = ct.c0.limbs[l].clone();
            let mut c1_l = ct.c1.limbs[l].clone();
            negacyclic_ntt_inverse(&mut c0_l, &self.ntt_tables[l]);
            negacyclic_ntt_inverse(&mut c1_l, &self.ntt_tables[l]);

            for i in 0..n {
                // c0: (c0[l][i] - c0[last][i] mod q_l) * q_last_inv mod q_l
                let c0_last_mod = c0_last[i] % q_l;
                let c0_diff = if c0_l[i] >= c0_last_mod {
                    c0_l[i] - c0_last_mod
                } else {
                    q_l - c0_last_mod + c0_l[i]
                };
                c0_new.limbs[l][i] = mod_mul_barrett(c0_diff, last_q_inv, &self.params.moduli[l]);

                // c1: same formula
                let c1_last_mod = c1_last[i] % q_l;
                let c1_diff = if c1_l[i] >= c1_last_mod {
                    c1_l[i] - c1_last_mod
                } else {
                    q_l - c1_last_mod + c1_l[i]
                };
                c1_new.limbs[l][i] = mod_mul_barrett(c1_diff, last_q_inv, &self.params.moduli[l]);
            }

            // Convert back to NTT domain
            negacyclic_ntt_forward(&mut c0_new.limbs[l], &self.ntt_tables[l]);
            negacyclic_ntt_forward(&mut c1_new.limbs[l], &self.ntt_tables[l]);
        }

        Ciphertext {
            c0: c0_new,
            c1: c1_new,
            scale: ct.scale / last_q as f64,
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

    #[test]
    fn test_crt_overflow_validation() {
        // Test with larger plaintext values that stress the CRT reconstruction
        let (ctx, sk) = make_ctx_and_key();
        let mut rng = StdRng::seed_from_u64(999);

        // Use values near the practical limit to stress CRT
        let x: Vec<f64> = (0..50).map(|i| (i as f64) * 10.0 - 250.0).collect();
        let p: Vec<f64> = (0..50).map(|i| (i as f64) * 5.0 - 125.0).collect();

        let ct = ctx.encrypt(&x, &sk, &mut rng);
        let ct_prod = ctx.ct_pt_mul(&ct, &p);
        let decoded = ctx.decrypt(&ct_prod, &sk);

        let expected: Vec<f64> = x.iter().zip(p.iter()).map(|(a, b)| a * b).collect();
        for i in 0..expected.len() {
            let err = (decoded[i] - expected[i]).abs();
            // Larger values produce more noise, so wider tolerance
            assert!(
                err < 10.0,
                "Slot {i}: decoded={}, expected={}, error={err}",
                decoded[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_ct_add() {
        let (ctx, sk) = make_ctx_and_key();
        let mut rng = StdRng::seed_from_u64(111);

        let a: Vec<f64> = (0..10).map(|i| i as f64 * 1.5).collect();
        let b: Vec<f64> = (0..10).map(|i| 10.0 - i as f64 * 0.3).collect();

        let ct_a = ctx.encrypt(&a, &sk, &mut rng);
        let ct_b = ctx.encrypt(&b, &sk, &mut rng);
        let ct_sum = ctx.ct_add(&ct_a, &ct_b);
        let decoded = ctx.decrypt(&ct_sum, &sk);

        let expected: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
        for i in 0..expected.len() {
            let err = (decoded[i] - expected[i]).abs();
            assert!(
                err < 1e-4,
                "ct_add slot {i}: decoded={}, expected={}, error={err}",
                decoded[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_ct_sub() {
        let (ctx, sk) = make_ctx_and_key();
        let mut rng = StdRng::seed_from_u64(222);

        let a: Vec<f64> = (0..10).map(|i| i as f64 * 2.0).collect();
        let b: Vec<f64> = (0..10).map(|i| i as f64 * 0.5).collect();

        let ct_a = ctx.encrypt(&a, &sk, &mut rng);
        let ct_b = ctx.encrypt(&b, &sk, &mut rng);
        let ct_diff = ctx.ct_sub(&ct_a, &ct_b);
        let decoded = ctx.decrypt(&ct_diff, &sk);

        let expected: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();
        for i in 0..expected.len() {
            let err = (decoded[i] - expected[i]).abs();
            assert!(
                err < 1e-4,
                "ct_sub slot {i}: decoded={}, expected={}, error={err}",
                decoded[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_pk_encrypt_decrypt_roundtrip() {
        let (ctx, sk) = make_ctx_and_key();
        let mut rng = StdRng::seed_from_u64(333);

        let pk = ctx.keygen_public(&sk, &mut rng);
        let z: Vec<f64> = (0..10).map(|i| i as f64 * 0.5).collect();
        let ct = ctx.encrypt_pk(&z, &pk, &mut rng);
        let decoded = ctx.decrypt(&ct, &sk);

        for i in 0..z.len() {
            let err = (decoded[i] - z[i]).abs();
            assert!(
                err < 1e-3,
                "pk encrypt slot {i}: decoded={}, expected={}, error={err}",
                decoded[i],
                z[i]
            );
        }
    }

    #[test]
    fn test_pk_encrypt_ct_pt_mul() {
        // Client encrypts with pk, server does ct×pt blind, client decrypts
        let (ctx, sk) = make_ctx_and_key();
        let mut rng = StdRng::seed_from_u64(444);

        let pk = ctx.keygen_public(&sk, &mut rng);

        // Client encrypts hidden states
        let h: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ct = ctx.encrypt_pk(&h, &pk, &mut rng);

        // Server does blind ct×pt multiply (no access to sk)
        let weights: Vec<f64> = vec![0.5, 0.5, 0.5, 0.5, 0.5];
        let ct_result = ctx.ct_pt_mul(&ct, &weights);

        // Client decrypts
        let decoded = ctx.decrypt(&ct_result, &sk);

        let expected: Vec<f64> = h.iter().zip(weights.iter()).map(|(a, b)| a * b).collect();
        for i in 0..expected.len() {
            let err = (decoded[i] - expected[i]).abs();
            assert!(
                err < 0.1,
                "pk ct×pt slot {i}: decoded={}, expected={}, error={err}",
                decoded[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_rescale() {
        let (ctx, sk) = make_ctx_and_key();
        let mut rng = StdRng::seed_from_u64(555);

        let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let p: Vec<f64> = vec![0.5, 0.5, 0.5, 0.5, 0.5];

        let ct = ctx.encrypt(&x, &sk, &mut rng);
        let ct_mul = ctx.ct_pt_mul(&ct, &p);

        // Before rescale: scale is Δ², 4 limbs
        assert!(ct_mul.scale > crate::params::SCALE * 1.5);
        assert_eq!(ct_mul.c0.limbs.len(), 4);

        // Rescale: drops 1 limb, scale → ~Δ
        let ct_rescaled = ctx.rescale(&ct_mul);
        assert_eq!(ct_rescaled.c0.limbs.len(), 3);

        // Decrypt the rescaled ciphertext
        // (Need to use only the first 3 limbs of sk for decryption)
        let decoded = ctx.decrypt(&ct_rescaled, &sk);

        let expected: Vec<f64> = x.iter().zip(p.iter()).map(|(a, b)| a * b).collect();
        for i in 0..expected.len() {
            let err = (decoded[i] - expected[i]).abs();
            assert!(
                err < 1.0, // rescale introduces some additional noise
                "rescale slot {i}: decoded={}, expected={}, error={err}",
                decoded[i],
                expected[i]
            );
        }
    }
}
