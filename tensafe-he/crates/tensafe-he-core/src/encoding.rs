//! CKKS encode/decode via the canonical embedding.
//!
//! Encode: z ∈ R^{N/2} → m(X) = round(Δ · σ^{-1}(z)) ∈ R_q
//! Decode: m(X) ∈ R_q → z = σ(m) / Δ ∈ R^{N/2}
//!
//! The canonical embedding σ maps polynomial m(X) to its evaluations at
//! the primitive 2N-th roots of unity: σ(m) = (m(ζ), m(ζ^3), m(ζ^5), ...).
//!
//! For real-valued CKKS (TenSafe), we only use the first N/2 slots (real parts).

use std::f64::consts::PI;

use crate::params::{CkksParams, SCALE};
use crate::rns::RnsPoly;

/// CKKS encoder/decoder for a given parameter set.
#[derive(Debug, Clone)]
pub struct CkksEncoder {
    /// Polynomial degree N.
    n: usize,
    /// Number of SIMD slots = N/2.
    num_slots: usize,
}

impl CkksEncoder {
    /// Create a new encoder for the given parameter set.
    pub fn new(params: &CkksParams) -> Self {
        let n = params.poly_degree;
        let num_slots = params.num_slots;

        Self {
            n,
            num_slots,
        }
    }

    /// Encode a real-valued vector z ∈ R^S into an RNS polynomial.
    ///
    /// Steps:
    /// 1. Extend z to complex: z_complex = z + 0i (conjugate-symmetric for real CKKS)
    /// 2. Apply inverse canonical embedding σ^{-1}(z) to get polynomial coefficients
    /// 3. Scale by Δ and round to integers
    /// 4. Reduce modulo each RNS prime
    ///
    /// The input vector length must be ≤ num_slots. Shorter vectors are zero-padded.
    pub fn encode(&self, z: &[f64], params: &CkksParams) -> RnsPoly {
        assert!(
            z.len() <= self.num_slots,
            "Input length {} exceeds slot count {}",
            z.len(),
            self.num_slots
        );

        // Pad to full slot count
        let mut z_full = vec![0.0f64; self.num_slots];
        z_full[..z.len()].copy_from_slice(z);

        // Apply inverse DFT (σ^{-1}) to get polynomial coefficients
        // For real-valued CKKS, the canonical embedding uses the odd-indexed
        // roots of X^{2N}-1, and the inverse is a specific DFT.
        let coeffs_f64 = self.inverse_canonical_embedding(&z_full);

        // Scale by Δ and round, then reduce to RNS representation
        let mut result = RnsPoly::zero(self.n, params.num_limbs);

        for l in 0..params.num_limbs {
            let q = params.moduli[l].value;
            for i in 0..self.n {
                let scaled = coeffs_f64[i] * SCALE;
                let rounded = scaled.round() as i64;
                result.limbs[l][i] = if rounded >= 0 {
                    (rounded as u64) % q
                } else {
                    q - ((-rounded) as u64 % q)
                };
            }
        }

        result
    }

    /// Decode an RNS polynomial back to a real-valued vector z ∈ R^S.
    ///
    /// Steps:
    /// 1. Convert from RNS to centered representation (signed integers)
    /// 2. Divide by the given scale to get float coefficients
    /// 3. Apply canonical embedding σ to get slot values
    ///
    /// The `scale` parameter should match the ciphertext's current scale:
    /// - After encrypt: scale = Δ
    /// - After ct×pt multiply: scale = Δ²
    pub fn decode(&self, poly: &RnsPoly, params: &CkksParams, scale: f64) -> Vec<f64> {
        // Use the first RNS limb for decoding (all limbs represent the same polynomial)
        let q = params.moduli[0].value;
        let half_q = q / 2;

        // Convert to centered representation and scale
        let coeffs_f64: Vec<f64> = poly.limbs[0]
            .iter()
            .map(|&c| {
                let signed = if c > half_q {
                    -((q - c) as f64)
                } else {
                    c as f64
                };
                signed / scale
            })
            .collect();

        // Apply canonical embedding (forward DFT at roots of unity)
        self.canonical_embedding(&coeffs_f64)
    }

    /// Inverse canonical embedding: σ^{-1}(z) → polynomial coefficients.
    /// This is essentially an inverse DFT at the primitive roots of X^N+1.
    ///
    /// For z ∈ C^{N/2} with conjugate symmetry (z_{N/2+i} = conj(z_i)):
    ///   m[j] = (1/N) · Σ_{k=0}^{N-1} z̃[k] · ζ^{-(2k+1)j}
    ///
    /// For real-valued z, the result m has real coefficients.
    fn inverse_canonical_embedding(&self, z: &[f64]) -> Vec<f64> {
        let n = self.n;
        let s = self.num_slots;
        assert_eq!(z.len(), s);

        // Build the full complex vector with conjugate symmetry:
        // The canonical embedding evaluates at ζ^{2k+1} for k=0..N-1.
        // Conjugate pair: slot k ↔ slot N-1-k (since conj(ζ^{2k+1}) = ζ^{2(N-1-k)+1}).
        // For real z: z̃[k] = z[k], z̃[N-1-k] = z[k].
        let mut z_real = vec![0.0f64; n];
        let z_imag = vec![0.0f64; n];
        for k in 0..s {
            z_real[k] = z[k];
            z_real[n - 1 - k] = z[k];
        }

        // Inverse DFT: m[j] = (1/N) · Σ_{k=0}^{N-1} z̃[k] · ζ^{-(2k+1)j}
        let mut coeffs = vec![0.0f64; n];
        let inv_n = 1.0 / n as f64;

        for j in 0..n {
            let mut sum_real = 0.0f64;
            for k in 0..n {
                // ζ^{-(2k+1)j} = cos(-angle) + i·sin(-angle)
                // angle = π(2k+1)j / N
                let angle = PI * (2 * k + 1) as f64 * j as f64 / n as f64;
                let cos_val = angle.cos();
                let sin_val = angle.sin();
                // z̃[k] · conj(ζ^{(2k+1)j}) = (z_r + i·z_i)(cos - i·sin)
                // Real part: z_r·cos + z_i·sin
                sum_real += z_real[k] * cos_val + z_imag[k] * sin_val;
            }
            coeffs[j] = sum_real * inv_n;
        }

        coeffs
    }

    /// Decode from pre-computed float64 polynomial coefficients.
    /// Used when coefficients have already been extracted via CRT reconstruction.
    pub fn decode_coefficients(&self, coeffs: &[f64]) -> Vec<f64> {
        self.canonical_embedding(coeffs)
    }

    /// Canonical embedding: polynomial coefficients → slot values.
    /// σ(m)[k] = m(ζ^{2k+1}) = Σ_{j=0}^{N-1} m[j] · ζ^{(2k+1)j}
    ///
    /// Returns the real parts of the first N/2 evaluations.
    fn canonical_embedding(&self, coeffs: &[f64]) -> Vec<f64> {
        let n = self.n;
        let s = self.num_slots;
        assert_eq!(coeffs.len(), n);

        let mut z = vec![0.0f64; s];

        for k in 0..s {
            let mut val_real = 0.0f64;
            for j in 0..n {
                // ζ^{(2k+1)j}: angle = π(2k+1)j / N
                let angle = PI * (2 * k + 1) as f64 * j as f64 / n as f64;
                val_real += coeffs[j] * angle.cos();
            }
            z[k] = val_real;
        }

        z
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_roundtrip() {
        // Use small parameters for testing
        let params = CkksParams::for_degree(8192);
        let encoder = CkksEncoder::new(&params);

        // Encode a simple vector
        let z: Vec<f64> = (0..10).map(|i| i as f64 * 0.1).collect();
        let poly = encoder.encode(&z, &params);
        let decoded = encoder.decode(&poly, &params, SCALE);

        // Check roundtrip accuracy
        for i in 0..z.len() {
            let err = (decoded[i] - z[i]).abs();
            assert!(
                err < 1e-5,
                "Slot {i}: decoded={}, expected={}, error={err}",
                decoded[i],
                z[i]
            );
        }
    }

    #[test]
    fn test_encode_zeros() {
        let params = CkksParams::for_degree(8192);
        let encoder = CkksEncoder::new(&params);

        let z = vec![0.0f64; 100];
        let poly = encoder.encode(&z, &params);
        let decoded = encoder.decode(&poly, &params, SCALE);

        for i in 0..z.len() {
            assert!(
                decoded[i].abs() < 1e-7,
                "Slot {i}: decoded={}, expected 0.0",
                decoded[i]
            );
        }
    }

    #[test]
    fn test_encode_large_values() {
        let params = CkksParams::for_degree(8192);
        let encoder = CkksEncoder::new(&params);

        let z: Vec<f64> = (0..50).map(|i| (i as f64 - 25.0) * 100.0).collect();
        let poly = encoder.encode(&z, &params);
        let decoded = encoder.decode(&poly, &params, SCALE);

        for i in 0..z.len() {
            let err = (decoded[i] - z[i]).abs();
            let rel_err = if z[i].abs() > 1e-10 {
                err / z[i].abs()
            } else {
                err
            };
            assert!(
                rel_err < 1e-5 || err < 1e-5,
                "Slot {i}: decoded={}, expected={}, rel_err={rel_err}",
                decoded[i],
                z[i]
            );
        }
    }
}
