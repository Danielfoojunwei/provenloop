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

/// Pre-computed tables for O(N log N) FFT-based canonical embedding.
///
/// The canonical embedding evaluates m(X) at ζ^{2k+1} for k=0..N-1,
/// where ζ = e^{πi/N}. This is a "twisted DFT":
///   m(ζ^{2k+1}) = FFT_N( m[j] · ζ^j )[k]
///
/// Pre-computing twist factors and twiddles eliminates all trig calls
/// from the hot path. For N=16384: ~114K multiply-adds vs ~268M trig calls.
#[derive(Debug, Clone)]
struct FftTables {
    n: usize,
    /// Twist factors: twist_re[j] = cos(πj/N), twist_im[j] = sin(πj/N)
    twist_re: Vec<f64>,
    twist_im: Vec<f64>,
    /// Bit-reversal permutation table for size N.
    bit_rev: Vec<usize>,
    /// FFT twiddle factors per stage: twiddle_re[s][k] = cos(-2πk/2^{s+1})
    twiddle_re: Vec<Vec<f64>>,
    /// FFT twiddle factors per stage: twiddle_im[s][k] = sin(-2πk/2^{s+1})
    twiddle_im: Vec<Vec<f64>>,
}

impl FftTables {
    fn new(n: usize) -> Self {
        let log_n = n.trailing_zeros() as usize;

        // Twist factors: ζ^j = e^{πij/N}
        let mut twist_re = vec![0.0; n];
        let mut twist_im = vec![0.0; n];
        for j in 0..n {
            let angle = PI * j as f64 / n as f64;
            twist_re[j] = angle.cos();
            twist_im[j] = angle.sin();
        }

        // Bit-reversal permutation
        let mut bit_rev = vec![0usize; n];
        for i in 0..n {
            let mut rev = 0usize;
            let mut val = i;
            for _ in 0..log_n {
                rev = (rev << 1) | (val & 1);
                val >>= 1;
            }
            bit_rev[i] = rev;
        }

        // Twiddle factors for each FFT stage
        let mut twiddle_re = Vec::with_capacity(log_n);
        let mut twiddle_im = Vec::with_capacity(log_n);
        for s in 0..log_n {
            let half_len = 1 << s;
            let mut tre = Vec::with_capacity(half_len);
            let mut tim = Vec::with_capacity(half_len);
            for k in 0..half_len {
                let angle = -2.0 * PI * k as f64 / (2 * half_len) as f64;
                tre.push(angle.cos());
                tim.push(angle.sin());
            }
            twiddle_re.push(tre);
            twiddle_im.push(tim);
        }

        Self { n, twist_re, twist_im, bit_rev, twiddle_re, twiddle_im }
    }

    /// In-place complex FFT (Cooley-Tukey radix-2 DIT).
    fn fft(&self, re: &mut [f64], im: &mut [f64]) {
        let n = self.n;
        debug_assert_eq!(re.len(), n);
        debug_assert_eq!(im.len(), n);

        // Bit-reversal permutation
        for i in 0..n {
            let j = self.bit_rev[i];
            if i < j {
                re.swap(i, j);
                im.swap(i, j);
            }
        }

        // Butterfly stages
        for s in 0..self.twiddle_re.len() {
            let half_len = 1 << s;
            let full_len = half_len << 1;
            for group_start in (0..n).step_by(full_len) {
                for k in 0..half_len {
                    let w_re = self.twiddle_re[s][k];
                    let w_im = self.twiddle_im[s][k];
                    let i0 = group_start + k;
                    let i1 = i0 + half_len;

                    let v_re = w_re * re[i1] - w_im * im[i1];
                    let v_im = w_re * im[i1] + w_im * re[i1];

                    let u_re = re[i0];
                    let u_im = im[i0];
                    re[i0] = u_re + v_re;
                    im[i0] = u_im + v_im;
                    re[i1] = u_re - v_re;
                    im[i1] = u_im - v_im;
                }
            }
        }
    }

    /// In-place complex inverse FFT (conjugate-FFT-conjugate-scale).
    #[allow(dead_code)]
    fn ifft(&self, re: &mut [f64], im: &mut [f64]) {
        let n = self.n;
        // Conjugate
        for v in im.iter_mut() { *v = -*v; }
        // Forward FFT
        self.fft(re, im);
        // Conjugate + scale by 1/N
        let inv_n = 1.0 / n as f64;
        for i in 0..n {
            re[i] *= inv_n;
            im[i] = -im[i] * inv_n;
        }
    }
}

/// CKKS encoder/decoder for a given parameter set.
#[derive(Debug, Clone)]
pub struct CkksEncoder {
    /// Polynomial degree N.
    n: usize,
    /// Number of SIMD slots = N/2.
    num_slots: usize,
    /// Pre-computed FFT tables for O(N log N) encode/decode.
    fft: FftTables,
}

impl CkksEncoder {
    /// Create a new encoder for the given parameter set.
    pub fn new(params: &CkksParams) -> Self {
        let n = params.poly_degree;
        let num_slots = params.num_slots;
        let fft = FftTables::new(n);

        Self {
            n,
            num_slots,
            fft,
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
    ///
    /// Uses O(N log N) FFT instead of O(N²) DFT.
    ///
    /// z[k] = Σ_j m[j]·ζ^{(2k+1)j} = Σ_j a[j]·ω^{-kj}  (where a[j] = m[j]·ζ^j, ω = e^{-2πi/N})
    ///
    /// Inverting: a[j] = (1/N)·FFT(z̃)[j], then m[j] = Re(a[j]·ζ^{-j})
    ///
    /// For real-valued z: z̃[k] = z[k], z̃[N-1-k] = z[k] (conjugate symmetry).
    fn inverse_canonical_embedding(&self, z: &[f64]) -> Vec<f64> {
        let n = self.n;
        let s = self.num_slots;
        assert_eq!(z.len(), s);

        // Build conjugate-symmetric complex vector
        let mut z_re = vec![0.0f64; n];
        let mut z_im = vec![0.0f64; n];
        for k in 0..s {
            z_re[k] = z[k];
            z_re[n - 1 - k] = z[k];
        }

        // FFT: a[j] = (1/N)·Σ_k z̃[k]·e^{-2πikj/N}
        self.fft.fft(&mut z_re, &mut z_im);

        // Scale by 1/N and un-twist: m[j] = Re(a[j]/N · ζ^{-j})
        // ζ^{-j} = cos(πj/N) - i·sin(πj/N)
        // Re((a_re + i·a_im)(cos - i·sin)) = a_re·cos + a_im·sin
        let inv_n = 1.0 / n as f64;
        let mut coeffs = vec![0.0f64; n];
        for j in 0..n {
            let a_re = z_re[j] * inv_n;
            let a_im = z_im[j] * inv_n;
            coeffs[j] = a_re * self.fft.twist_re[j] + a_im * self.fft.twist_im[j];
        }

        coeffs
    }

    /// Decode from pre-computed float64 polynomial coefficients.
    /// Used when coefficients have already been extracted via CRT reconstruction.
    pub fn decode_coefficients(&self, coeffs: &[f64]) -> Vec<f64> {
        self.canonical_embedding(coeffs)
    }

    /// Canonical embedding: polynomial coefficients → slot values.
    ///
    /// Uses O(N log N) FFT instead of O(N²) DFT.
    ///
    /// z[k] = Re(m(ζ^{2k+1})) = Re(Σ_j a[j]·ω^{-kj})  where a[j] = m[j]·ζ^j
    ///
    /// Computed as: z[k] = Re(conj(FFT(conj(a)))[k]) = Re(FFT(conj(a))[k])
    ///
    /// Returns the real parts of the first N/2 evaluations.
    fn canonical_embedding(&self, coeffs: &[f64]) -> Vec<f64> {
        let n = self.n;
        let s = self.num_slots;
        assert_eq!(coeffs.len(), n);

        // Conjugate-twist: b[j] = conj(a[j]) = m[j]·conj(ζ^j)
        //   b_re[j] = m[j]·cos(πj/N),  b_im[j] = -m[j]·sin(πj/N)
        let mut b_re = vec![0.0f64; n];
        let mut b_im = vec![0.0f64; n];
        for j in 0..n {
            b_re[j] = coeffs[j] * self.fft.twist_re[j];
            b_im[j] = -coeffs[j] * self.fft.twist_im[j];
        }

        // FFT(conj(a)): z[k] = Re(conj(FFT(conj(a))[k])) = Re(FFT(conj(a))[k])
        self.fft.fft(&mut b_re, &mut b_im);

        // Extract real parts of first N/2 values
        b_re.truncate(s);
        b_re
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
