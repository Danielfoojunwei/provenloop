//! Cryptographic sampling for CKKS.
//!
//! - Secret key: ternary distribution {-1, 0, 1}
//! - Error vectors: discrete Gaussian with σ ≈ 3.19
//! - Random polynomials: uniform in Z_q

use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Standard deviation for RLWE error distribution.
pub const ERROR_STD_DEV: f64 = 3.19;

/// Sample a ternary secret key: each coefficient ∈ {-1, 0, 1}.
/// The distribution is: P(-1) = 1/3, P(0) = 1/3, P(1) = 1/3.
///
/// Returns coefficients in [0, q) representation:
/// -1 is stored as q-1, 0 as 0, 1 as 1.
pub fn sample_ternary<R: Rng>(rng: &mut R, n: usize, q: u64) -> Vec<u64> {
    (0..n)
        .map(|_| {
            let r = rng.random_range(0u32..3);
            match r {
                0 => q - 1, // -1 mod q
                1 => 0,
                2 => 1,
                _ => unreachable!(),
            }
        })
        .collect()
}

/// Sample a discrete Gaussian error vector with standard deviation σ.
///
/// Returns coefficients in [0, q) representation (signed values mapped to mod q).
pub fn sample_gaussian<R: Rng>(rng: &mut R, n: usize, q: u64, sigma: f64) -> Vec<u64> {
    let normal = Normal::new(0.0, sigma).expect("Invalid sigma for Gaussian");

    (0..n)
        .map(|_| {
            let sample: f64 = normal.sample(rng);
            let rounded = sample.round() as i64;
            if rounded >= 0 {
                (rounded as u64) % q
            } else {
                q - ((-rounded) as u64 % q)
            }
        })
        .collect()
}

/// Sample a uniform random polynomial in Z_q^N.
/// Used for the 'a' component in RLWE encryption.
pub fn sample_uniform<R: Rng>(rng: &mut R, n: usize, q: u64) -> Vec<u64> {
    (0..n).map(|_| rng.random_range(0..q)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_ternary_distribution() {
        let mut rng = StdRng::seed_from_u64(42);
        let q = 97u64;
        let n = 10000;
        let samples = sample_ternary(&mut rng, n, q);

        // Count distribution
        let neg_ones = samples.iter().filter(|&&x| x == q - 1).count();
        let zeros = samples.iter().filter(|&&x| x == 0).count();
        let ones = samples.iter().filter(|&&x| x == 1).count();

        // Each should be roughly n/3
        let expected = n / 3;
        let tolerance = (n as f64 * 0.05) as usize; // 5% tolerance
        assert!(
            (neg_ones as isize - expected as isize).unsigned_abs() < tolerance,
            "neg_ones={neg_ones}, expected≈{expected}"
        );
        assert!(
            (zeros as isize - expected as isize).unsigned_abs() < tolerance,
            "zeros={zeros}, expected≈{expected}"
        );
        assert!(
            (ones as isize - expected as isize).unsigned_abs() < tolerance,
            "ones={ones}, expected≈{expected}"
        );

        // All values should be in {0, 1, q-1}
        for &s in &samples {
            assert!(s == 0 || s == 1 || s == q - 1, "Invalid ternary value: {s}");
        }
    }

    #[test]
    fn test_gaussian_distribution() {
        let mut rng = StdRng::seed_from_u64(42);
        let q = 1099511627777u64; // 40-bit prime
        let n = 10000;
        let sigma = ERROR_STD_DEV;
        let samples = sample_gaussian(&mut rng, n, q, sigma);

        // Convert back to signed for statistics
        let signed: Vec<f64> = samples
            .iter()
            .map(|&s| {
                if s > q / 2 {
                    -((q - s) as f64)
                } else {
                    s as f64
                }
            })
            .collect();

        let mean: f64 = signed.iter().sum::<f64>() / n as f64;
        let variance: f64 = signed.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let measured_sigma = variance.sqrt();

        // Mean should be close to 0
        assert!(
            mean.abs() < 0.2,
            "Gaussian mean too far from 0: {mean}"
        );
        // Sigma should be close to 3.19
        assert!(
            (measured_sigma - sigma).abs() < 0.3,
            "Gaussian sigma off: measured={measured_sigma}, expected={sigma}"
        );
    }

    #[test]
    fn test_uniform_range() {
        let mut rng = StdRng::seed_from_u64(42);
        let q = 97u64;
        let samples = sample_uniform(&mut rng, 1000, q);
        for &s in &samples {
            assert!(s < q, "Uniform sample {s} >= q={q}");
        }
    }
}
