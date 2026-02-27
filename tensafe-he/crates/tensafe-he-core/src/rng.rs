//! Custom ChaCha20-based CSPRNG for TenSafe-HE.
//!
//! Replaces the `rand` + `rand_distr` crates with a minimal, self-contained
//! implementation that covers exactly the API surface we need:
//!
//! - Uniform u32/u64 integers in a range (rejection sampling)
//! - Box-Muller Gaussian sampling
//! - Seedable from u64 (deterministic for tests)
//! - Seedable from OS entropy (`/dev/urandom`)
//!
//! The ChaCha20 quarter-round provides the raw pseudorandom bits.
//! See IETF RFC 8439 §2.1-2.3 for the ChaCha20 specification.

use std::f64::consts::PI;

/// ChaCha20-based cryptographically secure PRNG.
///
/// Generates 64 bytes (16 × u32) per block via 20 rounds of ChaCha quarter-rounds.
/// Outputs are consumed from the buffer; when exhausted, the counter increments
/// and a new block is generated.
#[derive(Debug, Clone)]
pub struct TenSafeRng {
    /// ChaCha20 state: [constant(4), key(8), counter(1), nonce(3)].
    state: [u32; 16],
    /// Output buffer from the last ChaCha20 block.
    buffer: [u32; 16],
    /// Current position in the buffer (0..16). When == 16, generate next block.
    index: usize,
}

impl TenSafeRng {
    /// Create a deterministic RNG from a 64-bit seed.
    ///
    /// The seed is expanded into the 256-bit ChaCha20 key by repeating it
    /// across the 8 key words. Nonce is zeroed. Counter starts at 0.
    ///
    /// This is intended for **reproducible tests**, not production encryption.
    pub fn from_seed(seed: u64) -> Self {
        let lo = seed as u32;
        let hi = (seed >> 32) as u32;

        // "expand 32-byte k" constant (RFC 8439 §2.3)
        let mut state = [0u32; 16];
        state[0] = 0x6170_7865; // "expa"
        state[1] = 0x3320_646e; // "nd 3"
        state[2] = 0x7962_2d32; // "2-by"
        state[3] = 0x6b20_6574; // "te k"

        // Key: repeat seed across 8 words
        for i in 0..4 {
            state[4 + i * 2] = lo;
            state[5 + i * 2] = hi;
        }

        // Counter = 0, Nonce = 0
        state[12] = 0;
        state[13] = 0;
        state[14] = 0;
        state[15] = 0;

        let mut rng = Self {
            state,
            buffer: [0u32; 16],
            index: 16, // force generation on first use
        };
        rng.refill();
        rng
    }

    /// Create an RNG seeded from OS entropy (`/dev/urandom` on Linux).
    ///
    /// Falls back to a time-based seed if `/dev/urandom` is unavailable.
    pub fn from_entropy() -> Self {
        let seed = read_os_entropy();
        Self::from_seed(seed)
    }

    /// Generate the next ChaCha20 block and reset the buffer index.
    fn refill(&mut self) {
        self.buffer = chacha20_block(&self.state);
        self.index = 0;
        // Increment the 32-bit counter (word 12)
        self.state[12] = self.state[12].wrapping_add(1);
    }

    /// Return the next u32 from the ChaCha20 stream.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        if self.index >= 16 {
            self.refill();
        }
        let val = self.buffer[self.index];
        self.index += 1;
        val
    }

    /// Return the next u64 from the ChaCha20 stream.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let lo = self.next_u32() as u64;
        let hi = self.next_u32() as u64;
        (hi << 32) | lo
    }

    /// Return a uniform random u32 in `[0, max)` using rejection sampling.
    ///
    /// Rejection ensures uniform distribution even when `max` doesn't divide 2^32.
    #[inline]
    pub fn range_u32(&mut self, max: u32) -> u32 {
        debug_assert!(max > 0, "range_u32: max must be > 0");
        // Threshold for rejection: largest multiple of max that fits in u32
        let threshold = (u32::MAX - max + 1) % max;
        loop {
            let r = self.next_u32();
            // Accept if r >= threshold (rejection zone is [0, threshold))
            if r >= threshold {
                return r % max;
            }
        }
    }

    /// Return a uniform random u64 in `[0, max)` using rejection sampling.
    #[inline]
    pub fn range_u64(&mut self, max: u64) -> u64 {
        debug_assert!(max > 0, "range_u64: max must be > 0");
        let threshold = (u64::MAX - max + 1) % max;
        loop {
            let r = self.next_u64();
            if r >= threshold {
                return r % max;
            }
        }
    }

    /// Return a f64 uniformly distributed in `(0, 1)`.
    ///
    /// Uses 53 bits of randomness (full f64 mantissa precision).
    #[inline]
    fn next_f64(&mut self) -> f64 {
        // Use 53 random bits for the mantissa: (u64 >> 11) / 2^53
        // Add 0.5 ULP to avoid exact 0.0 (needed for log in Box-Muller)
        let bits = self.next_u64() >> 11;
        (bits as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// Sample from a Gaussian distribution with mean 0 and standard deviation `sigma`.
    ///
    /// Uses the Box-Muller transform: given U1, U2 ~ Uniform(0,1),
    ///   Z = sqrt(-2 ln U1) * cos(2π U2)
    /// produces Z ~ Normal(0, 1), then scale by sigma.
    #[inline]
    pub fn gaussian(&mut self, sigma: f64) -> f64 {
        let u1 = self.next_f64();
        let u2 = self.next_f64();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        z * sigma
    }
}

/// ChaCha20 block function: 20 rounds of quarter-rounds on the state.
///
/// Returns the output block (state + working_state after 20 rounds).
fn chacha20_block(state: &[u32; 16]) -> [u32; 16] {
    let mut working = *state;

    // 20 rounds = 10 double-rounds (column round + diagonal round)
    for _ in 0..10 {
        // Column rounds
        quarter_round(&mut working, 0, 4, 8, 12);
        quarter_round(&mut working, 1, 5, 9, 13);
        quarter_round(&mut working, 2, 6, 10, 14);
        quarter_round(&mut working, 3, 7, 11, 15);
        // Diagonal rounds
        quarter_round(&mut working, 0, 5, 10, 15);
        quarter_round(&mut working, 1, 6, 11, 12);
        quarter_round(&mut working, 2, 7, 8, 13);
        quarter_round(&mut working, 3, 4, 9, 14);
    }

    // Add original state (mod 2^32)
    let mut output = [0u32; 16];
    for i in 0..16 {
        output[i] = working[i].wrapping_add(state[i]);
    }
    output
}

/// ChaCha20 quarter-round operation (RFC 8439 §2.1).
#[inline(always)]
fn quarter_round(s: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
    s[a] = s[a].wrapping_add(s[b]);
    s[d] ^= s[a];
    s[d] = s[d].rotate_left(16);

    s[c] = s[c].wrapping_add(s[d]);
    s[b] ^= s[c];
    s[b] = s[b].rotate_left(12);

    s[a] = s[a].wrapping_add(s[b]);
    s[d] ^= s[a];
    s[d] = s[d].rotate_left(8);

    s[c] = s[c].wrapping_add(s[d]);
    s[b] ^= s[c];
    s[b] = s[b].rotate_left(7);
}

/// Read 8 bytes from `/dev/urandom` and interpret as u64.
/// Falls back to a time-based seed on failure.
fn read_os_entropy() -> u64 {
    use std::io::Read;
    let mut buf = [0u8; 8];
    if let Ok(mut f) = std::fs::File::open("/dev/urandom") {
        if f.read_exact(&mut buf).is_ok() {
            return u64::from_le_bytes(buf);
        }
    }
    // Fallback: use the address of a stack variable XOR'd with a constant
    let fallback = &buf as *const _ as u64;
    fallback ^ 0xdeadbeef_cafebabe
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chacha20_rfc8439_vector() {
        // RFC 8439 §2.3.2 test vector
        let mut state = [0u32; 16];
        // Constants
        state[0] = 0x61707865;
        state[1] = 0x3320646e;
        state[2] = 0x79622d32;
        state[3] = 0x6b206574;
        // Key
        state[4] = 0x03020100;
        state[5] = 0x07060504;
        state[6] = 0x0b0a0908;
        state[7] = 0x0f0e0d0c;
        state[8] = 0x13121110;
        state[9] = 0x17161514;
        state[10] = 0x1b1a1918;
        state[11] = 0x1f1e1d1c;
        // Counter = 1
        state[12] = 0x00000001;
        // Nonce
        state[13] = 0x09000000;
        state[14] = 0x4a000000;
        state[15] = 0x00000000;

        let output = chacha20_block(&state);

        // Expected output from RFC 8439 §2.3.2
        assert_eq!(output[0], 0xe4e7f110);
        assert_eq!(output[1], 0x15593bd1);
        assert_eq!(output[2], 0x1fdd0f50);
        assert_eq!(output[3], 0xc47120a3);
        assert_eq!(output[4], 0xc7f4d1c7);
        assert_eq!(output[5], 0x0368c033);
        assert_eq!(output[6], 0x9aaa2204);
        assert_eq!(output[7], 0x4e6cd4c3);
        assert_eq!(output[8], 0x466482d2);
        assert_eq!(output[9], 0x09aa9f07);
        assert_eq!(output[10], 0x05d7c214);
        assert_eq!(output[11], 0xa2028bd9);
        assert_eq!(output[12], 0xd19c12b5);
        assert_eq!(output[13], 0xb94e16de);
        assert_eq!(output[14], 0xe883d0cb);
        assert_eq!(output[15], 0x4e3c50a2);
    }

    #[test]
    fn test_deterministic_seeding() {
        let mut rng1 = TenSafeRng::from_seed(42);
        let mut rng2 = TenSafeRng::from_seed(42);
        for _ in 0..1000 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_different_seeds_produce_different_output() {
        let mut rng1 = TenSafeRng::from_seed(42);
        let mut rng2 = TenSafeRng::from_seed(43);
        let mut same = 0;
        for _ in 0..100 {
            if rng1.next_u64() == rng2.next_u64() {
                same += 1;
            }
        }
        assert!(same < 5, "Different seeds produced too many identical values: {same}");
    }

    #[test]
    fn test_range_u32_uniform() {
        let mut rng = TenSafeRng::from_seed(42);
        let max = 3u32;
        let n = 30000;
        let mut counts = [0usize; 3];
        for _ in 0..n {
            let v = rng.range_u32(max);
            assert!(v < max);
            counts[v as usize] += 1;
        }
        // Each bucket should be ~10000 ± 5%
        for (i, &c) in counts.iter().enumerate() {
            let expected = n / max as usize;
            let tolerance = (n as f64 * 0.05) as usize;
            assert!(
                (c as isize - expected as isize).unsigned_abs() < tolerance,
                "Bucket {i}: count={c}, expected~{expected}"
            );
        }
    }

    #[test]
    fn test_range_u64_bounds() {
        let mut rng = TenSafeRng::from_seed(42);
        let q = 1099511627777u64; // 40-bit prime
        for _ in 0..10000 {
            let v = rng.range_u64(q);
            assert!(v < q, "range_u64 produced {v} >= {q}");
        }
    }

    #[test]
    fn test_gaussian_distribution() {
        let mut rng = TenSafeRng::from_seed(42);
        let sigma = 3.19;
        let n = 50000;
        let samples: Vec<f64> = (0..n).map(|_| rng.gaussian(sigma)).collect();

        let mean: f64 = samples.iter().sum::<f64>() / n as f64;
        let variance: f64 =
            samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let measured_sigma = variance.sqrt();

        assert!(
            mean.abs() < 0.1,
            "Gaussian mean too far from 0: {mean}"
        );
        assert!(
            (measured_sigma - sigma).abs() < 0.15,
            "Gaussian sigma off: measured={measured_sigma:.3}, expected={sigma}"
        );
    }

    #[test]
    fn test_next_f64_range() {
        let mut rng = TenSafeRng::from_seed(42);
        for _ in 0..10000 {
            let v = rng.next_f64();
            assert!(v > 0.0, "next_f64 produced {v} <= 0.0");
            assert!(v < 1.0, "next_f64 produced {v} >= 1.0");
        }
    }

    #[test]
    fn test_from_entropy_produces_output() {
        let mut rng = TenSafeRng::from_entropy();
        // Just verify it produces data without panicking
        let v = rng.next_u64();
        // Should not be all zeros (astronomically unlikely with a good RNG)
        let v2 = rng.next_u64();
        assert!(v != 0 || v2 != 0, "Entropy RNG produced all zeros");
    }
}
