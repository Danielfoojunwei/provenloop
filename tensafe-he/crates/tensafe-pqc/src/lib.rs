//! TenSafe Post-Quantum Cryptography.
//!
//! Implements ML-KEM-768 (Kyber) and ML-DSA-65 (Dilithium) using the
//! NTT infrastructure from `tensafe-he-core`. Both algorithms operate
//! on polynomial rings with NTT-friendly moduli, allowing us to reuse
//! the same Cooley-Tukey butterfly code.
//!
//! # NTT Parameter Sharing
//!
//! | Algorithm | Ring | Modulus q | Degree n | NTT size |
//! |-----------|------|-----------|----------|----------|
//! | CKKS (HE) | Z_q[X]/(X^N+1) | 40-60 bit primes | 8192-32768 | N |
//! | Kyber768 | Z_3329[X]/(X^256+1) | 3329 | 256 | 128* |
//! | Dilithium3 | Z_8380417[X]/(X^256+1) | 8380417 | 256 | 256 |
//!
//! *Kyber uses a 128-point NTT because 3329 ≡ 1 (mod 256) but 3329 ≢ 1 (mod 512).
//! Dilithium's 8380417 ≡ 1 (mod 512), so a full 256-point negacyclic NTT works.

pub mod ntt_pqc;
pub mod kyber;
pub mod dilithium;
