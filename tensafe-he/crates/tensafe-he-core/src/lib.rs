//! TenSafe-HE Core: CKKS homomorphic encryption library optimized for ZeRo-MOAI inference.
//!
//! This library implements the minimal CKKS operation set needed for TenSafe:
//! - Encode/Decode (canonical embedding)
//! - Encrypt/Decrypt (RLWE)
//! - ct × pt multiply (element-wise in NTT domain)
//! - ct + ct / ct - ct (ciphertext addition/subtraction)
//! - NTT/iNTT (negacyclic Number Theoretic Transform)
//!
//! NOT implemented (by design — ZeRo-MOAI eliminates the need):
//! - Ciphertext rotation (Galois automorphism)
//! - ct × ct multiply
//! - Relinearization / key switching
//! - Bootstrapping

pub mod params;
pub mod rns;
pub mod ntt;
pub mod sampling;
pub mod encoding;
pub mod ciphertext;
pub mod serialize;
