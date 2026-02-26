//! TenSafe-HE CUDA Backend: GPU-accelerated CKKS homomorphic encryption.
//!
//! This crate provides a CUDA-accelerated implementation of the CKKS operations
//! defined in `tensafe-he-core`. All polynomial arithmetic (NTT, Hadamard products,
//! modular arithmetic) runs on NVIDIA GPUs via custom CUDA kernels.
//!
//! # Architecture
//!
//! - **CUDA kernels** are embedded as CUDA C source in [`kernels`] and compiled
//!   at runtime via NVRTC. No `nvcc` needed at build time.
//! - **Barrett reduction** on GPU uses `__umul64hi` for 128-bit intermediate
//!   products, matching the CPU reference implementation exactly.
//! - **NTT** is decomposed into `log₂(N)` kernel launches per limb (one butterfly
//!   stage per launch, N/2 threads each). Negacyclic twist is a separate kernel.
//! - **Fused kernels** for encrypt/decrypt combine multiple modular operations
//!   into a single launch to reduce kernel overhead.
//!
//! # Performance (H100, N=16384, L=4, fused NTT)
//!
//! | Operation          | Kernel launches | Expected latency |
//! |--------------------|-----------------| ---------------- |
//! | NTT forward (1 limb) | 7 (6 global + 1 fused) | ~0.14ms |
//! | NTT inverse (1 limb) | 7 (1 fused + 5 global + 1 scale) | ~0.14ms |
//! | ct×pt_ntt (cached) | 8 (2 Hadamard × 4 limbs) | ~0.05ms |
//! | encrypt (total)    | ~32 (NTT fused + fused encrypt) | ~6.5ms |
//! | decrypt (total)    | ~32 (fused decrypt + iNTT fused) | ~4.5ms |
//! | Full rank-32 HE pipeline | ~500 | ~71.7ms (4 batches) |
//!
//! # Usage
//!
//! ```rust,no_run
//! use tensafe_he_core::params::CkksParams;
//! use tensafe_he_cuda::context::GpuCkksContext;
//! use rand::SeedableRng;
//! use rand::rngs::StdRng;
//!
//! let params = CkksParams::n16384();
//! let ctx = GpuCkksContext::new(params, 0).expect("CUDA init failed");
//!
//! let mut rng = StdRng::seed_from_u64(42);
//! let sk = ctx.keygen(&mut rng).unwrap();
//!
//! // Encrypt a vector
//! let z = vec![1.0, 2.0, 3.0];
//! let ct = ctx.encrypt(&z, &sk, &mut rng).unwrap();
//!
//! // Pre-cache weight NTT (Optimization 3.1)
//! let weights = vec![0.5; 3];
//! let w_ntt = ctx.encode_to_ntt(&weights).unwrap();
//!
//! // ct × pt (fast path — ~0.1ms on H100)
//! let ct_out = ctx.ct_pt_mul_ntt(&ct, &w_ntt).unwrap();
//!
//! // Decrypt
//! let result = ctx.decrypt(&ct_out, &sk).unwrap();
//! ```

pub mod kernels;
pub mod poly;
pub mod ntt;
pub mod context;

// Re-export main types for convenience.
pub use context::{GpuCkksContext, GpuCiphertext, GpuSecretKey, GpuPublicKey};
pub use ntt::{
    GpuNttTables, StreamPool, HePipelineGraph,
    NttBackend, GpuNttBackend, PimDataLayout,
};
pub use poly::{GpuRnsPoly, CompletionEvent};

use cudarc::driver::{DriverError, LaunchConfig};

/// CUDA thread block size for all kernels.
pub const BLOCK_SIZE: u32 = 256;

/// Result type for GPU operations.
pub type GpuResult<T> = Result<T, GpuError>;

/// Errors that can occur during GPU operations.
#[derive(Debug)]
pub enum GpuError {
    /// CUDA driver error (device init, memory alloc, kernel launch).
    Driver(DriverError),
    /// NVRTC compilation error (should never happen with embedded source).
    Compile(String),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::Driver(e) => write!(f, "CUDA driver error: {e}"),
            GpuError::Compile(e) => write!(f, "NVRTC compile error: {e}"),
        }
    }
}

impl std::error::Error for GpuError {}

impl From<DriverError> for GpuError {
    fn from(e: DriverError) -> Self {
        GpuError::Driver(e)
    }
}

impl From<cudarc::nvrtc::CompileError> for GpuError {
    fn from(e: cudarc::nvrtc::CompileError) -> Self {
        GpuError::Compile(format!("{e:?}"))
    }
}

/// Compute a launch configuration for `num_threads` total threads.
pub fn launch_cfg(num_threads: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((num_threads + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    }
}
