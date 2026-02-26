//! GPU NTT tables and transform operations.
//!
//! Pre-computes twiddle factors on the CPU, uploads them to GPU memory,
//! and provides methods to execute forward/inverse negacyclic NTT entirely on GPU.
//!
//! The twiddle factors use ψ (primitive 2N-th root), so the negacyclic twist
//! is baked into the butterfly — no separate twist kernel needed.
//!
//! Each NTT is log₂(N) butterfly kernel launches + 1 normalization for inverse.
//! For N=16384: 14 launches for forward, 15 for inverse, per RNS limb.

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync};
use std::sync::Arc;
use tensafe_he_core::ntt::NttTables;
use tensafe_he_core::params::Modulus;

use crate::kernels::MODULE;
use crate::{launch_cfg, GpuResult};

/// GPU-resident NTT tables for a single (N, q) pair.
///
/// Contains all pre-computed data needed to run negacyclic NTT on GPU:
/// - Twiddle factors (powers of ψ in bit-reversed order, baking in the negacyclic twist)
/// - Modulus parameters for Barrett reduction
pub struct GpuNttTables {
    /// Forward twiddle factors on GPU (N elements, bit-reversed order).
    pub forward_twiddles: CudaSlice<u64>,
    /// Inverse twiddle factors on GPU (N elements, bit-reversed order).
    pub inverse_twiddles: CudaSlice<u64>,
    /// N^{-1} mod q — normalization factor for iNTT.
    pub n_inv: u64,
    /// The modulus q.
    pub q: u64,
    /// Barrett constant for this modulus.
    pub barrett_hi: u64,
    /// log₂(N).
    pub log_n: u32,
    /// Polynomial degree N.
    pub n: usize,
}

impl GpuNttTables {
    /// Create GPU NTT tables from CPU tables + modulus.
    ///
    /// Pre-computes psi power arrays on CPU and uploads everything to GPU.
    pub fn from_cpu(
        dev: &Arc<CudaDevice>,
        cpu_tables: &NttTables,
        modulus: &Modulus,
    ) -> GpuResult<Self> {
        let n = cpu_tables.n;
        let q = cpu_tables.q;

        // Upload twiddle factors (ψ-based, negacyclic twist baked in)
        let forward_twiddles = dev.htod_copy(cpu_tables.forward_twiddles.clone())?;
        let inverse_twiddles = dev.htod_copy(cpu_tables.inverse_twiddles.clone())?;

        Ok(Self {
            forward_twiddles,
            inverse_twiddles,
            n_inv: cpu_tables.n_inv,
            q,
            barrett_hi: modulus.barrett_hi,
            log_n: cpu_tables.log_n,
            n,
        })
    }

    /// In-place negacyclic forward NTT on GPU (Cooley-Tukey, SEAL-style).
    ///
    /// Uses fused shared-memory kernel for stages where butterfly span fits
    /// in a thread block (last ~9 stages), reducing total launches from
    /// log_n (~14) to ~6.
    pub fn negacyclic_ntt_forward(
        &self,
        data: &mut CudaSlice<u64>,
        dev: &Arc<CudaDevice>,
    ) -> GpuResult<()> {
        let half_n = (self.n / 2) as u32;
        let butterfly_cfg = launch_cfg(half_n);
        let block_size = crate::BLOCK_SIZE;

        // Determine first fused stage: t <= BLOCK_SIZE
        // At stage s: t = N >> (s+1). Fuse when t <= block_size.
        // first_fused_stage = log_n - log2(block_size) - 1 + 1 = log_n - log2(block_size)
        let log_block = block_size.trailing_zeros();
        let first_fused = if self.log_n > log_block { self.log_n - log_block } else { 0 };

        // Global stages (t > block_size): individual kernel launches
        let mut t = (self.n >> 1) as u32;
        let mut m = 1u32;
        for _ in 0..first_fused {
            let f = dev.get_func(MODULE, "ntt_fwd_stage").unwrap();
            unsafe {
                f.launch(
                    butterfly_cfg,
                    (
                        &mut *data,
                        &self.forward_twiddles,
                        self.q,
                        self.barrett_hi,
                        t,
                        m,
                    ),
                )?;
            }
            t >>= 1;
            m <<= 1;
        }

        // Fused stages (t <= block_size): one shared-memory kernel launch
        if first_fused < self.log_n {
            let num_blocks = (self.n as u32) / (2 * block_size);
            let fused_cfg = cudarc::driver::LaunchConfig {
                grid_dim: (num_blocks.max(1), 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 2 * block_size * 8, // 2B u64 elements
            };
            let f = dev.get_func(MODULE, "ntt_fwd_fused").unwrap();
            unsafe {
                f.launch(
                    fused_cfg,
                    (
                        &mut *data,
                        &self.forward_twiddles,
                        self.q,
                        self.barrett_hi,
                        self.log_n,
                        first_fused,
                    ),
                )?;
            }
        }

        Ok(())
    }

    /// In-place negacyclic inverse NTT on GPU (Gentleman-Sande, SEAL-style).
    ///
    /// Uses fused shared-memory kernel for stages where butterfly span fits
    /// in a thread block (first ~9 stages for inverse), then individual
    /// launches for remaining stages.
    pub fn negacyclic_ntt_inverse(
        &self,
        data: &mut CudaSlice<u64>,
        dev: &Arc<CudaDevice>,
    ) -> GpuResult<()> {
        let half_n = (self.n / 2) as u32;
        let butterfly_cfg = launch_cfg(half_n);
        let block_size = crate::BLOCK_SIZE;

        // For inverse NTT: t starts at 1 and doubles.
        // Fuse stages where t <= block_size (first log2(block_size)+1 stages).
        let log_block = block_size.trailing_zeros();
        let num_fused = (log_block + 1).min(self.log_n);

        // Fused stages (t = 1 to block_size): one shared-memory kernel
        if num_fused > 0 {
            let num_blocks = (self.n as u32) / (2 * block_size);
            let fused_cfg = cudarc::driver::LaunchConfig {
                grid_dim: (num_blocks.max(1), 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 2 * block_size * 8,
            };
            let f = dev.get_func(MODULE, "ntt_inv_fused").unwrap();
            unsafe {
                f.launch(
                    fused_cfg,
                    (
                        &mut *data,
                        &self.inverse_twiddles,
                        self.q,
                        self.barrett_hi,
                        num_fused,
                    ),
                )?;
            }
        }

        // Global stages (t > block_size): individual kernel launches
        let mut t = 1u32 << num_fused;
        let mut m = (self.n >> 1) as u32 >> num_fused;
        for _ in num_fused..self.log_n {
            let f = dev.get_func(MODULE, "ntt_inv_stage").unwrap();
            unsafe {
                f.launch(
                    butterfly_cfg,
                    (
                        &mut *data,
                        &self.inverse_twiddles,
                        self.q,
                        self.barrett_hi,
                        t,
                        m,
                    ),
                )?;
            }
            t <<= 1;
            m >>= 1;
        }

        // Normalize by N^{-1} mod q
        let scale_cfg = launch_cfg(self.n as u32);
        let f = dev.get_func(MODULE, "poly_scale").unwrap();
        unsafe {
            f.launch(
                scale_cfg,
                (&mut *data, self.n_inv, self.q, self.barrett_hi, self.n as u32),
            )?;
        }

        Ok(())
    }
}
