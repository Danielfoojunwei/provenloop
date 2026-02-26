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
use crate::{launch_cfg, GpuResult, BLOCK_SIZE};

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
    /// The ψ-based twiddle factors bake the negacyclic twist into the butterfly.
    /// No separate twist kernel needed.
    ///
    /// Stage s: m = 1 << s groups, t = N >> (s+1) half-group size.
    /// Twiddle for group i = forward_twiddles[m + i].
    /// Each stage launches N/2 threads (one per butterfly).
    pub fn negacyclic_ntt_forward(
        &self,
        data: &mut CudaSlice<u64>,
        dev: &Arc<CudaDevice>,
    ) -> GpuResult<()> {
        let half_n = (self.n / 2) as u32;
        let butterfly_cfg = launch_cfg(half_n);

        let mut t = (self.n >> 1) as u32;
        let mut m = 1u32;

        for _ in 0..self.log_n {
            // tw_offset = m (twiddle index base for this stage)
            let f = dev.get_func(MODULE, "ntt_fwd_stage").unwrap();
            unsafe {
                f.launch(
                    butterfly_cfg,
                    (
                        data,
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
        Ok(())
    }

    /// In-place negacyclic inverse NTT on GPU (Gentleman-Sande, SEAL-style).
    ///
    /// Stage s: m = N >> (s+1) groups, t = 1 << s half-group size.
    /// Twiddle for group i = inverse_twiddles[m + i].
    /// Includes N^{-1} normalization.
    pub fn negacyclic_ntt_inverse(
        &self,
        data: &mut CudaSlice<u64>,
        dev: &Arc<CudaDevice>,
    ) -> GpuResult<()> {
        let half_n = (self.n / 2) as u32;
        let butterfly_cfg = launch_cfg(half_n);

        let mut t = 1u32;
        let mut m = (self.n >> 1) as u32;

        for _ in 0..self.log_n {
            // tw_offset = m (twiddle index base for this stage)
            let f = dev.get_func(MODULE, "ntt_inv_stage").unwrap();
            unsafe {
                f.launch(
                    butterfly_cfg,
                    (
                        data,
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
                (data, self.n_inv, self.q, self.barrett_hi, self.n as u32),
            )?;
        }

        Ok(())
    }
}
