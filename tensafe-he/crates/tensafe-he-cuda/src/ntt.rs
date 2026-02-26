//! GPU NTT tables and transform operations.
//!
//! Pre-computes twiddle factors and psi powers on the CPU, uploads them to GPU memory,
//! and provides methods to execute forward/inverse negacyclic NTT entirely on GPU.
//!
//! Each NTT (forward or inverse) is log₂(N) kernel launches for the butterfly stages,
//! plus 1-2 kernel launches for the negacyclic twist and normalization.
//!
//! For N=16384: 15 kernel launches for forward, 16 for inverse, per RNS limb.

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use std::sync::Arc;
use tensafe_he_core::ntt::NttTables;
use tensafe_he_core::params::Modulus;
use tensafe_he_core::rns::mod_mul;

use crate::kernels::MODULE;
use crate::{launch_cfg, GpuResult, BLOCK_SIZE};

/// GPU-resident NTT tables for a single (N, q) pair.
///
/// Contains all pre-computed data needed to run negacyclic NTT on GPU:
/// - Twiddle factors (bit-reversed, matching CPU layout)
/// - Psi powers for negacyclic twist
/// - Modulus parameters for Barrett reduction
pub struct GpuNttTables {
    /// Forward twiddle factors on GPU (N elements, bit-reversed order).
    pub forward_twiddles: CudaSlice<u64>,
    /// Inverse twiddle factors on GPU (N elements, bit-reversed order).
    pub inverse_twiddles: CudaSlice<u64>,
    /// Forward psi powers: [ψ^0, ψ^1, ..., ψ^{N-1}] on GPU.
    pub psi_powers: CudaSlice<u64>,
    /// Inverse psi powers: [ψ^{-0}, ψ^{-1}, ..., ψ^{-(N-1)}] on GPU.
    pub inv_psi_powers: CudaSlice<u64>,
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

        // Upload twiddle factors
        let forward_twiddles = dev.htod_copy(cpu_tables.forward_twiddles.clone())?;
        let inverse_twiddles = dev.htod_copy(cpu_tables.inverse_twiddles.clone())?;

        // Precompute psi power arrays for the twist operations.
        // CPU code uses: psi = forward_twiddles[1], psi_inv = inverse_twiddles[1]
        let psi = cpu_tables.forward_twiddles[1];
        let psi_inv = cpu_tables.inverse_twiddles[1];

        let mut psi_pows = Vec::with_capacity(n);
        let mut inv_psi_pows = Vec::with_capacity(n);
        let mut p = 1u64;
        let mut ip = 1u64;
        for _ in 0..n {
            psi_pows.push(p);
            inv_psi_pows.push(ip);
            p = mod_mul(p, psi, q);
            ip = mod_mul(ip, psi_inv, q);
        }

        let psi_powers = dev.htod_copy(psi_pows)?;
        let inv_psi_powers = dev.htod_copy(inv_psi_pows)?;

        Ok(Self {
            forward_twiddles,
            inverse_twiddles,
            psi_powers,
            inv_psi_powers,
            n_inv: cpu_tables.n_inv,
            q,
            barrett_hi: modulus.barrett_hi,
            log_n: cpu_tables.log_n,
            n,
        })
    }

    /// In-place forward NTT on GPU (Cooley-Tukey, no twist).
    pub fn ntt_forward(&self, data: &mut CudaSlice<u64>, dev: &Arc<CudaDevice>) -> GpuResult<()> {
        let half_n = (self.n / 2) as u32;
        let butterfly_cfg = launch_cfg(half_n);

        for s in 0..self.log_n {
            let t = 1u32 << s;
            // Twiddle offset for forward NTT stage s:
            //   offset(s) = 1 + N - (N >> s)
            let tw_offset = 1 + self.n as u32 - (self.n as u32 >> s);

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
                        tw_offset,
                    ),
                )?;
            }
        }
        Ok(())
    }

    /// In-place inverse NTT on GPU (Gentleman-Sande, no untwist).
    /// Includes the N^{-1} normalization.
    pub fn ntt_inverse(
        &self,
        data: &mut CudaSlice<u64>,
        dev: &Arc<CudaDevice>,
    ) -> GpuResult<()> {
        let half_n = (self.n / 2) as u32;
        let butterfly_cfg = launch_cfg(half_n);

        for s in 0..self.log_n {
            // Inverse NTT stage s: t = N >> (s+1), m = 1 << s
            let t = (self.n as u32) >> (s + 1);
            let tw_offset = 1u32 << s;

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
                        tw_offset,
                    ),
                )?;
            }
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

    /// Apply forward negacyclic twist: data[i] *= ψ^i mod q.
    pub fn apply_forward_twist(
        &self,
        data: &mut CudaSlice<u64>,
        dev: &Arc<CudaDevice>,
    ) -> GpuResult<()> {
        let cfg = launch_cfg(self.n as u32);
        let f = dev.get_func(MODULE, "fwd_twist").unwrap();
        unsafe {
            f.launch(
                cfg,
                (
                    data,
                    &self.psi_powers,
                    self.q,
                    self.barrett_hi,
                    self.n as u32,
                ),
            )?;
        }
        Ok(())
    }

    /// Apply inverse negacyclic twist: data[i] *= ψ^{-i} mod q.
    pub fn apply_inverse_twist(
        &self,
        data: &mut CudaSlice<u64>,
        dev: &Arc<CudaDevice>,
    ) -> GpuResult<()> {
        let cfg = launch_cfg(self.n as u32);
        let f = dev.get_func(MODULE, "inv_twist").unwrap();
        unsafe {
            f.launch(
                cfg,
                (
                    data,
                    &self.inv_psi_powers,
                    self.q,
                    self.barrett_hi,
                    self.n as u32,
                ),
            )?;
        }
        Ok(())
    }

    /// Full negacyclic forward NTT: twist → NTT.
    pub fn negacyclic_ntt_forward(
        &self,
        data: &mut CudaSlice<u64>,
        dev: &Arc<CudaDevice>,
    ) -> GpuResult<()> {
        self.apply_forward_twist(data, dev)?;
        self.ntt_forward(data, dev)?;
        Ok(())
    }

    /// Full negacyclic inverse NTT: iNTT → untwist.
    pub fn negacyclic_ntt_inverse(
        &self,
        data: &mut CudaSlice<u64>,
        dev: &Arc<CudaDevice>,
    ) -> GpuResult<()> {
        self.ntt_inverse(data, dev)?;
        self.apply_inverse_twist(data, dev)?;
        Ok(())
    }
}
