//! GPU-resident RNS polynomial.
//!
//! [`GpuRnsPoly`] mirrors [`tensafe_he_core::rns::RnsPoly`] but lives entirely in GPU memory.
//! Each RNS limb is a separate `CudaSlice<u64>` of N coefficients for coalesced access.

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync};
use std::sync::Arc;
use tensafe_he_core::params::Modulus;
use tensafe_he_core::rns::RnsPoly;

use crate::kernels::MODULE;
use crate::{launch_cfg, GpuResult};

/// GPU-resident RNS polynomial: L limbs × N coefficients, each limb on device.
pub struct GpuRnsPoly {
    /// One GPU buffer per RNS limb, each containing N u64 coefficients.
    pub limbs: Vec<CudaSlice<u64>>,
    /// Polynomial degree N.
    pub n: usize,
}

impl GpuRnsPoly {
    /// Allocate a zero polynomial on GPU.
    pub fn zero(dev: &Arc<CudaDevice>, n: usize, num_limbs: usize) -> GpuResult<Self> {
        let mut limbs = Vec::with_capacity(num_limbs);
        for _ in 0..num_limbs {
            limbs.push(dev.alloc_zeros::<u64>(n)?);
        }
        Ok(Self { limbs, n })
    }

    /// Upload a CPU [`RnsPoly`] to GPU memory.
    pub fn from_host(dev: &Arc<CudaDevice>, poly: &RnsPoly) -> GpuResult<Self> {
        let mut limbs = Vec::with_capacity(poly.limbs.len());
        for limb in &poly.limbs {
            limbs.push(dev.htod_copy(limb.clone())?);
        }
        Ok(Self { limbs, n: poly.n })
    }

    /// Download GPU polynomial to CPU [`RnsPoly`].
    pub fn to_host(&self, dev: &Arc<CudaDevice>) -> GpuResult<RnsPoly> {
        let mut limbs = Vec::with_capacity(self.limbs.len());
        for gpu_limb in &self.limbs {
            limbs.push(dev.dtoh_sync_copy(gpu_limb)?);
        }
        Ok(RnsPoly { limbs, n: self.n })
    }

    /// Number of RNS limbs.
    pub fn num_limbs(&self) -> usize {
        self.limbs.len()
    }

    /// Element-wise addition: out = self + other (mod q per limb).
    pub fn add(
        &self,
        other: &Self,
        moduli: &[Modulus],
        dev: &Arc<CudaDevice>,
    ) -> GpuResult<Self> {
        let mut out = Self::zero(dev, self.n, self.num_limbs())?;
        let cfg = launch_cfg(self.n as u32);

        for l in 0..self.num_limbs() {
            let f = dev.get_func(MODULE, "poly_add").unwrap();
            unsafe {
                f.launch(
                    cfg,
                    (
                        &mut out.limbs[l],
                        &self.limbs[l],
                        &other.limbs[l],
                        moduli[l].value,
                        self.n as u32,
                    ),
                )?;
            }
        }
        Ok(out)
    }

    /// Element-wise subtraction: out = self - other (mod q per limb).
    pub fn sub(
        &self,
        other: &Self,
        moduli: &[Modulus],
        dev: &Arc<CudaDevice>,
    ) -> GpuResult<Self> {
        let mut out = Self::zero(dev, self.n, self.num_limbs())?;
        let cfg = launch_cfg(self.n as u32);

        for l in 0..self.num_limbs() {
            let f = dev.get_func(MODULE, "poly_sub").unwrap();
            unsafe {
                f.launch(
                    cfg,
                    (
                        &mut out.limbs[l],
                        &self.limbs[l],
                        &other.limbs[l],
                        moduli[l].value,
                        self.n as u32,
                    ),
                )?;
            }
        }
        Ok(out)
    }

    /// Hadamard (element-wise) multiply: out = self ⊙ other (mod q per limb).
    /// Uses Barrett reduction on GPU.
    pub fn hadamard_mul(
        &self,
        other: &Self,
        moduli: &[Modulus],
        dev: &Arc<CudaDevice>,
    ) -> GpuResult<Self> {
        let mut out = Self::zero(dev, self.n, self.num_limbs())?;
        let cfg = launch_cfg(self.n as u32);

        for l in 0..self.num_limbs() {
            let f = dev.get_func(MODULE, "poly_hadamard").unwrap();
            unsafe {
                f.launch(
                    cfg,
                    (
                        &mut out.limbs[l],
                        &self.limbs[l],
                        &other.limbs[l],
                        moduli[l].value,
                        moduli[l].barrett_hi,
                        self.n as u32,
                    ),
                )?;
            }
        }
        Ok(out)
    }

    /// Negate all coefficients: out[i] = -self[i] mod q.
    pub fn negate(
        &self,
        moduli: &[Modulus],
        dev: &Arc<CudaDevice>,
    ) -> GpuResult<Self> {
        let mut out = Self::zero(dev, self.n, self.num_limbs())?;
        let cfg = launch_cfg(self.n as u32);

        for l in 0..self.num_limbs() {
            let f = dev.get_func(MODULE, "poly_negate").unwrap();
            unsafe {
                f.launch(
                    cfg,
                    (
                        &mut out.limbs[l],
                        &self.limbs[l],
                        moduli[l].value,
                        self.n as u32,
                    ),
                )?;
            }
        }
        Ok(out)
    }
}
