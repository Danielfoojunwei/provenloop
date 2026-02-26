//! GPU-accelerated CKKS context: encrypt, decrypt, ct×pt multiply.
//!
//! [`GpuCkksContext`] is the main entry point for TenSafe-HE on GPU.
//! It mirrors [`tensafe_he_core::ciphertext::CkksContext`] but runs all
//! polynomial arithmetic on the GPU via CUDA kernels.
//!
//! Workflow:
//! 1. Encoding/decoding stays on CPU (I/O-bound, not worth GPU transfer overhead)
//! 2. Sampling stays on CPU (random, then upload once)
//! 3. NTT, Hadamard products, encrypt/decrypt fused ops run on GPU

use cudarc::driver::{CudaDevice, LaunchAsync};
use std::sync::Arc;

use tensafe_he_core::encoding::CkksEncoder;
use tensafe_he_core::ntt::NttTables;
use tensafe_he_core::params::{CkksParams, SCALE};
use tensafe_he_core::rns::RnsPoly;
use tensafe_he_core::sampling::{sample_gaussian, sample_ternary, sample_uniform, ERROR_STD_DEV};

use crate::kernels::{KERNEL_NAMES, KERNEL_SOURCE, MODULE};
use crate::ntt::GpuNttTables;
use crate::poly::GpuRnsPoly;
use crate::{launch_cfg, GpuResult};

use rand::Rng;

/// GPU-resident CKKS ciphertext.
pub struct GpuCiphertext {
    /// First component c0 (NTT domain, on GPU).
    pub c0: GpuRnsPoly,
    /// Second component c1 (NTT domain, on GPU).
    pub c1: GpuRnsPoly,
    /// Current scale factor Δ.
    pub scale: f64,
}

/// GPU-resident secret key.
pub struct GpuSecretKey {
    /// Secret key polynomial in NTT domain (on GPU).
    pub s_ntt: GpuRnsPoly,
}

/// GPU-accelerated CKKS context.
///
/// Owns the CUDA device handle, compiled kernel module, pre-computed GPU NTT tables,
/// and CPU-side encoder. All heavy polynomial arithmetic dispatches to GPU kernels.
pub struct GpuCkksContext {
    /// CUDA device handle (shared reference).
    pub dev: Arc<CudaDevice>,
    /// CKKS parameter set.
    pub params: CkksParams,
    /// CPU-side NTT tables (used for table data, not computation).
    cpu_ntt_tables: Vec<NttTables>,
    /// GPU-resident NTT tables, one per RNS limb.
    pub gpu_ntt_tables: Vec<GpuNttTables>,
    /// CPU-side encoder (canonical embedding DFT).
    pub encoder: CkksEncoder,
}

impl GpuCkksContext {
    /// Create a new GPU CKKS context.
    ///
    /// - Initializes CUDA device `ordinal` (typically 0)
    /// - Compiles CUDA kernels via NVRTC
    /// - Pre-computes and uploads NTT tables to GPU
    ///
    /// This is an expensive one-time setup (~100ms for kernel compilation).
    pub fn new(params: CkksParams, device_ordinal: usize) -> GpuResult<Self> {
        let dev = CudaDevice::new(device_ordinal)?;

        // Compile and load CUDA kernels
        let ptx = cudarc::nvrtc::compile_ptx(KERNEL_SOURCE)?;
        dev.load_ptx(ptx, MODULE, KERNEL_NAMES)?;

        // Build CPU NTT tables (needed for twiddle factor data)
        let cpu_ntt_tables: Vec<NttTables> = params
            .moduli
            .iter()
            .map(|m| NttTables::new(params.poly_degree, m.value))
            .collect();

        // Upload NTT tables to GPU
        let gpu_ntt_tables: Vec<GpuNttTables> = cpu_ntt_tables
            .iter()
            .zip(params.moduli.iter())
            .map(|(tables, modulus)| GpuNttTables::from_cpu(&dev, tables, modulus))
            .collect::<Result<Vec<_>, _>>()?;

        let encoder = CkksEncoder::new(&params);

        Ok(Self {
            dev,
            params,
            cpu_ntt_tables,
            gpu_ntt_tables,
            encoder,
        })
    }

    /// Generate a secret key.
    ///
    /// Ternary coefficients are sampled on CPU, then transformed to NTT domain on GPU.
    pub fn keygen<R: Rng>(&self, rng: &mut R) -> GpuResult<GpuSecretKey> {
        let n = self.params.poly_degree;
        let num_limbs = self.params.num_limbs;

        // Sample ternary secret on CPU
        let s_ternary = sample_ternary(rng, n, 3);

        // Convert to each RNS limb representation
        let mut s_cpu = RnsPoly::zero(n, num_limbs);
        for l in 0..num_limbs {
            let q = self.params.moduli[l].value;
            for i in 0..n {
                s_cpu.limbs[l][i] = match s_ternary[i] {
                    0 => q - 1, // -1 mod q
                    1 => 0,     // 0
                    2 => 1,     // 1
                    _ => unreachable!(),
                };
            }
        }

        // Upload to GPU
        let mut s_ntt = GpuRnsPoly::from_host(&self.dev, &s_cpu)?;

        // Transform each limb to NTT domain on GPU
        for l in 0..num_limbs {
            self.gpu_ntt_tables[l].negacyclic_ntt_forward(&mut s_ntt.limbs[l], &self.dev)?;
        }

        Ok(GpuSecretKey { s_ntt })
    }

    /// Encrypt a real-valued vector.
    ///
    /// RLWE encryption: ct = (c0, c1) where
    ///   c0 = -a·s + m + e   (NTT domain)
    ///   c1 = a               (NTT domain)
    ///
    /// Encoding + sampling on CPU, NTT + fused encrypt kernel on GPU.
    pub fn encrypt<R: Rng>(
        &self,
        z: &[f64],
        sk: &GpuSecretKey,
        rng: &mut R,
    ) -> GpuResult<GpuCiphertext> {
        let n = self.params.poly_degree;
        let num_limbs = self.params.num_limbs;

        // 1. Encode plaintext on CPU
        let m_cpu = self.encoder.encode(z, &self.params);

        // 2. Sample a and e on CPU
        let mut a_cpu = RnsPoly::zero(n, num_limbs);
        let mut e_cpu = RnsPoly::zero(n, num_limbs);
        for l in 0..num_limbs {
            let q = self.params.moduli[l].value;
            a_cpu.limbs[l] = sample_uniform(rng, n, q);
            e_cpu.limbs[l] = sample_gaussian(rng, n, q, ERROR_STD_DEV);
        }

        // 3. Upload m, a, e to GPU
        let mut m_gpu = GpuRnsPoly::from_host(&self.dev, &m_cpu)?;
        let mut a_gpu = GpuRnsPoly::from_host(&self.dev, &a_cpu)?;
        let mut e_gpu = GpuRnsPoly::from_host(&self.dev, &e_cpu)?;

        // 4. NTT all three on GPU
        for l in 0..num_limbs {
            self.gpu_ntt_tables[l].negacyclic_ntt_forward(&mut m_gpu.limbs[l], &self.dev)?;
            self.gpu_ntt_tables[l].negacyclic_ntt_forward(&mut a_gpu.limbs[l], &self.dev)?;
            self.gpu_ntt_tables[l].negacyclic_ntt_forward(&mut e_gpu.limbs[l], &self.dev)?;
        }

        // 5. Fused encrypt kernel: c0 = -a*s + m + e
        let mut c0 = GpuRnsPoly::zero(&self.dev, n, num_limbs)?;
        let cfg = launch_cfg(n as u32);

        for l in 0..num_limbs {
            let f = self.dev.get_func(MODULE, "encrypt_fused").unwrap();
            unsafe {
                f.launch(
                    cfg,
                    (
                        &mut c0.limbs[l],
                        &a_gpu.limbs[l],
                        &sk.s_ntt.limbs[l],
                        &m_gpu.limbs[l],
                        &e_gpu.limbs[l],
                        self.params.moduli[l].value,
                        self.params.moduli[l].barrett_hi,
                        n as u32,
                    ),
                )?;
            }
        }

        Ok(GpuCiphertext {
            c0,
            c1: a_gpu,
            scale: SCALE,
        })
    }

    /// Decrypt a ciphertext to a real-valued vector.
    ///
    /// GPU: m_ntt = c0 + c1·s (fused kernel per limb)
    /// GPU: iNTT(m_ntt) per limb
    /// CPU: decode polynomial to float vector
    pub fn decrypt(&self, ct: &GpuCiphertext, sk: &GpuSecretKey) -> GpuResult<Vec<f64>> {
        let n = self.params.poly_degree;
        let num_limbs = self.params.num_limbs;

        // 1. Fused decrypt kernel: m = c0 + c1*s
        let mut m_gpu = GpuRnsPoly::zero(&self.dev, n, num_limbs)?;
        let cfg = launch_cfg(n as u32);

        for l in 0..num_limbs {
            let f = self.dev.get_func(MODULE, "decrypt_fused").unwrap();
            unsafe {
                f.launch(
                    cfg,
                    (
                        &mut m_gpu.limbs[l],
                        &ct.c0.limbs[l],
                        &ct.c1.limbs[l],
                        &sk.s_ntt.limbs[l],
                        self.params.moduli[l].value,
                        self.params.moduli[l].barrett_hi,
                        n as u32,
                    ),
                )?;
            }
        }

        // 2. Inverse NTT on GPU
        for l in 0..num_limbs {
            self.gpu_ntt_tables[l].negacyclic_ntt_inverse(&mut m_gpu.limbs[l], &self.dev)?;
        }

        // 3. Download to CPU and decode
        let m_cpu = m_gpu.to_host(&self.dev)?;
        Ok(self.encoder.decode(&m_cpu, &self.params))
    }

    /// Ciphertext × plaintext multiply.
    ///
    /// Encodes the plaintext, transforms to NTT domain on GPU, then performs
    /// element-wise Hadamard product in NTT domain.
    pub fn ct_pt_mul(
        &self,
        ct: &GpuCiphertext,
        pt: &[f64],
    ) -> GpuResult<GpuCiphertext> {
        let pt_ntt = self.encode_to_ntt(pt)?;
        self.ct_pt_mul_ntt(ct, &pt_ntt)
    }

    /// Ciphertext × plaintext multiply with pre-cached NTT plaintext.
    ///
    /// This is the fast path (Optimization 3.1): the plaintext NTT is computed
    /// once and reused across many inferences.
    ///
    /// ct' = (c0 ⊙ pt_ntt, c1 ⊙ pt_ntt)
    pub fn ct_pt_mul_ntt(
        &self,
        ct: &GpuCiphertext,
        pt_ntt: &GpuRnsPoly,
    ) -> GpuResult<GpuCiphertext> {
        let moduli = &self.params.moduli;
        Ok(GpuCiphertext {
            c0: ct.c0.hadamard_mul(pt_ntt, moduli, &self.dev)?,
            c1: ct.c1.hadamard_mul(pt_ntt, moduli, &self.dev)?,
            scale: ct.scale * ct.scale,
        })
    }

    /// Encode a plaintext vector and transform to NTT domain on GPU.
    ///
    /// This is Optimization 3.1: pre-compute NTT(weights) once per layer,
    /// cache on GPU, reuse across all inference requests.
    pub fn encode_to_ntt(&self, z: &[f64]) -> GpuResult<GpuRnsPoly> {
        // Encode on CPU
        let poly_cpu = self.encoder.encode(z, &self.params);

        // Upload to GPU
        let mut poly_gpu = GpuRnsPoly::from_host(&self.dev, &poly_cpu)?;

        // NTT on GPU
        for l in 0..self.params.num_limbs {
            self.gpu_ntt_tables[l].negacyclic_ntt_forward(&mut poly_gpu.limbs[l], &self.dev)?;
        }

        Ok(poly_gpu)
    }

    /// Upload a CPU ciphertext to GPU.
    pub fn upload_ciphertext(
        &self,
        ct: &tensafe_he_core::ciphertext::Ciphertext,
    ) -> GpuResult<GpuCiphertext> {
        Ok(GpuCiphertext {
            c0: GpuRnsPoly::from_host(&self.dev, &ct.c0)?,
            c1: GpuRnsPoly::from_host(&self.dev, &ct.c1)?,
            scale: ct.scale,
        })
    }

    /// Download a GPU ciphertext to CPU.
    pub fn download_ciphertext(
        &self,
        ct: &GpuCiphertext,
    ) -> GpuResult<tensafe_he_core::ciphertext::Ciphertext> {
        Ok(tensafe_he_core::ciphertext::Ciphertext {
            c0: ct.c0.to_host(&self.dev)?,
            c1: ct.c1.to_host(&self.dev)?,
            scale: ct.scale,
        })
    }
}
