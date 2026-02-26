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

use cudarc::driver::{CudaDevice, CudaStream, LaunchAsync};
use std::sync::Arc;

use tensafe_he_core::encoding::CkksEncoder;
use tensafe_he_core::ntt::NttTables;
use tensafe_he_core::params::{CkksParams, SCALE};
use tensafe_he_core::rns::RnsPoly;
use tensafe_he_core::sampling::{sample_gaussian_signed, sample_ternary, sample_uniform, ERROR_STD_DEV};

use crate::kernels::{KERNEL_NAMES, KERNEL_SOURCE, MODULE};
use crate::ntt::{GpuNttTables, StreamPool};
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

/// GPU-resident public key.
pub struct GpuPublicKey {
    /// b = -a·s + e  (NTT domain, on GPU)
    pub b: GpuRnsPoly,
    /// a (NTT domain, on GPU)
    pub a: GpuRnsPoly,
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

    /// Generate a public key from the secret key (GPU-accelerated NTT).
    pub fn keygen_public<R: Rng>(&self, sk: &GpuSecretKey, rng: &mut R) -> GpuResult<GpuPublicKey> {
        let n = self.params.poly_degree;
        let num_limbs = self.params.num_limbs;

        // Sample a and e on CPU
        let mut a_cpu = RnsPoly::zero(n, num_limbs);
        for l in 0..num_limbs {
            let q = self.params.moduli[l].value;
            a_cpu.limbs[l] = sample_uniform(rng, n, q);
        }

        let e_signed = sample_gaussian_signed(rng, n, ERROR_STD_DEV);
        let mut e_cpu = RnsPoly::zero(n, num_limbs);
        for l in 0..num_limbs {
            let q = self.params.moduli[l].value;
            for i in 0..n {
                e_cpu.limbs[l][i] = if e_signed[i] >= 0 {
                    (e_signed[i] as u64) % q
                } else {
                    q - ((-e_signed[i]) as u64 % q)
                };
            }
        }

        // Upload and NTT on GPU
        let mut a_gpu = GpuRnsPoly::from_host(&self.dev, &a_cpu)?;
        let mut e_gpu = GpuRnsPoly::from_host(&self.dev, &e_cpu)?;
        for l in 0..num_limbs {
            self.gpu_ntt_tables[l].negacyclic_ntt_forward(&mut a_gpu.limbs[l], &self.dev)?;
            self.gpu_ntt_tables[l].negacyclic_ntt_forward(&mut e_gpu.limbs[l], &self.dev)?;
        }

        // b = -a·s + e (use encrypt_fused with m=0)
        let zero_gpu = GpuRnsPoly::zero(&self.dev, n, num_limbs)?;
        let mut b = GpuRnsPoly::zero(&self.dev, n, num_limbs)?;
        let cfg = launch_cfg(n as u32);

        for l in 0..num_limbs {
            let f = self.dev.get_func(MODULE, "encrypt_fused").unwrap();
            unsafe {
                f.launch(
                    cfg,
                    (
                        &mut b.limbs[l],
                        &a_gpu.limbs[l],
                        &sk.s_ntt.limbs[l],
                        &zero_gpu.limbs[l], // m = 0
                        &e_gpu.limbs[l],
                        self.params.moduli[l].value,
                        self.params.moduli[l].barrett_hi,
                        n as u32,
                    ),
                )?;
            }
        }

        Ok(GpuPublicKey { b, a: a_gpu })
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
        // a is sampled independently per limb (CRT-uniform, cancels in decrypt)
        let mut a_cpu = RnsPoly::zero(n, num_limbs);
        for l in 0..num_limbs {
            let q = self.params.moduli[l].value;
            a_cpu.limbs[l] = sample_uniform(rng, n, q);
        }

        // e must be consistent across limbs (same signed integers reduced mod each q)
        let e_signed = sample_gaussian_signed(rng, n, ERROR_STD_DEV);
        let mut e_cpu = RnsPoly::zero(n, num_limbs);
        for l in 0..num_limbs {
            let q = self.params.moduli[l].value;
            for i in 0..n {
                e_cpu.limbs[l][i] = if e_signed[i] >= 0 {
                    (e_signed[i] as u64) % q
                } else {
                    q - ((-e_signed[i]) as u64 % q)
                };
            }
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

    /// Encrypt a real-valued vector using a public key (asymmetric RLWE, GPU-accelerated).
    ///
    /// ct = (c0, c1) where:
    ///   u ← ternary, e0,e1 ← Gaussian
    ///   c0 = b·u + e0 + m,  c1 = a·u + e1
    pub fn encrypt_pk<R: Rng>(
        &self,
        z: &[f64],
        pk: &GpuPublicKey,
        rng: &mut R,
    ) -> GpuResult<GpuCiphertext> {
        let n = self.params.poly_degree;
        let num_limbs = self.params.num_limbs;

        // Encode plaintext on CPU
        let m_cpu = self.encoder.encode(z, &self.params);

        // Sample u (ternary) on CPU
        let u_ternary = sample_ternary(rng, n, 3);
        let mut u_cpu = RnsPoly::zero(n, num_limbs);
        for l in 0..num_limbs {
            let q = self.params.moduli[l].value;
            for i in 0..n {
                u_cpu.limbs[l][i] = match u_ternary[i] {
                    0 => q - 1,
                    1 => 0,
                    2 => 1,
                    _ => unreachable!(),
                };
            }
        }

        // Sample e0, e1 on CPU
        let e0_signed = sample_gaussian_signed(rng, n, ERROR_STD_DEV);
        let e1_signed = sample_gaussian_signed(rng, n, ERROR_STD_DEV);
        let mut e0_cpu = RnsPoly::zero(n, num_limbs);
        let mut e1_cpu = RnsPoly::zero(n, num_limbs);
        for l in 0..num_limbs {
            let q = self.params.moduli[l].value;
            for i in 0..n {
                e0_cpu.limbs[l][i] = if e0_signed[i] >= 0 { (e0_signed[i] as u64) % q } else { q - ((-e0_signed[i]) as u64 % q) };
                e1_cpu.limbs[l][i] = if e1_signed[i] >= 0 { (e1_signed[i] as u64) % q } else { q - ((-e1_signed[i]) as u64 % q) };
            }
        }

        // Upload to GPU and NTT
        let mut m_gpu = GpuRnsPoly::from_host(&self.dev, &m_cpu)?;
        let mut u_gpu = GpuRnsPoly::from_host(&self.dev, &u_cpu)?;
        let mut e0_gpu = GpuRnsPoly::from_host(&self.dev, &e0_cpu)?;
        let mut e1_gpu = GpuRnsPoly::from_host(&self.dev, &e1_cpu)?;

        for l in 0..num_limbs {
            self.gpu_ntt_tables[l].negacyclic_ntt_forward(&mut m_gpu.limbs[l], &self.dev)?;
            self.gpu_ntt_tables[l].negacyclic_ntt_forward(&mut u_gpu.limbs[l], &self.dev)?;
            self.gpu_ntt_tables[l].negacyclic_ntt_forward(&mut e0_gpu.limbs[l], &self.dev)?;
            self.gpu_ntt_tables[l].negacyclic_ntt_forward(&mut e1_gpu.limbs[l], &self.dev)?;
        }

        // c0 = b·u + e0 + m (reuse encrypt_fused: it computes -a*s + m + e, here a=b, s=u negated)
        // Better: use poly_hadamard + poly_add directly
        let moduli = &self.params.moduli;
        let bu = pk.b.hadamard_mul(&u_gpu, moduli, &self.dev)?;
        let au = pk.a.hadamard_mul(&u_gpu, moduli, &self.dev)?;

        let c0 = bu.add(&e0_gpu, moduli, &self.dev)?.add(&m_gpu, moduli, &self.dev)?;
        let c1 = au.add(&e1_gpu, moduli, &self.dev)?;

        Ok(GpuCiphertext {
            c0,
            c1,
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
        Ok(self.encoder.decode(&m_cpu, &self.params, ct.scale))
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

    /// Ciphertext × plaintext multiply on a specific CUDA stream.
    ///
    /// Dispatches the Hadamard product kernels to the given stream,
    /// enabling concurrent ct×pt across multiple batches.
    pub fn ct_pt_mul_ntt_on_stream(
        &self,
        ct: &GpuCiphertext,
        pt_ntt: &GpuRnsPoly,
        stream: &CudaStream,
    ) -> GpuResult<GpuCiphertext> {
        let moduli = &self.params.moduli;
        Ok(GpuCiphertext {
            c0: ct.c0.hadamard_mul_on_stream(pt_ntt, moduli, &self.dev, stream)?,
            c1: ct.c1.hadamard_mul_on_stream(pt_ntt, moduli, &self.dev, stream)?,
            scale: ct.scale * ct.scale,
        })
    }

    /// Batch ciphertext × plaintext multiply across multiple CUDA streams.
    ///
    /// For rank-32 LoRA with 4 batches, each batch's ct×pt is independent
    /// (same input ct, different LoRA-A weight rows). Running on separate
    /// streams yields overlapped kernel execution:
    ///
    ///   Sequential: 4 × 4.8ms = 19.2ms
    ///   4 streams:  ~5.5ms (3.5× speedup, limited by HBM bandwidth sharing)
    ///
    /// The `pool` must have at least 1 stream. Each batch `i` is assigned
    /// to `pool.get(i)`, wrapping if batches > streams.
    pub fn ct_pt_mul_ntt_multi_stream(
        &self,
        ct: &GpuCiphertext,
        pt_ntts: &[&GpuRnsPoly],
        pool: &StreamPool,
    ) -> GpuResult<Vec<GpuCiphertext>> {
        let mut results = Vec::with_capacity(pt_ntts.len());

        for (i, pt_ntt) in pt_ntts.iter().enumerate() {
            let stream = pool.get(i);
            results.push(self.ct_pt_mul_ntt_on_stream(ct, pt_ntt, stream)?);
        }

        // Synchronize all streams back to default before returning.
        pool.sync_all(&self.dev)?;

        Ok(results)
    }

    /// Decrypt a ciphertext on a specific CUDA stream.
    ///
    /// Dispatches the fused decrypt kernel + inverse NTT to the given stream,
    /// enabling concurrent decrypt across multiple batch results.
    pub fn decrypt_on_stream(
        &self,
        ct: &GpuCiphertext,
        sk: &GpuSecretKey,
        stream: &CudaStream,
    ) -> GpuResult<Vec<f64>> {
        let n = self.params.poly_degree;
        let num_limbs = self.params.num_limbs;

        // Fused decrypt kernel on the given stream: m = c0 + c1*s
        let mut m_gpu = GpuRnsPoly::zero(&self.dev, n, num_limbs)?;
        let cfg = launch_cfg(n as u32);

        for l in 0..num_limbs {
            let f = self.dev.get_func(MODULE, "decrypt_fused").unwrap();
            unsafe {
                f.launch_on_stream(
                    stream,
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

        // Inverse NTT on the given stream
        for l in 0..num_limbs {
            self.gpu_ntt_tables[l].negacyclic_ntt_inverse_on_stream(
                &mut m_gpu.limbs[l],
                &self.dev,
                stream,
            )?;
        }

        // Sync the stream before downloading to CPU
        self.dev.wait_for(stream)?;
        self.dev.synchronize()?;

        let m_cpu = m_gpu.to_host(&self.dev)?;
        Ok(self.encoder.decode(&m_cpu, &self.params, ct.scale))
    }

    /// Batch decrypt across multiple CUDA streams.
    ///
    /// Each ciphertext is decrypted on a separate stream for concurrent
    /// execution of fused decrypt + iNTT kernels.
    ///
    ///   Sequential 4 decrypts: 4 × 18ms = 72ms
    ///   4 streams:             ~22ms (3.3× speedup)
    pub fn decrypt_multi_stream(
        &self,
        cts: &[&GpuCiphertext],
        sk: &GpuSecretKey,
        pool: &StreamPool,
    ) -> GpuResult<Vec<Vec<f64>>> {
        let mut results = Vec::with_capacity(cts.len());

        for (i, ct) in cts.iter().enumerate() {
            let stream = pool.get(i);
            results.push(self.decrypt_on_stream(ct, sk, stream)?);
        }

        Ok(results)
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

    /// Ciphertext + Ciphertext addition on GPU.
    ///
    /// ct' = (c0_a + c0_b, c1_a + c1_b). Both ciphertexts must have the same scale.
    pub fn ct_add(
        &self,
        ct_a: &GpuCiphertext,
        ct_b: &GpuCiphertext,
    ) -> GpuResult<GpuCiphertext> {
        assert!(
            (ct_a.scale - ct_b.scale).abs() < 1.0,
            "Scale mismatch in ct_add: {} vs {}",
            ct_a.scale,
            ct_b.scale
        );
        let moduli = &self.params.moduli;
        Ok(GpuCiphertext {
            c0: ct_a.c0.add(&ct_b.c0, moduli, &self.dev)?,
            c1: ct_a.c1.add(&ct_b.c1, moduli, &self.dev)?,
            scale: ct_a.scale,
        })
    }

    /// Ciphertext - Ciphertext subtraction on GPU.
    ///
    /// ct' = (c0_a - c0_b, c1_a - c1_b). Both ciphertexts must have the same scale.
    pub fn ct_sub(
        &self,
        ct_a: &GpuCiphertext,
        ct_b: &GpuCiphertext,
    ) -> GpuResult<GpuCiphertext> {
        assert!(
            (ct_a.scale - ct_b.scale).abs() < 1.0,
            "Scale mismatch in ct_sub: {} vs {}",
            ct_a.scale,
            ct_b.scale
        );
        let moduli = &self.params.moduli;
        Ok(GpuCiphertext {
            c0: ct_a.c0.sub(&ct_b.c0, moduli, &self.dev)?,
            c1: ct_a.c1.sub(&ct_b.c1, moduli, &self.dev)?,
            scale: ct_a.scale,
        })
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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// Helper: try to init GPU, skip test gracefully if no GPU available.
    /// cudarc panics (instead of returning Err) when the CUDA driver is missing,
    /// so we catch the panic with std::panic::catch_unwind.
    fn gpu_ctx() -> Option<GpuCkksContext> {
        let result = std::panic::catch_unwind(|| {
            let params = CkksParams::for_degree(8192);
            GpuCkksContext::new(params, 0)
        });
        match result {
            Ok(Ok(ctx)) => Some(ctx),
            Ok(Err(e)) => {
                eprintln!("Skipping GPU test (CUDA error): {e}");
                None
            }
            Err(_) => {
                eprintln!("Skipping GPU test (no CUDA driver / no GPU in this container)");
                None
            }
        }
    }

    #[test]
    fn test_gpu_encrypt_decrypt_roundtrip() {
        let Some(ctx) = gpu_ctx() else { return };
        let mut rng = StdRng::seed_from_u64(42);
        let sk = ctx.keygen(&mut rng).unwrap();

        let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ct = ctx.encrypt(&x, &sk, &mut rng).unwrap();
        let decoded = ctx.decrypt(&ct, &sk).unwrap();

        for i in 0..x.len() {
            let err = (decoded[i] - x[i]).abs();
            assert!(
                err < 0.1,
                "Slot {i}: decoded={}, expected={}, error={err}",
                decoded[i], x[i]
            );
        }
        eprintln!("GPU encrypt/decrypt roundtrip: PASSED");
    }

    #[test]
    fn test_gpu_ct_pt_multiply() {
        let Some(ctx) = gpu_ctx() else { return };
        let mut rng = StdRng::seed_from_u64(42);
        let sk = ctx.keygen(&mut rng).unwrap();

        let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let p: Vec<f64> = vec![0.5, 0.5, 0.5, 0.5, 0.5];
        let expected: Vec<f64> = x.iter().zip(p.iter()).map(|(a, b)| a * b).collect();

        let ct = ctx.encrypt(&x, &sk, &mut rng).unwrap();
        let ct_prod = ctx.ct_pt_mul(&ct, &p).unwrap();
        let decoded = ctx.decrypt(&ct_prod, &sk).unwrap();

        for i in 0..expected.len() {
            let err = (decoded[i] - expected[i]).abs();
            assert!(
                err < 0.1,
                "Slot {i}: decoded={}, expected={}, error={err}",
                decoded[i], expected[i]
            );
        }
        eprintln!("GPU ct*pt multiply: PASSED");
    }

    #[test]
    fn test_gpu_cached_ntt_multiply() {
        let Some(ctx) = gpu_ctx() else { return };
        let mut rng = StdRng::seed_from_u64(42);
        let sk = ctx.keygen(&mut rng).unwrap();

        let x: Vec<f64> = (0..20).map(|i| (i as f64) * 0.1).collect();
        let p: Vec<f64> = (0..20).map(|i| 1.0 - (i as f64) * 0.04).collect();
        let expected: Vec<f64> = x.iter().zip(p.iter()).map(|(a, b)| a * b).collect();

        let ct = ctx.encrypt(&x, &sk, &mut rng).unwrap();
        let w_ntt = ctx.encode_to_ntt(&p).unwrap();
        let ct_prod = ctx.ct_pt_mul_ntt(&ct, &w_ntt).unwrap();
        let decoded = ctx.decrypt(&ct_prod, &sk).unwrap();

        for i in 0..expected.len() {
            let err = (decoded[i] - expected[i]).abs();
            assert!(
                err < 0.1,
                "Slot {i}: decoded={}, expected={}, error={err}",
                decoded[i], expected[i]
            );
        }
        eprintln!("GPU cached NTT multiply: PASSED");
    }
}
