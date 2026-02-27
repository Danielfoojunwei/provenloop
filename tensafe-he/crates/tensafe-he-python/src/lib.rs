//! Python bindings for TenSafe-HE CKKS homomorphic encryption.
//!
//! Exposes the core CKKS operations to Python via PyO3:
//! - `CkksContext`: parameter setup, key generation
//! - `TenSafeCiphertext`: encrypted data with `__mul__` and `__add__`
//! - `encrypt()` / `decrypt()`: RLWE encrypt/decrypt
//!
//! This replaces CuKKS/OpenFHE/Pyfhel with our own Rust CKKS implementation.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use tensafe_he_core::ciphertext::{CkksContext as CoreContext, Ciphertext, PublicKey, SecretKey};
use tensafe_he_core::params::CkksParams;
use tensafe_he_core::rng::TenSafeRng;

/// Python-visible CKKS context. Holds parameters, NTT tables, encoder.
#[pyclass]
struct CkksContext {
    ctx: CoreContext,
    poly_degree: usize,
    num_slots: usize,
}

#[pymethods]
impl CkksContext {
    /// Create a CKKS context with the given polynomial degree.
    ///
    /// Args:
    ///     poly_mod_degree: Polynomial ring degree (8192, 16384, or 32768).
    ///
    /// Returns:
    ///     CkksContext ready for key generation and encryption.
    #[new]
    #[pyo3(signature = (poly_mod_degree=16384))]
    fn new(poly_mod_degree: usize) -> PyResult<Self> {
        let params = match poly_mod_degree {
            8192 => CkksParams::for_degree(8192),
            16384 => CkksParams::n16384(),
            32768 => CkksParams::for_degree(32768),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported poly_mod_degree: {poly_mod_degree}. Use 8192, 16384, or 32768."
                )));
            }
        };
        let num_slots = params.poly_degree / 2;
        let poly_degree = params.poly_degree;
        let ctx = CoreContext::new(params);
        Ok(Self {
            ctx,
            poly_degree,
            num_slots,
        })
    }

    /// Create a context optimized for a given multiplicative depth.
    ///
    /// depth=3 → poly_n=32768 (256-bit security, 16384 SIMD slots)
    /// depth=1 → poly_n=16384 (192-bit security, 8192 SIMD slots)
    #[staticmethod]
    #[pyo3(signature = (depth=3))]
    fn for_depth(depth: usize) -> PyResult<Self> {
        let poly_n = match depth {
            1 => 8192,
            2 => 16384,
            3 | 4 | 5 => 32768,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported depth: {depth}. Use 1-5."
                )));
            }
        };
        Self::new(poly_n)
    }

    /// Number of SIMD slots (N/2).
    #[getter]
    fn num_slots(&self) -> usize {
        self.num_slots
    }

    /// Polynomial degree N.
    #[getter]
    fn poly_mod_degree(&self) -> usize {
        self.poly_degree
    }

    /// Generate a secret key.
    #[pyo3(signature = (seed=None))]
    fn keygen(&self, seed: Option<u64>) -> TenSafeSecretKey {
        let mut rng = match seed {
            Some(s) => TenSafeRng::from_seed(s),
            None => TenSafeRng::from_entropy(),
        };
        let sk = self.ctx.keygen(&mut rng);
        TenSafeSecretKey { inner: sk }
    }

    /// Generate a public key from a secret key.
    #[pyo3(signature = (sk, seed=None))]
    fn keygen_public(&self, sk: &TenSafeSecretKey, seed: Option<u64>) -> TenSafePublicKey {
        let mut rng = match seed {
            Some(s) => TenSafeRng::from_seed(s),
            None => TenSafeRng::from_entropy(),
        };
        let pk = self.ctx.keygen_public(&sk.inner, &mut rng);
        TenSafePublicKey { inner: pk }
    }

    /// Encrypt a float64 numpy array using the secret key.
    ///
    /// The array is padded/truncated to `num_slots` elements.
    #[pyo3(signature = (data, sk, seed=None))]
    fn encrypt<'py>(
        &self,
        _py: Python<'py>,
        data: PyReadonlyArray1<'py, f64>,
        sk: &TenSafeSecretKey,
        seed: Option<u64>,
    ) -> PyResult<TenSafeCiphertext> {
        let slice = data.as_slice()?;
        let mut padded = vec![0.0f64; self.num_slots];
        let n = slice.len().min(self.num_slots);
        padded[..n].copy_from_slice(&slice[..n]);

        let mut rng = match seed {
            Some(s) => TenSafeRng::from_seed(s),
            None => TenSafeRng::from_entropy(),
        };
        let ct = self.ctx.encrypt(&padded, &sk.inner, &mut rng);
        Ok(TenSafeCiphertext {
            inner: ct,
            num_slots: self.num_slots,
            cached_ctx: None,
        })
    }

    /// Encrypt using a public key (asymmetric encryption).
    #[pyo3(signature = (data, pk, seed=None))]
    fn encrypt_pk<'py>(
        &self,
        _py: Python<'py>,
        data: PyReadonlyArray1<'py, f64>,
        pk: &TenSafePublicKey,
        seed: Option<u64>,
    ) -> PyResult<TenSafeCiphertext> {
        let slice = data.as_slice()?;
        let mut padded = vec![0.0f64; self.num_slots];
        let n = slice.len().min(self.num_slots);
        padded[..n].copy_from_slice(&slice[..n]);

        let mut rng = match seed {
            Some(s) => TenSafeRng::from_seed(s),
            None => TenSafeRng::from_entropy(),
        };
        let ct = self.ctx.encrypt_pk(&padded, &pk.inner, &mut rng);
        Ok(TenSafeCiphertext {
            inner: ct,
            num_slots: self.num_slots,
            cached_ctx: None,
        })
    }

    /// Decrypt a ciphertext to a numpy float64 array.
    fn decrypt<'py>(
        &self,
        py: Python<'py>,
        ct: &TenSafeCiphertext,
        sk: &TenSafeSecretKey,
    ) -> Bound<'py, PyArray1<f64>> {
        let decoded = self.ctx.decrypt(&ct.inner, &sk.inner);
        PyArray1::from_vec(py, decoded)
    }

    /// Ciphertext × plaintext multiply.
    fn ct_pt_mul<'py>(
        &self,
        _py: Python<'py>,
        ct: &TenSafeCiphertext,
        plaintext: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<TenSafeCiphertext> {
        let slice = plaintext.as_slice()?;
        let mut padded = vec![0.0f64; self.num_slots];
        let n = slice.len().min(self.num_slots);
        padded[..n].copy_from_slice(&slice[..n]);

        let result = self.ctx.ct_pt_mul(&ct.inner, &padded);
        Ok(TenSafeCiphertext {
            inner: result,
            num_slots: self.num_slots,
            cached_ctx: None,
        })
    }

    /// Ciphertext + Ciphertext addition.
    fn ct_add(&self, ct_a: &TenSafeCiphertext, ct_b: &TenSafeCiphertext) -> TenSafeCiphertext {
        let result = self.ctx.ct_add(&ct_a.inner, &ct_b.inner);
        TenSafeCiphertext {
            inner: result,
            num_slots: self.num_slots,
            cached_ctx: None,
        }
    }

    /// Ciphertext - Ciphertext subtraction.
    fn ct_sub(&self, ct_a: &TenSafeCiphertext, ct_b: &TenSafeCiphertext) -> TenSafeCiphertext {
        let result = self.ctx.ct_sub(&ct_a.inner, &ct_b.inner);
        TenSafeCiphertext {
            inner: result,
            num_slots: self.num_slots,
            cached_ctx: None,
        }
    }

    /// Rescale after multiplication (drop last RNS limb, halve the scale).
    fn rescale(&self, ct: &TenSafeCiphertext) -> TenSafeCiphertext {
        let result = self.ctx.rescale(&ct.inner);
        TenSafeCiphertext {
            inner: result,
            num_slots: self.num_slots,
            cached_ctx: None,
        }
    }

    /// Batch encrypt multiple numpy arrays (one encrypt call per array).
    /// Returns a list of ciphertexts. More efficient than individual calls
    /// as RNG state is maintained across encryptions.
    #[pyo3(signature = (arrays, sk, seed=None))]
    fn batch_encrypt<'py>(
        &self,
        _py: Python<'py>,
        arrays: Vec<PyReadonlyArray1<'py, f64>>,
        sk: &TenSafeSecretKey,
        seed: Option<u64>,
    ) -> PyResult<Vec<TenSafeCiphertext>> {
        let mut rng = match seed {
            Some(s) => TenSafeRng::from_seed(s),
            None => TenSafeRng::from_entropy(),
        };

        let mut results = Vec::with_capacity(arrays.len());
        for arr in &arrays {
            let slice = arr.as_slice()?;
            let mut padded = vec![0.0f64; self.num_slots];
            let n = slice.len().min(self.num_slots);
            padded[..n].copy_from_slice(&slice[..n]);

            let ct = self.ctx.encrypt(&padded, &sk.inner, &mut rng);
            results.push(TenSafeCiphertext {
                inner: ct,
                num_slots: self.num_slots,
                cached_ctx: None,
            });
        }
        Ok(results)
    }

    /// Batch decrypt multiple ciphertexts. Returns a list of numpy arrays.
    fn batch_decrypt<'py>(
        &self,
        py: Python<'py>,
        cts: Vec<PyRef<'py, TenSafeCiphertext>>,
        sk: &TenSafeSecretKey,
    ) -> Vec<Bound<'py, PyArray1<f64>>> {
        cts.iter()
            .map(|ct| {
                let decoded = self.ctx.decrypt(&ct.inner, &sk.inner);
                PyArray1::from_vec(py, decoded)
            })
            .collect()
    }

    /// Check if this backend is available (always true for Rust native).
    #[staticmethod]
    fn is_available() -> bool {
        true
    }

    /// Return backend metadata.
    fn get_backend_info(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("backend", "tensafe-he-native")?;
            dict.set_item("version", env!("CARGO_PKG_VERSION"))?;
            dict.set_item("language", "Rust")?;
            dict.set_item("gpu_accelerated", false)?;
            dict.set_item("poly_mod_degree", self.poly_degree)?;
            dict.set_item("num_slots", self.num_slots)?;
            dict.set_item("num_limbs", self.ctx.params.num_limbs)?;
            Ok(dict.into_any().unbind())
        })
    }
}

/// Python-visible secret key.
#[pyclass]
struct TenSafeSecretKey {
    inner: SecretKey,
}

/// Python-visible public key.
#[pyclass]
struct TenSafePublicKey {
    inner: PublicKey,
}

/// Python-visible CKKS ciphertext with operator overloading.
#[pyclass]
struct TenSafeCiphertext {
    inner: Ciphertext,
    num_slots: usize,
    /// Cached context for operator overloading (avoids re-creating per __mul__ call).
    /// Lazily initialized on first use.
    cached_ctx: Option<CoreContext>,
}

impl TenSafeCiphertext {
    /// Get or create a CoreContext for this ciphertext's parameters.
    /// Uses cached version if available, otherwise creates and caches a new one.
    fn ensure_ctx(&mut self) {
        if self.cached_ctx.is_none() {
            let poly_degree = self.inner.c0.limbs[0].len();
            let params = CkksParams::for_degree(poly_degree);
            self.cached_ctx = Some(CoreContext::new(params));
        }
    }
}

#[pymethods]
impl TenSafeCiphertext {
    /// Ciphertext × plaintext (numpy array). Returns a new ciphertext.
    ///
    /// This is the ZeRo-MOAI hot path: `ct_prod = ct_rep * packed_pt`
    fn __mul__<'py>(
        &mut self,
        _py: Python<'py>,
        other: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<TenSafeCiphertext> {
        let slice = other.as_slice()?;

        self.ensure_ctx();
        let num_slots = self.num_slots;
        let mut padded = vec![0.0f64; num_slots];
        let n = slice.len().min(num_slots);
        padded[..n].copy_from_slice(&slice[..n]);

        let ctx = self.cached_ctx.as_ref().unwrap();
        let result = ctx.ct_pt_mul(&self.inner, &padded);
        Ok(TenSafeCiphertext {
            inner: result,
            num_slots,
            cached_ctx: None,
        })
    }

    /// Ciphertext + Ciphertext addition.
    fn __add__(&mut self, other: &TenSafeCiphertext) -> TenSafeCiphertext {
        self.ensure_ctx();
        let num_slots = self.num_slots;
        let ctx = self.cached_ctx.as_ref().unwrap();
        let result = ctx.ct_add(&self.inner, &other.inner);
        TenSafeCiphertext {
            inner: result,
            num_slots,
            cached_ctx: None,
        }
    }

    /// Ciphertext - Ciphertext subtraction.
    fn __sub__(&mut self, other: &TenSafeCiphertext) -> TenSafeCiphertext {
        self.ensure_ctx();
        let num_slots = self.num_slots;
        let ctx = self.cached_ctx.as_ref().unwrap();
        let result = ctx.ct_sub(&self.inner, &other.inner);
        TenSafeCiphertext {
            inner: result,
            num_slots,
            cached_ctx: None,
        }
    }

    /// Current scale factor.
    #[getter]
    fn scale(&self) -> f64 {
        self.inner.scale
    }

    /// Number of RNS limbs.
    #[getter]
    fn num_limbs(&self) -> usize {
        self.inner.c0.limbs.len()
    }
}

/// Python module: `import tensafe_he`
#[pymodule]
fn tensafe_he(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CkksContext>()?;
    m.add_class::<TenSafeCiphertext>()?;
    m.add_class::<TenSafeSecretKey>()?;
    m.add_class::<TenSafePublicKey>()?;
    Ok(())
}
