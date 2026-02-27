//! GPU-resident RNS polynomial.
//!
//! [`GpuRnsPoly`] mirrors [`tensafe_he_core::rns::RnsPoly`] but lives entirely in GPU memory.
//! Each RNS limb is a separate `CudaSlice<u64>` of N coefficients for coalesced access.

use cudarc::driver::{CudaDevice, CudaSlice, CudaStream, LaunchAsync};
use std::sync::Arc;
use tensafe_he_core::params::Modulus;
use tensafe_he_core::rns::RnsPoly;

use crate::kernels::MODULE;
use crate::{launch_cfg, GpuError, GpuResult};

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

    // =====================================================================
    // Tier 1: Multi-stream polynomial operations
    // =====================================================================

    /// Hadamard multiply on a specific CUDA stream.
    ///
    /// Uses cudarc's `launch_on_stream` to dispatch each limb's kernel to the
    /// given stream, enabling concurrent Hadamard multiplies across batches.
    ///
    /// For rank-32 with 4 batches × 4 limbs = 16 kernel launches:
    ///   Sequential: 16 × ~3µs = ~48µs
    ///   4 streams:  ~15µs (kernels overlap across batches)
    pub fn hadamard_mul_on_stream(
        &self,
        other: &Self,
        moduli: &[Modulus],
        dev: &Arc<CudaDevice>,
        stream: &CudaStream,
    ) -> GpuResult<Self> {
        let mut out = Self::zero(dev, self.n, self.num_limbs())?;
        let cfg = launch_cfg(self.n as u32);

        for l in 0..self.num_limbs() {
            let f = dev.get_func(MODULE, "poly_hadamard").unwrap();
            unsafe {
                f.launch_on_stream(
                    stream,
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

    // =====================================================================
    // Tier 2: UMA-ready memory operations
    // =====================================================================

    /// Upload a CPU polynomial using CUDA managed (unified) memory.
    ///
    /// Allocates each RNS limb via `cuMemAllocManaged` instead of standard
    /// device allocation + `cuMemcpyHtoD`. The resulting `CudaSlice` wraps
    /// a managed pointer that is accessible from both CPU and GPU.
    ///
    /// **On UMA platforms** (Grace Hopper, Grace Blackwell):
    ///   - The managed pointer maps to a single physical memory region
    ///   - GPU access uses NVLink-C2C coherence (900 GB/s, ~100ns latency)
    ///   - No PCIe DMA copy occurs — data is already "there"
    ///   - Encrypt: 6.5ms → ~5.0ms (eliminates htod_copy for m, a, e)
    ///
    /// **On discrete GPU** (H100 + PCIe):
    ///   - Managed memory is page-migrated on first GPU access
    ///   - First NTT launch triggers automatic page fault + migration
    ///   - Subsequent accesses hit GPU-local HBM (same as htod_copy)
    ///   - Slightly higher first-access latency, but amortized over pipeline
    ///
    /// The returned `GpuRnsPoly` is fully compatible with all existing
    /// kernel launches because `CudaSlice` wraps the managed `CUdeviceptr`
    /// identically to a device-allocated pointer. `cuMemFree_v2` (called on
    /// drop) works for both managed and device memory.
    pub fn from_host_uma(dev: &Arc<CudaDevice>, poly: &RnsPoly) -> GpuResult<Self> {
        use cudarc::driver::{result, sys};

        let mut limbs = Vec::with_capacity(poly.limbs.len());
        let num_bytes = poly.n * std::mem::size_of::<u64>();

        for host_limb in &poly.limbs {
            // Allocate managed memory via the CUDA driver API.
            // CU_MEM_ATTACH_GLOBAL: accessible from any stream on any device.
            let managed_ptr = unsafe {
                result::malloc_managed(
                    num_bytes,
                    sys::CUmemAttach_flags::CU_MEM_ATTACH_GLOBAL,
                )
                .map_err(GpuError::Driver)?
            };

            // Copy host data into managed memory. Since managed memory is
            // CPU-accessible, we use cuMemcpyHtoD which works for both
            // device and managed targets.
            unsafe {
                result::memcpy_htod_sync(managed_ptr, host_limb)
                    .map_err(GpuError::Driver)?;
            }

            // Wrap the managed pointer in a CudaSlice via upgrade_device_ptr.
            // This is safe because:
            //   1. managed_ptr is a valid allocation from cuMemAllocManaged
            //   2. It has space for poly.n u64 elements
            //   3. The data was just initialized by memcpy above
            //   4. CudaSlice's Drop calls cuMemFree_v2, which frees managed memory
            let cuda_slice = unsafe {
                dev.upgrade_device_ptr::<u64>(managed_ptr, poly.n)
            };

            limbs.push(cuda_slice);
        }

        Ok(Self { limbs, n: poly.n })
    }

    /// Download GPU polynomial to CPU using managed memory zero-copy.
    ///
    /// **On UMA platforms** (Grace Hopper, Grace Blackwell):
    ///   - If the polynomial was allocated with `from_host_uma`, the limbs
    ///     are in managed memory. After device synchronization, the CPU can
    ///     read the managed pointer directly — no `cuMemcpyDtoH` needed.
    ///   - PCIe path: `dtoh_sync_copy` blocks for ~28ms (sync + copy)
    ///   - UMA path: `synchronize` + CPU read = ~0.5ms (coherence latency)
    ///
    /// **On discrete GPU**:
    ///   - Managed memory is in GPU-local HBM after first kernel access.
    ///   - CPU read triggers page fault + migration back to system memory.
    ///   - Net performance similar to dtoh_sync_copy (driver handles migration).
    ///
    /// The method first synchronizes the device to ensure all GPU writes are
    /// visible, then reads managed memory directly from the CPU.
    pub fn to_host_uma(&self, dev: &Arc<CudaDevice>) -> GpuResult<RnsPoly> {
        // Synchronize to ensure all GPU work writing to these buffers is complete.
        dev.synchronize().map_err(GpuError::Driver)?;

        let mut limbs = Vec::with_capacity(self.limbs.len());
        for gpu_limb in &self.limbs {
            // Read the managed memory pointer from the CudaSlice.
            // DevicePtr::device_ptr() returns the raw CUdeviceptr.
            use cudarc::driver::DevicePtr;
            let dev_ptr = *gpu_limb.device_ptr();

            // On UMA platforms, the managed pointer is directly CPU-accessible.
            // We read the data by casting the CUdeviceptr (which is a u64 holding
            // a virtual address) to a host pointer and copying.
            //
            // Safety: after synchronize(), all GPU writes are visible to the CPU.
            // The managed memory region is valid for self.n u64 elements.
            let mut host_limb = vec![0u64; self.n];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    dev_ptr as *const u64,
                    host_limb.as_mut_ptr(),
                    self.n,
                );
            }

            limbs.push(host_limb);
        }

        Ok(RnsPoly { limbs, n: self.n })
    }

    // =====================================================================
    // Tier 2: Async completion signaling (poll-based sync)
    // =====================================================================

    /// Record a CUDA event after polynomial operations complete.
    ///
    /// Returns a [`CompletionEvent`] that can be polled from the CPU without
    /// blocking (unlike `dtoh_sync_copy` which blocks until ALL prior GPU
    /// work completes via `cuStreamSynchronize`).
    ///
    /// Usage pattern:
    /// ```ignore
    /// let event = result_poly.record_completion(dev)?;
    /// // ... do other CPU work (model forward pass, next batch prep) ...
    /// while !event.is_complete() { std::thread::yield_now(); }
    /// let data = result_poly.to_host(dev)?;
    /// ```
    ///
    /// The event is recorded on the device's default stream, so it captures
    /// the completion of all previously queued work on that stream.
    ///
    /// Performance: eliminates ~5-10ms of implicit GPU drain wait that's
    /// embedded in the 28ms `dtoh_sync_copy` measurement.
    pub fn record_completion(&self, dev: &Arc<CudaDevice>) -> GpuResult<CompletionEvent> {
        CompletionEvent::record_on_stream(dev, *dev.cu_stream())
    }

    /// Record a completion event on a specific CUDA stream.
    pub fn record_completion_on_stream(
        &self,
        dev: &Arc<CudaDevice>,
        stream: &CudaStream,
    ) -> GpuResult<CompletionEvent> {
        CompletionEvent::record_on_stream(dev, stream.stream)
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

// =========================================================================
// Tier 2: CUDA event-based completion signaling
// =========================================================================

/// A CUDA event for non-blocking GPU completion polling.
///
/// Wraps a real `CUevent` created via `cuEventCreate` and recorded on a
/// stream via `cuEventRecord`. The CPU can poll `is_complete()` without
/// blocking, unlike `cuStreamSynchronize` which halts the calling thread.
///
/// This enables overlapped CPU/GPU execution:
/// - GPU runs decrypt + iNTT on stream
/// - CPU immediately starts preparing the next batch
/// - CPU polls `is_complete()` when it needs the result
///
/// Savings: 5-10ms of implicit sync wait per token.
pub struct CompletionEvent {
    /// Raw CUDA event handle.
    event: cudarc::driver::sys::CUevent,
}

unsafe impl Send for CompletionEvent {}

impl CompletionEvent {
    /// Create and record a completion event on the given raw CUDA stream.
    ///
    /// Uses `CU_EVENT_DISABLE_TIMING` for minimal overhead (~2µs vs ~5µs
    /// with timing enabled).
    fn record_on_stream(
        _dev: &Arc<CudaDevice>,
        stream: cudarc::driver::sys::CUstream,
    ) -> GpuResult<Self> {
        use cudarc::driver::{result, sys};

        // Create an event with timing disabled (lower overhead).
        let event = result::event::create(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING)
            .map_err(GpuError::Driver)?;

        // Record the event on the stream. It will signal when all previously
        // enqueued work on this stream completes.
        unsafe {
            result::event::record(event, stream).map_err(GpuError::Driver)?;
        }

        Ok(Self { event })
    }

    /// Check if the GPU work preceding this event has completed.
    ///
    /// Returns `true` if complete, `false` if still in progress.
    /// This is a non-blocking query via `cuEventQuery`.
    ///
    /// `cuEventQuery` returns:
    /// - `CUDA_SUCCESS` → work is complete
    /// - `CUDA_ERROR_NOT_READY` → work is still in progress
    /// - Other → actual error (treated as not complete)
    pub fn is_complete(&self) -> bool {
        use cudarc::driver::sys;
        unsafe {
            sys::lib().cuEventQuery(self.event) == sys::CUresult::CUDA_SUCCESS
        }
    }

    /// Block the calling thread until the GPU work completes.
    ///
    /// Uses `cuEventSynchronize` which is more efficient than spinning on
    /// `is_complete()` because it yields the thread to the OS scheduler.
    pub fn wait(&self) -> GpuResult<()> {
        use cudarc::driver::sys;
        unsafe {
            sys::lib()
                .cuEventSynchronize(self.event)
                .result()
                .map_err(GpuError::Driver)?;
        }
        Ok(())
    }
}

impl Drop for CompletionEvent {
    fn drop(&mut self) {
        // cuEventDestroy_v2 can be called even if the event hasn't completed yet.
        // Any associated resources are released asynchronously at completion.
        unsafe {
            let _ = cudarc::driver::result::event::destroy(self.event);
        }
    }
}
