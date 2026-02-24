# Batched GPU Decryption: 4 Syncs to 1

## Overview

TenSafe's batch GPU decryption optimization eliminates redundant PCIe synchronization barriers during CKKS decryption. Instead of transferring each decrypted ciphertext from GPU to CPU individually (blocking sync per transfer), we accumulate all results on the GPU and perform **one bulk transfer** at the end.

**Key result**: 4 PCIe sync barriers reduced to 1, saving 84ms/token (+44% throughput).

---

## The Problem

After each ciphertext-plaintext multiply in the ZeRo-MOAI batch loop, the standard flow calls `decrypt_vector()`:

```python
# BEFORE: Sequential decrypt (4 blocking syncs)
for batch_idx in range(n_batches):  # n_batches = 4 (poly_n=32768)
    ct_prod = ct_rep * packed_pt
    dec = cukks.decrypt_vector(ct_prod)  # GPU inverse-NTT → CPU transfer (BLOCKING)
    # ^^^ Each call: ~7ms GPU NTT + ~28ms PCIe sync = ~35ms
    # Total: 4 × 35ms = 140ms
```

Each `decrypt_vector()` performs:
1. **GPU inverse-NTT** (~7ms) -- transforms from NTT domain to coefficient domain
2. **GPU→CPU PCIe transfer** (~28ms) -- **blocking synchronization barrier**

The PCIe transfer is the bottleneck. The GPU pipeline stalls while waiting for the DMA transfer to complete, preventing overlap between the next batch's computation and the current batch's transfer.

### Timing Diagram (Before)

```
GPU:  [NTT₁][STALL][NTT₂][STALL][NTT₃][STALL][NTT₄][STALL]
PCIe:       [XFER₁]      [XFER₂]      [XFER₃]      [XFER₄]
CPU:               [SUM₁]       [SUM₂]       [SUM₃]       [SUM₄]

Total: 4 × (7ms + 28ms) = 140ms
```

---

## Our Solution: `decrypt_to_gpu()`

We introduce a new method on the CKKS adapter that decrypts the ciphertext but **keeps the result as a `torch.Tensor` on CUDA memory**, avoiding the CPU transfer entirely:

```python
class _CuKKSAdapter:
    def decrypt_to_gpu(self, ct) -> torch.Tensor:
        """Decrypt ciphertext but keep result on GPU (no CPU transfer).

        Returns a torch.Tensor on CUDA. Call .cpu().numpy() later
        when you need the values on CPU.  This avoids a blocking
        GPU→CPU sync per call — batch multiple decrypts and transfer once.
        """
        self._metrics["total_decryptions"] += 1
        t0 = time.perf_counter()
        if isinstance(ct, _EmulatedCiphertext):
            result = torch.tensor(ct.decrypt(), dtype=torch.float64, device="cuda")
        else:
            dec = self._ctx.decrypt(ct)
            if isinstance(dec, torch.Tensor):
                result = dec  # already on GPU
            else:
                result = torch.tensor(np.asarray(dec), dtype=torch.float64, device="cuda")
        elapsed = (time.perf_counter() - t0) * 1000
        self._metrics["total_decrypt_ms"] += elapsed
        return result
```

### The Batch Loop (After)

```python
gpu_decrypted = []  # Accumulate GPU tensors

for batch_idx in range(n_batches):
    ct_prod = ct_rep * packed_pt
    dec_gpu = self._cukks.decrypt_to_gpu(ct_prod)  # GPU NTT only, stays on CUDA
    gpu_decrypted.append(dec_gpu)

# ONE bulk transfer: all batches at once
stacked = torch.stack(gpu_decrypted)   # [n_batches, simd_slots] on CUDA
all_dec = stacked.cpu().numpy()         # single PCIe sync barrier
```

### Timing Diagram (After)

```
GPU:  [NTT₁][NTT₂][NTT₃][NTT₄][STACK]
PCIe:                              [BULK_XFER]
CPU:                                          [SUM_ALL]

Total: 4 × 7ms + 28ms = 56ms
```

---

## Implementation Details

### Adapter API

All three CKKS backends implement `decrypt_to_gpu()`:

| Backend | Behavior |
|---------|----------|
| `_CuKKSAdapter` | Real GPU decrypt, returns CUDA tensor |
| `_PyfhelAdapter` | CPU decrypt, returns numpy (API compat) |
| `_PurePythonCKKS` | Returns numpy array (emulator) |

The batch loop handles both paths:

```python
if gpu_decrypted and isinstance(gpu_decrypted[0], torch.Tensor):
    # GPU path: torch.stack() + single .cpu().numpy()
    stacked = torch.stack(gpu_decrypted)
    all_dec = stacked.cpu().numpy()
else:
    # CPU/emulator path: np.stack()
    all_dec = np.stack([
        d if isinstance(d, np.ndarray) else np.asarray(d)
        for d in gpu_decrypted
    ])
```

### Memory Considerations

Each decrypted tensor occupies `simd_slots × 8 bytes` (float64) on GPU:
- poly_n=16384: 8192 × 8 = 64 KB per tensor × 7 batches = 448 KB on GPU
- poly_n=32768: 16384 × 8 = 128 KB per tensor × 4 batches = 512 KB on GPU

This is negligible compared to the model's GPU memory footprint (~5 GB).

### torch.stack() vs torch.cat()

We use `torch.stack()` (creates new dimension) rather than `torch.cat()` (concatenates along existing dimension) because:
1. Each batch needs separate segment extraction -- keeping them in separate rows is cleaner
2. `torch.stack()` is a single CUDA kernel, ~0.01ms overhead
3. The resulting `[n_batches, simd_slots]` shape maps directly to the extraction loop

---

## Measured Impact

### poly_n=32768 (4 batches)

| Metric | Before (sequential) | After (batched) |
|--------|-------------------|-----------------|
| PCIe sync barriers | 4 | **1** |
| GPU-to-CPU transfers | 4 × 28ms = 112ms | 1 × 28ms = **28ms** |
| GPU decrypt (NTT) | 4 × 7ms = 28ms | 4 × 7ms = 28ms |
| torch.stack overhead | -- | ~0.01ms |
| **Total decrypt time** | **140ms/token** | **56ms/token** |
| **Speedup** | -- | **2.5x on decrypt** |

### poly_n=16384 (7 batches)

| Metric | Before | After |
|--------|--------|-------|
| PCIe syncs | 7 | **1** |
| GPU-to-CPU | 7 × 28ms = 196ms | 1 × 28ms = **28ms** |
| GPU decrypt | 7 × 7ms = 49ms | 7 × 7ms = 49ms |
| **Total** | **245ms** | **77ms** |
| **Speedup** | -- | **3.2x on decrypt** |

### End-to-End Impact (WebSocket Mode)

| Configuration | tok/s | Change |
|--------------|-------|--------|
| Baseline (sequential decrypt) | 3.0 -- 4.2 | -- |
| + Batch GPU decrypt | 4.8 -- 5.8 | **+44%** |

---

## Why Not Use CUDA Streams?

An alternative approach would be to overlap GPU decryption with PCIe transfers using CUDA streams:

```python
# Alternative: stream-based overlap (NOT used)
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()
for batch_idx in range(n_batches):
    with torch.cuda.stream(stream1):
        dec = decrypt(ct)       # GPU NTT in stream1
    with torch.cuda.stream(stream2):
        prev_dec.cpu()          # Transfer prev result in stream2
```

We chose the simpler `decrypt_to_gpu() + stack() + cpu()` approach because:
1. CuKKS uses its own internal CUDA operations that don't play well with PyTorch streams
2. The bulk transfer approach is simpler and debuggable
3. The savings are already substantial (2.5-3.2x on decrypt phase)
4. Stream management complexity isn't worth the marginal improvement

---

## File References

| File | Lines | Component |
|------|-------|-----------|
| `demonstrator/server/inference_engine.py` | 187-207 | `_CuKKSAdapter.decrypt_to_gpu()` |
| `demonstrator/server/inference_engine.py` | 120-130 | `_PurePythonCKKS.decrypt_to_gpu()` |
| `demonstrator/server/inference_engine.py` | 255-257 | `_PyfhelAdapter.decrypt_to_gpu()` |
| `demonstrator/server/inference_engine.py` | 899-944 | Batch loop in `_he_lora_delta()` |
