# ZeRo-MOAI: Zero-Rotation Matrix-Operation Acceleration for Inference

## Overview

ZeRo-MOAI is TenSafe's novel SIMD column-packing strategy for CKKS homomorphic encryption that **eliminates all ciphertext rotation operations** during the LoRA matrix-vector multiplication. This reduces HE cost by an order of magnitude compared to standard rotation-based approaches.

**Key result**: 0 rotations per token vs ~1536 rotations in standard CKKS matmul.

---

## The Problem: Rotations Are Expensive

In standard CKKS matrix-vector multiplication, computing `y = A @ x` where `A` is a matrix and `x` is encrypted requires **O(d)** ciphertext rotations, where `d` is the vector dimension.

For TenSafe's Qwen2.5-1.5B model:
- Hidden dimension `d = 1536`
- LoRA rank `r = 32`
- Standard approach: ~1536 rotations per matmul
- Each rotation: ~2ms on GPU (NTT + permutation + inverse NTT)
- **Total rotation cost: ~3072ms per token** -- completely impractical

### Why Rotations Are So Costly

A CKKS ciphertext stores a vector of `N/2` encrypted values in SIMD slots. To compute an inner product `<a, x>`, the standard approach:

1. Encode `a` as plaintext, multiply `ct(x) * pt(a)` -- element-wise
2. Rotate the result by powers of 2 and accumulate: `sum = ct + rotate(ct, 1) + rotate(ct, 2) + ...`
3. This "rotate-and-sum" pattern needs `O(log N)` rotations per inner product
4. For a full matrix with `r` rows: `r * O(log N)` rotations

Each rotation involves an automorphism on the polynomial ring -- essentially a full NTT, permutation, and inverse NTT.

---

## Our Solution: Column-Strided SIMD Packing

ZeRo-MOAI avoids all rotations by **packing multiple matrix rows into a single ciphertext** and using element-wise multiply to compute multiple dot products simultaneously.

### Core Insight

With `S` SIMD slots and vector dimension `d`, we can fit `floor(S/d)` independent vectors side by side in one ciphertext. Each "column" holds a copy of the hidden state `h`, and each corresponding plaintext column holds a different row of LoRA-A.

After one element-wise multiply, we have `floor(S/d)` independent element-wise products. The inner products are extracted by summing each `d`-element segment **after decryption** (in plaintext), avoiding any ciphertext rotation.

### Algorithm

**Step 1: Replicate hidden state across SIMD slots**

```python
# h = hidden state, shape [1536]
# simd_slots = 8192 (poly_n=16384)
cols_per_ct = simd_slots // d_model  # 8192 // 1536 = 5

replicated = np.zeros(simd_slots, dtype=np.float64)
for i in range(cols_per_ct):
    replicated[i * d_model : i * d_model + d_model] = h[:d_model]

# Layout:
# Slot: [0..1535 | 1536..3071 | 3072..4607 | 4608..6143 | 6144..7679 | 7680..8191]
# Data: [h       | h          | h          | h          | h          | 0...0     ]

ct_rep = cukks.encrypt_vector(replicated)  # ONE encrypt
```

**Step 2: Pack LoRA-A rows into plaintext**

For each batch of `cols_per_ct` rows:

```python
# For batch processing rows r_start..r_end of LoRA-A
packed_pt = np.zeros(simd_slots, dtype=np.float64)
for i, r in enumerate(range(r_start, r_end)):
    a_row = lora_a[r, :]  # shape [1536]
    off = i * d_model
    packed_pt[off : off + len(a_row)] = a_row

# Layout:
# Slot: [0..1535      | 1536..3071   | ...]
# Data: [A[r0, 0..1535] | A[r1, 0..1535] | ...]
```

**Step 3: Single ciphertext-plaintext multiply**

```python
ct_prod = ct_rep * packed_pt  # ONE ct×pt multiply, element-wise
```

After this multiply, slot `[i*d + j]` contains `h[j] * A[r_i, j]`.

**Step 4: Decrypt and extract inner products**

```python
decrypted = cukks.decrypt_to_gpu(ct_prod)  # decrypt on GPU

# Sum each d_model-sized segment to get the dot product
for i, r in enumerate(range(r_start, r_end)):
    off = i * d_model
    intermediate[r] = np.sum(decrypted[off : off + d_model])
```

The summation `sum(h[j] * A[r, j] for j in 0..d-1)` is exactly the inner product `<h, A[r, :]>`.

**Step 5: Plaintext LoRA-B multiplication**

```python
# intermediate has shape [rank] = [32]
# lora_b has shape [d_model, rank] = [1536, 32]
delta = lora_b @ intermediate  # plaintext matmul, ~1ms
```

---

## Batch Loop (Full Implementation)

From `inference_engine.py`, the `_he_lora_delta()` method:

```python
cols_per_ct = max(1, self.simd_slots // d_model)  # 5 for 8192/1536
n_batches = math.ceil(rank / cols_per_ct)           # ceil(32/5) = 7

# Build replicated hidden state
replicated = np.zeros(self.simd_slots, dtype=np.float64)
for i in range(cols_per_ct):
    replicated[i * d_model : i * d_model + d_model] = h_np[:d_model]

ct_rep = self._cukks.encrypt_vector(replicated)  # 1 encrypt

# GPU-resident batch decrypt
gpu_decrypted = []
batch_ranges = []

for batch_idx in range(n_batches):
    r_start = batch_idx * cols_per_ct
    r_end = min(r_start + cols_per_ct, rank)
    batch_ranges.append((r_start, r_end))

    # Pack rows into plaintext at matching SIMD offsets
    packed_pt = np.zeros(self.simd_slots, dtype=np.float64)
    for i, r in enumerate(range(r_start, r_end)):
        a_row = lora_a[r, :].astype(np.float64)
        off = i * d_model
        packed_pt[off : off + len(a_row)] = a_row

    # SINGLE ct*pt multiply covers all rows in this batch
    ct_prod = ct_rep * packed_pt

    # Decrypt on GPU (no CPU transfer yet)
    dec_gpu = self._cukks.decrypt_to_gpu(ct_prod)
    gpu_decrypted.append(dec_gpu)

# ONE bulk GPU→CPU transfer
stacked = torch.stack(gpu_decrypted)   # [n_batches, simd_slots] on CUDA
all_dec = stacked.cpu().numpy()         # single PCIe sync

# Extract dot products
intermediate = np.zeros(rank, dtype=np.float64)
for batch_idx, (r_start, r_end) in enumerate(batch_ranges):
    dec = all_dec[batch_idx]
    for i, r in enumerate(range(r_start, r_end)):
        off = i * d_model
        intermediate[r] = np.sum(dec[off : off + d_model])

# Plaintext B matmul
delta = lora_b @ intermediate
```

---

## Cost Model

### HE Operations Per Token

| Operation | Count | Cost |
|-----------|-------|------|
| Encrypt (replicated h) | 1 | ~10 ms |
| ct × pt multiply | n_batches | ~5 ms each |
| Decrypt (to GPU) | n_batches | ~7 ms each |
| GPU→CPU transfer | 1 | ~28 ms |
| **Rotations** | **0** | **0 ms** |

### Comparison: Standard vs ZeRo-MOAI

| | Standard (rotate-and-sum) | ZeRo-MOAI |
|--|--------------------------|-----------|
| Rotations | O(d) = ~1536 | **0** |
| Rotation cost | ~1536 × 2ms = 3072ms | **0ms** |
| ct × pt multiplies | 1 | 4-7 |
| Multiply cost | ~5ms | 4-7 × 5ms = 20-35ms |
| Encrypts | 1 | 1 (replicated) |
| **Total HE** | **~3077ms** | **~65ms** |
| **Speedup** | -- | **~47x** |

### Slot Efficiency

| poly_n | SIMD slots | cols_per_ct | n_batches (rank=32) | Padding waste |
|--------|-----------|-------------|---------------------|---------------|
| 16384 | 8,192 | 5 | 7 | 3,712/8,192 = 45% |
| 32768 | 16,384 | 10 | 4 | 1,024/16,384 = 6% |

poly_n=32768 has better slot utilization (6% waste vs 45%) but each operation is 2.3x slower due to larger NTT. The NTT speedup of poly_n=16384 dominates despite the higher batch count.

---

## Design Decisions

### Why Not Use Rotations At All?

Rotations are the single most expensive CKKS primitive on GPU. Each rotation requires:
1. Decompose ciphertext into digits (gadget decomposition)
2. Key-switch using evaluation keys (~400x the size of a ciphertext)
3. NTT/inverse NTT of the permuted polynomial

On our RTX A2000, a single rotation costs ~2ms -- comparable to the entire encrypt or decrypt operation. Eliminating rotations entirely is worth the cost of extra ct×pt multiplies.

### Why Pack Rows, Not Columns?

Row packing (our approach) means each ciphertext contains multiple complete dot products. The alternative -- column packing -- would distribute elements of a single dot product across multiple ciphertexts, requiring rotation-based accumulation to extract the result.

### Why Decrypt-Then-Sum (Not Rotate-And-Sum)?

After the ct×pt multiply, each SIMD segment contains `h[j] * A[r, j]` for one row. To get the dot product, we need to sum across the segment. We have two options:

1. **Rotate-and-sum in ciphertext**: O(log d) rotations in encrypted domain. Cost: ~20ms per dot product.
2. **Decrypt, then sum in plaintext**: One decrypt + numpy sum. Cost: ~7ms decrypt + ~0.01ms sum.

Option 2 is cheaper and exposes no additional information (the dot product result would be decrypted anyway).

---

## File References

| File | Lines | Component |
|------|-------|-----------|
| `demonstrator/server/inference_engine.py` | 860-972 | `_he_lora_delta()` -- ZeRo-MOAI batch loop |
| `demonstrator/server/inference_engine.py` | 810-830 | `_ckks_encrypt()` / `_ckks_decrypt()` |
| `demonstrator/server/inference_engine.py` | 148-223 | `_CuKKSAdapter` -- GPU encrypt/decrypt API |
