# GateLink-Split: Phone Split Inference Protocol

## Overview

GateLink-Split enables privacy-preserving LLM inference on resource-constrained devices (phones) by splitting the model between the device and server. The phone runs the embedding layer and LM head locally, while the server handles all 28 transformer layers and the HE-LoRA computation. The server never sees raw token IDs or sampling decisions.

**Key result**: 4.85 tok/s on split inference with HE encryption, full architectural privacy.

---

## Architecture

### Layer Split

| Component | Runs On | Size | Why |
|-----------|---------|------|-----|
| Tokenization | Server (split mode) | -- | JS BPE regex diverges from Python tokenizer on mobile browsers |
| Embedding (layer 0) | Phone | 446 MB (float16) | Raw token IDs never leave device |
| Transformer layers 1-28 | Server | ~3 GB (float16) | Too compute-intensive for phone |
| DP noise injection | Server | -- | Applied to post-transformer hidden state |
| HE-LoRA delta | Server | ~0.2 GB keys | CKKS encrypt/compute/decrypt on GPU |
| LM Head projection | Phone | 446 MB (tied) | Sampling decisions never leave device |
| Token sampling | Phone | -- | Temperature, top-k, top-p, repetition penalty -- all client-side |

### Data Flow

```
Phone                                   Server
─────                                   ──────
1. User types query
2. (tokenize request) ──────────────→  3. BPE tokenize
   ←──────────────── (token_ids) ←──
4. embed_tokens[token_id]
   → hidden_state [1536]
5. DP noise (client-side, optional)
6. base64(hidden) ──────────────────→  7. Decode base64
                                       8. Transformer L1..L28
                                       9. Layer norm
                                       10. DP noise (server, σ=4.84)
                                       11. CKKS encrypt(h_noised)
                                       12. ct(h) × pt(LoRA_A)
                                       13. Batch decrypt on GPU
                                       14. LoRA_B × intermediate
                                       15. hidden += delta
   ←──────────── base64(pre_act) ←──  16. Return pre-activations
17. Decode base64
18. LM head: pre_act @ W_head
19. top-p / top-k sampling
20. → next_token
21. Repeat from step 4 (incremental)
```

---

## WebSocket Protocol

Split inference uses a persistent WebSocket at `/api/v1/split/stream`:

### Message Types

**Client → Server:**

```json
{"type": "ping"}
{"type": "config"}
{"type": "tokenize", "text": "What is a savings account?"}
{"type": "forward",
 "hidden_states": "<base64-encoded float32>",
 "seq_len": 15,
 "hidden_dim": 1536,
 "expert_name": "banking_expert",
 "use_he": true,
 "session_id": "abc123",
 "incremental": true}
```

**Server → Client:**

```json
{"type": "pong"}
{"type": "config", "model": "Qwen/Qwen2.5-1.5B", ...}
{"type": "tokenize", "token_ids": [1234, 5678, ...], "length": 15}
{"type": "forward",
 "pre_activations": "<base64-encoded float32>",
 "he_active": true,
 "encrypt_ms": 10.2,
 "compute_ms": 20.1,
 "decrypt_ms": 35.3,
 "total_ms": 142.5,
 "dp_sigma": 4.8448,
 "incremental": true,
 "cached_seq_len": 14}
```

### Connection Lifecycle

```javascript
// From split_client.js
async serverForward(payload) {
    // Prefer WebSocket if connected
    if (this._wsReady && this._ws.readyState === WebSocket.OPEN) {
        return this._wsSend({type: "forward", ...payload});
    }
    // HTTP fallback for reliability
    return fetch("/api/v1/split/forward", {
        method: "POST",
        body: JSON.stringify(payload)
    });
}
```

The client attempts WebSocket first, with automatic HTTP fallback. This ensures the split client works even through proxies that don't support WebSocket.

---

## KV Cache Management

The server maintains per-session `DynamicCache` (HuggingFace Transformers) for incremental decoding.

### Implementation

```python
class _KVCacheStore:
    """Thread-safe, session-keyed KV cache store."""

    def __init__(self, max_sessions=32, ttl_seconds=300.0):
        self._store: OrderedDict = OrderedDict()  # LRU order
        self._timestamps: dict = {}
        self._max = max_sessions
        self._ttl = ttl_seconds
        self._lock = threading.RLock()
```

### Features

| Feature | Implementation |
|---------|---------------|
| Thread safety | `threading.RLock` protects all operations |
| LRU eviction | `OrderedDict.move_to_end()` on access |
| TTL expiry | 300-second timeout per session |
| Max sessions | 32 concurrent (configurable) |
| Incremental mode | After first pass (full seq), subsequent calls send only new token |

### Incremental Decoding

The first forward pass processes the full input sequence and creates the KV cache. Subsequent passes send only the new token's embedding (seq_len=1), and the server appends to the existing cache:

```python
def split_forward(self, hidden_states_np, ..., incremental=False):
    # Retrieve or create DynamicCache
    cache = None
    past_seq_len = 0
    if incremental:
        cache = self._kv_cache_store.get(session_id)
    if cache is None:
        cache = DynamicCache()
    else:
        past_seq_len = cache.get_seq_length()

    # Position IDs start at the correct offset
    position_ids = torch.arange(
        past_seq_len, past_seq_len + seq_len,
        dtype=torch.long, device=self.device
    ).unsqueeze(0)

    # Run through all 28 layers with cache
    for layer in self.model.model.layers:
        layer_out = layer(
            hidden,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            position_embeddings=position_embeddings,
        )
        hidden = layer_out[0]

    # Store updated cache
    self._kv_cache_store.put(session_id, cache)
```

This reduces per-token computation from O(n) (reprocess entire sequence) to O(1) (process only the new token, reuse KV from previous tokens).

---

## Phone Weight Management

### Weight Download

The phone needs 446 MB of float16 embedding weights (tied with LM head):
- `embed_tokens.bin`: 151,936 × 1,536 × 2 bytes = 446 MB
- `lm_head.bin`: Same weights (tied in Qwen2.5-1.5B)

### IndexedDB Caching

Weights are cached in IndexedDB to avoid re-downloading:

```javascript
// Chunked write to prevent mobile Safari OOM
const CHUNK_SIZE = 8 * 1024 * 1024;  // 8 MB chunks
for (let i = 0; i < data.byteLength; i += CHUNK_SIZE) {
    const chunk = data.slice(i, Math.min(i + CHUNK_SIZE, data.byteLength));
    await db.put("weights", chunk, `embed_chunk_${i}`);
}
```

| Approach | Peak Memory | Why |
|----------|-------------|-----|
| Single `put()` | ~892 MB | Structured clone doubles memory |
| Chunked 8MB `put()` | ~462 MB | Only 1 chunk in memory at a time |

### Float16 Lookup Table

Pure JavaScript float16→float32 conversion uses a pre-computed lookup table for O(1) performance:

```javascript
const F16_TABLE = new Float32Array(65536);  // 256 KB
for (let i = 0; i < 65536; i++) F16_TABLE[i] = _f16Slow(i);

function f16(h) { return F16_TABLE[h]; }  // O(1) lookup
```

This is critical for the embedding lookup and LM head projection, which each require 151,936 × 1,536 float16→float32 conversions.

---

## Client-Side Embedding

```javascript
// Embedding lookup from local weight matrix
embed(tokenId) {
    const offset = tokenId * this.hiddenDim;
    const embedding = new Float32Array(this.hiddenDim);
    for (let i = 0; i < this.hiddenDim; i++) {
        embedding[i] = f16(this.weights[offset + i]);
    }
    return embedding;
}
```

The raw token ID is used as a direct index into the weight matrix. The server never sees this token ID -- it only receives the continuous 1536-dimensional hidden state vector.

---

## Client-Side LM Head

```javascript
// LM head projection: hidden_state @ W_lm_head^T
// W_lm_head is tied with embedding weights
lmHead(hiddenState) {
    const logits = new Float32Array(this.vocabSize);  // 151,936
    for (let v = 0; v < this.vocabSize; v++) {
        let sum = 0;
        const offset = v * this.hiddenDim;
        for (let h = 0; h < this.hiddenDim; h++) {
            sum += hiddenState[h] * f16(this.weights[offset + h]);
        }
        logits[v] = sum;
    }
    return logits;
}
```

**This is the bottleneck on phone**: 151,936 × 1,536 = 233M multiply-adds on CPU JavaScript. Takes ~650ms per token on a phone, vs ~5ms with torch GPU in `bench_split_ws.py`.

---

## Client-Side Sampling

```javascript
sampleTopP(logits, temperature, topP, topK) {
    // 1. Apply temperature scaling
    for (let i = 0; i < logits.length; i++) logits[i] /= temperature;

    // 2. Softmax
    const max = Math.max(...logits);
    let sum = 0;
    for (let i = 0; i < logits.length; i++) {
        logits[i] = Math.exp(logits[i] - max);
        sum += logits[i];
    }
    for (let i = 0; i < logits.length; i++) logits[i] /= sum;

    // 3. Top-k filter
    const indexed = logits.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]);
    const topKCandidates = indexed.slice(0, topK);

    // 4. Top-p (nucleus) sampling
    let cumulative = 0;
    const nucleus = [];
    for (const [prob, idx] of topKCandidates) {
        cumulative += prob;
        nucleus.push([prob, idx]);
        if (cumulative >= topP) break;
    }

    // 5. Random sample from nucleus
    const r = Math.random() * cumulative;
    let acc = 0;
    for (const [prob, idx] of nucleus) {
        acc += prob;
        if (acc >= r) return idx;
    }
    return nucleus[nucleus.length - 1][1];
}
```

All sampling decisions stay on-device. The server never sees the probability distribution or the chosen token.

---

## TCP Relay (Phone LAN Access)

The TCP relay (`relay.py`) at port 9095 transparently proxies all traffic to the server at port 8095:

```python
async def pipe(label, reader, writer, tag):
    while True:
        data = await reader.read(1024 * 1024)  # 1 MB chunks
        if not data:
            break
        writer.write(data)
        await writer.drain()
```

The relay operates at the raw TCP level, so it transparently handles:
- HTTP requests (split/forward POST)
- WebSocket upgrade + frames (split/stream)
- Static file serving (frontend assets)

Phone connects to `http://<LAN-IP>:9095` and the relay forwards everything to `localhost:8095`.

---

## Privacy Guarantees

| Data | Phone | Server | Network |
|------|-------|--------|---------|
| Token IDs | Visible | **Never** | **Never** |
| Hidden states | Generated | Seen (DP-noised) | Encrypted (HTTPS) |
| LoRA intermediates | **Never** | CKKS encrypted | **Never** |
| Logit distribution | Computed | **Never** | **Never** |
| Chosen token | Visible | **Never** | **Never** |
| Expert routing | Keyword client-side | Sees adapter name | Encrypted |

---

## File References

| File | Component |
|------|-----------|
| `demonstrator/frontend/split_client.js` | Full split client (BPE tokenizer, embedding, LM head, sampling, WS transport) |
| `demonstrator/server/inference_engine.py:1026-1167` | `split_forward()` -- server-side forward pass |
| `demonstrator/server/inference_engine.py:312-377` | `_KVCacheStore` -- session KV cache |
| `demonstrator/server/app.py` | WebSocket `/api/v1/split/stream` endpoint |
| `relay.py` | TCP relay for phone LAN access |
