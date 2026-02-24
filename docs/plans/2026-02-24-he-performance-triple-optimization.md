# HE Performance Triple Optimization Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Triple the tok/s of the TenSafe demonstrator from ~3.0 to ~6-8 by eliminating GPU↔CPU sync barriers, HTTP per-token overhead, and halving NTT cost.

**Architecture:** Three independent optimizations that stack multiplicatively:
- Track A: Batch decrypt on GPU — keep all decrypts GPU-resident, one bulk transfer
- Track B: Split WebSocket protocol — persistent connection replaces HTTP POST per token
- Track C: Lower poly_n 32768→16384 — halves NTT cost per operation

**Tech Stack:** Python (inference_engine.py, app.py), JavaScript (split_client.js, app.js), CuKKS (OpenFHE GPU CKKS), FastAPI WebSocket, PyTorch CUDA

**Current baseline (measured):**
- WebSocket mode: 3.0 tok/s (333ms/tok). HE = 245ms (73.5%)
- Split mode: 0.92 tok/s (1084ms/tok). HTTP overhead = ~700ms/tok
- Base model (no HE): 17.9 tok/s (56ms/tok)

---

## Track A: Batch Decrypt on GPU

**Problem:** `_he_lora_delta` calls `decrypt_vector()` 4 times per token. Each call does:
1. GPU-side inverse NTT (~7ms)
2. GPU→CPU data transfer over PCIe (~28ms)
3. Total: ~35ms × 4 = **140ms/token** (42% of all time)

The GPU→CPU sync is the real bottleneck — 4 separate blocking `.cpu()` calls.

**Solution:** Decrypt all 4 batches on GPU (staying in CUDA memory), then do ONE bulk `.cpu().numpy()` transfer at the end. Saves 3 PCIe sync barriers (~84ms).

**Expected improvement:** HE decrypt from ~140ms → ~56ms (= ~84ms saved) → tok/s from 3.0 to ~4.0

### Task A1: Add `decrypt_to_gpu()` to `_CuKKSAdapter`

**Files:**
- Modify: `demonstrator/server/inference_engine.py:135-188` (class `_CuKKSAdapter`)

**Step 1: Add the method to `_CuKKSAdapter`**

After the existing `decrypt_vector()` method (line 172), add:

```python
def decrypt_to_gpu(self, ct) -> torch.Tensor:
    """Decrypt ciphertext but keep result on GPU (no CPU transfer).

    Returns a torch.Tensor on CUDA. Call .cpu().numpy() later
    when you need the values on CPU.  This avoids a blocking
    GPU→CPU sync per call — batch multiple decrypts and transfer once.
    """
    self._metrics["total_decryptions"] += 1
    t0 = time.perf_counter()
    if isinstance(ct, _EmulatedCiphertext):
        # Emulator path: just move to GPU tensor for API compat
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

**Step 2: Add the same method to `_PurePythonCKKS` for emulator compat**

After line 117, add:

```python
def decrypt_to_gpu(self, ct):
    """Emulator version — returns numpy (no GPU). API compat only."""
    self._metrics["total_decryptions"] += 1
    t0 = time.perf_counter()
    if isinstance(ct, _EmulatedCiphertext):
        result = ct.decrypt()
    else:
        result = np.asarray(ct, dtype=np.float64)
    elapsed = (time.perf_counter() - t0) * 1000
    self._metrics["total_decrypt_ms"] += elapsed
    return result
```

**Step 3: Add the same to `_PyfhelAdapter`**

After line 218, add:

```python
def decrypt_to_gpu(self, ct):
    """Pyfhel CPU version — returns numpy. API compat only."""
    return self.decrypt_vector(ct)  # CPU backend, no GPU benefit
```

**Step 4: Verify server still starts**

Run: `curl http://127.0.0.1:8095/health`
Expected: `{"status":"ok","engine_ready":true,...}`

**Step 5: Commit**

```
git add demonstrator/server/inference_engine.py
git commit -m "feat: add decrypt_to_gpu() to all CKKS adapters for batched GPU decrypt"
```

---

### Task A2: Rewrite `_he_lora_delta` batch loop to use GPU-resident decrypts

**Files:**
- Modify: `demonstrator/server/inference_engine.py:847-872` (the batch loop in `_he_lora_delta`)

**Step 1: Replace the per-batch decrypt loop**

Replace lines 845-872 (the `intermediate = ...` through the batch loop) with:

```python
        intermediate = np.zeros(rank, dtype=np.float64)

        # ---- GPU-resident batch decrypt optimisation ----
        #
        # Instead of 4 separate decrypt_vector() calls (each blocking
        # GPU→CPU sync, ~35ms each = 140ms), we:
        #   1. Compute all ct×pt products on GPU
        #   2. Decrypt each on GPU (stays in CUDA memory, ~7ms each)
        #   3. ONE bulk .cpu().numpy() transfer at the end (~28ms)
        #
        # Cost: 1 encrypt + 4 GPU decrypts + 1 transfer = ~66ms
        # vs old: 1 encrypt + 4 (GPU decrypt + transfer) = ~150ms
        #
        gpu_decrypted = []  # list of GPU tensors
        batch_ranges = []   # (r_start, r_end) per batch

        t_dec_total = time.perf_counter()
        for batch_idx in range(n_batches):
            r_start = batch_idx * cols_per_ct
            r_end = min(r_start + cols_per_ct, rank)
            n_cols = r_end - r_start
            batch_ranges.append((r_start, r_end))

            # Pack n_cols A-rows into one plaintext, each at its slot offset
            packed_pt = np.zeros(self.simd_slots, dtype=np.float64)
            for i, r in enumerate(range(r_start, r_end)):
                a_row = lora_a[r, :].astype(np.float64)
                off = i * d_model
                packed_pt[off : off + len(a_row)] = a_row

            # Single ct-pt multiply covers all n_cols rows at once
            ct_prod = ct_rep * packed_pt
            ops += n_cols

            # Decrypt on GPU — NO CPU transfer yet
            dec_gpu = self._cukks.decrypt_to_gpu(ct_prod)
            gpu_decrypted.append(dec_gpu)

        # ONE bulk GPU→CPU transfer for all batches
        if gpu_decrypted and isinstance(gpu_decrypted[0], torch.Tensor):
            stacked = torch.stack(gpu_decrypted)     # [n_batches, simd_slots]
            all_dec = stacked.cpu().numpy()           # ONE PCIe transfer
        else:
            # Emulator/Pyfhel path — already numpy
            all_dec = np.stack([
                d if isinstance(d, np.ndarray) else np.asarray(d)
                for d in gpu_decrypted
            ])

        decrypt_ms_total = (time.perf_counter() - t_dec_total) * 1000

        # Extract dot-product sum for each column in each batch
        for batch_idx, (r_start, r_end) in enumerate(batch_ranges):
            dec = all_dec[batch_idx]
            for i, r in enumerate(range(r_start, r_end)):
                off = i * d_model
                intermediate[r] = np.sum(dec[off : off + d_model])

        ops += n_batches
```

**Step 2: Run split selftest to verify correctness**

Run: `curl -s http://127.0.0.1:8095/api/v1/split/selftest | python3 -m json.tool | grep verdict`
Expected: `"verdict": "PASS — split pipeline generates clean output with DP noise"`

**Step 3: Run WebSocket selftest for tok/s comparison**

Run: `curl -s http://127.0.0.1:8095/api/v1/selftest`
Record: `tokens_per_second` (should be higher than baseline 3.0)

**Step 4: Run chat/compare for timing breakdown**

Run: `curl -s -X POST http://127.0.0.1:8095/api/v1/chat/compare -H "Content-Type: application/json" -d '{"query":"What is compound interest","max_tokens":16}'`
Record: adapted.tok_s and adapted.time_ms (compare to baseline 3.3-4.2)

**Step 5: Commit**

```
git add demonstrator/server/inference_engine.py
git commit -m "perf: batch GPU decrypts in _he_lora_delta — 4 syncs → 1 bulk transfer"
```

---

## Track B: Split WebSocket Protocol

**Problem:** Each split token requires a full HTTP POST/response cycle:
- ~12 KB request (base64 hidden states + JSON)
- ~13 KB response (base64 pre-activations + metrics)
- HTTP header overhead: ~2 KB per round-trip
- TCP keepalive / connection reuse still has per-request parsing
- On phone over LAN: ~50-100ms overhead per token

**Solution:** New WebSocket endpoint `/api/v1/split/stream` with persistent binary-capable connection. Phone opens WS once, sends incremental hidden states per frame, server returns pre-activations per frame.

**Expected improvement:** Split mode from 0.92 → ~2.5-3.0 tok/s (eliminates ~700ms HTTP overhead)

### Task B1: Add WebSocket split/stream endpoint to server

**Files:**
- Modify: `demonstrator/server/app.py` (add new WS handler after existing chat/stream handler, ~line 219)

**Step 1: Add the WS endpoint**

After line 219 (end of `chat_stream` handler), add:

```python
# ======================================================================
# WebSocket split inference (persistent connection, binary-capable)
# ======================================================================

@app.websocket("/api/v1/split/stream")
async def split_stream(ws: WebSocket):
    """WebSocket endpoint for split inference.

    Eliminates per-token HTTP overhead by keeping a persistent connection.
    Protocol:
      Client → Server:  {"type": "forward", "hidden_states_b64": "...", ...}
      Server → Client:  {"type": "result", "pre_activations_b64": "...", ...}
      Client → Server:  {"type": "ping"}
      Server → Client:  {"type": "pong"}
    """
    await ws.accept()
    logger.info("Split-WS client connected")
    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type", "forward")

            if msg_type == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
                continue

            if msg_type == "config":
                # Return split config over WS (avoids separate HTTP call)
                cfg = {
                    "type": "config",
                    "model": engine.moe_config.get("model", ""),
                    "hidden_dim": engine.model.config.hidden_size,
                    "vocab_size": engine.tokenizer.vocab_size,
                    "num_layers": len(engine.model.model.layers),
                    "client_layers": 1,
                    "dp_epsilon": engine._dp_epsilon,
                    "dp_delta": engine._dp_delta,
                    "dp_sigma": round(engine._dp_sigma, 6),
                    "dp_sensitivity": engine._dp_sensitivity,
                    "experts": list(engine.adapters.keys()),
                    "expert_keywords": {
                        n: engine.moe_config["experts"][n].get("gate_keywords", [])
                        for n in engine.adapters
                        if n in engine.moe_config.get("experts", {})
                    },
                    "simd_slots": engine.simd_slots,
                    "he_active": engine.he_ctx is not None,
                }
                await ws.send_text(json.dumps(cfg))
                continue

            if msg_type == "tokenize":
                text = str(msg.get("text", ""))[:10000]
                ids = engine.tokenizer.encode(text)
                await ws.send_text(json.dumps({
                    "type": "token_ids",
                    "token_ids": ids,
                    "length": len(ids),
                }))
                continue

            if msg_type != "forward":
                await ws.send_text(json.dumps({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}",
                }))
                continue

            # ---- forward pass (same logic as HTTP split/forward) ----
            client_ip = ws.client.host if ws.client else "unknown"
            if not _check_rate_limit(client_ip):
                await ws.send_text(json.dumps({
                    "type": "error",
                    "message": "Rate limit exceeded",
                }))
                continue

            try:
                h_b64 = msg.get("hidden_states_b64", "")
                seq_len = min(max(int(msg.get("seq_len", 1)), 1), 4096)
                hidden_dim = int(msg.get("hidden_dim", engine.model.config.hidden_size))
                expert_name = str(msg.get("expert_name", "shared_attention"))[:128]
                use_he = bool(msg.get("use_he", True))
                session_id = str(msg.get("session_id", "default"))[:128]
                incremental = bool(msg.get("incremental", False))

                h_bytes = base64.b64decode(h_b64)
                h_np = np.frombuffer(h_bytes, dtype=np.float32).reshape(seq_len, hidden_dim)

                async with _gen_semaphore:
                    result = engine.split_forward(
                        hidden_states=h_np,
                        seq_len=seq_len,
                        hidden_dim=hidden_dim,
                        expert_name=expert_name,
                        use_he=use_he,
                        session_id=session_id,
                    )

                pre_np = result["pre_activations"]
                pre_b64 = base64.b64encode(pre_np.astype(np.float32).tobytes()).decode()

                await ws.send_text(json.dumps({
                    "type": "result",
                    "pre_activations_b64": pre_b64,
                    "seq_len": pre_np.shape[0] if pre_np.ndim > 1 else 1,
                    "he_active": result.get("he_active", False),
                    "encrypt_ms": result.get("encrypt_ms", 0),
                    "compute_ms": result.get("compute_ms", 0),
                    "decrypt_ms": result.get("decrypt_ms", 0),
                    "he_operations": result.get("he_operations", 0),
                    "dp_sigma": result.get("dp_sigma", 0),
                    "total_ms": result.get("total_ms", 0),
                    "incremental": result.get("incremental", False),
                    "cached_seq_len": result.get("cached_seq_len", 0),
                    "layers_computed": result.get("layers_computed", 0),
                }, default=str))

            except Exception as exc:
                logger.exception(f"Split-WS forward error: {exc}")
                await ws.send_text(json.dumps({
                    "type": "error",
                    "message": str(exc)[:500],
                }))

    except WebSocketDisconnect:
        logger.info("Split-WS client disconnected")
    except Exception as exc:
        logger.exception(f"Split-WS error: {exc}")
        try:
            await ws.send_text(json.dumps({
                "type": "error",
                "message": str(exc)[:500],
            }))
            await ws.close(code=1011, reason=str(exc)[:120])
        except Exception:
            pass
```

**Step 2: Add base64 import if not already present**

Check line 1 of app.py — ensure `import base64` is present. If not, add it.

**Step 3: Verify server starts and WS endpoint exists**

Run: `curl -s http://127.0.0.1:8095/health`
Expected: Server healthy (WS endpoints don't show in health but should not break startup)

Test WS with wscat or Python:
```python
import asyncio, websockets, json
async def test():
    async with websockets.connect("ws://127.0.0.1:8095/api/v1/split/stream") as ws:
        await ws.send(json.dumps({"type": "ping"}))
        r = await ws.recv()
        print(r)  # {"type": "pong"}
asyncio.run(test())
```

**Step 4: Commit**

```
git add demonstrator/server/app.py
git commit -m "feat: add WebSocket split/stream endpoint for persistent split connections"
```

---

### Task B2: Add WebSocket transport to `split_client.js`

**Files:**
- Modify: `demonstrator/frontend/split_client.js:630-663` (serverForward method)
- Modify: `demonstrator/frontend/split_client.js:400-471` (initialize method)

**Step 1: Add WS connection management to the class**

In the constructor (around line 385), add WS state:

```javascript
    this._ws = null;          // WebSocket connection (persistent)
    this._wsReady = false;    // true when WS is open and ready
    this._wsPending = null;   // {resolve, reject} for current forward call
    this._wsQueue = [];       // queued messages while connecting
```

**Step 2: Add WS connect/disconnect methods**

After the constructor, add:

```javascript
  /** Open persistent WebSocket for split inference. */
  async _wsConnect() {
    if (this._ws && this._ws.readyState === WebSocket.OPEN) return;

    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const url = proto + "//" + location.host + "/api/v1/split/stream";

    return new Promise((resolve, reject) => {
      const ws = new WebSocket(url);
      ws.onopen = () => {
        this._ws = ws;
        this._wsReady = true;
        console.log("[Split-WS] Connected");
        resolve();
      };
      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);
        if (msg.type === "pong") return; // keepalive response
        if (msg.type === "error" && this._wsPending) {
          this._wsPending.reject(new Error(msg.message));
          this._wsPending = null;
          return;
        }
        if (this._wsPending) {
          this._wsPending.resolve(msg);
          this._wsPending = null;
        }
      };
      ws.onerror = (err) => {
        console.warn("[Split-WS] Error, falling back to HTTP", err);
        this._wsReady = false;
        if (this._wsPending) {
          this._wsPending.reject(new Error("WebSocket error"));
          this._wsPending = null;
        }
        reject(err);
      };
      ws.onclose = () => {
        console.log("[Split-WS] Closed");
        this._wsReady = false;
        this._ws = null;
      };
    });
  }

  /** Send a message over WS and wait for response. */
  _wsSend(msg) {
    return new Promise((resolve, reject) => {
      if (!this._ws || this._ws.readyState !== WebSocket.OPEN) {
        reject(new Error("WebSocket not connected"));
        return;
      }
      this._wsPending = { resolve, reject };
      this._ws.send(JSON.stringify(msg));
      // Timeout after 30s
      setTimeout(() => {
        if (this._wsPending) {
          this._wsPending.reject(new Error("WS forward timeout (30s)"));
          this._wsPending = null;
        }
      }, 30000);
    });
  }
```

**Step 3: Modify `serverForward()` to prefer WS with HTTP fallback**

Replace the existing `serverForward` method (lines 632-663) with:

```javascript
  async serverForward(noisedFlat, seqLen, expert, useHE, sessionId, incremental) {
    const b64 = ab2b64(noisedFlat.buffer);

    if (this._abort && this._abort.signal.aborted) {
      throw new Error("Generation aborted");
    }

    const payload = {
      hidden_states_b64: b64,
      seq_len: seqLen,
      hidden_dim: this.HD,
      expert_name: expert,
      use_he: useHE,
      session_id: sessionId || "split_" + Date.now(),
      incremental: !!incremental,
    };

    // Prefer WebSocket if connected (eliminates HTTP overhead)
    if (this._wsReady && this._ws && this._ws.readyState === WebSocket.OPEN) {
      try {
        return await this._wsSend({ type: "forward", ...payload });
      } catch (e) {
        console.warn("[Split-WS] Forward failed, falling back to HTTP:", e.message);
        this._wsReady = false;
      }
    }

    // HTTP fallback (original path)
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000);
    try {
      const resp = await fetch("/api/v1/split/forward", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      if (!resp.ok) throw new Error("Server " + resp.status + ": " + (await resp.text()).slice(0, 200));
      return resp.json();
    } finally {
      clearTimeout(timeoutId);
    }
  }
```

**Step 4: Connect WS during initialization**

In the `initialize()` method, after `this.ready = true;` (around line 470), add:

```javascript
    // Open persistent WebSocket for split inference (non-blocking)
    this._wsConnect().catch(e =>
      console.warn("[Split-WS] Initial connect failed, will use HTTP:", e.message)
    );
```

**Step 5: Connect WS at start of `generate()` if not connected**

At the top of the `generate()` method (around line 679, after `this._abort = new AbortController()`), add:

```javascript
    // Ensure WS is connected for this generation session
    if (!this._wsReady) {
      try { await this._wsConnect(); } catch(e) {
        console.warn("[Split-WS] Connect failed, using HTTP fallback");
      }
    }
```

**Step 6: Bump cache bust version**

In `index.html`, change `?v=9` to `?v=10` on both script tags.

**Step 7: Verify phone can still do split inference**

Restart server, hard-reload phone, run a query. Should connect via WS now.
Check browser console for `[Split-WS] Connected`.

**Step 8: Commit**

```
git add demonstrator/frontend/split_client.js demonstrator/frontend/index.html
git commit -m "feat: WebSocket transport for split inference — eliminates HTTP overhead"
```

---

### Task B3: Add WebSocket relay support

**Files:**
- Modify: `relay.py` (must forward WebSocket connections, not just TCP)

**Step 1: Check current relay**

The current relay is a raw TCP relay (`asyncio` socket forwarding). WebSocket runs over TCP, so the raw TCP relay should already forward WS frames transparently. However, the WS upgrade handshake must be forwarded correctly.

**Verify:** On phone, navigate to the app. Check if `[Split-WS] Connected` appears in console.

If WS works through the TCP relay already → skip to commit.
If not → the relay needs to be HTTP/WS-aware. In that case, replace with an `aiohttp` or `websockets` proxy. But typically raw TCP relay handles WS fine because WS is just HTTP upgrade + framed TCP.

**Step 2: Commit if changes were needed**

```
git add relay.py
git commit -m "fix: ensure TCP relay passes WebSocket upgrade frames"
```

---

## Track C: Lower poly_n 32768 → 16384

**Problem:** CuKKS uses `for_depth(3)` which auto-selects poly_n=32768 (16384 SIMD slots). NTT on 32768-element polynomials is ~2× slower than 16384.

**Solution:** Use `InferenceConfig(poly_mod_degree=16384)` directly instead of `for_depth(3)`.

**Trade-offs:**
- Security: ~256-bit → ~192-bit (still above 128-bit minimum requirement)
- SIMD slots: 16384 → 8192 (cols_per_ct: 10→5, batches: 4→7)
- NTT cost per op: ~2× faster
- Net per-token: fewer ms despite more batches (NTT speedup > batch count increase)

**Expected improvement:** HE from ~245ms → ~120-150ms/tok → tok/s from 3.0 to ~4.5

### Task C1: Make poly_n configurable via environment variable

**Files:**
- Modify: `demonstrator/server/inference_engine.py:468-502` (CuKKS init)

**Step 1: Read poly_n from env with fallback**

Replace lines 478-483 (the `for_depth(3)` block) with:

```python
            # Allow poly_n override: TENSAFE_POLY_N=16384 for faster NTT
            # Default: for_depth(3) → poly_n=32768 (256-bit security)
            # Override: poly_n=16384 → ~192-bit security (still > 128-bit min)
            poly_n_override = int(os.environ.get("TENSAFE_POLY_N", "0"))
            if poly_n_override in (8192, 16384, 32768, 65536):
                from cukks.context import InferenceConfig
                logger.info(
                    f"CuKKS: using explicit poly_n={poly_n_override} "
                    f"(from TENSAFE_POLY_N env var)"
                )
                inf_cfg = InferenceConfig(
                    poly_mod_degree=poly_n_override,
                    scale_bits=hec.get("scale_bits", 40),
                )
                ctx = _cukks_pkg.CKKSInferenceContext(inf_cfg)
            else:
                # Default: auto-select for depth=3 → poly_n=32768
                ctx = _cukks_pkg.CKKSInferenceContext.for_depth(3)
```

**Step 2: Ensure `import os` is present at top of file**

Check imports at top. `os` should already be imported. If not, add it.

**Step 3: Test with default (no env var) — should be identical to current**

Run: `curl -s http://127.0.0.1:8095/health`
Run: `curl -s http://127.0.0.1:8095/api/v1/split/config | python3 -m json.tool | grep simd`
Expected: `"simd_slots": 16384` (unchanged from current)

**Step 4: Commit**

```
git add demonstrator/server/inference_engine.py
git commit -m "feat: configurable poly_n via TENSAFE_POLY_N env var (default unchanged)"
```

---

### Task C2: Benchmark with poly_n=16384

**Step 1: Restart server with TENSAFE_POLY_N=16384**

```bash
# Kill current server, restart with lower poly_n
TENSAFE_POLY_N=16384 python3 -m demonstrator.server.app --port 8095
```

**Step 2: Verify new parameters**

Run: `curl -s http://127.0.0.1:8095/api/v1/split/config`
Expected: `"simd_slots": 8192` (halved from 16384)

**Step 3: Run split selftest — verify output quality**

Run: `curl -s http://127.0.0.1:8095/api/v1/split/selftest`
Expected: `"verdict": "PASS"` with coherent English text

**Step 4: Run chat/compare benchmark**

Run: `curl -s -X POST http://127.0.0.1:8095/api/v1/chat/compare -H "Content-Type: application/json" -d '{"query":"What is compound interest","max_tokens":16}'`
Record: adapted.tok_s (should be higher than 3.3 baseline)

**Step 5: Run WebSocket selftest for aggregate timing**

Run: `curl -s http://127.0.0.1:8095/api/v1/selftest`
Record: tokens_per_second, total_compute_ms, total_decrypt_ms

**Step 6: Compare results**

| Metric | poly_n=32768 (baseline) | poly_n=16384 | Change |
|---|---|---|---|
| SIMD slots | 16384 | 8192 | -50% |
| Batches | 4 | 7 | +75% |
| tok/s | 3.0 | ? | ? |
| decrypt ms/tok | 140 | ? | ? |
| compute ms/tok | 105 | ? | ? |

If improvement is < 10%, revert to poly_n=32768 and skip Track C.
If improvement is > 15%, keep poly_n=16384 as new default.

**Step 7: If keeping, update comment in moe_config.json**

Update the `note_cukks` field to reflect new default.

**Step 8: Commit**

```
git add demonstrator/server/inference_engine.py demonstrator/adapters/tgsp/moe_config.json
git commit -m "perf: benchmark poly_n=16384 — [X]% improvement in HE throughput"
```

---

## Verification: Combined Benchmark

After all three tracks are implemented:

### Final Benchmark Script

```bash
# 1. Split selftest (correctness)
curl -s http://127.0.0.1:8095/api/v1/split/selftest | python3 -c "
import json,sys; d=json.load(sys.stdin)
print('Split selftest:', d['verdict'])
print('  dp_sigma:', d.get('dp_sigma'))
print('  tokens:', d['total_tokens'])
"

# 2. WebSocket selftest (tok/s)
curl -s http://127.0.0.1:8095/api/v1/selftest | python3 -c "
import json,sys; d=json.load(sys.stdin)
a=d.get('aggregate',{})
print('WS tok/s:', a.get('tokens_per_second'))
print('  decrypt_ms total:', a.get('total_decrypt_ms'))
print('  compute_ms total:', a.get('total_compute_ms'))
"

# 3. Compare endpoint (base vs adapted)
curl -s -X POST http://127.0.0.1:8095/api/v1/chat/compare \
  -H "Content-Type: application/json" \
  -d '{"query":"What is compound interest","max_tokens":32}' | python3 -c "
import json,sys; d=json.load(sys.stdin)
print('Base tok/s:', d['base']['tok_s'])
print('Adapted tok/s:', d['adapted']['tok_s'])
print('HE ops:', d['adapted']['he_operations'])
"
```

**Expected Final Results (all three tracks combined):**

| Metric | Before | After (projected) |
|---|---|---|
| WebSocket tok/s | 3.0 | 5.0-7.0 |
| Split tok/s | 0.92 | 2.5-4.0 |
| HE decrypt ms/tok | 140 | 30-50 |
| HE compute ms/tok | 105 | 50-80 |
| Security level | 256-bit | 192-bit (if Track C kept) |
| HE backend | CuKKS GPU (real) | CuKKS GPU (real) |
| DP sigma | 4.84 | 4.84 (unchanged) |

---

## Implementation Order

1. **Track A first** (highest impact, safest — no API changes, no security changes)
2. **Track C second** (benchmark only — env var toggle, easy to revert)
3. **Track B last** (most code, needs phone testing, but HTTP fallback ensures safety)

Each track is independently deployable and testable. They stack multiplicatively.
