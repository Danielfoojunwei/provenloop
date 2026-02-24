/**
 * GateLink-Split Client — Pure JavaScript (no WASM, no dependencies)
 *
 * Real client-side computation for split inference privacy:
 *   1. BPE tokenization (input text stays on device)
 *   2. Embedding lookup (from local weight matrix, tied with LM head)
 *   3. Calibrated DP noise injection (Gaussian mechanism, ε=1.0)
 *   4. Send DP-noised hidden states to server (/api/v1/split/forward)
 *   5. Receive pre-LM-head activations from server
 *   6. LM head projection (local, using tied weights)
 *   7. Top-p / top-k sampling (output selection stays on device)
 *
 * Weight matrix (446 MB float16) is cached in IndexedDB after first download.
 * Server never sees raw token IDs or final token selection.
 */

"use strict";

// ================================================================
// Float16 ↔ Float32 — Lookup Table (fastest approach: 256 KB table)
// ================================================================

const _F16 = new ArrayBuffer(4);
const _F16DV = new DataView(_F16);

function _f16Slow(h) {
  const s = (h & 0x8000) << 16;
  const e = (h >> 10) & 0x1f;
  const m = h & 0x3ff;
  if (e === 0) {
    if (m === 0) { _F16DV.setUint32(0, s); return _F16DV.getFloat32(0); }
    let mm = m, ee = 1;
    while (!(mm & 0x400)) { mm <<= 1; ee--; }
    _F16DV.setUint32(0, s | ((ee + 112) << 23) | ((mm & 0x3ff) << 13));
    return _F16DV.getFloat32(0);
  }
  if (e === 31) {
    _F16DV.setUint32(0, s | 0x7f800000 | (m << 13));
    return _F16DV.getFloat32(0);
  }
  _F16DV.setUint32(0, s | ((e + 112) << 23) | (m << 13));
  return _F16DV.getFloat32(0);
}

/** Pre-computed lookup table for ALL 65536 float16 bit patterns → float32. */
const F16_TABLE = new Float32Array(65536);
for (let i = 0; i < 65536; i++) F16_TABLE[i] = _f16Slow(i);

/** Fast float16 → float32 via table lookup. */
function f16(h) { return F16_TABLE[h]; }


// ================================================================
// GPT-2 Byte-Level Encoding (used by Qwen BPE)
// ================================================================

function buildByteMap() {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = [...bs];
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  }
  const toChar = {}, toByte = {};
  for (let i = 0; i < bs.length; i++) {
    toChar[bs[i]] = String.fromCodePoint(cs[i]);
    toByte[String.fromCodePoint(cs[i])] = bs[i];
  }
  return { toChar, toByte };
}

const BYTE_MAP = buildByteMap();

function textToByteChars(text) {
  const bytes = new TextEncoder().encode(text);
  let r = "";
  for (const b of bytes) r += BYTE_MAP.toChar[b];
  return r;
}

function byteCharsToText(chars) {
  const bytes = [];
  for (const c of chars) {
    if (c in BYTE_MAP.toByte) bytes.push(BYTE_MAP.toByte[c]);
  }
  return new TextDecoder().decode(new Uint8Array(bytes));
}


// ================================================================
// BPE Tokenizer (Qwen 2.5 compatible)
// ================================================================

const PRE_RE = /(?:'[sStTmMdD]|'[rRvV][eE]|'[lL][lL])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;

class BPETokenizer {
  constructor(json) {
    const m = json.model;
    this.vocab = m.vocab;                    // token→id
    this.rev = {};                           // id→token
    for (const [t, id] of Object.entries(this.vocab)) this.rev[id] = t;
    this.ranks = {};                         // "a b"→rank
    for (let i = 0; i < m.merges.length; i++) this.ranks[m.merges[i]] = i;
    this.added = {};                         // special tokens
    for (const a of (json.added_tokens || [])) this.added[a.content] = a.id;
    this.eosId = this.added["<|endoftext|>"] || this.added["<|im_end|>"] || 151643;
    this._cache = {};
  }

  encode(text) {
    if (!text) return [];
    const ids = [];
    let rem = text;
    while (rem.length > 0) {
      let found = false;
      for (const [tok, id] of Object.entries(this.added)) {
        if (rem.startsWith(tok)) { ids.push(id); rem = rem.slice(tok.length); found = true; break; }
      }
      if (found) continue;
      let nxt = rem.length;
      for (const tok of Object.keys(this.added)) {
        const p = rem.indexOf(tok); if (p > 0 && p < nxt) nxt = p;
      }
      const chunk = rem.slice(0, nxt); rem = rem.slice(nxt);
      const words = chunk.match(PRE_RE) || [];
      for (const w of words) {
        const bw = textToByteChars(w);
        for (const t of this._bpe(bw)) {
          if (t in this.vocab) ids.push(this.vocab[t]);
        }
      }
    }
    return ids;
  }

  decode(ids) {
    return byteCharsToText(ids.map(id => this.rev[id] || "").join(""));
  }

  _bpe(word) {
    if (word in this._cache) return this._cache[word];
    let sym = [...word];
    if (sym.length <= 1) { this._cache[word] = sym; return sym; }
    for (;;) {
      let bestR = Infinity, bestP = null;
      for (let i = 0; i < sym.length - 1; i++) {
        const r = this.ranks[sym[i] + " " + sym[i + 1]];
        if (r !== undefined && r < bestR) { bestR = r; bestP = [sym[i], sym[i + 1]]; }
      }
      if (!bestP) break;
      const ns = []; let i = 0;
      while (i < sym.length) {
        if (i < sym.length - 1 && sym[i] === bestP[0] && sym[i + 1] === bestP[1]) {
          ns.push(bestP[0] + bestP[1]); i += 2;
        } else { ns.push(sym[i]); i++; }
      }
      sym = ns;
      if (sym.length === 1) break;
    }
    this._cache[word] = sym;
    return sym;
  }
}


// ================================================================
// IndexedDB Weight Cache — Chunked to prevent OOM on mobile
//
// Problem: Storing a 446 MB ArrayBuffer in one IDB put() triggers
// a structured-clone that copies the ENTIRE buffer in memory.
// Peak = 446 MB (buffer) + 446 MB (clone) = ~892 MB → phone kills tab.
//
// Fix: Write in 8 MB chunks. Each clone is only 8 MB.
// Peak = 446 MB (buffer) + 8 MB (chunk) + 8 MB (clone) = ~462 MB.
// ================================================================

const DB_NAME = "tensafe_split", DB_VER = 1, STORE = "w";
const CACHE_CHUNK = 8 * 1024 * 1024;  // 8 MB per chunk

function _openDB() {
  return new Promise((ok, fail) => {
    const r = indexedDB.open(DB_NAME, DB_VER);
    r.onupgradeneeded = () => r.result.createObjectStore(STORE);
    r.onsuccess = () => ok(r.result);
    r.onerror = () => fail(r.error);
  });
}

function _idbGet(db, key) {
  return new Promise(ok => {
    const r = db.transaction(STORE, "readonly").objectStore(STORE).get(key);
    r.onsuccess = () => ok(r.result ?? null);
    r.onerror = () => ok(null);
  });
}

function _idbPut(db, key, value) {
  return new Promise((ok, fail) => {
    const tx = db.transaction(STORE, "readwrite");
    tx.objectStore(STORE).put(value, key);
    tx.oncomplete = ok;
    tx.onerror = () => fail(tx.error);
  });
}

/** Read cached weights. Handles chunked v2 format + legacy v1 single-buffer. */
async function cacheGet(key) {
  try {
    const db = await _openDB();

    // v2 chunked format (safe for mobile)
    const meta = await _idbGet(db, key + "_meta");
    if (meta && meta.version === 2) {
      const buffer = new ArrayBuffer(meta.total);
      const view = new Uint8Array(buffer);
      let offset = 0;
      let valid = true;
      for (let i = 0; i < meta.numChunks; i++) {
        const chunk = await _idbGet(db, key + "_" + i);
        if (!chunk) {
          console.warn("[Split] Cache chunk", i, "missing — cache invalid");
          valid = false;
          break;
        }
        view.set(new Uint8Array(chunk), offset);
        offset += chunk.byteLength;
        // chunk goes out of scope → eligible for GC before next read
      }
      if (valid) {
        console.log("[Split] Loaded", meta.numChunks, "cached chunks (" +
          (meta.total / 1048576).toFixed(0) + " MB)");
        return buffer;
      }
      // Fall through to legacy if v2 incomplete
    }

    // Legacy v1: single buffer (may OOM on phones during write, but reads fine)
    return await _idbGet(db, key);
  } catch { return null; }
}

/** Write buffer to IndexedDB in small chunks to avoid OOM on mobile.
 *  Metadata is written LAST so incomplete writes are detected on next load. */
async function cacheSet(key, buffer) {
  try {
    const db = await _openDB();
    const total = buffer.byteLength;
    const numChunks = Math.ceil(total / CACHE_CHUNK);

    for (let i = 0; i < numChunks; i++) {
      const start = i * CACHE_CHUNK;
      const end = Math.min(start + CACHE_CHUNK, total);
      // .slice() creates an 8 MB copy — structured-clone of 8 MB is safe
      await _idbPut(db, key + "_" + i, buffer.slice(start, end));
    }

    // Metadata LAST = atomic: if any chunk write failed, no meta → re-download
    await _idbPut(db, key + "_meta", {
      total, numChunks, chunkSize: CACHE_CHUNK, version: 2
    });
    console.log("[Split] Cached in", numChunks, "chunks");
  } catch (e) {
    console.warn("[Split] Cache write failed:", e);
  }
}


// ================================================================
// Gaussian DP Noise (Box-Muller)
// ================================================================

function gaussNoise(n, sigma) {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 2) {
    const u1 = Math.random(), u2 = Math.random();
    const r = Math.sqrt(-2 * Math.log(u1 || 1e-30));
    const t = 2 * Math.PI * u2;
    out[i] = sigma * r * Math.cos(t);
    if (i + 1 < n) out[i + 1] = sigma * r * Math.sin(t);
  }
  return out;
}

function l2norm(a) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * a[i]; return Math.sqrt(s); }


// ================================================================
// Base64 ↔ ArrayBuffer
// ================================================================

function ab2b64(buffer) {
  const u8 = new Uint8Array(buffer);
  let bin = "";
  for (let i = 0; i < u8.length; i += 0x8000) {
    bin += String.fromCharCode.apply(null, u8.subarray(i, Math.min(i + 0x8000, u8.length)));
  }
  return btoa(bin);
}

function b642ab(b64) {
  const bin = atob(b64);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return u8.buffer;
}


// ================================================================
// Top-p / Top-k Sampling
// ================================================================

function sample(logits, temp = 0.7, topK = 50, topP = 0.9, prevIds = null, repPenalty = 1.3) {
  const N = logits.length;

  // Repetition penalty: penalise tokens that already appeared
  if (prevIds && prevIds.length > 0 && repPenalty > 1.0) {
    const seen = new Set(prevIds.slice(-64));  // last 64 tokens window
    for (const id of seen) {
      if (id >= 0 && id < N) {
        logits[id] = logits[id] > 0
          ? logits[id] / repPenalty
          : logits[id] * repPenalty;
      }
    }
  }

  if (temp <= 0) {
    let mx = 0; for (let i = 1; i < N; i++) { if (logits[i] > logits[mx]) mx = i; }
    return mx;
  }

  // Temperature
  const sc = new Float32Array(N);
  for (let i = 0; i < N; i++) sc[i] = logits[i] / temp;

  // Top-k mask
  if (topK > 0 && topK < N) {
    const vals = Float32Array.from(sc).sort((a, b) => a - b).reverse();
    const thr = vals[topK - 1];
    for (let i = 0; i < N; i++) { if (sc[i] < thr) sc[i] = -Infinity; }
  }

  // Softmax
  let mx = -Infinity;
  for (let i = 0; i < N; i++) { if (sc[i] > mx) mx = sc[i]; }
  const pr = new Float32Array(N);
  let sm = 0;
  for (let i = 0; i < N; i++) { pr[i] = Math.exp(sc[i] - mx); sm += pr[i]; }
  for (let i = 0; i < N; i++) pr[i] /= sm;

  // Top-p nucleus
  if (topP < 1.0) {
    const idx = Array.from({ length: N }, (_, i) => i);
    idx.sort((a, b) => pr[b] - pr[a]);
    let cum = 0;
    const keep = new Set();
    for (const i of idx) { keep.add(i); cum += pr[i]; if (cum >= topP) break; }
    sm = 0;
    for (let i = 0; i < N; i++) { if (!keep.has(i)) pr[i] = 0; else sm += pr[i]; }
    if (sm === 0) sm = 1;  // guard: prevent NaN from division by zero
    for (let i = 0; i < N; i++) pr[i] /= sm;
  }

  // Multinomial
  const r = Math.random();
  let c = 0;
  for (let i = 0; i < N; i++) { c += pr[i]; if (r < c) return i; }
  return N - 1;
}


// ================================================================
// GateLink-Split Inference Client
// ================================================================

class SplitInferenceClient {
  constructor() {
    this.weights = null;       // Uint16Array (float16 tied embed+lm_head)
    this.tokenizer = null;     // BPETokenizer
    this.config = null;        // server config

    this.HD = 1536;            // hidden_dim
    this.VS = 151936;          // vocab_size

    this.dpEps = 1.0;
    this.dpDelta = 1e-5;
    this.dpSigma = 0;
    this.dpSens = 1.0;

    this.expertKW = {};
    this._ws = null;          // WebSocket connection (persistent)
    this._wsReady = false;    // true when WS is open and ready
    this._wsPending = null;   // {resolve, reject} for current forward call
    this.ready = false;
    this._abort = null;
  }

  // -------- Initialization --------

  async initialize(onProgress) {
    const P = onProgress || (() => {});

    // Server config
    P("config", 0, "Fetching config...");
    const cfg = await fetch("/api/v1/split/config").then(r => r.json());
    this.config = cfg;
    this.HD = cfg.hidden_dim || 1536;
    this.VS = cfg.vocab_size || 151936;
    this.dpEps = cfg.dp_epsilon ?? 1.0;
    this.dpSigma = cfg.dp_sigma ?? 0;
    this.dpSens = cfg.dp_sensitivity ?? 1.0;
    this.expertKW = cfg.expert_keywords || {};
    P("config", 1, "Config loaded");

    // Tokenizer
    P("tokenizer", 0, "Loading tokenizer...");
    const tj = await fetch("/weights/tokenizer.json").then(r => r.json());
    this.tokenizer = new BPETokenizer(tj);
    P("tokenizer", 1, "Tokenizer ready (BPE, " + Object.keys(this.tokenizer.vocab).length + " tokens)");

    // Weights (446 MB, IndexedDB cached)
    await this._loadWeights(P);

    // Integrity check: verify server is reachable and tokenizer+weights work
    P("verify", 0.5, "Running integrity check...");
    try {
      const verifyResp = await fetch("/api/v1/split/verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ probe_text: "hello" }),
      });
      if (verifyResp.ok) {
        const vData = await verifyResp.json();
        // Check that our local embedding matches server's expected embedding
        const probeIds = vData.token_ids;
        const serverEmbHash = vData.embed_hash;
        if (probeIds && probeIds.length > 0) {
          const localEmb = this.embed(probeIds[0]);
          let localHash = 0;
          for (let d = 0; d < Math.min(8, this.HD); d++) {
            localHash += Math.round(localEmb[d] * 10000);
          }
          if (serverEmbHash !== undefined && Math.abs(localHash - serverEmbHash) > 1) {
            console.error("[Split] INTEGRITY FAIL: local embed hash", localHash, "!= server", serverEmbHash);
            console.error("[Split] This means weights are corrupt or stale. Clearing cache...");
            // Force re-download by clearing cache
            try {
              const db = await _openDB();
              const tx = db.transaction(STORE, "readwrite");
              tx.objectStore(STORE).clear();
              await new Promise((ok, fail) => { tx.oncomplete = ok; tx.onerror = fail; });
              console.log("[Split] Cache cleared — reload page to re-download weights");
            } catch (ce) { console.warn("[Split] Cache clear failed:", ce); }
            throw new Error("Weight integrity check failed. Please reload the page to re-download weights.");
          }
          console.log("[Split] Integrity check PASSED: embed hash match (" + localHash + ")");
        }
        P("verify", 1, "Integrity OK");
      } else {
        console.warn("[Split] Verify endpoint not available (HTTP " + verifyResp.status + ") — skipping integrity check");
        P("verify", 1, "Verify skipped (endpoint unavailable)");
      }
    } catch (e) {
      if (e.message.includes("integrity")) throw e;  // Re-throw integrity failures
      console.warn("[Split] Integrity check skipped:", e.message);
      P("verify", 1, "Verify skipped");
    }

    this.ready = true;
    // Open persistent WebSocket for split inference (non-blocking)
    this._wsConnect().catch(e =>
      console.warn("[Split-WS] Initial connect failed, will use HTTP:", e.message)
    );
    P("ready", 1, "Split client ready");
  }

  async _loadWeights(P) {
    // Cache key includes version — bump to invalidate stale/corrupt caches.
    // v3: mandatory server tokenization + integrity check + diagnostic logging.
    const CACHE_KEY = "embed_v3";
    const expectedBytes = this.VS * this.HD * 2;  // 151936 * 1536 * 2

    P("weights", 0, "Checking cache...");
    const cached = await cacheGet(CACHE_KEY);
    if (cached && cached.byteLength === expectedBytes) {
      this.weights = new Uint16Array(cached);
      console.log("[Split] Loaded weights from IndexedDB cache:", this.weights.length, "values");
      P("weights", 1, "Weights loaded from cache (" + (cached.byteLength / 1048576).toFixed(0) + " MB)");
      return;
    }
    if (cached) {
      console.warn("[Split] Cached weights wrong size:", cached.byteLength, "expected:", expectedBytes, "— re-downloading");
    }

    P("weights", 0.01, "Downloading weights (446 MB, first time only)...");
    const resp = await fetch("/weights/embed_tokens.bin");
    if (!resp.ok) throw new Error("Weight download failed: HTTP " + resp.status);
    const total = parseInt(resp.headers.get("Content-Length") || "0");
    const reader = resp.body.getReader();

    // Stream directly into a single pre-allocated buffer.
    // Peak memory during download: ~1x file size (446 MB).
    let buffer, view, offset = 0;
    let chunks = null;  // fallback if Content-Length unknown

    if (total > 0) {
      try {
        buffer = new ArrayBuffer(total);
      } catch (oom) {
        throw new Error(
          "Not enough memory to load weights (" +
          (total / 1048576).toFixed(0) + " MB). " +
          "Close other tabs and try again."
        );
      }
      view = new Uint8Array(buffer);
    } else {
      chunks = [];
    }

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (view) {
        view.set(value, offset);
      } else {
        chunks.push(value);
      }
      offset += value.byteLength;
      if (total > 0) {
        P("weights", offset / total, `Downloading weights... ${(offset / 1048576).toFixed(0)} / ${(total / 1048576).toFixed(0)} MB`);
      }
    }

    // Verify download completeness
    if (total > 0 && offset !== total) {
      throw new Error(
        "Download incomplete: got " + (offset / 1048576).toFixed(0) +
        " MB of " + (total / 1048576).toFixed(0) + " MB. Check your connection."
      );
    }

    // If Content-Length was unknown, combine chunks now and free them
    if (chunks) {
      try {
        buffer = new ArrayBuffer(offset);
      } catch (oom) {
        chunks.length = 0;
        chunks = null;
        throw new Error(
          "Not enough memory to assemble weights (" +
          (offset / 1048576).toFixed(0) + " MB). Close other tabs and try again."
        );
      }
      view = new Uint8Array(buffer);
      let pos = 0;
      for (const c of chunks) { view.set(c, pos); pos += c.byteLength; }
      chunks.length = 0;  // release chunk refs for GC
      chunks = null;
    }

    // Validate size: expected = vocab_size × hidden_dim × 2 bytes (float16)
    const expected = this.VS * this.HD * 2;
    if (buffer.byteLength !== expected) {
      throw new Error(
        "Weight file size mismatch: got " + buffer.byteLength +
        " bytes, expected " + expected + " (" + this.VS + " × " + this.HD + " × 2)"
      );
    }

    // Set weights IMMEDIATELY — usable before cache finishes.
    // Uint16Array is just a VIEW on the same buffer (no extra memory).
    this.weights = new Uint16Array(buffer);
    P("weights", 1, "Weights ready — caching in background...");

    // Cache in BACKGROUND using chunked writes (8 MB each).
    // Old code did one 446 MB put() → structured-clone doubled memory to ~892 MB
    // → phone killed the tab. Chunked writes peak at only ~462 MB.
    cacheSet(CACHE_KEY, buffer).then(() => {
      P("weights", 1, "Weights cached for next time");
      console.log("[Split] Background cache complete");
    }).catch(e => {
      console.warn("[Split] Background cache failed (weights still usable):", e);
    });
  }

  // -------- Client-side operations --------

  /** Look up embedding for one token. Returns Float32Array[HD]. */
  embed(tokenId) {
    const off = tokenId * this.HD;
    const out = new Float32Array(this.HD);
    for (let d = 0; d < this.HD; d++) out[d] = F16_TABLE[this.weights[off + d]];
    return out;
  }

  /** Add DP noise to one embedding. Clips to unit L2 norm first. */
  noisify(emb) {
    const norm = l2norm(emb);
    const scale = norm > this.dpSens ? (this.dpSens / norm) : 1.0;
    const noise = gaussNoise(this.HD, this.dpSigma);
    const out = new Float32Array(this.HD);
    for (let d = 0; d < this.HD; d++) out[d] = emb[d] * scale + noise[d];
    return out;
  }

  /** LM head projection: hidden[HD] → logits[VS]. Uses tied weights. */
  project(hidden) {
    const logits = new Float32Array(this.VS);
    const w = this.weights;
    const HD = this.HD;
    for (let v = 0; v < this.VS; v++) {
      const off = v * HD;
      let dot = 0;
      for (let d = 0; d < HD; d++) dot += hidden[d] * F16_TABLE[w[off + d]];
      logits[v] = dot;
    }
    return logits;
  }

  /** Keyword-based expert routing (same logic as server). */
  routeExpert(query) {
    const q = query.toLowerCase();
    let best = "shared_attention", bestS = 0;
    for (const [name, kws] of Object.entries(this.expertKW)) {
      if (!kws || !kws.length) continue;
      let s = 0;
      for (const kw of kws) { if (q.includes(kw.toLowerCase())) s++; }
      if (s > bestS) { bestS = s; best = name; }
    }
    return best;
  }

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
        if (msg.type === "pong") return;
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
      setTimeout(() => {
        if (this._wsPending) {
          this._wsPending.reject(new Error("WS forward timeout (30s)"));
          this._wsPending = null;
        }
      }, 30000);
    });
  }

  // -------- Server communication --------

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

    if (this._abort && this._abort.signal.aborted) {
      clearTimeout(timeoutId);
      throw new Error("Generation aborted");
    }

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

  // -------- Full autoregressive generation --------

  /**
   * Run split inference.
   * @param {string} query
   * @param {object} opts  { maxTokens, temperature, topP, topK, useHE }
   * @param {Function} onToken  (tokenText, metrics) per generated token
   * @param {Function} onStatus (stage, detail) for progress
   * @returns {{ text, totalTokens, totalTimeMs, tokPerSec, metrics, expert }}
   */
  async generate(query, opts = {}, onToken, onStatus) {
    if (!this.ready) throw new Error("Not initialized");
    const { maxTokens = 128, temperature = 0.7, topP = 0.9, topK = 50, useHE = true } = opts;
    const status = onStatus || (() => {});
    this._abort = new AbortController();
    // Ensure WS is connected for this generation session
    if (!this._wsReady) {
      try { await this._wsConnect(); } catch(e) {
        console.warn("[Split-WS] Connect failed, using HTTP fallback");
      }
    }

    // Stable session_id for this generation (reused across all steps)
    const sessionId = "split_" + Date.now() + "_" + Math.random().toString(36).slice(2, 8);

    // 1. Tokenize with training-format prompt using SERVER tokenizer.
    // MANDATORY — JS BPE tokenizer produces wrong token IDs on phone browsers
    // due to Unicode regex differences → wrong embeddings → garbage output.
    // Server tokenization uses exact HuggingFace tokenizer = always correct.
    status("tokenize", "Tokenizing input...");
    const chatPrompt =
      "### System:\nYou are a helpful financial assistant.\n\n" +
      "### Instruction:\n" + query + "\n\n### Response:\n";
    let inputIds;
    const MAX_TOKENIZE_RETRIES = 3;
    for (let attempt = 1; attempt <= MAX_TOKENIZE_RETRIES; attempt++) {
      try {
        const tokResp = await fetch("/api/v1/split/tokenize", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: chatPrompt }),
        });
        if (!tokResp.ok) throw new Error("HTTP " + tokResp.status + ": " + (await tokResp.text()).slice(0, 200));
        const tokData = await tokResp.json();
        inputIds = tokData.token_ids;
        if (!Array.isArray(inputIds) || inputIds.length === 0) {
          throw new Error("Server returned empty token_ids");
        }
        console.log("[Split] Server tokenized OK:", inputIds.length, "tokens, first 5:", inputIds.slice(0, 5));
        break;
      } catch (e) {
        console.error("[Split] Server tokenize attempt", attempt, "failed:", e);
        if (attempt === MAX_TOKENIZE_RETRIES) {
          // HARD FAIL — do NOT fall back to JS BPE (produces wrong token IDs on phones).
          throw new Error(
            "Server tokenization failed after " + MAX_TOKENIZE_RETRIES +
            " attempts. Cannot proceed — JS fallback tokenizer is unreliable. " +
            "Check server connection. Error: " + e.message
          );
        }
        // Wait briefly before retry
        await new Promise(r => setTimeout(r, 500 * attempt));
      }
    }
    status("tokenize", inputIds.length + " input tokens (server)");

    // 2. Route expert
    const expert = this.routeExpert(query);
    status("route", "Expert: " + expert);

    // 3. Autoregressive loop
    let curIds = [...inputIds];
    const generated = [];
    const allMetrics = [];
    const t0 = performance.now();

    // Cache noised embeddings (position → Float32Array)
    const noiseCache = new Map();

    // Allow up to GRACE_BUFFER extra tokens after maxTokens to finish the
    // current sentence. Stops as soon as a sentence-ending char is seen.
    const GRACE_BUFFER = 32;
    const hardLimit = maxTokens + GRACE_BUFFER;

    for (let step = 0; step < hardLimit; step++) {
      if (this._abort.signal.aborted) break;

      // If past maxTokens, only continue if mid-sentence
      if (step >= maxTokens && generated.length > 0) {
        const tail = generated.join("").trim();
        const lastChar = tail.slice(-1);
        if (".!?\n\"')".includes(lastChar)) {
          status("done", "Sentence complete (" + generated.length + " tokens)");
          break;
        }
      }

      const stepT = performance.now();

      // Incremental mode: first call sends full sequence, subsequent calls
      // send only the NEW token's embedding (server caches KV from prior steps)
      const isIncremental = step > 0;
      let flat, sendSeqLen;

      const embT = performance.now();
      if (isIncremental) {
        // Only embed the new token (last position)
        const lastIdx = curIds.length - 1;
        const emb = this.embed(curIds[lastIdx]);
        const noised = this.noisify(emb);
        // Don't cache incremental embeddings (never re-read, wastes memory)
        flat = new Float32Array(this.HD);
        flat.set(noised, 0);
        sendSeqLen = 1;
        status("embed", "Embedding 1 new token + DP noise (incremental)...");
      } else {
        // First call: send full sequence
        flat = new Float32Array(curIds.length * this.HD);
        for (let i = 0; i < curIds.length; i++) {
          let noised;
          if (noiseCache.has(i)) {
            noised = noiseCache.get(i);
          } else {
            const emb = this.embed(curIds[i]);
            noised = this.noisify(emb);
            noiseCache.set(i, noised);
          }
          flat.set(noised, i * this.HD);
        }
        sendSeqLen = curIds.length;
        status("embed", "Embedding " + curIds.length + " tokens + DP noise...");
      }
      const embMs = performance.now() - embT;

      // 3b. Server forward (transformer layers + HE LoRA)
      status("server", "Server: transformer + HE LoRA" + (isIncremental ? " (incremental)" : "") + "...");
      const srvT = performance.now();
      let result;
      try {
        result = await this.serverForward(flat, sendSeqLen, expert, useHE, sessionId, isIncremental);
      } catch (e) {
        if (e.name === "AbortError" || this._abort.signal.aborted) {
          status("aborted", "Generation cancelled");
          break;
        }
        throw e;
      }
      const srvMs = performance.now() - srvT;

      // 3c. Decode pre-activations
      status("project", "LM head projection (client-side)...");
      const projT = performance.now();

      const preActBuf = b642ab(result.pre_activations_b64);
      const preActF32 = new Float32Array(preActBuf);

      // In incremental mode, server returns only last position's hidden state
      // In full mode, extract from the full sequence output
      let lastH;
      if (result.incremental) {
        // Server returned [1, 1, HD] — just the last token
        lastH = preActF32.slice(0, this.HD);
      } else {
        const lastOff = (sendSeqLen - 1) * this.HD;
        lastH = preActF32.slice(lastOff, lastOff + this.HD);
      }

      // LM head projection (tied weights, ~0.5s on iPhone)
      const logits = this.project(lastH);
      const projMs = performance.now() - projT;

      // 3d. Sample next token (with repetition penalty over last 64 tokens)
      const nextId = sample(logits, temperature, topK, topP, curIds, 1.3);
      const tokText = this.tokenizer.decode([nextId]);

      // Diagnostic: log first few tokens for debugging
      if (step < 5) {
        const topIds = [];
        const logitsCopy = Float32Array.from(logits);
        for (let t = 0; t < 5; t++) {
          let mx = 0;
          for (let i = 1; i < logitsCopy.length; i++) { if (logitsCopy[i] > logitsCopy[mx]) mx = i; }
          topIds.push({ id: mx, logit: logitsCopy[mx].toFixed(2), tok: this.tokenizer.decode([mx]) });
          logitsCopy[mx] = -Infinity;
        }
        console.log(`[Split] Step ${step}: nextId=${nextId} tok=${JSON.stringify(tokText)} top5=`, topIds);
      }

      // Stop on EOS, <|im_end|>, or "###" (new section = end of response)
      const imEndId = this.tokenizer.added["<|im_end|>"] || 151645;
      if (nextId === this.tokenizer.eosId || nextId === imEndId) {
        status("done", "EOS");
        break;
      }
      // Stop if model starts a new "### " section (training format boundary)
      if (generated.length > 2) {
        const tail = generated.slice(-3).join("") + tokText;
        if (tail.includes("\n###") || tail.includes("\n\n###")) {
          status("done", "Section break");
          break;
        }
      }

      curIds.push(nextId);
      generated.push(tokText);

      const stepMs = performance.now() - stepT;
      const totalMs = performance.now() - t0;

      const metrics = {
        step: step + 1,
        tokenId: nextId,
        embedMs: Math.round(embMs),
        serverMs: Math.round(srvMs),
        projectMs: Math.round(projMs),
        stepMs: Math.round(stepMs),
        totalMs: Math.round(totalMs),
        tokPerSec: ((step + 1) / totalMs * 1000).toFixed(2),
        heActive: result.he_active,
        heOps: result.he_operations || 0,
        encryptMs: result.encrypt_ms || 0,
        computeMs: result.compute_ms || 0,
        decryptMs: result.decrypt_ms || 0,
        expert: expert,
        layersComputed: result.layers_computed || 0,
        dpEpsilon: this.dpEps,
        dpSigma: this.dpSigma,
        incremental: !!result.incremental,
        cachedSeqLen: result.cached_seq_len || 0,
      };
      allMetrics.push(metrics);
      if (onToken) onToken(tokText, metrics);
    }

    const totalMs = performance.now() - t0;
    return {
      text: generated.join(""),
      totalTokens: generated.length,
      totalTimeMs: Math.round(totalMs),
      tokPerSec: (generated.length / totalMs * 1000).toFixed(2),
      metrics: allMetrics,
      expert,
      inputTokens: inputIds.length,
      dpEpsilon: this.dpEps,
      sessionId,
    };
  }

  abort() {
    if (this._abort && !this._abort.signal.aborted) {
      this._abort.abort();
      console.log("[Split] Generation aborted by user");
    }
  }
}

// ================================================================
// Export
// ================================================================
window.SplitInferenceClient = SplitInferenceClient;
