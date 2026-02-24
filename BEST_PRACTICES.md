# TenSafe Finance Demonstrator — Best Practices & System Guide

> **Version 2.0** | Last updated after the 17-fix audit (55/55 tests green)

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Base Model Selection](#2-base-model-selection)
3. [TG Tinker SDK Usage](#3-tg-tinker-sdk-usage)
4. [Training Pipeline](#4-training-pipeline)
5. [Expert Routing (Gated MoE)](#5-expert-routing-gated-moe)
6. [GateLink-Split Inference](#6-gatelink-split-inference)
7. [Homomorphic Encryption (CKKS)](#7-homomorphic-encryption-ckks)
8. [Security & Privacy](#8-security--privacy)
9. [Deployment & Configuration](#9-deployment--configuration)
10. [QA & Verification](#10-qa--verification)
11. [Performance Tuning](#11-performance-tuning)
12. [FAQ / Troubleshooting](#12-faq--troubleshooting)

---

## 1. System Overview

### Architecture

```
  Training Pipeline                       Inference Pipeline
  ==================                      ==================

  Finance Data                            User Query
      |                                       |
  [SFT Training]                         [Frontend app.js]
      |                                       |
  [RL (REINFORCE)]                    +-------+-------+
      |                               |               |
  [MoE Assembly]                  WebSocket      GateLink-Split
      |                          Streaming       (split_client.js)
  [TGSP Packaging]                    |               |
      |                          FastAPI          Client embed
  moe_config.json                 Server            + DP noise
      |                               |               |
  Inference Engine  <---------  FinanceInferenceEngine |
      |                          |       |             |
  HE-LoRA Deltas             CKKS HE  DP Noise    Server forward
      |                          |       |         (28 layers + HE)
  Gated MoE Routing              |       |             |
      |                          v       v             v
  Token Stream -------> Dashboard Metrics       Client LM head
                                                + Sampling
```

### Component Summary

| Component | Purpose | Key File(s) |
|-----------|---------|-------------|
| **TG Tinker SDK** | Training orchestration, DP, LoRA | `tensafe.core.orchestrator` |
| **SFT Trainer** | Supervised fine-tuning on finance data | `demonstrator/training/train_sft.py` |
| **RL Trainer** | REINFORCE policy optimization | `demonstrator/training/train_rl.py` |
| **MoE Assembler** | Package adapters into TGSP format | `demonstrator/training/assemble_moe.py` |
| **Inference Engine** | HE-encrypted LoRA inference + MoE | `demonstrator/server/inference_engine.py` |
| **FastAPI Server** | WebSocket streaming + REST API | `demonstrator/server/app.py` |
| **Split Client** | Client-side embedding + sampling (JS) | `demonstrator/frontend/split_client.js` |
| **Frontend** | Dashboard with HE metrics | `demonstrator/frontend/app.js` |
| **CKKS HE** | Homomorphic encryption for LoRA | CuKKS / Pyfhel / Emulator |
| **TGSP** | Signed adapter packages | `tensafe.lora_to_tgsp_converter` |
| **RVUv2** | Adapter safety screening | `tensafe.tgsp_adapter_registry` |

### Directory Layout

```
TenSafe_Project/
  demonstrator/
    server/
      app.py                  # FastAPI server (endpoints)
      inference_engine.py     # Core engine (HE, DP, MoE, split)
    frontend/
      app.js                  # Dashboard + WebSocket client
      split_client.js         # GateLink-Split JS client
      index.html              # Frontend entry point
    training/
      train_sft.py            # SFT training script
      train_rl.py             # RL REINFORCE training
      assemble_moe.py         # MoE assembly + TGSP conversion
      data_loading.py         # Dataset loaders
      reward_fn.py            # RL reward function
    adapters/
      tgsp/                   # TGSP output + moe_config.json
      banking_expert/         # SFT checkpoints
      investment_expert/      # SFT checkpoints
      shared_attention/       # SFT checkpoints
    scripts/
      qa_verify.py            # 34 QA tests
      regression_gatelink.py  # 21 GateLink regression tests
      benchmark_throughput.py # Throughput benchmark
  TenSafe_Extracted/
    src/tensafe/              # Core SDK (orchestrator, TGSP, RLVR)
  Adapter-Safety-Tensafe/
    src/                      # RVUv2 safety screening
```

---

## 2. Base Model Selection

### Why Qwen 2.5-1.5B

The demonstrator uses **Qwen/Qwen2.5-1.5B** as its base model. This choice is driven by the constraints of privacy-preserving split inference on mobile devices:

| Criterion | Qwen 2.5-1.5B | Why It Matters |
|-----------|---------------|----------------|
| **Parameters** | 1.5B | Fits in 1.5 GB (phone profile) |
| **Layers** | 28 | Enough depth for expert routing quality |
| **Hidden dim** | 1536 | Fits in 8192 SIMD slots (CKKS) with room for batching |
| **Vocab size** | 151,936 | Full multilingual BPE coverage |
| **Architecture** | Transformer (causal LM) | Native KV cache, LoRA-friendly attention layers |

### Comparison With Alternatives

| Model | Params | Hidden Dim | Layers | SIMD Fit | Phone Viable |
|-------|--------|-----------|--------|----------|-------------|
| Qwen 2.5-0.5B | 0.5B | 896 | 24 | Yes | Yes, but quality too low for finance |
| **Qwen 2.5-1.5B** | **1.5B** | **1536** | **28** | **Yes** | **Yes (recommended)** |
| Qwen 2.5-3B | 3B | 2048 | 36 | Marginal | No (>3 GB) |
| Llama-3.2-1B | 1.3B | 2048 | 16 | Marginal | Yes, but fewer layers hurt routing |
| Phi-3-mini | 3.8B | 3072 | 32 | No | No (too large) |

### When to Consider Upgrading

- **Server/workstation profile** with >8 GB VRAM: Qwen 2.5-3B for better reasoning
- **If hidden_dim > 8192**: You lose single-ciphertext CKKS packing; need to split across multiple ciphertexts
- **Multi-language finance**: Qwen's tokenizer handles CJK natively, which is an advantage over Llama

---

## 3. TG Tinker SDK Usage

### Step 1: Install

```bash
pip install tensafe-platform tensafe
# Or from source:
cd TenSafe_Extracted && pip install -e src/
```

### Step 2: Authenticate

```bash
tensorguard tinker auth --api-key YOUR_KEY
```

### Step 3: Create a Training Client

```python
from tensafe.core.orchestrator import OrchestratorConfig, TenSafeOrchestrator

config = OrchestratorConfig(
    model_name_or_path="Qwen/Qwen2.5-1.5B",
    torch_dtype="float16",
    device_map="cuda",

    # LoRA
    lora_enabled=True,
    lora_rank=32,
    lora_alpha=64.0,
    lora_dropout=0.0,
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],

    # Optimizer
    optimizer="adamw",
    learning_rate=1e-4,
    weight_decay=0.01,
    max_grad_norm=1.0,

    # Differential Privacy
    dp_enabled=True,
    dp_noise_multiplier=1.0,
    dp_target_epsilon=8.0,
    dp_target_delta=1e-5,
)

orchestrator = TenSafeOrchestrator(
    config=config,
    orchestrator_id="my-adapter",
)
orchestrator.initialize()
```

### Step 4: OrchestratorConfig Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_name_or_path` | str | — | HuggingFace model ID or local path |
| `torch_dtype` | str | `"float16"` | Model precision (`"float16"`, `"bfloat16"`, `"float32"`) |
| `device_map` | str | `"cuda"` | Device placement (`"cuda"`, `"cpu"`, `"auto"`) |
| `lora_enabled` | bool | `True` | Enable LoRA adapters |
| `lora_rank` | int | `32` | LoRA rank (r). Higher = more capacity, more memory |
| `lora_alpha` | float | `64.0` | LoRA scaling factor. Typically `2 * rank` |
| `lora_target_modules` | list | `["q_proj", ...]` | Which layers get LoRA adapters |
| `learning_rate` | float | `1e-4` | AdamW learning rate |
| `dp_enabled` | bool | `True` | Enable differential privacy |
| `dp_noise_multiplier` | float | `1.0` | DP noise scale multiplier |
| `dp_target_epsilon` | float | `8.0` | Target privacy budget per epoch |

### Step 5: Training Loop Pattern

```python
for batch in dataloader:
    # Forward + backward (accumulates gradients)
    fb = orchestrator.forward_backward(batch, sample_rate)

    if step % grad_accum == 0:
        # Optimizer step (clips gradients, adds DP noise)
        opt = orchestrator.optim_step(clip=True, sample_rate=sample_rate)
```

### Step 6: Checkpoint Save / Load

```python
# Save (includes optimizer state for resume)
state_bytes = orchestrator.save_state(include_optimizer=True)
with open("checkpoint.pt", "wb") as f:
    f.write(state_bytes)

# Load
with open("checkpoint.pt", "rb") as f:
    orchestrator.load_state(f.read())
```

### CLI Commands (Quick Reference)

| Command | Description |
|---------|-------------|
| `tensorguard tinker auth` | Authenticate with API key |
| `tensorguard tinker create-client` | Create a new training client |
| `tensorguard tinker forward-backward` | Run forward + backward pass |
| `tensorguard tinker optim-step` | Run optimizer step |
| `tensorguard tinker sample` | Generate text samples |
| `tensorguard tinker save-state` | Save checkpoint |
| `tensorguard tinker load-state` | Resume from checkpoint |

### Error Handling Best Practices

1. **Always wrap training in try/except** — Save an emergency checkpoint on OOM or crash
2. **Use crash-resilient checkpointing** — Write to `.tmp` then rename (atomic)
3. **Keep only 2 most recent checkpoints** — Saves disk space
4. **Enable gradient checkpointing** — Saves ~60% activation VRAM:
   ```python
   orchestrator._ml_backend._model.gradient_checkpointing_enable()
   ```

---

## 4. Training Pipeline

### 4.1 Data Preparation

**Sources:**
- Sujet-Finance-Instruct-177k (instruction-following finance data)
- financial_phrasebank (sentiment-labeled finance sentences)

**Splits:**
- `banking`: 30,000 samples (loans, deposits, regulations)
- `investment`: 20,000 samples (portfolio, markets, analysis)
- `combined`: 50,000 samples (all finance, used for shared_attention)

**Format:** Each sample is a dict with `instruction`, `input`, `output` fields, converted to a single `text` field for tokenization.

**Data loading functions** (in `demonstrator/training/data_loading.py`):
- `load_banking_dataset()` returns 30K banking samples
- `load_investment_dataset()` returns 20K investment samples
- `load_combined_finance_dataset()` returns 50K combined samples
- `load_rl_prompts(n_prompts)` returns finance query prompts for RL

### 4.2 SFT Training (Step-by-Step)

**Three adapters are trained independently:**

| Adapter | Target Modules | Dataset | Description |
|---------|---------------|---------|-------------|
| `banking_expert` | q/k/v/o_proj | 30K banking | Loans, deposits, regulations |
| `investment_expert` | q/k/v/o_proj | 20K investment | Portfolio, markets, analysis |
| `shared_attention` | q/k/v/o_proj + gate/up_proj | 50K combined | General financial reasoning |

**Hyperparameters:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| LoRA rank | 32 | Balances capacity and memory |
| LoRA alpha | 64.0 | Standard 2x rank scaling |
| Learning rate | 1e-4 (default) | AdamW with weight_decay=0.01 |
| Batch size | 1 | Per-GPU; effective batch = batch_size * grad_accum |
| Gradient accumulation | 8 | Effective batch size = 8 |
| Max steps | 2,000 | ~3 epochs over the dataset |
| Max sequence length | 512 | Truncated/padded |
| Checkpoint interval | 250 optim steps | Crash-resilient |

**Step-by-step:**

```bash
# Step 1: Train all three adapters
python demonstrator/training/train_sft.py --adapter all

# Step 2: Train only one adapter (e.g., banking)
python demonstrator/training/train_sft.py --adapter banking_expert

# Step 3: Custom hyperparams
python demonstrator/training/train_sft.py \
    --adapter all \
    --max-steps 3000 \
    --rank 32 \
    --alpha 64.0 \
    --batch-size 1 \
    --grad-accum 8 \
    --lr 1e-4 \
    --max-seq-length 512

# Step 4: Re-train an adapter that already has adapter_final.pt
python demonstrator/training/train_sft.py \
    --adapter banking_expert \
    --no-skip-completed
```

**Outputs per adapter:**
- `adapters/{name}/adapter_final.pt` — Final LoRA weights (no optimizer, smaller file)
- `adapters/{name}/training_metrics.json` — Loss curve, grad norms, timing
- `adapters/{name}/checkpoint_step_*.pt` — Resume checkpoints (auto-cleaned after completion)

**Crash Recovery:** If training crashes (OOM, power loss), simply re-run the same command. The script auto-detects the latest checkpoint and resumes from there.

### 4.3 RL Fine-Tuning (Step-by-Step)

RL training sharpens domain quality using REINFORCE with a rule-based reward function.

**Prerequisites:** SFT training must be complete (needs `adapter_final.pt` for each adapter).

**Reward Function** (in `demonstrator/training/reward_fn.py`):

| Component | Weight | What It Measures |
|-----------|--------|-----------------|
| Format | 0.4 | Proper structure, sentence length, coherence |
| Terminology | 0.3 | Correct use of domain-specific finance terms |
| Relevance | 0.2 | Answer stays on topic relative to the prompt |
| Safety | 0.1 | No harmful financial advice, disclaimers present |

**RLVR Config:**

| Parameter | Value |
|-----------|-------|
| Algorithm | REINFORCE |
| Learning rate | 1e-5 (10x lower than SFT) |
| RL steps | 500 |
| Rollout batch size | 1 |
| Max new tokens | 64 |
| Temperature | 0.7 |
| Entropy coefficient | 0.01 |
| KL coefficient | 0.01 |
| Baseline decay | 0.99 |
| Checkpoint interval | 100 steps |

**Step-by-step:**

```bash
# Step 1: Run RL on all adapters
python demonstrator/training/train_rl.py --adapter all

# Step 2: RL on a single adapter
python demonstrator/training/train_rl.py --adapter banking_expert

# Step 3: Custom settings
python demonstrator/training/train_rl.py \
    --adapter all \
    --sft-dir demonstrator/adapters \
    --output-dir demonstrator/adapters \
    --rl-steps 500 \
    --rank 32 \
    --alpha 64.0
```

**Outputs:**
- `adapters/{name}_rl/adapter_rl_final.pt` — RL-refined LoRA weights
- `adapters/{name}_rl/rl_metrics.json` — Reward curve, policy loss, entropy

### 4.4 MoE Assembly & TGSP Packaging

Assembles the trained adapters into the gated MoE configuration consumed by the inference engine.

**What it does:**
1. Picks best checkpoint per adapter (RL > SFT)
2. Extracts LoRA-only weights into PEFT directory format
3. Converts to TGSP signed packages via `LoRAToTGSPConverter`
4. Writes `moe_config.json` with expert routing, HE config, and GateLink config

**Step-by-step:**

```bash
# Step 1: Assemble (default paths)
python demonstrator/training/assemble_moe.py

# Step 2: Custom paths
python demonstrator/training/assemble_moe.py \
    --adapters-dir demonstrator/adapters \
    --output-dir demonstrator/adapters/tgsp
```

**Outputs:**
- `adapters/tgsp/banking_expert.tgsp` — Signed adapter package
- `adapters/tgsp/investment_expert.tgsp` — Signed adapter package
- `adapters/tgsp/shared_attention.tgsp` — Signed adapter package
- `adapters/tgsp/moe_config.json` — Full MoE configuration
- `adapters/tgsp/tgsp_results.json` — Conversion results + manifest hashes

### Full Pipeline Summary

```bash
# Complete training pipeline from scratch:
cd TenSafe_Project

# 1. SFT (trains all 3 adapters)
python demonstrator/training/train_sft.py --adapter all

# 2. RL (refines all 3 adapters)
python demonstrator/training/train_rl.py --adapter all

# 3. Assembly (packages into TGSP + writes moe_config.json)
python demonstrator/training/assemble_moe.py

# 4. Start inference server
uvicorn demonstrator.server.app:app --host 0.0.0.0 --port 8000
```

---

## 5. Expert Routing (Gated MoE)

### Gate Types

| Gate Type | Behavior | Used By |
|-----------|----------|---------|
| `step` | Keyword-match: if query contains any keyword, gate activates | `banking_expert`, `investment_expert` |
| `none` | Always active, no gating | `shared_attention` |

### Keyword Lists

**Banking Expert:**
`bank`, `deposit`, `loan`, `mortgage`, `credit`, `savings`, `checking`, `interest rate`, `refinance`

**Investment Expert:**
`invest`, `portfolio`, `stock`, `bond`, `etf`, `dividend`, `market`, `allocation`, `risk`

**Shared Attention:**
Always active (`always_active: true`). Handles general finance queries that do not match a specialist.

### How Routing Works

1. User query is lowercased
2. Each expert's keyword list is checked against the query
3. Expert with the most keyword matches wins
4. If no keywords match, `shared_attention` is used (always-active fallback)
5. The winning expert's LoRA weights are applied during inference

### Adding a New Expert

1. **Create adapter config** in `train_sft.py`:
   ```python
   ADAPTER_CONFIGS["insurance_expert"] = {
       "dataset_fn": load_insurance_dataset,
       "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
       "description": "Insurance domain expert",
   }
   ```

2. **Add target modules** in `train_rl.py`:
   ```python
   _TARGET_MODULES["insurance_expert"] = ["q_proj", "k_proj", "v_proj", "o_proj"]
   ```

3. **Train:** `python demonstrator/training/train_sft.py --adapter insurance_expert`

4. **RL refine:** `python demonstrator/training/train_rl.py --adapter insurance_expert`

5. **Add to `assemble_moe.py`** expert list and keyword config:
   ```python
   "insurance_expert": {
       "gate_type": "step",
       "gate_keywords": ["insurance", "premium", "coverage", "claim", "policy", "deductible"],
       "always_active": False,
   }
   ```

6. **Re-assemble:** `python demonstrator/training/assemble_moe.py`

---

## 6. GateLink-Split Inference

### 6.1 Architecture

GateLink-Split divides inference between client and server for privacy:

```
CLIENT (browser/device)                SERVER (FastAPI)
=======================                ================

1. Tokenize query                      (never sees raw text)
2. Embed tokens (local weights)
3. Add DP noise (Gaussian, eps=1.0)
4. Send noised hidden states -------->  5. Run 28 transformer layers
                                        6. Apply HE-encrypted LoRA deltas
                                        7. Return pre-LM-head states
8. Receive pre-activations <---------
9. LM head projection (local)          (never sees final tokens)
10. Top-p/top-k sampling (local)
11. Repeat from step 2 for next token
```

**Privacy guarantees:**
- Server never sees raw token IDs or final token choices
- Hidden states are DP-noised before transmission
- LoRA computations happen under CKKS encryption

### 6.2 Incremental Mode (Step-by-Step)

Without incremental mode, the client sends the FULL growing sequence every step, causing O(n^2) work. Incremental mode uses server-side KV caching:

**Step 0 (first token):**
1. Client embeds all input tokens, adds DP noise
2. Sends full sequence to server with `incremental=false`
3. Server runs all 28 layers, caches KV states, returns pre-activations

**Step 1+ (subsequent tokens):**
1. Client embeds only the NEW token, adds DP noise
2. Sends single token to server with `incremental=true`
3. Server retrieves cached KV, processes just the new token, updates cache
4. Returns only the new token's pre-activations

**KV Cache Store Properties:**
- `OrderedDict`-based with LRU eviction
- Max 32 concurrent sessions
- 300-second TTL (auto-expires idle sessions)
- Thread-safe (uses `threading.Lock`)

### 6.3 Device Profiles

| Profile | DP Epsilon (eps) | Client Layers (K) | Memory Budget | Use Case |
|---------|-----------------|-------------------|---------------|----------|
| **phone** | 1.0 | 1 | 1.5 GB | Mobile deployment (default) |
| **laptop** | 4.0 | 1 | 4 GB | Browser-based inference |
| **workstation** | 8.0 | 2 | 8 GB | Desktop GPU inference |
| **server** | 16.0 | 4 | 16+ GB | Data center deployment |

Lower epsilon = stronger privacy guarantee but more noise. The phone profile (eps=1.0) provides the strongest privacy.

**Configuring the profile:** Set in `moe_config.json` under `gatelink_config`:
```json
{
  "gatelink_config": {
    "device_profile": "phone",
    "client_layers": 1,
    "dp_epsilon": 1.0,
    "max_epsilon": 10.0,
    "max_lora_rank": 32
  }
}
```

### 6.4 Client-Side Usage

**JavaScript API:**

```javascript
// Initialize (downloads 446 MB weight matrix, cached in IndexedDB)
const client = new SplitInferenceClient();
await client.initialize((stage, progress, msg) => {
    console.log(`[${stage}] ${(progress * 100).toFixed(0)}%: ${msg}`);
});

// Generate text
const result = await client.generate(
    "What are the best investment strategies for retirement?",
    {
        maxTokens: 128,
        temperature: 0.7,
        topP: 0.9,
        topK: 50,
        useHE: true,
    },
    (tokenText, metrics) => {
        // Called per generated token
        process.stdout.write(tokenText);
    },
    (stage, detail) => {
        // Status updates: "tokenize", "route", "embed", "server", "project", "done"
    }
);

console.log(`Generated ${result.totalTokens} tokens in ${result.totalTimeMs}ms`);
```

**Bootstrap endpoint:** The client fetches its configuration from `GET /api/v1/split/config`, which returns hidden_dim, vocab_size, DP parameters, expert keywords, and HE status.

---

## 7. Homomorphic Encryption (CKKS)

### Scheme Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Scheme | CKKS | Approximate arithmetic on encrypted data |
| `poly_modulus_degree` | 16,384 | Ring dimension (security + capacity) |
| `coeff_modulus_bits` | [60, 40, 40, 60] | Modulus chain for leveled operations |
| `scale_bits` | 40 | Precision: 2^40 scale |
| `simd_slots` | 8,192 | Vectors packed per ciphertext (n/2) |
| Column packing | Enabled | ZeRo-MOAI strategy (zero rotations) |

### How HE-LoRA Works

For each token, the LoRA delta is computed under encryption:

1. **Extract** hidden state h from transformer output (1536-dim)
2. **Add DP noise** (Gaussian mechanism calibrated to epsilon)
3. **Encrypt** h into CKKS ciphertext (8192 SIMD slots, 1536 used)
4. **ct-pt matmul** for LoRA A: `ct_h * packed_A` (zero rotations, column-packed)
5. **Decrypt** intermediate result to plaintext
6. **Plaintext matmul** for LoRA B: `intermediate * B` (no encryption needed)
7. **Add** delta to hidden state: `h' = h + delta`

### 3-Tier CKKS Fallback

The engine tries three backends in order:

| Tier | Backend | Speed | Encryption |
|------|---------|-------|-----------|
| 1 | **CuKKS** (GPU, OpenFHE) | Fastest | Real CKKS |
| 2 | **Pyfhel** (CPU) | Medium | Real CKKS |
| 3 | **Pure-Python emulator** | Slowest | Emulated (same math, no encryption) |

The system automatically falls back if a backend is unavailable. All three produce identical LoRA delta values.

### When HE Adds Value vs Overhead

**Use HE (default):**
- Production deployments with untrusted servers
- When demonstrating privacy-preserving inference
- GateLink-Split mode (server processes encrypted LoRA)

**Disable HE (for development):**
- Local development/debugging (set `use_he: false` in requests)
- Benchmarking base model quality without HE overhead
- When running on CPU without Pyfhel installed (emulator adds latency)

---

## 8. Security & Privacy

### 8.1 TGSP Package Security

**TGSP** (TensorGuard Secure Package) is the signed adapter format:

- **Magic bytes:** `TGSP\x01\x00` (identifies format + version)
- **Dual signatures:** Ed25519 (classical) + Dilithium3 (post-quantum)
- **Manifest hash:** SHA-256 over all adapter weights
- **Verification:** `LoRAToTGSPConverter` validates on conversion; `TGSPAdapterRegistry` validates on load

**RVUv2 Safety Screening** (three layers):

| Layer | Check | Purpose |
|-------|-------|---------|
| 1 | Allowlist | Only known adapter IDs pass |
| 2 | Safe LoRA SVD | Singular value decomposition check — rejects weights that deviate too far from base model |
| 3 | Mahalanobis OOD | Out-of-distribution detection — catches poisoned adapters |

### 8.2 Differential Privacy

**Gaussian Mechanism:**
```
sigma = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon
```

Where:
- `sensitivity` = 1.0 (L2 norm of hidden state, clipped)
- `delta` = 1e-5
- `epsilon` = per-request privacy budget (1.0 for phone profile)

**Advanced Composition Theorem:** Multi-query budget tracking. After k queries with per-query epsilon, the total privacy loss is bounded by:
```
total_epsilon <= sqrt(2k * ln(1/delta)) * epsilon + k * epsilon * (e^epsilon - 1)
```

**Per-Session Tracking:**
- Each `session_id` has its own privacy budget
- Budget is tracked via `PrivacyBudgetTracker`
- When `max_epsilon` (default 10.0) is exhausted, further queries are rejected

**Reset endpoint:** `POST /api/v1/privacy/reset` clears all session budgets (demo use only).

### 8.3 Privacy Best Practices

1. **Never set epsilon > 16 in production** — Higher epsilon means weaker privacy
2. **Monitor budget** via `GET /api/v1/metrics` (check `dp_budget_remaining`)
3. **Use session isolation** — Each user should have a unique `session_id`
4. **Phone profile for maximum privacy** — epsilon=1.0 is the strongest setting
5. **DP noise is added ONCE per request** — The `track_budget` parameter ensures budget is not double-counted (fixed in the recent audit)

---

## 9. Deployment & Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MOE_CONFIG_PATH` | `demonstrator/adapters/tgsp/moe_config.json` | Path to MoE configuration |
| `DEVICE` | `cuda` (if available) | PyTorch device (`cuda`, `cpu`) |
| `TG_ENVIRONMENT` | — | Set to `production` to disable `/docs` Swagger UI |

### Starting the Server

```bash
# Development
uvicorn demonstrator.server.app:app --host 0.0.0.0 --port 8000 --reload

# Production
uvicorn demonstrator.server.app:app --host 0.0.0.0 --port 8000 --workers 1
```

> **Note:** Use `--workers 1` because the inference engine holds GPU state and is not fork-safe.

### API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check (`{"status": "ok", "engine_ready": true}`) |
| `/api/v1/chat/stream` | WebSocket | Streaming chat with per-token HE metrics |
| `/api/v1/chat/compare` | POST | Base model vs LoRA-adapted comparison |
| `/api/v1/split/forward` | POST | GateLink-Split server-side forward pass |
| `/api/v1/split/config` | GET | Split inference client configuration |
| `/api/v1/metrics` | GET | Live system metrics (HE, DP, adapters, GPU) |
| `/api/v1/privacy/reset` | POST | Reset DP privacy budget (demo only) |
| `/` | GET | Static frontend (served from `demonstrator/frontend/`) |

### WebSocket Chat Request Format

```json
{
  "query": "What is a good savings strategy?",
  "max_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "use_he": true,
  "session_id": "user_123"
}
```

### Split Forward Request Format

```json
{
  "hidden_states_b64": "<base64-encoded float32 array>",
  "seq_len": 10,
  "hidden_dim": 1536,
  "expert_name": "banking_expert",
  "use_he": true,
  "session_id": "split_abc123",
  "incremental": false
}
```

### Frontend

The frontend is a static SPA served from `demonstrator/frontend/`. It includes:
- **WebSocket streaming chat** with real-time token-by-token display
- **HE metrics dashboard** showing encryption/decryption/compute timing per token
- **Pipeline visualization** with stage-by-stage timing
- **Expert routing indicators** showing which adapter is active
- **Comparison mode** (base model vs LoRA-adapted side-by-side)
- **GateLink-Split mode** toggle (uses `split_client.js` for client-side inference)

---

## 10. QA & Verification

### Test Suites

| Suite | Tests | What It Covers |
|-------|-------|---------------|
| `qa_verify.py` | 34 | Engine init, CKKS HE, DP noise, adapter loading, streaming, split forward, metrics, privacy budget |
| `regression_gatelink.py` | 21 | GateLink-specific: incremental mode, KV cache, session isolation, DP budget per-session, payload validation |

### Running Tests

```bash
cd C:\Users\lover\Downloads\TenSafe_Project

# Run both suites
python demonstrator/scripts/qa_verify.py
python demonstrator/scripts/regression_gatelink.py

# Run both in sequence
python demonstrator/scripts/qa_verify.py && python demonstrator/scripts/regression_gatelink.py
```

**Expected output:** `34/34 PASSED` and `21/21 PASSED` (55 total).

### Throughput Benchmark

```bash
python demonstrator/scripts/benchmark_throughput.py
```

This benchmarks:
- Tokens per second for WebSocket streaming mode
- Tokens per second for GateLink-Split mode (with and without incremental)
- HE overhead per token
- DP noise injection time

### Adding Custom Tests

Follow the pattern in `qa_verify.py`:

```python
def test_my_feature():
    """Description of what this tests."""
    engine = FinanceInferenceEngine(moe_config_path=CONFIG_PATH, device="cpu")
    engine.initialize()

    # Your test logic
    result = engine.some_method(...)
    assert result is not None, "Expected non-None result"
    assert result["key"] == expected_value, f"Got {result['key']}"

    return True  # Test passed
```

---

## 11. Performance Tuning

### LoRA Rank Tradeoffs

| Rank | Parameters | Quality | Memory | HE Cost |
|------|-----------|---------|--------|---------|
| 8 | ~200K | Lower | Minimal | Fastest |
| 16 | ~400K | Good | Low | Fast |
| **32** | **~800K** | **Best (recommended)** | **Medium** | **Medium** |
| 64 | ~1.6M | Marginal gain | High | Slow |

Rank 32 is the sweet spot: fits within a single CKKS ciphertext operation while providing good adaptation quality.

### Incremental Mode Gains

| Mode | Time per Token (step 50) | Scaling |
|------|-------------------------|---------|
| Full recompute | ~500ms (grows with sequence) | O(n) per token, O(n^2) total |
| **Incremental** | **~50ms (constant)** | **O(1) per token, O(n) total** |

Always use incremental mode for GateLink-Split generation longer than a few tokens.

### HE Overhead

| Operation | CuKKS (GPU) | Pyfhel (CPU) | Emulator |
|-----------|-------------|-------------|----------|
| Encrypt (per token) | ~2ms | ~8ms | <0.1ms |
| ct-pt Matmul | ~1ms | ~5ms | <0.1ms |
| Decrypt | ~1ms | ~4ms | <0.1ms |

**To disable HE for development:** Set `use_he: false` in your WebSocket request or split forward request.

### GPU vs CPU

| Configuration | tok/s (streaming) | Recommended For |
|--------------|-------------------|-----------------|
| CUDA + CuKKS | 15-25 tok/s | Production |
| CUDA + Emulator | 20-30 tok/s | Dev/debugging |
| CPU + Pyfhel | 2-5 tok/s | CI/testing |
| CPU + Emulator | 3-8 tok/s | Local dev |

### Memory Optimization

1. **Gradient checkpointing** during training saves ~60% activation VRAM
2. **float16 inference** — The model loads in half precision by default
3. **LoRA-only storage** — Final checkpoints strip the optimizer (much smaller files)
4. **KV cache TTL** — Idle sessions auto-expire after 300 seconds
5. **GPU cleanup** — Call `torch.cuda.empty_cache()` between adapter trainings

### Monitoring

The `/api/v1/metrics` endpoint returns:

```json
{
  "engine_ready": true,
  "model": "Qwen/Qwen2.5-1.5B",
  "he_active": true,
  "simd_slots": 8192,
  "adapters": ["banking_expert", "investment_expert", "shared_attention"],
  "gpu": {
    "name": "NVIDIA RTX 4090",
    "mem_used_mb": 2048.5,
    "mem_total_mb": 24564.0
  },
  "differential_privacy": {
    "dp_epsilon_per_request": 1.0,
    "dp_sigma": 4.6852,
    "dp_total_epsilon_spent": 3.1623,
    "dp_total_requests": 10,
    "dp_max_epsilon": 10.0,
    "dp_budget_remaining": 6.8377
  }
}
```

---

## 12. FAQ / Troubleshooting

### General

**Q: What is TenSafe?**
A: TenSafe is a privacy-preserving ML framework that combines homomorphic encryption (CKKS), differential privacy, and LoRA adapters to enable secure inference on sensitive financial data. The demonstrator showcases this with a Qwen 2.5-1.5B model fine-tuned for finance.

**Q: What models are supported?**
A: Any HuggingFace causal LM that supports LoRA. The demonstrator is built around Qwen 2.5-1.5B, but the training pipeline and inference engine work with any model by changing `model_name_or_path`. The CKKS SIMD slot alignment works best when hidden_dim <= 8192.

**Q: Can I use this with non-finance domains?**
A: Yes. Replace the dataset loaders, reward function, and expert keywords. The HE/DP/TGSP infrastructure is domain-agnostic.

### Training

**Q: How long does SFT training take?**
A: With a single GPU (e.g., RTX 4090), each adapter takes roughly 30-60 minutes for 2000 steps. All three adapters sequentially take 2-3 hours.

**Q: Can I add a new domain expert?**
A: Yes. See [Section 5: Adding a New Expert](#adding-a-new-expert). You need: a dataset loader, adapter config in train_sft.py, target modules in train_rl.py, keywords in assemble_moe.py, then re-train and re-assemble.

**Q: What if training crashes mid-way?**
A: Simply re-run the same command. The script detects the latest checkpoint (saved every 250 optim steps for SFT, every 100 for RL) and resumes automatically. Emergency checkpoints are also saved on OOM.

**Q: Should I use RL training?**
A: RL training is optional but recommended. It improves domain-specific quality (format, terminology, relevance) by 10-30% based on the reward function. SFT alone produces usable adapters.

### Inference

**Q: Why is GateLink-Split mode slower than streaming?**
A: Split mode has additional overhead: client-side embedding (using float16 lookup tables), base64 encoding/decoding, HTTP round-trips per token, and client-side LM head projection (151,936 vocab x 1536 hidden dim). Incremental mode significantly reduces the per-token cost after the first step.

**Q: What happens when the DP privacy budget is exhausted?**
A: The privacy tracker rejects further queries for that session. In the demo, you can reset it via `POST /api/v1/privacy/reset`. In production, users would need to start a new session or wait for administrative reset.

**Q: What does `use_he: false` do?**
A: It skips the CKKS encryption/decryption steps. The LoRA delta is still computed (the math is identical), but without the encryption wrapper. Useful for development and benchmarking base model quality.

**Q: The split client says "Downloading weights (446 MB)"**
A: The client needs the tied embedding/LM-head weight matrix (151,936 tokens x 1536 dims x 2 bytes float16 = 446 MB). This is downloaded once and cached in IndexedDB. Subsequent page loads use the cache.

### Security

**Q: Is the pure-Python CKKS emulator safe for production?**
A: **No.** The emulator performs the same mathematical operations but does not actually encrypt data. It exists for development/testing when CuKKS or Pyfhel are unavailable. Production deployments must use CuKKS or Pyfhel.

**Q: How do I rotate TGSP signing keys?**
A: Regenerate keys via `LoRAToTGSPConverter(auto_generate_keys=True)` and re-run `assemble_moe.py`. The converter generates new Ed25519 + Dilithium3 key pairs. Existing .tgsp files signed with old keys will fail validation.

**Q: What is RVUv2?**
A: Rapid Verification Unit v2. It screens adapter packages through three layers: (1) allowlist check, (2) Safe LoRA SVD analysis (rejects weights that deviate excessively from the base model), and (3) Mahalanobis out-of-distribution detection (catches poisoned adapters).

### Troubleshooting

**Q: Garbled output from split mode (e.g., random characters)**
A: This was caused by missing KV cache — each call reprocessed from scratch with different DP noise, causing position-dependent corruption. **Fixed** by adding incremental mode with server-side KV caching. Ensure you are using the latest code with `_KVCacheStore`.

**Q: `ImportError: No module named 'tensafe_platform.crypto.cukks_backend'`**
A: CuKKS requires the C++ OpenFHE extension to be compiled. The engine automatically falls back to Pyfhel, then to the pure-Python emulator. This is normal in development environments.

**Q: CUDA Out of Memory during training**
A: Try these steps in order:
1. Reduce `--batch-size` to 1 (if not already)
2. Ensure gradient checkpointing is enabled (automatic in train_sft.py)
3. Reduce `--max-seq-length` from 512 to 256
4. Reduce `--rank` from 32 to 16
5. Use `--grad-accum 16` to maintain effective batch size with smaller micro-batches

**Q: Privacy budget shows incorrect epsilon after generation**
A: This was a double-counting bug where `_add_dp_noise()` consumed budget on every token AND once per request. **Fixed** by passing `track_budget=False` for per-token calls, with budget consumed only once per request.

**Q: `epsilon_spent` shows as "?" in the dashboard**
A: This was a JavaScript falsy-zero bug where `0.0 || "?"` evaluated to `"?"`. **Fixed** by using the nullish coalescing operator (`??`) instead of logical OR (`||`).

**Q: How do I check if the server is healthy?**
A: `curl http://localhost:8000/health` should return `{"status": "ok", "engine_ready": true}`.

---

*This document reflects the state after the 17-fix audit. All 55 tests (34 QA + 21 regression) pass.*
