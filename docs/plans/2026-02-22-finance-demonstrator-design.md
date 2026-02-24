# TenSafe Finance Demonstrator Design

## Overview

End-to-end demonstrator showcasing TenSafe's privacy-preserving ML platform:
- Qwen 1.5B base model with 3 LoRA adapters (2 gated experts + 1 shared attention)
- Trained via SFT + REINFORCE RL on HuggingFace finance datasets
- Served via GateLink-Split protocol with full CKKS homomorphic encryption
- iPhone 15 Pro web app interface with live metrics, encryption visualization, and comparison mode

## Hardware

- **GPU:** NVIDIA A2000 (6GB VRAM) in WSL Docker
- **HE Compute:** CPU (Pyfhel/SEAL) — does not consume GPU VRAM
- **Client:** iPhone 15 Pro via Safari over local WiFi
- **Constraint:** Training and inference are sequential (not concurrent)

## Architecture

```
WSL Docker Container (port 8000)
├── FastAPI Server
│   ├── /api/v1/tg-tinker/*    — Training API (TG Tinker)
│   ├── /api/v1/chat/stream    — WebSocket streaming chat
│   ├── /api/v1/chat/compare   — Base vs LoRA comparison
│   ├── /api/v1/metrics        — Live system metrics
│   └── /                      — Static web frontend
├── Training Mode
│   ├── TenSafeOrchestrator (Qwen 1.5B + LoRA rank 32)
│   ├── RLVRTrainer (REINFORCE algorithm)
│   └── TGSP Converter (adapter packaging)
└── Inference Mode
    ├── GateLink-Split Server (K=1 for phone profile)
    ├── UnifiedHEBackend (CKKS, N=16384, 8192 SIMD slots)
    ├── TGSP Adapter Registry (3 encrypted adapters)
    └── Gated MoE Router (step-function gates)
```

## Training Pipeline

### Phase 1: SFT (Supervised Fine-Tuning)

Three LoRA adapters, all rank 32, alpha 64.0:

| Adapter | Modules | Dataset | Samples |
|---------|---------|---------|---------|
| Expert 1 (Banking) | q,k,v,o_proj | sujet-ai/Sujet-Finance-Instruct-177k (banking filter) | ~30k |
| Expert 2 (Investment) | q,k,v,o_proj | FiQA + financial_phrasebank (investment filter) | ~20k |
| Shared Attention | q,k,v,o_proj, gate_proj, up_proj | Combined full finance dataset | ~50k |

Training config per adapter:
- Batch size: 1, gradient accumulation: 8 (effective batch 8)
- Gradient checkpointing: enabled
- Optimizer: AdamW, lr=1e-4, weight_decay=0.01
- Max steps: 2000 per adapter
- DP: enabled, noise_multiplier=1.0, target_epsilon=8.0
- Precision: float16

### Phase 2: RL (REINFORCE)

Rule-based reward function:
```
R = 0.4 * format_score + 0.3 * terminology_score + 0.2 * relevance_score + 0.1 * safety_score
```

- format_score: Structured response, no hallucinated numbers
- terminology_score: Correct financial terms (from curated glossary)
- relevance_score: Answer addresses the question topic
- safety_score: Includes appropriate disclaimers for investment advice

Config:
- Algorithm: REINFORCE with baseline
- Steps: 500 per adapter
- Temperature: 0.7, KL coefficient: 0.01
- Max new tokens: 256

### Phase 3: Gated MoE Assembly

- Gate type: step function `g(x) = step(w_g^T x + b_g)`
- Expert 1 gate: trained to activate on banking queries
- Expert 2 gate: trained to activate on investment queries
- Shared attention LoRA: always active (no gate, additive)

### Phase 4: TGSP Packaging

Each adapter converted to TGSP format:
- Ed25519 + Dilithium3 hybrid signatures
- Kyber768 + ChaCha20Poly1305 encryption
- SHA256 manifest integrity
- Auto-generated keys for demo

## HE Configuration

- Scheme: CKKS
- poly_modulus_degree: 16384 (8192 SIMD slots)
- coeff_modulus_bits: [60, 40, 40, 60]
- scale: 2^40
- ZeRo-MOAI column packing: all 8192 slots utilized
- Rotations: 0 (zero-rotation guarantee)
- GPU batching: batch all layer projections per token

## GateLink-Split Protocol (iPhone Profile)

- Client layers (K): 1
- DP epsilon: 1.0
- Max LoRA rank: 32 (overridden from default 4 for demo)
- RAM budget: ~1.5 GB (Safari tab)
- Expected throughput: variable (A2000 + CPU HE)

Flow per token:
1. Server computes base model layers 1..N
2. Server computes encrypted LoRA deltas via CKKS
3. Server sends encrypted gate pre-activation (~32 KB) to client
4. Client decrypts, evaluates gate, returns bit
5. Server applies gated delta
6. Client receives pre-activations, runs LM head locally
7. Client samples next token

## Web App Interface

Mobile-first layout for iPhone 15 Pro Safari:

### Chat Panel
- Streaming token-by-token via WebSocket
- Each message shows encryption badge:
  - Input: "Encrypted | CKKS | 8192 slots | XX KB ciphertext"
  - Output: "Decrypted | Expert: Banking/Investment | Gate: 0/1"

### Metrics Panel (always visible)
- tok/s (tokens per second)
- Per-token latency (ms)
- HE operations count
- Rotations count (should be 0)
- Encrypt / Compute / Decrypt / Network time breakdown

### Compare Mode
- Toggle to run same query twice: base model vs LoRA-adapted
- Side-by-side response display
- Quality difference visible (generic vs domain-specific)

### Expert Routing Display
- Shows which expert gate fired
- Gate confidence value
- Shared attention LoRA contribution

### Encryption Pipeline Visualization
- Animated flow: Plaintext -> CKKS Encrypt -> HE Compute -> Decrypt -> Response
- Real ciphertext sizes and operation timing

### Settings (slide-out)
- Temperature, top_p, top_k
- DP epsilon remaining
- Device profile info
- Toggle HE on/off for latency comparison

## Data Flow

```
iPhone Safari                    WSL Docker (FastAPI)
─────────────                    ───────────────────
User types query
     │
     ├──WebSocket──────────────> Receive plaintext query
     │                           │
     │                           ├─ Tokenize + embed
     │                           ├─ CKKS encrypt activations (CPU)
     │                           │  (8192 SIMD slots, max packed)
     │                           ├─ Compute HE-LoRA deltas (CPU)
     │                           │  (zero-rotation column-packed matmul)
     │                           ├─ Evaluate gates (per-expert)
     │                           ├─ Apply gated deltas
     │                           ├─ CKKS decrypt result (CPU)
     │                           ├─ LM head → logits (GPU)
     │                           ├─ Sample next token
     │                           ├─ Collect metrics
     │                           │
     │<──WebSocket token+metrics─┤ Stream token + per-token metrics
     │                           │
     │  (repeat until EOS)       │
     │                           │
     │<──WebSocket final─────────┤ Final metrics summary
```

## File Structure (New Files)

```
TenSafe_Project/
├── demonstrator/
│   ├── Dockerfile                    # Docker build
│   ├── docker-compose.yml            # WSL deployment
│   ├── requirements.txt              # Python deps
│   ├── server/
│   │   ├── app.py                    # FastAPI main
│   │   ├── chat.py                   # WebSocket chat endpoint
│   │   ├── compare.py                # Base vs LoRA comparison
│   │   ├── metrics.py                # Live metrics collection
│   │   └── inference_engine.py       # GateLink-Split + HE inference
│   ├── training/
│   │   ├── train_sft.py              # SFT training script
│   │   ├── train_rl.py               # REINFORCE RL training
│   │   ├── reward_fn.py              # Finance reward function
│   │   ├── datasets.py               # HuggingFace dataset loading
│   │   └── assemble_moe.py           # Gated MoE assembly + TGSP
│   ├── frontend/
│   │   ├── index.html                # Main page
│   │   ├── app.js                    # Chat + metrics logic
│   │   └── styles.css                # Mobile-first styles
│   └── scripts/
│       ├── train.sh                  # Full training pipeline
│       └── serve.sh                  # Start inference server
```
