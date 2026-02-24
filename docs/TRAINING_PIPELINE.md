# Training Pipeline: SFT + REINFORCE + MoE Assembly

## Overview

TenSafe's training pipeline produces three domain-specialized LoRA adapters through a three-stage process:

1. **Stage 1: Supervised Fine-Tuning (SFT)** -- Train each adapter on domain-specific financial data
2. **Stage 2: Reinforcement Learning (REINFORCE)** -- Refine with a 4-axis domain reward function
3. **Stage 3: MoE Assembly** -- Package into TGSP format and generate MoE config

All stages are crash-resilient with checkpoint-resume support and GPU memory management.

---

## Stage 1: Supervised Fine-Tuning

### Configuration

```python
# Per-adapter training configuration
Optimizer:      AdamW (weight_decay=0.01, max_grad_norm=1.0)
Learning rate:  1e-4
Batch size:     1 (gradient accumulation = 8, effective batch = 8)
Max steps:      2000 optimization steps
Max seq len:    512 tokens
LoRA rank:      32
LoRA alpha:     64 (scaling factor = alpha/rank = 2.0)
DP:             noise_multiplier=1.0, target_epsilon=8.0, delta=1e-5
Checkpoints:    Every 250 optim steps, keep 2 most recent
```

### Three Expert Adapters

| Expert | Target Modules | Training Data | Description |
|--------|---------------|---------------|-------------|
| `banking_expert` | q, k, v, o_proj | Banking domain Q&A | Loans, deposits, regulations |
| `investment_expert` | q, k, v, o_proj | Investment domain Q&A | Portfolio, markets, analysis |
| `shared_attention` | q, k, v, o_proj, gate_proj, up_proj | Combined finance | General financial reasoning |

Note: `shared_attention` targets 6 modules (including MLP gate/up projections) vs 4 for domain experts. This gives it a broader adaptation surface for fallback routing.

### Crash Resilience

The training loop implements multiple layers of fault tolerance:

**1. Skip-completed detection:**
```python
def _is_adapter_complete(adapter_dir: Path) -> bool:
    final_pt = adapter_dir / "adapter_final.pt"
    metrics_json = adapter_dir / "training_metrics.json"
    return final_pt.exists() and metrics_json.exists()
```

**2. Checkpoint resume:**
```python
def _find_latest_checkpoint(adapter_dir: Path) -> Path | None:
    checkpoints = sorted(
        adapter_dir.glob("checkpoint_step_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    return checkpoints[-1] if checkpoints else None
```

**3. Atomic writes (crash-safe):**
```python
# Write to temp file first, then rename
tmp_path = path.with_suffix(".pt.tmp")
with open(tmp_path, "wb") as f:
    f.write(state)
tmp_path.rename(path)  # Atomic on POSIX
```

**4. OOM emergency save:**
```python
except torch.cuda.OutOfMemoryError:
    logger.error("CUDA OOM! Saving emergency checkpoint...")
    _gpu_cleanup()
    _save_resume_checkpoint(orchestrator, adapter_dir, ...)
    raise
```

**5. Checkpoint pruning (keep last 2):**
```python
all_ckpts = sorted(adapter_dir.glob("checkpoint_step_*.pt"), ...)
for old_ckpt in all_ckpts[:-2]:
    old_ckpt.unlink(missing_ok=True)
```

### GPU Memory Management

```python
def _gpu_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
```

Called:
- Before model load (clean slate)
- After each checkpoint save (defragment)
- After each adapter completes (free for next adapter)
- On OOM (emergency free before save)

Gradient checkpointing saves ~60% activation VRAM:
```python
backend._model.gradient_checkpointing_enable()
```

### Training Loop

```python
for epoch in range(100):  # up to 100 epochs, early-stop at max_steps
    for batch in dataloader:
        batch = {k: v.to("cuda") for k, v in batch.items()}

        # Forward + backward (through TenSafeOrchestrator)
        fb_metrics = orchestrator.forward_backward(batch, sample_rate)

        if (global_step + 1) % grad_accum == 0:
            # Optimizer step with DP noise
            opt_metrics = orchestrator.optim_step(True, sample_rate)
            optim_steps += 1

            # Log every 25 optim steps
            if optim_steps % 25 == 0:
                logger.info(f"step={global_step} loss={fb_metrics.loss:.4f}")

            # Checkpoint every 250 optim steps
            if optim_steps % CHECKPOINT_INTERVAL == 0:
                _save_resume_checkpoint(...)

        global_step += 1
```

---

## Stage 2: REINFORCE RL

### Configuration

```python
Algorithm:      REINFORCE (policy gradient)
Learning rate:  1e-5 (10x lower than SFT)
Rollout batch:  1
Max new tokens: 64
Temperature:    0.7
Top-p:          0.9
Reward scale:   1.0
Baseline:       Exponential moving average (decay=0.99)
Entropy coeff:  0.01
KL coeff:       0.01
Total steps:    500
Eval interval:  50
Save interval:  100
DP:             noise_multiplier=1.0, target_epsilon=8.0
```

### Process

1. Load SFT-trained adapter (`adapter_final.pt`)
2. Initialize `RLVRTrainer` with `finance_reward` function
3. For 500 steps: sample prompt → generate response → score → update
4. Save final RL adapter (`adapter_rl_final.pt`)

```python
for step in range(rl_steps):
    batch = prompts[pidx : pidx + rollout_batch_size]
    m = trainer.step(batch)  # generate → reward → policy gradient

    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # aggressive memory cleanup

    if step % CHECKPOINT_INTERVAL == 0:
        trainer.save_checkpoint(...)
```

---

## Reward Function: 4-Axis Financial Scoring

The reward function (`reward_fn.py`) returns a scalar in [0, 1]:

### Axis Weights

| Axis | Weight | Purpose |
|------|--------|---------|
| **Format** | 0.4 | Structural quality of response |
| **Terminology** | 0.3 | Domain-specific financial terms |
| **Relevance** | 0.2 | Addresses the prompt |
| **Safety** | 0.1 | Disclaimers for investment advice |

### Format Scoring (40%)

```python
def _score_format(response):
    score = 0.0
    wc = len(response.split())

    # Length: 20-500 words (0.3)
    if 20 <= wc <= 500: score += 0.3
    elif 10 <= wc < 20: score += 0.1

    # Structure: bullets, numbers, newlines (0.3)
    if "\n" in response or "- " in response or "1." in response:
        score += 0.3

    # No hallucinated decimals (0.2)
    if not re.findall(r"\d{1,2}\.\d{3,}%", response):
        score += 0.2

    # Proper ending punctuation (0.2)
    if response.strip()[-1] in ".!?)\"'":
        score += 0.2

    return min(1.0, score)
```

### Terminology Scoring (30%)

```python
ALL_FINANCE_TERMS = BANKING_TERMS | INVESTMENT_TERMS  # 60+ terms

def _score_terminology(response):
    n = sum(1 for t in ALL_FINANCE_TERMS if t in response.lower())
    if n >= 5: return 1.0
    if n >= 3: return 0.7
    if n >= 1: return 0.4
    return 0.0
```

Financial glossary includes:
- **Banking**: deposit, withdrawal, APR, APY, FDIC, mortgage, amortization, collateral, escrow, etc.
- **Investment**: portfolio, diversification, ETF, dividend, Sharpe ratio, beta, DCF, EPS, etc.

### Relevance Scoring (20%)

```python
def _score_relevance(prompt, response):
    pw = set(re.findall(r"\b[a-z]{4,}\b", prompt.lower()))
    rw = set(re.findall(r"\b[a-z]{4,}\b", response.lower()))
    return min(1.0, len(pw & rw) / len(pw) * 1.5)
```

Measures word overlap between prompt and response (4+ character words), with 1.5x scaling to reward partial coverage.

### Safety Scoring (10%)

```python
def _score_safety(prompt, response):
    needs = any(kw in prompt.lower() for kw in
                ["invest", "buy", "sell", "portfolio", "stock", "recommend"])
    if not needs: return 1.0  # not investment advice, no disclaimer needed

    for rx in _DISCLAIMER_RES:  # "not financial advice", "consult professional", etc.
        if rx.search(response): return 1.0
    return 0.2  # penalty for missing disclaimer on investment queries
```

---

## Stage 3: MoE Assembly

### Process

1. **Select best checkpoint**: RL final > SFT final for each adapter
2. **Extract LoRA weights**: Parse full orchestrator state, extract only `lora_A` and `lora_B` tensors
3. **Create PEFT directory**: Standard PEFT format (`adapter_config.json` + `adapter_model.bin`)
4. **Convert to TGSP**: Cryptographically signed package via `LoRAToTGSPConverter`
5. **Generate MoE config**: JSON consumed by the inference engine

### LoRA Extraction

```python
def _extract_lora_to_peft_dir(full_state_path, peft_dir):
    state = torch.load(full_state_path, map_location="cpu")
    model_state = state["model_state_dict"]

    lora_state = {}
    for key, tensor in model_state.items():
        if "lora_" in key.lower():
            lora_state[key] = tensor

    # Save as PEFT directory
    torch.save(lora_state, peft_dir / "adapter_model.bin")
    # + adapter_config.json with rank, alpha, target_modules
```

### TGSP Conversion

```python
converter = LoRAToTGSPConverter(auto_generate_keys=True)
result = converter.convert(
    input_path=str(peft_dir),
    output_path=str(tgsp_out),
    model_name=f"tensafe-finance-{name}",
    model_version="1.0.0",
    validate=True,
    metadata={"domain": "finance", "rank": 32, ...},
)
```

TGSP (TenSafe Guard-Signed Package) includes:
- Compressed LoRA weights
- Cryptographic signatures (integrity verification)
- Metadata (rank, alpha, target modules, training config)
- Manifest hash for tamper detection

### Generated MoE Config

```json
{
  "model": "Qwen/Qwen2.5-1.5B",
  "experts": {
    "banking_expert": {
      "tgsp_path": "...",
      "checkpoint_path": "...",
      "gate_type": "step",
      "gate_keywords": ["bank", "deposit", "loan", "mortgage", ...],
      "always_active": false
    },
    "investment_expert": {
      "gate_keywords": ["invest", "portfolio", "stock", "bond", ...],
      "always_active": false
    },
    "shared_attention": {
      "gate_type": "none",
      "always_active": true
    }
  },
  "he_config": {
    "scheme": "ckks",
    "poly_modulus_degree": 16384,
    "coeff_modulus_bits": [60, 40, 40, 60],
    "scale_bits": 40,
    "use_column_packing": true
  },
  "gatelink_config": {
    "dp_epsilon": 1.0,
    "max_epsilon": 10.0,
    "max_lora_rank": 32
  }
}
```

---

## Expert Routing (Keyword Step-Gate)

At inference time, queries are routed to the appropriate expert:

```python
def route_expert(self, query):
    q = query.lower()
    best, best_score = "shared_attention", 0

    for name, adp in self.adapters.items():
        if adp["always_active"]:
            continue
        score = sum(1 for kw in adp["gate_keywords"] if kw in q)
        if score > best_score:
            best, best_score = name, score

    return best  # falls back to shared_attention if no keywords match
```

Routing is deterministic: count keyword matches, select highest-scoring expert. `shared_attention` is always the fallback.

---

## CLI Usage

### Training All Adapters

```bash
# Stage 1: SFT
python -m demonstrator.training.train_sft \
  --adapter all \
  --max-steps 2000 \
  --rank 32 \
  --alpha 64.0

# Stage 2: RL
python -m demonstrator.training.train_rl \
  --adapter all \
  --rl-steps 500

# Stage 3: Assembly
python -m demonstrator.training.assemble_moe \
  --adapters-dir demonstrator/adapters \
  --output-dir demonstrator/adapters/tgsp
```

### Training Individual Adapter

```bash
python -m demonstrator.training.train_sft --adapter banking_expert
python -m demonstrator.training.train_rl --adapter banking_expert
```

### Resume After Crash

Simply re-run the same command. The training loop automatically:
1. Checks for completed adapters (skips if `adapter_final.pt` exists)
2. Finds latest checkpoint (`checkpoint_step_*.pt`)
3. Loads resume metadata (step count, loss, metrics)
4. Continues from where it left off

---

## File References

| File | Component |
|------|-----------|
| `demonstrator/training/train_sft.py` | SFT training loop with crash resilience |
| `demonstrator/training/train_rl.py` | REINFORCE RL training |
| `demonstrator/training/reward_fn.py` | 4-axis financial reward function |
| `demonstrator/training/data_loading.py` | Dataset loading utilities |
| `demonstrator/training/assemble_moe.py` | MoE assembly + TGSP conversion |
| `demonstrator/server/inference_engine.py:790-806` | Expert routing (keyword step-gate) |
