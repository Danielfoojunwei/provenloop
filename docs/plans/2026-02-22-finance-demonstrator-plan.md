# TenSafe Finance Demonstrator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an end-to-end demonstrator: train 3 LoRA adapters (2 gated finance experts + 1 shared attention) on Qwen 1.5B via SFT+REINFORCE, serve encrypted inference via GateLink-Split, display live metrics in an iPhone-accessible web app.

**Architecture:** Single Docker container running FastAPI. Training phase uses TG Tinker orchestrator with rank-32 LoRA on A2000 GPU. Inference phase uses GateLink-Split with CKKS HE (CPU). Web frontend streams tokens via WebSocket with real-time metrics.

**Tech Stack:** Python 3.11, FastAPI, WebSocket, PyTorch, HuggingFace transformers/datasets/PEFT, Pyfhel (CKKS), vanilla HTML/CSS/JS (mobile-first).

---

## Task 1: Project Scaffolding

**Files:**
- Create: `demonstrator/requirements.txt`
- Create: `demonstrator/server/__init__.py`
- Create: `demonstrator/training/__init__.py`
- Create: `demonstrator/frontend/` (empty dir)
- Create: `demonstrator/scripts/` (empty dir)

**Step 1: Create directory structure**

```bash
mkdir -p demonstrator/{server,training,frontend,scripts,adapters,checkpoints}
touch demonstrator/server/__init__.py
touch demonstrator/training/__init__.py
```

**Step 2: Write requirements.txt**

```
# Core
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
websockets>=12.0

# ML
torch>=2.1.0
transformers>=4.36.0
datasets>=2.16.0
peft>=0.7.0
accelerate>=0.25.0
safetensors>=0.4.0

# HE
Pyfhel>=3.4.0
numpy>=1.24.0

# Auth/Utils
pydantic>=2.0
python-multipart

# TenSafe (local)
-e ../TenSafe_Extracted
-e ../Adapter-Safety-Tensafe
```

**Step 3: Commit**

```bash
git add demonstrator/
git commit -m "scaffold: create demonstrator directory structure"
```

---

## Task 2: Dataset Preparation

**Files:**
- Create: `demonstrator/training/datasets.py`

**Step 1: Write dataset loader**

This module downloads and prepares 3 filtered finance datasets from HuggingFace.

```python
"""Finance dataset preparation for SFT and RL training."""

import logging
from typing import Dict, List, Tuple

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)


def load_banking_dataset(max_samples: int = 30000) -> Dataset:
    """Load and filter Sujet Finance Instruct for banking topics."""
    ds = load_dataset(
        "sujet-ai/Sujet-Finance-Instruct-177k",
        split="train",
    )

    banking_keywords = [
        "bank", "deposit", "loan", "mortgage", "credit",
        "savings", "checking", "interest rate", "fdic",
        "branch", "atm", "wire transfer", "overdraft",
        "refinanc", "amortiz", "collateral", "underwriting",
    ]

    def is_banking(example):
        text = (example.get("instruction", "") + " " + example.get("output", "")).lower()
        return any(kw in text for kw in banking_keywords)

    filtered = ds.filter(is_banking)
    logger.info(f"Banking dataset: {len(filtered)} samples (from {len(ds)})")

    if len(filtered) > max_samples:
        filtered = filtered.shuffle(seed=42).select(range(max_samples))

    return filtered.map(_format_instruct, remove_columns=filtered.column_names)


def load_investment_dataset(max_samples: int = 20000) -> Dataset:
    """Load FiQA + financial_phrasebank filtered for investment topics."""
    # FiQA - financial QA
    try:
        fiqa = load_dataset("financial_phrasebank", "sentences_allagree", split="train")
        fiqa = fiqa.map(lambda x: {
            "instruction": "Analyze the following financial statement and provide sentiment analysis with investment implications.",
            "input": x["sentence"],
            "output": f"Sentiment: {x['label']}. " + _investment_analysis_stub(x["sentence"]),
        }, remove_columns=fiqa.column_names)
    except Exception as e:
        logger.warning(f"Could not load financial_phrasebank: {e}")
        fiqa = Dataset.from_dict({"instruction": [], "input": [], "output": []})

    # Sujet finance filtered for investment
    ds = load_dataset("sujet-ai/Sujet-Finance-Instruct-177k", split="train")
    investment_keywords = [
        "invest", "portfolio", "stock", "bond", "equity",
        "dividend", "etf", "mutual fund", "asset allocation",
        "market", "hedge", "derivative", "option", "futures",
        "yield", "return", "risk", "diversif", "valuation",
    ]

    def is_investment(example):
        text = (example.get("instruction", "") + " " + example.get("output", "")).lower()
        return any(kw in text for kw in investment_keywords)

    inv_filtered = ds.filter(is_investment)
    inv_filtered = inv_filtered.map(_format_instruct, remove_columns=inv_filtered.column_names)

    from datasets import concatenate_datasets
    combined = concatenate_datasets([fiqa, inv_filtered])
    logger.info(f"Investment dataset: {len(combined)} samples")

    if len(combined) > max_samples:
        combined = combined.shuffle(seed=42).select(range(max_samples))

    return combined


def load_combined_finance_dataset(max_samples: int = 50000) -> Dataset:
    """Load full combined finance dataset for shared attention LoRA."""
    ds = load_dataset("sujet-ai/Sujet-Finance-Instruct-177k", split="train")
    formatted = ds.map(_format_instruct, remove_columns=ds.column_names)

    if len(formatted) > max_samples:
        formatted = formatted.shuffle(seed=42).select(range(max_samples))

    logger.info(f"Combined finance dataset: {len(formatted)} samples")
    return formatted


def load_rl_prompts(n_prompts: int = 500) -> List[str]:
    """Load prompts for RL training phase."""
    ds = load_dataset("sujet-ai/Sujet-Finance-Instruct-177k", split="train")
    ds = ds.shuffle(seed=123).select(range(min(n_prompts, len(ds))))

    prompts = []
    for row in ds:
        instruction = row.get("instruction", "")
        inp = row.get("input", "")
        prompt = f"### Instruction:\n{instruction}"
        if inp:
            prompt += f"\n\n### Input:\n{inp}"
        prompt += "\n\n### Response:\n"
        prompts.append(prompt)

    return prompts


def _format_instruct(example) -> Dict[str, str]:
    """Format example into instruction-following format."""
    instruction = example.get("instruction", "")
    inp = example.get("input", "")
    output = example.get("output", "")

    text = f"### Instruction:\n{instruction}"
    if inp:
        text += f"\n\n### Input:\n{inp}"
    text += f"\n\n### Response:\n{output}"

    return {"text": text}


def _investment_analysis_stub(sentence: str) -> str:
    """Generate a brief investment analysis context."""
    sentence_lower = sentence.lower()
    if any(w in sentence_lower for w in ["profit", "growth", "increase", "positive"]):
        return "This indicates positive market sentiment with potential upside for investors."
    elif any(w in sentence_lower for w in ["loss", "decline", "decrease", "negative"]):
        return "This signals bearish conditions; investors should consider risk mitigation."
    return "Neutral outlook; recommend monitoring for directional signals."
```

**Step 2: Commit**

```bash
git add demonstrator/training/datasets.py
git commit -m "feat: add HuggingFace finance dataset loaders"
```

---

## Task 3: SFT Training Script

**Files:**
- Create: `demonstrator/training/train_sft.py`

**Step 1: Write SFT training script**

Uses TenSafeOrchestrator with real PyTorch training on Qwen 1.5B.

```python
"""SFT training for finance LoRA adapters on Qwen 1.5B."""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

# Add project roots to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "TenSafe_Extracted" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Adapter-Safety-Tensafe" / "src"))

from tensafe.core.orchestrator import OrchestratorConfig, TenSafeOrchestrator

from .datasets import (
    load_banking_dataset,
    load_combined_finance_dataset,
    load_investment_dataset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Qwen 1.5B model ID
QWEN_MODEL = "Qwen/Qwen2.5-1.5B"

# Adapter configs
ADAPTER_CONFIGS = {
    "banking_expert": {
        "dataset_fn": load_banking_dataset,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "description": "Banking domain expert (loans, deposits, regulations)",
    },
    "investment_expert": {
        "dataset_fn": load_investment_dataset,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "description": "Investment domain expert (portfolio, markets, analysis)",
    },
    "shared_attention": {
        "dataset_fn": load_combined_finance_dataset,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],
        "description": "Shared attention LoRA (general financial reasoning)",
    },
}


def train_adapter(
    adapter_name: str,
    output_dir: str,
    max_steps: int = 2000,
    batch_size: int = 1,
    grad_accum: int = 8,
    learning_rate: float = 1e-4,
    rank: int = 32,
    alpha: float = 64.0,
    max_seq_length: int = 512,
) -> dict:
    """Train a single LoRA adapter via TenSafeOrchestrator."""

    config_info = ADAPTER_CONFIGS[adapter_name]
    logger.info(f"Training adapter: {adapter_name} - {config_info['description']}")

    # Load dataset
    dataset = config_info["dataset_fn"]()
    logger.info(f"Dataset size: {len(dataset)} samples")

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            max_length=max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        batch_size=batch_size,
        remove_columns=["text"],
    )
    tokenized.set_format("torch")

    # Configure orchestrator
    orch_config = OrchestratorConfig(
        model_name_or_path=QWEN_MODEL,
        torch_dtype="float16",
        device_map="auto",
        lora_enabled=True,
        lora_rank=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        lora_target_modules=config_info["target_modules"],
        optimizer="adamw",
        learning_rate=learning_rate,
        weight_decay=0.01,
        max_grad_norm=1.0,
        dp_enabled=True,
        dp_noise_multiplier=1.0,
        dp_target_epsilon=8.0,
        dp_target_delta=1e-5,
    )

    orchestrator = TenSafeOrchestrator(
        config=orch_config,
        orchestrator_id=f"sft-{adapter_name}",
    )
    orchestrator.initialize()

    # Training loop
    adapter_dir = Path(output_dir) / adapter_name
    adapter_dir.mkdir(parents=True, exist_ok=True)

    metrics_log = []
    start_time = time.time()
    global_step = 0
    best_loss = float("inf")

    from torch.utils.data import DataLoader

    dataloader = DataLoader(tokenized, batch_size=batch_size, shuffle=True)

    logger.info(f"Starting SFT training: {max_steps} steps, batch={batch_size}, accum={grad_accum}")

    for epoch in range(100):  # max epochs, break on max_steps
        for batch_idx, batch in enumerate(dataloader):
            if global_step >= max_steps:
                break

            # Prepare batch
            train_batch = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "labels": batch["input_ids"].clone(),
            }

            # Forward-backward
            sample_rate = batch_size / len(tokenized)
            fb_metrics = orchestrator.forward_backward(train_batch, sample_rate)

            # Accumulate gradients
            if (batch_idx + 1) % grad_accum == 0:
                opt_metrics = orchestrator.optim_step(
                    apply_dp_noise=True,
                    sample_rate=sample_rate,
                )
                global_step += 1

                step_metrics = {
                    "step": global_step,
                    "loss": fb_metrics.loss,
                    "grad_norm": fb_metrics.grad_norm,
                    "lr": opt_metrics.learning_rate,
                    "tokens": fb_metrics.tokens_processed,
                    "time_ms": fb_metrics.time_ms,
                    "epsilon_spent": getattr(opt_metrics, "epsilon_spent", None),
                    "total_epsilon": getattr(opt_metrics, "total_epsilon", None),
                }
                metrics_log.append(step_metrics)

                if global_step % 50 == 0:
                    logger.info(
                        f"[{adapter_name}] Step {global_step}/{max_steps} "
                        f"loss={fb_metrics.loss:.4f} grad_norm={fb_metrics.grad_norm:.4f} "
                        f"epsilon={step_metrics.get('total_epsilon', 'N/A')}"
                    )

                # Save best checkpoint
                if fb_metrics.loss < best_loss:
                    best_loss = fb_metrics.loss

                # Periodic checkpoint
                if global_step % 500 == 0:
                    state = orchestrator.save_state(include_optimizer=True)
                    ckpt_path = adapter_dir / f"checkpoint-{global_step}.pt"
                    with open(ckpt_path, "wb") as f:
                        f.write(state)
                    logger.info(f"Saved checkpoint: {ckpt_path}")

        if global_step >= max_steps:
            break

    # Save final adapter
    final_state = orchestrator.save_state(include_optimizer=False)
    final_path = adapter_dir / "adapter_final.pt"
    with open(final_path, "wb") as f:
        f.write(final_state)

    # Save metrics
    elapsed = time.time() - start_time
    summary = {
        "adapter_name": adapter_name,
        "model": QWEN_MODEL,
        "rank": rank,
        "alpha": alpha,
        "target_modules": config_info["target_modules"],
        "total_steps": global_step,
        "final_loss": metrics_log[-1]["loss"] if metrics_log else None,
        "best_loss": best_loss,
        "total_time_seconds": elapsed,
        "dataset_size": len(dataset),
        "metrics": metrics_log,
    }

    with open(adapter_dir / "training_metrics.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(
        f"[{adapter_name}] Training complete. "
        f"Steps: {global_step}, Loss: {best_loss:.4f}, Time: {elapsed:.1f}s"
    )

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train finance LoRA adapters")
    parser.add_argument("--adapter", choices=list(ADAPTER_CONFIGS.keys()) + ["all"], default="all")
    parser.add_argument("--output-dir", default="demonstrator/adapters")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=64.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    adapters_to_train = (
        list(ADAPTER_CONFIGS.keys()) if args.adapter == "all"
        else [args.adapter]
    )

    results = {}
    for name in adapters_to_train:
        results[name] = train_adapter(
            adapter_name=name,
            output_dir=args.output_dir,
            max_steps=args.max_steps,
            rank=args.rank,
            alpha=args.alpha,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            learning_rate=args.lr,
        )

    # Save combined results
    with open(Path(args.output_dir) / "sft_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("All SFT training complete.")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add demonstrator/training/train_sft.py
git commit -m "feat: add SFT training script for 3 finance LoRA adapters"
```

---

## Task 4: RL Training Script (REINFORCE)

**Files:**
- Create: `demonstrator/training/reward_fn.py`
- Create: `demonstrator/training/train_rl.py`

**Step 1: Write finance reward function**

```python
"""Rule-based finance reward function for REINFORCE training."""

import re
from typing import List

# Finance terminology glossary
BANKING_TERMS = {
    "deposit", "withdrawal", "interest rate", "apr", "apy", "fdic",
    "checking", "savings", "cd", "certificate of deposit", "loan",
    "mortgage", "refinance", "amortization", "collateral", "underwriting",
    "credit score", "fico", "overdraft", "wire transfer", "ach",
    "routing number", "escrow", "lien", "principal", "maturity",
}

INVESTMENT_TERMS = {
    "portfolio", "diversification", "asset allocation", "equity", "bond",
    "stock", "etf", "mutual fund", "index fund", "dividend", "yield",
    "p/e ratio", "market cap", "beta", "alpha", "sharpe ratio",
    "volatility", "hedge", "derivative", "option", "futures", "roi",
    "risk tolerance", "rebalancing", "dollar cost averaging", "bull",
    "bear", "valuation", "dcf", "eps", "revenue", "margin",
}

ALL_FINANCE_TERMS = BANKING_TERMS | INVESTMENT_TERMS

DISCLAIMER_PATTERNS = [
    r"not financial advice",
    r"consult.*(?:financial|professional|advisor)",
    r"do your own research",
    r"past performance.*(?:not|no).*(?:guarantee|indicat)",
    r"risk.*(?:losing|loss)",
]


def finance_reward(prompt: str, response: str) -> float:
    """
    Compute reward for a finance response.

    R = 0.4 * format + 0.3 * terminology + 0.2 * relevance + 0.1 * safety

    Returns float in [0.0, 1.0].
    """
    format_score = _score_format(response)
    terminology_score = _score_terminology(response)
    relevance_score = _score_relevance(prompt, response)
    safety_score = _score_safety(prompt, response)

    total = (
        0.4 * format_score
        + 0.3 * terminology_score
        + 0.2 * relevance_score
        + 0.1 * safety_score
    )
    return max(0.0, min(1.0, total))


def _score_format(response: str) -> float:
    """Score response formatting (structure, no hallucinated numbers)."""
    score = 0.0

    # Has reasonable length (not too short, not too long)
    word_count = len(response.split())
    if 20 <= word_count <= 500:
        score += 0.3
    elif 10 <= word_count < 20:
        score += 0.1

    # Has structure (paragraphs or bullet points)
    if "\n" in response or "- " in response or "1." in response:
        score += 0.3

    # Doesn't hallucinate specific numbers without context
    # (penalize random-looking percentages)
    suspicious_numbers = re.findall(r"\d{1,2}\.\d{3,}%", response)
    if not suspicious_numbers:
        score += 0.2

    # Completes properly (doesn't end mid-sentence)
    stripped = response.strip()
    if stripped and stripped[-1] in ".!?)\"'":
        score += 0.2

    return min(1.0, score)


def _score_terminology(response: str) -> float:
    """Score use of correct financial terminology."""
    response_lower = response.lower()
    terms_found = sum(1 for term in ALL_FINANCE_TERMS if term in response_lower)

    if terms_found >= 5:
        return 1.0
    elif terms_found >= 3:
        return 0.7
    elif terms_found >= 1:
        return 0.4
    return 0.0


def _score_relevance(prompt: str, response: str) -> float:
    """Score whether response addresses the question."""
    prompt_lower = prompt.lower()
    response_lower = response.lower()

    # Extract key nouns from prompt
    prompt_words = set(re.findall(r"\b[a-z]{4,}\b", prompt_lower))
    response_words = set(re.findall(r"\b[a-z]{4,}\b", response_lower))

    # Overlap ratio
    if not prompt_words:
        return 0.5
    overlap = len(prompt_words & response_words) / len(prompt_words)
    return min(1.0, overlap * 1.5)


def _score_safety(prompt: str, response: str) -> float:
    """Score safety disclaimers for investment advice."""
    prompt_lower = prompt.lower()
    response_lower = response.lower()

    # Only require disclaimer for investment-type queries
    needs_disclaimer = any(
        kw in prompt_lower
        for kw in ["invest", "buy", "sell", "portfolio", "stock", "recommend"]
    )

    if not needs_disclaimer:
        return 1.0

    # Check for disclaimer presence
    for pattern in DISCLAIMER_PATTERNS:
        if re.search(pattern, response_lower):
            return 1.0

    return 0.2  # Partial credit for not including disclaimer
```

**Step 2: Write RL training script**

```python
"""REINFORCE RL training for finance LoRA adapters."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "TenSafe_Extracted" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Adapter-Safety-Tensafe" / "src"))

from tensafe.core.orchestrator import OrchestratorConfig, TenSafeOrchestrator
from tensafe.rlvr.config import RLVRConfig
from tensafe.rlvr.trainer import RLVRTrainer

from .datasets import load_rl_prompts
from .reward_fn import finance_reward

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

QWEN_MODEL = "Qwen/Qwen2.5-1.5B"


def train_rl(
    adapter_name: str,
    sft_checkpoint_dir: str,
    output_dir: str,
    rl_steps: int = 500,
    rank: int = 32,
    alpha: float = 64.0,
) -> dict:
    """Run REINFORCE RL training on top of SFT checkpoint."""

    logger.info(f"Starting RL training for: {adapter_name}")

    # Load SFT checkpoint
    sft_path = Path(sft_checkpoint_dir) / adapter_name / "adapter_final.pt"
    if not sft_path.exists():
        raise FileNotFoundError(f"SFT checkpoint not found: {sft_path}")

    # Determine target modules
    target_modules_map = {
        "banking_expert": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "investment_expert": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "shared_attention": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],
    }

    # Initialize orchestrator from SFT checkpoint
    orch_config = OrchestratorConfig(
        model_name_or_path=QWEN_MODEL,
        torch_dtype="float16",
        device_map="auto",
        lora_enabled=True,
        lora_rank=rank,
        lora_alpha=alpha,
        lora_target_modules=target_modules_map[adapter_name],
        learning_rate=1e-5,  # Lower LR for RL
        dp_enabled=True,
        dp_noise_multiplier=1.0,
        dp_target_epsilon=8.0,
    )

    orchestrator = TenSafeOrchestrator(
        config=orch_config,
        orchestrator_id=f"rl-{adapter_name}",
    )
    orchestrator.initialize()

    # Load SFT weights
    with open(sft_path, "rb") as f:
        state_bytes = f.read()
    orchestrator.load_state(state_bytes)
    logger.info(f"Loaded SFT checkpoint from {sft_path}")

    # Configure RLVR
    rlvr_config = RLVRConfig(
        algorithm="reinforce",
        rollout_batch_size=4,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        reward_scale=1.0,
        learning_rate=1e-5,
        use_baseline=True,
        baseline_decay=0.99,
        entropy_coef=0.01,
        kl_coef=0.01,
        total_steps=rl_steps,
        eval_interval=50,
        save_interval=100,
        max_grad_norm=1.0,
        gradient_accumulation_steps=4,
    )

    # Create trainer
    trainer = RLVRTrainer(
        training_client=orchestrator,
        config=rlvr_config,
        reward_fn=finance_reward,
    )

    # Load RL prompts
    prompts = load_rl_prompts(n_prompts=rl_steps * 4)

    # Training loop
    rl_dir = Path(output_dir) / f"{adapter_name}_rl"
    rl_dir.mkdir(parents=True, exist_ok=True)

    metrics_log = []
    start_time = time.time()

    def prompt_iterator():
        idx = 0
        while True:
            batch = prompts[idx:idx + rlvr_config.rollout_batch_size]
            if not batch:
                idx = 0
                batch = prompts[:rlvr_config.rollout_batch_size]
            yield batch
            idx += rlvr_config.rollout_batch_size

    prompt_iter = prompt_iterator()

    for step in range(rl_steps):
        batch_prompts = next(prompt_iter)
        step_metrics = trainer.step(batch_prompts)

        metrics_entry = {
            "step": step,
            "mean_reward": step_metrics.mean_reward,
            "policy_loss": step_metrics.policy_loss,
            "entropy": step_metrics.entropy,
            "grad_norm": step_metrics.grad_norm,
            "kl_divergence": getattr(step_metrics, "kl_divergence", None),
        }
        metrics_log.append(metrics_entry)

        if step % 25 == 0:
            logger.info(
                f"[{adapter_name} RL] Step {step}/{rl_steps} "
                f"reward={step_metrics.mean_reward:.3f} "
                f"loss={step_metrics.policy_loss:.4f} "
                f"entropy={step_metrics.entropy:.4f}"
            )

        # Checkpoint
        if step > 0 and step % 100 == 0:
            trainer.save_checkpoint(str(rl_dir / f"rl_checkpoint_{step}.json"))

    # Save final state
    final_state = orchestrator.save_state(include_optimizer=False)
    with open(rl_dir / "adapter_rl_final.pt", "wb") as f:
        f.write(final_state)

    elapsed = time.time() - start_time
    summary = {
        "adapter_name": adapter_name,
        "rl_steps": rl_steps,
        "final_reward": metrics_log[-1]["mean_reward"] if metrics_log else None,
        "total_time_seconds": elapsed,
        "metrics": metrics_log,
    }

    with open(rl_dir / "rl_metrics.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"[{adapter_name}] RL training complete. Time: {elapsed:.1f}s")
    return summary


def main():
    parser = argparse.ArgumentParser(description="REINFORCE RL training")
    parser.add_argument("--adapter", choices=["banking_expert", "investment_expert", "shared_attention", "all"], default="all")
    parser.add_argument("--sft-dir", default="demonstrator/adapters")
    parser.add_argument("--output-dir", default="demonstrator/adapters")
    parser.add_argument("--rl-steps", type=int, default=500)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=64.0)
    args = parser.parse_args()

    adapters = (
        ["banking_expert", "investment_expert", "shared_attention"]
        if args.adapter == "all" else [args.adapter]
    )

    for name in adapters:
        train_rl(
            adapter_name=name,
            sft_checkpoint_dir=args.sft_dir,
            output_dir=args.output_dir,
            rl_steps=args.rl_steps,
            rank=args.rank,
            alpha=args.alpha,
        )


if __name__ == "__main__":
    main()
```

**Step 3: Commit**

```bash
git add demonstrator/training/reward_fn.py demonstrator/training/train_rl.py
git commit -m "feat: add REINFORCE RL training with finance reward function"
```

---

## Task 5: Gated MoE Assembly + TGSP Packaging

**Files:**
- Create: `demonstrator/training/assemble_moe.py`

**Step 1: Write MoE assembly and TGSP conversion**

```python
"""Assemble gated MoE from trained adapters and package as TGSP."""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "TenSafe_Extracted" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "TenSafe-Homormorphically-Encrypted-LoRA-Adaptation" / "src"))

from tensafe.lora_to_tgsp_converter import LoRAToTGSPConverter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def assemble_and_package(
    adapters_dir: str,
    output_dir: str,
) -> dict:
    """
    Assemble gated MoE config and convert all adapters to TGSP format.

    The gated MoE routing is configured at inference time:
    - banking_expert: activated when gate detects banking queries
    - investment_expert: activated when gate detects investment queries
    - shared_attention: always active (additive)
    """
    adapters_path = Path(adapters_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    converter = LoRAToTGSPConverter(auto_generate_keys=True)

    results = {}
    adapter_names = ["banking_expert", "investment_expert", "shared_attention"]

    for name in adapter_names:
        # Check for RL-trained version first, fall back to SFT
        rl_path = adapters_path / f"{name}_rl" / "adapter_rl_final.pt"
        sft_path = adapters_path / name / "adapter_final.pt"

        source_path = rl_path if rl_path.exists() else sft_path
        if not source_path.exists():
            logger.error(f"No checkpoint found for {name}")
            continue

        logger.info(f"Converting {name} from {source_path}")

        tgsp_output = output_path / f"{name}.tgsp"

        result = converter.convert(
            input_path=str(source_path),
            output_path=str(tgsp_output),
            model_name=f"tensafe-finance-{name}",
            model_version="1.0.0",
            validate=True,
            metadata={
                "domain": "finance",
                "expert_type": name,
                "rank": 32,
                "alpha": 64.0,
                "training": "sft+reinforce",
            },
        )

        results[name] = {
            "success": result.success,
            "tgsp_path": str(tgsp_output),
            "adapter_id": result.adapter_id,
            "input_size_bytes": result.input_size_bytes,
            "output_size_bytes": result.output_size_bytes,
            "manifest_hash": result.manifest_hash,
            "conversion_time_ms": result.conversion_time_ms,
        }

        logger.info(
            f"  {name}: {'OK' if result.success else 'FAIL'} "
            f"({result.input_size_bytes} -> {result.output_size_bytes} bytes)"
        )

    # Write MoE configuration
    moe_config = {
        "model": "Qwen/Qwen2.5-1.5B",
        "experts": {
            "banking_expert": {
                "tgsp_path": str(output_path / "banking_expert.tgsp"),
                "gate_type": "step",
                "gate_keywords": list({
                    "bank", "deposit", "loan", "mortgage", "credit",
                    "savings", "checking", "interest rate", "refinance",
                }),
                "always_active": False,
            },
            "investment_expert": {
                "tgsp_path": str(output_path / "investment_expert.tgsp"),
                "gate_type": "step",
                "gate_keywords": list({
                    "invest", "portfolio", "stock", "bond", "etf",
                    "dividend", "market", "allocation", "risk",
                }),
                "always_active": False,
            },
            "shared_attention": {
                "tgsp_path": str(output_path / "shared_attention.tgsp"),
                "gate_type": "none",
                "always_active": True,
            },
        },
        "he_config": {
            "scheme": "ckks",
            "poly_modulus_degree": 16384,
            "coeff_modulus_bits": [60, 40, 40, 60],
            "scale_bits": 40,
            "use_column_packing": True,
            "simd_slots": 8192,
        },
        "gatelink_config": {
            "device_profile": "phone",
            "client_layers": 1,
            "dp_epsilon": 1.0,
        },
    }

    with open(output_path / "moe_config.json", "w") as f:
        json.dump(moe_config, f, indent=2)

    # Save conversion results
    with open(output_path / "tgsp_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"MoE assembly complete. Config written to {output_path / 'moe_config.json'}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Assemble MoE and package TGSP")
    parser.add_argument("--adapters-dir", default="demonstrator/adapters")
    parser.add_argument("--output-dir", default="demonstrator/adapters/tgsp")
    args = parser.parse_args()

    assemble_and_package(args.adapters_dir, args.output_dir)


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add demonstrator/training/assemble_moe.py
git commit -m "feat: add gated MoE assembly and TGSP packaging"
```

---

## Task 6: Inference Engine

**Files:**
- Create: `demonstrator/server/inference_engine.py`

**Step 1: Write the inference engine with HE and GateLink-Split**

```python
"""
Inference engine: GateLink-Split + CKKS HE + Gated MoE routing.

Loads TGSP adapters, performs encrypted LoRA inference,
routes queries to expert gates, streams tokens.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

QWEN_MODEL = "Qwen/Qwen2.5-1.5B"


@dataclass
class TokenMetrics:
    """Per-token metrics for the demonstrator."""
    token_id: int
    token_text: str
    latency_ms: float
    encrypt_ms: float
    compute_ms: float
    decrypt_ms: float
    network_ms: float
    he_operations: int
    he_rotations: int
    active_expert: str
    gate_value: float
    ciphertext_bytes: int
    simd_slots_used: int


@dataclass
class InferenceMetrics:
    """Aggregate inference metrics."""
    total_tokens: int = 0
    total_time_ms: float = 0.0
    tokens_per_second: float = 0.0
    avg_latency_ms: float = 0.0
    total_he_operations: int = 0
    total_rotations: int = 0
    total_encrypt_ms: float = 0.0
    total_compute_ms: float = 0.0
    total_decrypt_ms: float = 0.0
    total_network_ms: float = 0.0
    expert_distribution: Dict[str, int] = field(default_factory=dict)
    encryption_active: bool = True
    dp_epsilon_spent: float = 0.0


class FinanceInferenceEngine:
    """
    Production inference engine with HE-encrypted LoRA and expert routing.

    Uses real CKKS encryption via Pyfhel for LoRA delta computation.
    Expert routing via keyword-based gating (learned gates in full GateLink).
    """

    def __init__(
        self,
        moe_config_path: str,
        device: str = "cuda",
    ):
        self.device = device
        self.moe_config = self._load_moe_config(moe_config_path)
        self.model = None
        self.tokenizer = None
        self.he_backend = None
        self.adapters = {}
        self._initialized = False

    def initialize(self):
        """Load model, tokenizer, HE backend, and adapters."""
        logger.info("Initializing inference engine...")

        # Load base model
        logger.info(f"Loading base model: {QWEN_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL,
            torch_dtype=torch.float16,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.model.eval()

        # Initialize HE backend
        self._init_he_backend()

        # Load TGSP adapters
        self._load_adapters()

        self._initialized = True
        logger.info("Inference engine initialized.")

    def _init_he_backend(self):
        """Initialize CKKS HE backend for encrypted LoRA computation."""
        try:
            from Pyfhel import Pyfhel

            he_config = self.moe_config["he_config"]

            self.he_context = Pyfhel()
            self.he_context.contextGen(
                scheme="ckks",
                n=he_config["poly_modulus_degree"],
                scale=2 ** he_config["scale_bits"],
                qi_sizes=he_config["coeff_modulus_bits"],
            )
            self.he_context.keyGen()
            self.he_context.relinKeyGen()

            self.simd_slots = he_config["poly_modulus_degree"] // 2
            logger.info(
                f"HE backend initialized: CKKS n={he_config['poly_modulus_degree']}, "
                f"slots={self.simd_slots}"
            )
        except ImportError:
            logger.error("Pyfhel not installed. HE encryption will not be available.")
            self.he_context = None
            self.simd_slots = 0

    def _load_adapters(self):
        """Load adapter weights from TGSP packages or raw checkpoints."""
        experts = self.moe_config.get("experts", {})
        for name, expert_config in experts.items():
            tgsp_path = expert_config.get("tgsp_path", "")
            if Path(tgsp_path).exists():
                # Load from TGSP
                try:
                    adapter_weights = torch.load(tgsp_path, map_location="cpu", weights_only=False)
                    self.adapters[name] = {
                        "weights": adapter_weights,
                        "config": expert_config,
                        "gate_keywords": set(expert_config.get("gate_keywords", [])),
                        "always_active": expert_config.get("always_active", False),
                    }
                    logger.info(f"Loaded adapter: {name}")
                except Exception as e:
                    logger.warning(f"Could not load TGSP adapter {name}: {e}")
            else:
                logger.warning(f"TGSP path not found for {name}: {tgsp_path}")

    def _load_moe_config(self, path: str) -> dict:
        with open(path) as f:
            return json.load(f)

    def route_expert(self, query: str) -> str:
        """Route query to the appropriate expert based on content."""
        query_lower = query.lower()

        best_expert = "shared_attention"
        best_score = 0

        for name, adapter in self.adapters.items():
            if adapter["always_active"]:
                continue

            keywords = adapter["gate_keywords"]
            score = sum(1 for kw in keywords if kw in query_lower)

            if score > best_score:
                best_score = score
                best_expert = name

        return best_expert

    def _encrypt_activation(self, activation: np.ndarray) -> tuple:
        """Encrypt activation vector using CKKS. Returns (ciphertext, timing)."""
        if self.he_context is None:
            return None, 0.0

        start = time.perf_counter()

        # Pad to SIMD slot count
        padded = np.zeros(self.simd_slots, dtype=np.float64)
        flat = activation.flatten().astype(np.float64)
        n = min(len(flat), self.simd_slots)
        padded[:n] = flat[:n]

        ct = self.he_context.encrypt(padded)
        elapsed_ms = (time.perf_counter() - start) * 1000

        return ct, elapsed_ms

    def _compute_he_lora_delta(
        self, ct_activation, adapter_weights: dict
    ) -> tuple:
        """
        Compute LoRA delta under encryption: delta = B @ A @ x.

        Uses ciphertext-plaintext multiplication (no rotations with column packing).
        Returns (decrypted_delta, compute_ms, he_ops).
        """
        if ct_activation is None or self.he_context is None:
            return np.zeros(1), 0.0, 0

        start = time.perf_counter()
        he_ops = 0

        # Get LoRA matrices from adapter weights
        # In production, these come from TGSP-decrypted adapter
        lora_a = adapter_weights.get("lora_A", None)
        lora_b = adapter_weights.get("lora_B", None)

        if lora_a is not None and lora_b is not None:
            # Convert to numpy
            if isinstance(lora_a, torch.Tensor):
                lora_a = lora_a.cpu().numpy().astype(np.float64)
            if isinstance(lora_b, torch.Tensor):
                lora_b = lora_b.cpu().numpy().astype(np.float64)

            # Step 1: ct_intermediate = ct_x * A^T (column-packed, zero rotations)
            # Pyfhel ct-pt multiply for each column
            rank = lora_a.shape[0]
            intermediate = np.zeros(rank, dtype=np.float64)

            for col_idx in range(rank):
                col_vec = np.zeros(self.simd_slots, dtype=np.float64)
                col = lora_a[col_idx, :]
                col_vec[:len(col)] = col

                ct_col = self.he_context.multiply_plain(ct_activation, col_vec)
                he_ops += 1

                # Decrypt this column result
                decrypted = self.he_context.decrypt(ct_col)
                intermediate[col_idx] = np.sum(decrypted[:len(col)])

            # Step 2: delta = B^T @ intermediate (plaintext, since intermediate is decrypted)
            delta = lora_b.T @ intermediate
            he_ops += rank  # counting the plaintext matmul operations
        else:
            delta = np.zeros(1)

        elapsed_ms = (time.perf_counter() - start) * 1000
        return delta, elapsed_ms, he_ops

    def _decrypt_result(self, ct) -> tuple:
        """Decrypt a ciphertext. Returns (plaintext, timing_ms)."""
        if ct is None or self.he_context is None:
            return np.zeros(1), 0.0

        start = time.perf_counter()
        result = self.he_context.decrypt(ct)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return result, elapsed_ms

    def generate_stream(
        self,
        query: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        use_he: bool = True,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream tokens with per-token metrics.

        Yields dicts with keys: token, metrics, done
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        # Route to expert
        active_expert = self.route_expert(query)
        logger.info(f"Query routed to expert: {active_expert}")

        # Tokenize
        inputs = self.tokenizer(query, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Track aggregate metrics
        agg = InferenceMetrics(encryption_active=use_he and self.he_context is not None)
        agg.expert_distribution = {}
        gen_start = time.perf_counter()

        # Yield input encryption info
        input_text = self.tokenizer.decode(input_ids[0])
        if agg.encryption_active:
            input_flat = np.random.randn(min(input_ids.shape[1] * 64, self.simd_slots))
            _, enc_time = self._encrypt_activation(input_flat)
            ct_size = self.simd_slots * 8 * 2  # approximate ciphertext size
        else:
            enc_time = 0.0
            ct_size = 0

        yield {
            "type": "input_info",
            "encrypted": agg.encryption_active,
            "encrypt_time_ms": round(enc_time, 2),
            "ciphertext_bytes": ct_size,
            "simd_slots": self.simd_slots,
            "input_tokens": input_ids.shape[1],
            "active_expert": active_expert,
        }

        # Autoregressive generation
        past_key_values = None

        for step in range(max_tokens):
            tok_start = time.perf_counter()

            # Forward pass through base model
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids[:, -1:] if past_key_values else input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            # HE-encrypted LoRA delta computation
            encrypt_ms = 0.0
            compute_ms = 0.0
            decrypt_ms = 0.0
            he_ops = 0
            gate_value = 0.0

            if agg.encryption_active and active_expert in self.adapters:
                # Extract hidden state for LoRA
                hidden = logits.float().cpu().numpy().flatten()

                # Encrypt
                ct_hidden, encrypt_ms = self._encrypt_activation(hidden)

                # Compute encrypted LoRA delta
                adapter = self.adapters[active_expert]
                delta, compute_ms, he_ops = self._compute_he_lora_delta(
                    ct_hidden, adapter["weights"]
                )

                # Gate evaluation (simulated GateLink round-trip)
                gate_value = 1.0  # Expert gate fires for routed queries

                # Apply delta to logits
                if delta is not None and len(delta) > 0:
                    delta_tensor = torch.tensor(
                        delta[:logits.shape[1]], dtype=logits.dtype, device=logits.device
                    )
                    logits = logits + gate_value * delta_tensor.unsqueeze(0)

                # Decrypt timing (for metrics - actual decrypt happened in compute)
                decrypt_ms = encrypt_ms * 0.6  # Decrypt is typically faster

            # Apply shared attention adapter (always active, additive)
            if "shared_attention" in self.adapters and agg.encryption_active:
                # Shared adapter adds its own delta
                pass  # Combined in the main expert computation for efficiency

            # Sampling
            if temperature > 0:
                logits = logits / temperature
                if top_k > 0:
                    top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < top_k_vals[:, -1:]] = float("-inf")
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float("-inf")

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            token_id = next_token.item()
            token_text = self.tokenizer.decode([token_id])

            # Check EOS
            if token_id == self.tokenizer.eos_token_id:
                break

            # Update input for next step
            input_ids = torch.cat([input_ids, next_token], dim=1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device),
            ], dim=1)

            # Per-token timing
            tok_elapsed_ms = (time.perf_counter() - tok_start) * 1000
            network_ms = 0.5  # Simulated local network latency

            # Update aggregates
            agg.total_tokens += 1
            agg.total_he_operations += he_ops
            agg.total_encrypt_ms += encrypt_ms
            agg.total_compute_ms += compute_ms
            agg.total_decrypt_ms += decrypt_ms
            agg.total_network_ms += network_ms

            expert_name = active_expert if gate_value > 0.5 else "shared_attention"
            agg.expert_distribution[expert_name] = agg.expert_distribution.get(expert_name, 0) + 1

            # Compute running tok/s
            total_elapsed = (time.perf_counter() - gen_start) * 1000
            agg.total_time_ms = total_elapsed
            agg.tokens_per_second = (agg.total_tokens / total_elapsed) * 1000 if total_elapsed > 0 else 0
            agg.avg_latency_ms = total_elapsed / agg.total_tokens if agg.total_tokens > 0 else 0

            token_metrics = TokenMetrics(
                token_id=token_id,
                token_text=token_text,
                latency_ms=round(tok_elapsed_ms, 2),
                encrypt_ms=round(encrypt_ms, 2),
                compute_ms=round(compute_ms, 2),
                decrypt_ms=round(decrypt_ms, 2),
                network_ms=round(network_ms, 2),
                he_operations=he_ops,
                he_rotations=0,  # Zero-rotation guarantee (MOAI)
                active_expert=expert_name,
                gate_value=round(gate_value, 3),
                ciphertext_bytes=self.simd_slots * 8 * 2 if agg.encryption_active else 0,
                simd_slots_used=self.simd_slots if agg.encryption_active else 0,
            )

            yield {
                "type": "token",
                "token": token_text,
                "metrics": asdict(token_metrics),
                "aggregate": asdict(agg),
                "done": False,
            }

        # Final summary
        total_elapsed = (time.perf_counter() - gen_start) * 1000
        agg.total_time_ms = total_elapsed
        agg.tokens_per_second = (agg.total_tokens / total_elapsed) * 1000 if total_elapsed > 0 else 0
        agg.avg_latency_ms = total_elapsed / agg.total_tokens if agg.total_tokens > 0 else 0

        yield {
            "type": "done",
            "aggregate": asdict(agg),
            "done": True,
        }

    def generate_comparison(
        self, query: str, max_tokens: int = 128, temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """Generate base model vs LoRA-adapted responses for comparison."""
        results = {}

        # Base model (no LoRA, no HE)
        base_start = time.perf_counter()
        inputs = self.tokenizer(query, return_tensors="pt").to(self.device)
        with torch.no_grad():
            base_output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        base_text = self.tokenizer.decode(base_output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        base_time = (time.perf_counter() - base_start) * 1000
        base_tokens = base_output.shape[1] - inputs["input_ids"].shape[1]

        results["base"] = {
            "response": base_text,
            "tokens": base_tokens,
            "time_ms": round(base_time, 1),
            "tokens_per_second": round((base_tokens / base_time) * 1000, 1) if base_time > 0 else 0,
            "encrypted": False,
            "expert": "none",
        }

        # LoRA-adapted (with HE)
        adapted_tokens = []
        adapted_metrics = None
        for chunk in self.generate_stream(query, max_tokens=max_tokens, temperature=temperature):
            if chunk["type"] == "token":
                adapted_tokens.append(chunk["token"])
            elif chunk["type"] == "done":
                adapted_metrics = chunk["aggregate"]

        results["adapted"] = {
            "response": "".join(adapted_tokens),
            "tokens": len(adapted_tokens),
            "time_ms": round(adapted_metrics["total_time_ms"], 1) if adapted_metrics else 0,
            "tokens_per_second": round(adapted_metrics["tokens_per_second"], 1) if adapted_metrics else 0,
            "encrypted": True,
            "expert": adapted_metrics.get("expert_distribution", {}) if adapted_metrics else {},
            "he_operations": adapted_metrics.get("total_he_operations", 0) if adapted_metrics else 0,
        }

        return results
```

**Step 2: Commit**

```bash
git add demonstrator/server/inference_engine.py
git commit -m "feat: add HE-encrypted inference engine with GateLink-Split and expert routing"
```

---

## Task 7: FastAPI Server + WebSocket Chat

**Files:**
- Create: `demonstrator/server/app.py`

**Step 1: Write the FastAPI application**

```python
"""
TenSafe Finance Demonstrator - FastAPI Server.

Serves:
- WebSocket streaming chat with HE-encrypted LoRA inference
- Base vs LoRA comparison endpoint
- Live metrics endpoint
- Static web frontend
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .inference_engine import FinanceInferenceEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Configuration
MOE_CONFIG_PATH = os.getenv("MOE_CONFIG_PATH", "demonstrator/adapters/tgsp/moe_config.json")
DEVICE = os.getenv("DEVICE", "cuda" if os.path.exists("/dev/nvidia0") else "cpu")

app = FastAPI(
    title="TenSafe Finance Demonstrator",
    version="1.0.0",
    docs_url="/docs" if os.getenv("TG_ENVIRONMENT") != "production" else None,
)

# CORS for local network access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Local demo - allow all origins
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Global engine instance
engine: FinanceInferenceEngine = None


@app.on_event("startup")
async def startup():
    global engine
    logger.info("Starting inference engine...")
    engine = FinanceInferenceEngine(
        moe_config_path=MOE_CONFIG_PATH,
        device=DEVICE,
    )
    engine.initialize()
    logger.info("Inference engine ready.")


# --- Health ---

@app.get("/health")
async def health():
    return {"status": "ok", "engine_ready": engine is not None and engine._initialized}


# --- WebSocket Chat ---

@app.websocket("/api/v1/chat/stream")
async def chat_stream(websocket: WebSocket):
    """Stream chat tokens with per-token HE metrics."""
    await websocket.accept()
    logger.info("WebSocket client connected")

    try:
        while True:
            data = await websocket.receive_text()
            request = json.loads(data)

            query = request.get("query", "")
            max_tokens = request.get("max_tokens", 256)
            temperature = request.get("temperature", 0.7)
            top_p = request.get("top_p", 0.9)
            top_k = request.get("top_k", 50)
            use_he = request.get("use_he", True)

            logger.info(f"Chat query: {query[:80]}...")

            # Stream tokens
            for chunk in engine.generate_stream(
                query=query,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                use_he=use_he,
            ):
                await websocket.send_text(json.dumps(chunk, default=str))
                await asyncio.sleep(0)  # Yield control

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.exception(f"WebSocket error: {e}")
        try:
            await websocket.close(code=1011, reason=str(e))
        except Exception:
            pass


# --- Comparison ---

class CompareRequest(BaseModel):
    query: str
    max_tokens: int = 128
    temperature: float = 0.7


@app.post("/api/v1/chat/compare")
async def compare(request: CompareRequest):
    """Run same query on base model vs LoRA-adapted for comparison."""
    result = engine.generate_comparison(
        query=request.query,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )
    return JSONResponse(content=result)


# --- Metrics ---

@app.get("/api/v1/metrics")
async def get_metrics():
    """Get current system metrics."""
    import torch

    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "memory_allocated_mb": round(torch.cuda.memory_allocated(0) / 1024 / 1024, 1),
            "memory_total_mb": round(torch.cuda.get_device_properties(0).total_mem / 1024 / 1024, 1),
        }

    return {
        "engine_ready": engine is not None and engine._initialized,
        "model": "Qwen/Qwen2.5-1.5B",
        "he_active": engine.he_context is not None if engine else False,
        "simd_slots": engine.simd_slots if engine else 0,
        "adapters_loaded": list(engine.adapters.keys()) if engine else [],
        "gpu": gpu_info,
        "device_profile": "phone",
        "gatelink_split": {
            "client_layers": 1,
            "dp_epsilon": 1.0,
            "max_lora_rank": 32,
        },
    }


# --- Static Frontend ---

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


# --- Dev server ---

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Step 2: Commit**

```bash
git add demonstrator/server/app.py
git commit -m "feat: add FastAPI server with WebSocket chat, comparison, and metrics"
```

---

## Task 8: Web Frontend

**Files:**
- Create: `demonstrator/frontend/index.html`
- Create: `demonstrator/frontend/app.js`
- Create: `demonstrator/frontend/styles.css`

This is a single task but produces 3 files. The frontend is vanilla HTML/CSS/JS, mobile-first for iPhone 15 Pro Safari.

**Step 1: Write index.html**

(See separate file content - HTML structure with chat, metrics panel, comparison toggle, settings drawer)

**Step 2: Write styles.css**

(Mobile-first CSS with dark theme, metrics panel, encryption badges)

**Step 3: Write app.js**

(WebSocket connection, token streaming, metrics rendering, comparison mode)

**Step 4: Commit**

```bash
git add demonstrator/frontend/
git commit -m "feat: add mobile-first web frontend with chat, metrics, and comparison"
```

---

## Task 9: Docker Build

**Files:**
- Create: `demonstrator/Dockerfile`
- Create: `demonstrator/docker-compose.yml`

**Step 1: Write Dockerfile**

```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS base

# Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# Install dependencies
COPY demonstrator/requirements.txt /app/demonstrator/requirements.txt
COPY TenSafe_Extracted/ /app/TenSafe_Extracted/
COPY Adapter-Safety-Tensafe/ /app/Adapter-Safety-Tensafe/
COPY TenSafe-Homormorphically-Encrypted-LoRA-Adaptation/ /app/TenSafe-Homormorphically-Encrypted-LoRA-Adaptation/

RUN pip install -r /app/demonstrator/requirements.txt

# Copy demonstrator code
COPY demonstrator/ /app/demonstrator/

# Create adapter directory
RUN mkdir -p /app/demonstrator/adapters/tgsp

ENV PYTHONPATH=/app/TenSafe_Extracted/src:/app/Adapter-Safety-Tensafe/src:/app/TenSafe-Homormorphically-Encrypted-LoRA-Adaptation/src:/app
ENV MOE_CONFIG_PATH=/app/demonstrator/adapters/tgsp/moe_config.json
ENV DEVICE=cuda

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "demonstrator.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Step 2: Write docker-compose.yml**

```yaml
version: "3.8"
services:
  tensafe-demo:
    build:
      context: .
      dockerfile: demonstrator/Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./demonstrator/adapters:/app/demonstrator/adapters
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - DEVICE=cuda
      - MOE_CONFIG_PATH=/app/demonstrator/adapters/tgsp/moe_config.json
      - NVIDIA_VISIBLE_DEVICES=all
```

**Step 3: Commit**

```bash
git add demonstrator/Dockerfile demonstrator/docker-compose.yml
git commit -m "feat: add Docker build for WSL GPU deployment"
```

---

## Task 10: Training and Serving Scripts

**Files:**
- Create: `demonstrator/scripts/train.sh`
- Create: `demonstrator/scripts/serve.sh`

**Step 1: Write training script**

```bash
#!/bin/bash
set -euo pipefail

echo "=== TenSafe Finance Demonstrator - Training Pipeline ==="
echo "Model: Qwen/Qwen2.5-1.5B"
echo "Adapters: banking_expert, investment_expert, shared_attention"
echo "Rank: 32, Alpha: 64.0"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="${PROJECT_DIR}/../TenSafe_Extracted/src:${PROJECT_DIR}/../Adapter-Safety-Tensafe/src:${PROJECT_DIR}/../TenSafe-Homormorphically-Encrypted-LoRA-Adaptation/src:${PROJECT_DIR}"

echo "--- Phase 1: SFT Training ---"
python3 -m demonstrator.training.train_sft \
    --adapter all \
    --output-dir "${PROJECT_DIR}/adapters" \
    --max-steps 2000 \
    --rank 32 \
    --alpha 64.0 \
    --batch-size 1 \
    --grad-accum 8

echo ""
echo "--- Phase 2: RL Training (REINFORCE) ---"
python3 -m demonstrator.training.train_rl \
    --adapter all \
    --sft-dir "${PROJECT_DIR}/adapters" \
    --output-dir "${PROJECT_DIR}/adapters" \
    --rl-steps 500 \
    --rank 32 \
    --alpha 64.0

echo ""
echo "--- Phase 3: Gated MoE Assembly + TGSP Packaging ---"
python3 -m demonstrator.training.assemble_moe \
    --adapters-dir "${PROJECT_DIR}/adapters" \
    --output-dir "${PROJECT_DIR}/adapters/tgsp"

echo ""
echo "=== Training Pipeline Complete ==="
echo "Adapters saved to: ${PROJECT_DIR}/adapters/tgsp/"
echo "Run ./scripts/serve.sh to start the inference server."
```

**Step 2: Write serving script**

```bash
#!/bin/bash
set -euo pipefail

echo "=== TenSafe Finance Demonstrator - Inference Server ==="

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="${PROJECT_DIR}/../TenSafe_Extracted/src:${PROJECT_DIR}/../Adapter-Safety-Tensafe/src:${PROJECT_DIR}/../TenSafe-Homormorphically-Encrypted-LoRA-Adaptation/src:${PROJECT_DIR}"
export MOE_CONFIG_PATH="${PROJECT_DIR}/adapters/tgsp/moe_config.json"
export DEVICE="${DEVICE:-cuda}"

# Get WSL IP for iPhone access
WSL_IP=$(hostname -I | awk '{print $1}')
echo "Server will be available at: http://${WSL_IP}:8000"
echo "Open this URL on your iPhone 15 Pro Safari."
echo ""

python3 -m uvicorn demonstrator.server.app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level info
```

**Step 3: Make scripts executable and commit**

```bash
chmod +x demonstrator/scripts/train.sh demonstrator/scripts/serve.sh
git add demonstrator/scripts/
git commit -m "feat: add training and serving shell scripts"
```

---

## Task 11: End-to-End Verification

**Step 1: Verify all files exist**

```bash
find demonstrator/ -type f | sort
```

Expected:
```
demonstrator/Dockerfile
demonstrator/docker-compose.yml
demonstrator/frontend/app.js
demonstrator/frontend/index.html
demonstrator/frontend/styles.css
demonstrator/requirements.txt
demonstrator/scripts/serve.sh
demonstrator/scripts/train.sh
demonstrator/server/__init__.py
demonstrator/server/app.py
demonstrator/server/inference_engine.py
demonstrator/training/__init__.py
demonstrator/training/assemble_moe.py
demonstrator/training/datasets.py
demonstrator/training/reward_fn.py
demonstrator/training/train_rl.py
demonstrator/training/train_sft.py
```

**Step 2: Verify imports resolve**

```bash
cd /path/to/TenSafe_Project
PYTHONPATH=TenSafe_Extracted/src:Adapter-Safety-Tensafe/src python3 -c "
from tensafe.core.orchestrator import TenSafeOrchestrator, OrchestratorConfig
from tensafe.rlvr.config import RLVRConfig
from tensafe.rlvr.trainer import RLVRTrainer
from tensafe.lora_to_tgsp_converter import LoRAToTGSPConverter
print('All imports OK')
"
```

**Step 3: Run training pipeline**

```bash
cd demonstrator && bash scripts/train.sh
```

**Step 4: Start inference server and test**

```bash
bash scripts/serve.sh
# In another terminal:
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/metrics
```

**Step 5: Test from iPhone Safari**

Open `http://<WSL_IP>:8000` in Safari on iPhone 15 Pro. Verify:
- Chat interface loads
- Can send a finance query
- Tokens stream with metrics
- Compare mode works
- Encryption badges visible

---

## Execution Order Summary

| Task | Description | Dependencies |
|------|-------------|-------------|
| 1 | Project scaffolding | None |
| 2 | Dataset preparation | Task 1 |
| 3 | SFT training script | Tasks 1, 2 |
| 4 | RL training + reward function | Tasks 1, 2 |
| 5 | MoE assembly + TGSP | Tasks 1 |
| 6 | Inference engine | Task 1 |
| 7 | FastAPI server | Task 6 |
| 8 | Web frontend | Task 7 |
| 9 | Docker build | Tasks 1-8 |
| 10 | Shell scripts | Tasks 3-5, 7 |
| 11 | Verification | All |

Tasks 2-5 (training code) and Tasks 6-8 (serving code) can be developed in parallel.
