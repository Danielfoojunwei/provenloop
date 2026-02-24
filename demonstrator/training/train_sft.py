"""SFT training for finance LoRA adapters on Qwen 1.5B.

Uses TenSafeOrchestrator with real PyTorch training.
Three adapters: banking_expert, investment_expert, shared_attention.
All rank 32, alpha 64.0, max SIMD / GPU batching.

Crash-resilient: periodic checkpoints, resume support, gradient
checkpointing, GPU memory cleanup, emergency save on OOM/crash.
"""

import argparse
import gc
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Resolve project roots
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "TenSafe_Extracted" / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "Adapter-Safety-Tensafe" / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from tensafe.core.orchestrator import OrchestratorConfig, TenSafeOrchestrator

from demonstrator.training.data_loading import (
    load_banking_dataset,
    load_combined_finance_dataset,
    load_investment_dataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

QWEN_MODEL = "Qwen/Qwen2.5-1.5B"

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
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
        ],
        "description": "Shared attention LoRA (general financial reasoning)",
    },
}

# Checkpoint interval (in optim steps) — saves full state for resume
CHECKPOINT_INTERVAL = 250


def _gpu_cleanup():
    """Aggressively free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _is_adapter_complete(adapter_dir: Path) -> bool:
    """Check if an adapter has already been fully trained."""
    final_pt = adapter_dir / "adapter_final.pt"
    metrics_json = adapter_dir / "training_metrics.json"
    if not final_pt.exists() or not metrics_json.exists():
        return False
    try:
        with open(metrics_json) as f:
            meta = json.load(f)
        return meta.get("total_steps", 0) > 0
    except (json.JSONDecodeError, KeyError):
        return False


def _find_latest_checkpoint(adapter_dir: Path) -> Path | None:
    """Find the most recent resumable checkpoint in adapter_dir."""
    checkpoints = sorted(
        adapter_dir.glob("checkpoint_step_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    return checkpoints[-1] if checkpoints else None


def _save_checkpoint(
    orchestrator: TenSafeOrchestrator,
    adapter_dir: Path,
    tag,
    include_opt: bool = True,
):
    """Save orchestrator state to disk."""
    state = orchestrator.save_state(include_optimizer=include_opt)
    path = adapter_dir / f"adapter_{tag}.pt"
    # Write to temp file first, then rename — avoids corrupt checkpoint on crash
    tmp_path = path.with_suffix(".pt.tmp")
    with open(tmp_path, "wb") as f:
        f.write(state)
    tmp_path.rename(path)
    logger.info(f"Saved checkpoint: {path} ({len(state) / 1024 / 1024:.1f} MB)")


def _save_resume_checkpoint(
    orchestrator: TenSafeOrchestrator,
    adapter_dir: Path,
    global_step: int,
    optim_steps: int,
    metrics_log: list,
    best_loss: float,
    start_time: float,
):
    """Save a full resumable checkpoint with metadata."""
    # Save orchestrator state (includes optimizer for proper resume)
    state = orchestrator.save_state(include_optimizer=True)
    ckpt_path = adapter_dir / f"checkpoint_step_{global_step}.pt"
    tmp_path = ckpt_path.with_suffix(".pt.tmp")
    with open(tmp_path, "wb") as f:
        f.write(state)
    tmp_path.rename(ckpt_path)

    # Save resume metadata
    resume_meta = {
        "global_step": global_step,
        "optim_steps": optim_steps,
        "best_loss": best_loss,
        "elapsed_before_resume": time.time() - start_time,
        "metrics_log": metrics_log,
    }
    meta_path = adapter_dir / f"checkpoint_step_{global_step}.meta.json"
    with open(meta_path, "w") as f:
        json.dump(resume_meta, f, indent=2, default=str)

    logger.info(
        f"Resume checkpoint: step={global_step} optim={optim_steps} "
        f"loss={metrics_log[-1]['loss']:.4f} ({len(state) / 1024 / 1024:.1f} MB)"
    )

    # Keep only the 2 most recent checkpoints to save disk space
    all_ckpts = sorted(
        adapter_dir.glob("checkpoint_step_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    for old_ckpt in all_ckpts[:-2]:
        old_ckpt.unlink(missing_ok=True)
        old_meta = old_ckpt.with_suffix("").with_suffix(".meta.json")
        if old_meta.exists():
            old_meta.unlink(missing_ok=True)
        logger.info(f"Removed old checkpoint: {old_ckpt.name}")


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
    skip_completed: bool = True,
) -> dict:
    """Train a single LoRA adapter through TenSafeOrchestrator.

    Crash-resilient: saves resume checkpoints every CHECKPOINT_INTERVAL
    optim steps, supports resuming from latest checkpoint, enables
    gradient checkpointing for VRAM savings, and catches OOM to save
    an emergency checkpoint before exiting.
    """

    cfg = ADAPTER_CONFIGS[adapter_name]
    adapter_dir = Path(output_dir) / adapter_name
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 0. Skip if already complete
    # ------------------------------------------------------------------
    if skip_completed and _is_adapter_complete(adapter_dir):
        logger.info(f"[{adapter_name}] Already complete — skipping.")
        with open(adapter_dir / "training_metrics.json") as f:
            return json.load(f)

    logger.info(f"=== Training {adapter_name}: {cfg['description']} ===")

    # Check for resumable checkpoint
    resume_ckpt = _find_latest_checkpoint(adapter_dir)
    resume_meta = None
    if resume_ckpt:
        meta_path = resume_ckpt.with_suffix("").with_suffix(".meta.json")
        if meta_path.exists():
            with open(meta_path) as f:
                resume_meta = json.load(f)
            logger.info(
                f"Found resume checkpoint: {resume_ckpt.name} "
                f"(step={resume_meta['global_step']}, "
                f"optim={resume_meta['optim_steps']})"
            )

    # ------------------------------------------------------------------
    # 1. Load & tokenise dataset
    # ------------------------------------------------------------------
    dataset = cfg["dataset_fn"]()
    logger.info(f"Dataset size: {len(dataset)} samples")

    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(examples):
        tokens = tokenizer(
            examples["text"],
            max_length=max_seq_length,
            truncation=True,
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized.set_format("torch")

    dataloader = DataLoader(
        tokenized, batch_size=batch_size, shuffle=True, drop_last=True,
    )

    # ------------------------------------------------------------------
    # 2. Configure & initialise orchestrator
    # ------------------------------------------------------------------
    _gpu_cleanup()  # Clean slate before loading model

    orch_config = OrchestratorConfig(
        model_name_or_path=QWEN_MODEL,
        torch_dtype="float16",
        device_map="cuda",
        lora_enabled=True,
        lora_rank=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        lora_target_modules=cfg["target_modules"],
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

    # Enable gradient checkpointing — saves ~60% activation VRAM
    try:
        backend = orchestrator._ml_backend
        if hasattr(backend, "_model"):
            backend._model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled (saves ~60% activation VRAM)")
    except Exception as e:
        logger.warning(f"Could not enable gradient checkpointing: {e}")

    # ------------------------------------------------------------------
    # 2b. Resume from checkpoint if available
    # ------------------------------------------------------------------
    metrics_log = []
    global_step = 0
    optim_steps = 0
    best_loss = float("inf")
    elapsed_before = 0.0

    if resume_ckpt and resume_meta:
        try:
            with open(resume_ckpt, "rb") as f:
                state_bytes = f.read()
            orchestrator.load_state(state_bytes)
            global_step = resume_meta["global_step"]
            optim_steps = resume_meta["optim_steps"]
            best_loss = resume_meta["best_loss"]
            metrics_log = resume_meta.get("metrics_log", [])
            elapsed_before = resume_meta.get("elapsed_before_resume", 0.0)
            logger.info(
                f"Resumed from step {global_step} (optim={optim_steps}, "
                f"best_loss={best_loss:.4f})"
            )
            del state_bytes
            _gpu_cleanup()
        except Exception as e:
            logger.warning(f"Failed to resume from checkpoint: {e}. Starting fresh.")
            global_step = 0
            optim_steps = 0
            best_loss = float("inf")
            metrics_log = []

    # ------------------------------------------------------------------
    # 3. Training loop (crash-resilient)
    # ------------------------------------------------------------------
    sample_rate = batch_size / len(tokenized)
    start_time = time.time()

    logger.info(
        f"SFT config: steps={max_steps} bs={batch_size} accum={grad_accum} "
        f"lr={learning_rate} rank={rank} alpha={alpha} "
        f"(resuming from step {global_step})"
    )

    # Log GPU memory baseline
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024 / 1024
        props = torch.cuda.get_device_properties(0)
        total = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / 1024 / 1024
        logger.info(f"GPU memory: {alloc:.0f} / {total:.0f} MB used")

    try:
        for epoch in range(100):
            for batch in dataloader:
                if global_step >= max_steps:
                    break

                # Move batch tensors to GPU
                batch = {
                    k: v.to("cuda", non_blocking=True) if hasattr(v, "to") else v
                    for k, v in batch.items()
                }

                fb_metrics = orchestrator.forward_backward(batch, sample_rate)

                if (global_step + 1) % grad_accum == 0:
                    opt_metrics = orchestrator.optim_step(True, sample_rate)
                    optim_steps += 1

                    entry = {
                        "step": global_step,
                        "optim_step": optim_steps,
                        "loss": fb_metrics.loss,
                        "grad_norm": fb_metrics.grad_norm,
                        "lr": opt_metrics.learning_rate,
                        "tokens": fb_metrics.tokens_processed,
                        "time_ms": fb_metrics.time_ms,
                        "epsilon_spent": getattr(opt_metrics, "epsilon_spent", None),
                        "total_epsilon": getattr(opt_metrics, "total_epsilon", None),
                    }
                    metrics_log.append(entry)

                    if optim_steps % 25 == 0 or optim_steps == 1:
                        gpu_mb = (
                            torch.cuda.memory_allocated() / 1024 / 1024
                            if torch.cuda.is_available() else 0
                        )
                        logger.info(
                            f"[{adapter_name}] step={global_step}/{max_steps} "
                            f"optim={optim_steps} "
                            f"loss={fb_metrics.loss:.4f} "
                            f"gnorm={fb_metrics.grad_norm:.4f} "
                            f"eps={entry.get('total_epsilon', '-')} "
                            f"gpu={gpu_mb:.0f}MB"
                        )

                    if fb_metrics.loss < best_loss:
                        best_loss = fb_metrics.loss

                    # Periodic resume checkpoint
                    if optim_steps % CHECKPOINT_INTERVAL == 0:
                        _save_resume_checkpoint(
                            orchestrator, adapter_dir, global_step,
                            optim_steps, metrics_log, best_loss, start_time,
                        )
                        _gpu_cleanup()  # Free any fragmented memory

                global_step += 1

            if global_step >= max_steps:
                break

    except torch.cuda.OutOfMemoryError:
        logger.error(
            f"[{adapter_name}] CUDA OOM at step {global_step}! "
            "Saving emergency checkpoint..."
        )
        _gpu_cleanup()
        try:
            _save_resume_checkpoint(
                orchestrator, adapter_dir, global_step,
                optim_steps, metrics_log, best_loss, start_time,
            )
            logger.info("Emergency checkpoint saved. Re-run to resume.")
        except Exception as save_err:
            logger.error(f"Failed to save emergency checkpoint: {save_err}")
        raise

    except Exception as e:
        logger.error(
            f"[{adapter_name}] Error at step {global_step}: {e}. "
            "Saving emergency checkpoint..."
        )
        try:
            _save_resume_checkpoint(
                orchestrator, adapter_dir, global_step,
                optim_steps, metrics_log, best_loss, start_time,
            )
            logger.info("Emergency checkpoint saved. Re-run to resume.")
        except Exception as save_err:
            logger.error(f"Failed to save emergency checkpoint: {save_err}")
        raise

    # ------------------------------------------------------------------
    # 4. Save final adapter (LoRA only, no optimizer — smaller file)
    # ------------------------------------------------------------------
    _save_checkpoint(orchestrator, adapter_dir, "final", include_opt=False)

    elapsed = (time.time() - start_time) + elapsed_before
    summary = {
        "adapter_name": adapter_name,
        "model": QWEN_MODEL,
        "rank": rank,
        "alpha": alpha,
        "target_modules": cfg["target_modules"],
        "total_steps": global_step,
        "final_loss": metrics_log[-1]["loss"] if metrics_log else None,
        "best_loss": best_loss,
        "total_time_seconds": round(elapsed, 1),
        "dataset_size": len(dataset),
        "metrics": metrics_log,
    }

    with open(adapter_dir / "training_metrics.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(
        f"[{adapter_name}] DONE  steps={global_step} "
        f"best_loss={best_loss:.4f} time={elapsed:.0f}s"
    )

    # ------------------------------------------------------------------
    # 5. Cleanup GPU memory for next adapter
    # ------------------------------------------------------------------
    del orchestrator
    _gpu_cleanup()
    logger.info(f"[{adapter_name}] GPU memory released.")

    # Clean up resume checkpoints now that final is saved
    for ckpt in adapter_dir.glob("checkpoint_step_*"):
        ckpt.unlink(missing_ok=True)

    return summary


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Train finance LoRA adapters (SFT)")
    parser.add_argument(
        "--adapter",
        choices=list(ADAPTER_CONFIGS) + ["all"],
        default="all",
    )
    parser.add_argument("--output-dir", default="demonstrator/adapters")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=64.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument(
        "--no-skip-completed", action="store_true",
        help="Re-train adapters even if adapter_final.pt exists",
    )
    args = parser.parse_args()

    names = list(ADAPTER_CONFIGS) if args.adapter == "all" else [args.adapter]
    skip = not args.no_skip_completed

    results = {}
    for name in names:
        results[name] = train_adapter(
            adapter_name=name,
            output_dir=args.output_dir,
            max_steps=args.max_steps,
            rank=args.rank,
            alpha=args.alpha,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            learning_rate=args.lr,
            max_seq_length=args.max_seq_length,
            skip_completed=skip,
        )

    out_path = Path(args.output_dir) / "sft_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Combined SFT results saved to {out_path}")


if __name__ == "__main__":
    main()
