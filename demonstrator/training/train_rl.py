"""REINFORCE RL training on top of SFT checkpoints.

Loads the SFT-trained adapter, then applies REINFORCE with the
rule-based finance_reward function to sharpen domain quality.

Crash-resilient: skip completed, resume from checkpoint, emergency save,
GPU memory cleanup between adapters.
"""

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "TenSafe_Extracted" / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "Adapter-Safety-Tensafe" / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from tensafe.core.orchestrator import OrchestratorConfig, TenSafeOrchestrator
from tensafe.rlvr.config import RLVRConfig
from tensafe.rlvr.trainer import RLVRTrainer

from demonstrator.training.data_loading import load_rl_prompts
from demonstrator.training.reward_fn import finance_reward

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

QWEN_MODEL = "Qwen/Qwen2.5-1.5B"

_TARGET_MODULES = {
    "banking_expert": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "investment_expert": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "shared_attention": [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
    ],
}

CHECKPOINT_INTERVAL = 100


def _gpu_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _is_rl_complete(rl_dir: Path) -> bool:
    final = rl_dir / "adapter_rl_final.pt"
    metrics = rl_dir / "rl_metrics.json"
    if not final.exists() or not metrics.exists():
        return False
    try:
        with open(metrics) as f:
            meta = json.load(f)
        return meta.get("rl_steps", 0) > 0
    except (json.JSONDecodeError, KeyError):
        return False


def train_rl(
    adapter_name: str,
    sft_dir: str,
    output_dir: str,
    rl_steps: int = 500,
    rank: int = 30,
    alpha: float = 64.0,
    skip_completed: bool = True,
) -> dict:
    """Run REINFORCE on a single adapter (crash-resilient)."""

    rl_dir = Path(output_dir) / f"{adapter_name}_rl"
    rl_dir.mkdir(parents=True, exist_ok=True)

    # Skip if complete
    if skip_completed and _is_rl_complete(rl_dir):
        logger.info(f"[{adapter_name}] RL already complete — skipping.")
        with open(rl_dir / "rl_metrics.json") as f:
            return json.load(f)

    logger.info(f"=== RL training: {adapter_name} ===")

    sft_path = Path(sft_dir) / adapter_name / "adapter_final.pt"
    if not sft_path.exists():
        raise FileNotFoundError(f"SFT checkpoint missing: {sft_path}")

    _gpu_cleanup()

    # Orchestrator (lower LR for RL)
    orch_config = OrchestratorConfig(
        model_name_or_path=QWEN_MODEL,
        torch_dtype="float16",
        device_map="cuda",
        lora_enabled=True,
        lora_rank=rank,
        lora_alpha=alpha,
        lora_target_modules=_TARGET_MODULES[adapter_name],
        learning_rate=1e-5,
        dp_enabled=True,
        dp_noise_multiplier=1.0,
        dp_target_epsilon=8.0,
    )
    orchestrator = TenSafeOrchestrator(
        config=orch_config,
        orchestrator_id=f"rl-{adapter_name}",
    )
    orchestrator.initialize()

    # Enable gradient checkpointing
    try:
        backend = orchestrator._ml_backend
        if hasattr(backend, "_model"):
            backend._model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
    except Exception as e:
        logger.warning(f"Could not enable gradient checkpointing: {e}")

    with open(sft_path, "rb") as f:
        orchestrator.load_state(f.read())
    logger.info(f"Loaded SFT weights from {sft_path}")

    # RLVR config
    rlvr_config = RLVRConfig(
        algorithm="reinforce",
        rollout_batch_size=1,
        max_new_tokens=64,
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
        gradient_accumulation_steps=1,
    )

    trainer = RLVRTrainer(
        training_client=orchestrator,
        config=rlvr_config,
        reward_fn=finance_reward,
    )

    prompts = load_rl_prompts(n_prompts=rl_steps * rlvr_config.rollout_batch_size)

    metrics_log = []
    t0 = time.time()
    pidx = 0

    try:
        for step in range(rl_steps):
            batch = prompts[pidx: pidx + rlvr_config.rollout_batch_size]
            pidx += rlvr_config.rollout_batch_size
            if pidx >= len(prompts):
                pidx = 0

            m = trainer.step(batch)

            # Aggressively free GPU memory between steps
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            entry = {
                "step": step,
                "mean_reward": m.mean_reward,
                "policy_loss": m.policy_loss,
                "entropy": m.entropy,
                "grad_norm": m.grad_norm,
            }
            metrics_log.append(entry)

            if step % 25 == 0:
                logger.info(
                    f"[{adapter_name} RL] step={step}/{rl_steps} "
                    f"reward={m.mean_reward:.3f} loss={m.policy_loss:.4f}"
                )

            if step > 0 and step % CHECKPOINT_INTERVAL == 0:
                trainer.save_checkpoint(str(rl_dir / f"rl_ckpt_{step}.json"))
                # Also save orchestrator state for resume
                state = orchestrator.save_state(include_optimizer=True)
                ckpt_path = rl_dir / f"rl_state_{step}.pt"
                tmp = ckpt_path.with_suffix(".pt.tmp")
                with open(tmp, "wb") as f:
                    f.write(state)
                tmp.rename(ckpt_path)
                del state
                _gpu_cleanup()

    except torch.cuda.OutOfMemoryError:
        logger.error(f"[{adapter_name}] CUDA OOM at RL step {step}!")
        _gpu_cleanup()
        raise

    except Exception as e:
        logger.error(f"[{adapter_name}] RL error at step {step}: {e}")
        raise

    # Save final
    final = orchestrator.save_state(include_optimizer=False)
    final_path = rl_dir / "adapter_rl_final.pt"
    tmp = final_path.with_suffix(".pt.tmp")
    with open(tmp, "wb") as f:
        f.write(final)
    tmp.rename(final_path)

    elapsed = time.time() - t0
    summary = {
        "adapter_name": adapter_name,
        "rl_steps": rl_steps,
        "final_reward": metrics_log[-1]["mean_reward"] if metrics_log else None,
        "total_time_seconds": round(elapsed, 1),
        "metrics": metrics_log,
    }
    with open(rl_dir / "rl_metrics.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"[{adapter_name}] RL done. time={elapsed:.0f}s")

    # Cleanup GPU for next adapter
    del orchestrator
    _gpu_cleanup()

    # Remove intermediate checkpoints
    for ckpt in rl_dir.glob("rl_state_*.pt"):
        ckpt.unlink(missing_ok=True)
    for ckpt in rl_dir.glob("rl_ckpt_*.json"):
        ckpt.unlink(missing_ok=True)

    return summary


# ======================================================================
# CLI
# ======================================================================

def main():
    p = argparse.ArgumentParser(description="REINFORCE RL training")
    p.add_argument("--adapter", choices=list(_TARGET_MODULES) + ["all"], default="all")
    p.add_argument("--sft-dir", default="demonstrator/adapters")
    p.add_argument("--output-dir", default="demonstrator/adapters")
    p.add_argument("--rl-steps", type=int, default=500)
    p.add_argument("--rank", type=int, default=30)
    p.add_argument("--alpha", type=float, default=64.0)
    p.add_argument("--no-skip-completed", action="store_true")
    args = p.parse_args()

    names = list(_TARGET_MODULES) if args.adapter == "all" else [args.adapter]
    skip = not args.no_skip_completed
    for name in names:
        train_rl(name, args.sft_dir, args.output_dir, args.rl_steps, args.rank, args.alpha, skip)


if __name__ == "__main__":
    main()
