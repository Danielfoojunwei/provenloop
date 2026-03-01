"""Assemble gated MoE from trained adapters and package as TGSP.

Picks best checkpoint per adapter (RL > SFT), extracts LoRA-only
weights (much smaller than full state), converts to TGSP,
writes the moe_config.json consumed by the inference engine.
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "TenSafe_Extracted" / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from tensafe.lora_to_tgsp_converter import LoRAToTGSPConverter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _extract_lora_to_peft_dir(full_state_path: Path, peft_dir: Path) -> bool:
    """Extract LoRA weights from full orchestrator state into PEFT directory format.

    The TGSP converter natively supports PEFT directories with
    adapter_config.json + adapter_model.bin.
    """
    import io
    import json as _json

    logger.info(f"Extracting LoRA weights from {full_state_path.name}...")

    with open(full_state_path, "rb") as f:
        state_bytes = f.read()

    buffer = io.BytesIO(state_bytes)
    state = torch.load(buffer, map_location="cpu", weights_only=True)
    del state_bytes  # free memory

    model_state = state.get("model_state_dict", {})

    # Extract only LoRA parameters
    lora_state = {}
    target_modules = set()
    rank = 30

    for key, tensor in model_state.items():
        if "lora_" in key.lower():
            lora_state[key] = tensor
            # Infer rank from lora_A shape
            if "lora_A" in key and hasattr(tensor, "shape"):
                rank = tensor.shape[0]
            # Extract module name
            for mod in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"]:
                if mod in key:
                    target_modules.add(mod)

    if not lora_state:
        logger.warning(f"No LoRA keys found in {full_state_path.name}")
        return False

    # Create PEFT-compatible directory
    peft_dir.mkdir(parents=True, exist_ok=True)

    # Save adapter_model.bin (flat dict of LoRA tensors)
    torch.save(lora_state, peft_dir / "adapter_model.bin")

    # Save adapter_config.json
    config = {
        "peft_type": "LORA",
        "base_model_name_or_path": "Qwen/Qwen2.5-1.5B",
        "r": rank,
        "lora_alpha": 64,  # M6: use explicit alpha (rank*2 gave 60 for rank=30)
        "target_modules": sorted(target_modules),
        "lora_dropout": 0.0,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    with open(peft_dir / "adapter_config.json", "w") as f:
        _json.dump(config, f, indent=2)

    bin_size = (peft_dir / "adapter_model.bin").stat().st_size
    logger.info(
        f"  Extracted {len(lora_state)} LoRA params to PEFT dir "
        f"({bin_size / 1024 / 1024:.1f} MB, rank={rank}, "
        f"modules={sorted(target_modules)})"
    )
    return True


def assemble_and_package(adapters_dir: str, output_dir: str) -> dict:
    adapters_path = Path(adapters_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    converter = LoRAToTGSPConverter(auto_generate_keys=True)
    results = {}

    for name in ("banking_expert", "investment_expert", "shared_attention"):
        # Prefer RL checkpoint over SFT
        rl = adapters_path / f"{name}_rl" / "adapter_rl_final.pt"
        sft = adapters_path / name / "adapter_final.pt"
        src = rl if rl.exists() else sft

        if not src.exists():
            logger.error(f"No checkpoint for {name} at {sft} or {rl}")
            continue

        # Extract LoRA weights into PEFT directory format (converter has 1GB limit)
        peft_dir = out / f"{name}_peft"
        if not _extract_lora_to_peft_dir(src, peft_dir):
            logger.error(f"Failed to extract LoRA weights for {name}")
            continue

        logger.info(f"Converting {name} from PEFT dir {peft_dir}")
        tgsp_out = out / f"{name}.tgsp"

        try:
            result = converter.convert(
                input_path=str(peft_dir),
                output_path=str(tgsp_out),
                model_name=f"tensafe-finance-{name}",
                model_version="1.0.0",
                validate=True,
                metadata={
                    "domain": "finance",
                    "expert_type": name,
                    "rank": 30,
                    "alpha": 64.0,
                    "training": "sft+reinforce",
                },
            )

            results[name] = {
                "success": result.success,
                "tgsp_path": str(tgsp_out),
                "adapter_id": result.adapter_id,
                "input_bytes": result.input_size_bytes,
                "output_bytes": result.output_size_bytes,
                "manifest_hash": result.manifest_hash,
                "conversion_ms": result.conversion_time_ms,
            }
            status = "OK" if result.success else "FAIL"
            logger.info(f"  {name}: {status}")
        except Exception as e:
            logger.error(f"  {name}: conversion error: {e}")
            results[name] = {"success": False, "error": str(e)}

        # Clean up intermediate PEFT directory
        shutil.rmtree(peft_dir, ignore_errors=True)

    # ------------------------------------------------------------------ #
    # MoE config consumed by the inference engine
    # ------------------------------------------------------------------ #
    moe_config = {
        "model": "Qwen/Qwen2.5-1.5B",
        "experts": {
            "banking_expert": {
                "tgsp_path": str(out / "banking_expert.tgsp"),
                "checkpoint_path": str(
                    adapters_path / "banking_expert_rl" / "adapter_rl_final.pt"
                ),
                "gate_type": "step",
                "gate_keywords": [
                    "bank", "deposit", "loan", "mortgage", "credit",
                    "savings", "checking", "interest rate", "refinance",
                ],
                "always_active": False,
            },
            "investment_expert": {
                "tgsp_path": str(out / "investment_expert.tgsp"),
                "checkpoint_path": str(
                    adapters_path / "investment_expert_rl" / "adapter_rl_final.pt"
                ),
                "gate_type": "step",
                "gate_keywords": [
                    "invest", "portfolio", "stock", "bond", "etf",
                    "dividend", "market", "allocation", "risk",
                ],
                "always_active": False,
            },
            "shared_attention": {
                "tgsp_path": str(out / "shared_attention.tgsp"),
                "checkpoint_path": str(
                    adapters_path / "shared_attention_rl" / "adapter_rl_final.pt"
                ),
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
            "max_epsilon": 10.0,
            "max_lora_rank": 30,
        },
    }

    with open(out / "moe_config.json", "w") as f:
        json.dump(moe_config, f, indent=2)

    with open(out / "tgsp_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"MoE config -> {out / 'moe_config.json'}")
    return results


def main():
    p = argparse.ArgumentParser(description="Assemble MoE + TGSP")
    p.add_argument("--adapters-dir", default="demonstrator/adapters")
    p.add_argument("--output-dir", default="demonstrator/adapters/tgsp")
    assemble_and_package(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
