#!/usr/bin/env python3
"""TenSafe CLI -- command-line interface for TGSP adapter management.

Commands:
    tensafe init       Scaffold a new adapter project
    tensafe train      Train adapter from data
    tensafe screen     Run RVUv2 safety screening
    tensafe pack       Package as signed TGSP
    tensafe verify     Verify TGSP integrity
    tensafe export     Export to plain LoRA format
    tensafe import     Import existing LoRA to TGSP
    tensafe validate   Run TenSafe validation suite
    tensafe publish    Publish to marketplace

Usage:
    python -m tensafe.cli.main train --data ./corpus --rank 30 --alpha 64
    python -m tensafe.cli.main verify my_adapter.tgsp
    python -m tensafe.cli.main export my_adapter.tgsp --format safetensors
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Ensure the project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tensafe")


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_init(args):
    """Scaffold a new adapter project."""
    project_dir = Path(args.output_dir) / args.name if args.output_dir != "." else Path(args.name)
    project_dir.mkdir(parents=True, exist_ok=True)

    # Create project structure
    (project_dir / "data").mkdir(exist_ok=True)
    (project_dir / "checkpoints").mkdir(exist_ok=True)
    (project_dir / "output").mkdir(exist_ok=True)

    # Write project config
    config = {
        "name": args.name,
        "domain": args.domain,
        "base_model": getattr(args, "base_model", "Qwen/Qwen2.5-1.5B"),
        "lora_config": {
            "rank": getattr(args, "rank", 30),
            "alpha": getattr(args, "alpha", 64),
            "dropout": 0.0,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            "bias": "none",
            "task_type": "CAUSAL_LM",
        },
        "training": {
            "max_steps": 2000,
            "batch_size": 1,
            "grad_accum": 8,
            "learning_rate": 1e-4,
        },
        "dp": {
            "noise_multiplier": 1.0,
            "target_epsilon": 8.0,
            "target_delta": 1e-5,
        },
        "creator": getattr(args, "creator", "") or "",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    config_path = project_dir / "tensafe_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Write SKILL.md template
    rank = getattr(args, "rank", 30)
    alpha = getattr(args, "alpha", 64)
    skill_md = f"""# {args.name}

## What I Do
[Describe what this adapter specializes in]

## When to Use Me
Load this adapter when the task involves:
- [trigger keyword 1]
- [trigger keyword 2]

## My Capabilities
- [capability 1]
- [capability 2]

## Input / Output
- **Input:** [expected input format]
- **Output:** [expected output format]

## Compose With
- [other adapter names this works well with]

## Compliance
- [HIPAA, SOC 2, etc. if applicable]

## Creator
[Your name and organization]
"""
    with open(project_dir / "SKILL.md", "w") as f:
        f.write(skill_md)

    print(f"Adapter project scaffolded at: {project_dir}/")
    print(f"  tensafe_config.json -- adapter configuration (rank={rank}, alpha={alpha})")
    print(f"  SKILL.md            -- embedded skill description (EDIT THIS)")
    print(f"  data/               -- put training data here")
    print(f"  checkpoints/        -- training checkpoints")
    print(f"  output/             -- packaged TGSP files")
    print()
    print(f"Next: add data to {project_dir}/data/ then run:")
    print(f"  tensafe train --data {project_dir}/data/ --rank {rank} --alpha {alpha}")


def cmd_train(args):
    """Train adapter from data."""
    from tensafe.skills.tg_tinker.tinker import TGTinker

    tinker = TGTinker(
        base_model=getattr(args, "base_model", "Qwen/Qwen2.5-1.5B"),
        output_dir=args.output_dir,
        default_rank=args.rank,
        default_alpha=args.alpha,
    )

    result = tinker.create_adapter(
        data_path=args.data,
        domain=getattr(args, "domain", None),
        name=getattr(args, "name", None),
        signing_key=getattr(args, "signing_key", None),
        export_format=getattr(args, "export_format", None),
        rank=args.rank,
        alpha=args.alpha,
        max_steps=getattr(args, "max_steps", 2000),
        creator=getattr(args, "creator", None) or "TG Tinker CLI",
        description=getattr(args, "description", None) or "",
        tags=args.tags.split(",") if getattr(args, "tags", None) else None,
    )

    # Write result summary
    result_path = Path(args.output_dir) / "train_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        serializable = {k: v for k, v in result.items()}
        json.dump(serializable, f, indent=2, default=str)

    if result.get("success"):
        print()
        print("Adapter created successfully!")
        print(f"  TGSP: {result.get('tgsp_path', 'N/A')}")
        print(f"  Domain: {result.get('domain', 'N/A')}")
        qa = result.get("evaluation", {}).get("qa_verify_score", "N/A")
        print(f"  Quality (qa_verify): {qa}")
        print(f"  Steps completed: {result.get('steps_completed', [])}")
        if result.get("export_path"):
            print(f"  Export: {result['export_path']}")
        return 0
    else:
        print()
        print("Adapter creation FAILED.")
        for err in result.get("errors", []):
            print(f"  ERROR: {err}")
        return 1


def cmd_screen(args):
    """Run RVUv2 safety screening."""
    from tensafe.skills.tg_tinker.tinker import _stub_screen_rvu

    print(f"Screening: {args.path}")
    result = _stub_screen_rvu(args.path)

    print()
    print("RVUv2 Safety Screening Results:")
    print(f"  Overall: {'PASS' if result.passed else 'FAIL'}")
    print(f"  Toxicity: {result.toxicity_score:.4f}")
    print(f"  Bias: {result.bias_score:.4f}")
    print(f"  IP/PII: {result.ip_score:.4f}")

    if result.warnings:
        print()
        print("  Warnings:")
        for w in result.warnings:
            print(f"    - {w}")

    for layer_name, layer_result in result.layer_results.items():
        status = "PASS" if layer_result.get("passed") else "FAIL"
        score = layer_result.get("score", "N/A")
        threshold = layer_result.get("threshold", "N/A")
        print(f"  {layer_name}: {status} (score={score}, threshold={threshold})")

    return 0 if result.passed else 1


def cmd_pack(args):
    """Package adapter as signed TGSP."""
    import uuid
    from tensafe.skills.tg_tinker.tinker import package_tgsp, AdapterManifest
    from tensafe.skills.tg_tinker.data_analyzer import LoraConfig

    checkpoint_dir = getattr(args, "checkpoint_dir", None) or "."
    if not Path(checkpoint_dir).exists():
        print(f"ERROR: Checkpoint directory not found: {checkpoint_dir}")
        return 1

    # Load training metrics if available
    metrics_path = Path(checkpoint_dir) / "training_metrics.json"
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    # Load project config if available
    config = {}
    for config_name in ("tensafe_config.json", "tensafe.json"):
        config_path = Path(checkpoint_dir).parent / config_name
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            break

    name = getattr(args, "name", None) or config.get("name", Path(checkpoint_dir).stem)
    domain = getattr(args, "domain", None) or config.get("domain", "general")
    creator = getattr(args, "creator", None) or config.get("creator", "unknown")
    rank = getattr(args, "rank", 30)
    alpha = getattr(args, "alpha", 64)

    lora_config_dict = config.get("lora_config", {
        "rank": rank,
        "alpha": alpha,
        "dropout": 0.0,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "bias": "none",
        "task_type": "CAUSAL_LM",
    })

    manifest_data = AdapterManifest(
        adapter_id=str(uuid.uuid4()),
        name=name,
        domain=domain,
        base_model=config.get("base_model", "Qwen/Qwen2.5-1.5B"),
        rank=lora_config_dict.get("rank", rank),
        alpha=lora_config_dict.get("alpha", alpha),
        creator=creator,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        qa_verify_score=metrics.get("qa_verify_score", 0.0),
        rvu_passed=metrics.get("rvu_passed", False),
        dp_epsilon=metrics.get("epsilon_spent", 0.0),
        dp_budget_ok=metrics.get("dp_budget_ok", False),
        training_steps=metrics.get("total_steps", 0),
        training_loss=metrics.get("final_loss", 0.0),
        lora_config=lora_config_dict,
        description=getattr(args, "description", None) or f"{domain} adapter",
        tags=[domain, "lora", "tgsp"],
    )

    signing_key = getattr(args, "signing_key", None) if getattr(args, "sign", False) else None

    output_path = getattr(args, "output", None) or str(
        Path(checkpoint_dir).parent / f"{name}.tgsp"
    )

    # Embed skill_doc if provided
    skill_doc_path = getattr(args, "skill_doc", None)
    if skill_doc_path and Path(skill_doc_path).exists():
        with open(skill_doc_path) as f:
            manifest_data.description = f.read()

    tgsp_path = package_tgsp(
        checkpoint_dir=checkpoint_dir,
        manifest_data=manifest_data,
        signing_key=signing_key,
        output_path=output_path,
    )

    print()
    print(f"TGSP package created: {tgsp_path}")
    print(f"  Name: {name}")
    print(f"  Domain: {domain}")
    print(f"  Signed: {'yes' if signing_key else 'no'}")
    file_size = Path(tgsp_path).stat().st_size
    print(f"  Size: {file_size / 1024:.1f} KB")

    return 0


def cmd_verify(args):
    """Verify TGSP integrity."""
    from tensafe.skills.tg_tinker.export import verify_tgsp

    tgsp_path = args.file
    if not Path(tgsp_path).exists():
        print(f"ERROR: File not found: {tgsp_path}")
        return 1

    valid, details = verify_tgsp(tgsp_path)

    print()
    print(f"TGSP Verification: {tgsp_path}")
    print(f"  Valid: {'YES' if valid else 'NO'}")
    print(f"  Magic bytes: {'OK' if details.get('magic_ok') else 'FAIL'}")
    print(f"  Manifest: {'OK' if details.get('manifest_ok') else 'FAIL'}")
    print(f"  Hash: {'OK' if details.get('hash_ok') else 'FAIL'}")
    print(f"  Signed: {'yes' if details.get('signed') else 'no'}")
    print(f"  Creator: {details.get('creator', 'unknown')}")
    print(f"  Format version: {details.get('format_version', 'unknown')}")

    if details.get("errors"):
        print()
        print("  Errors:")
        for err in details["errors"]:
            print(f"    - {err}")

    if valid and details.get("manifest"):
        manifest = details["manifest"]
        print()
        print(f"  Adapter ID: {manifest.get('adapter_id', 'N/A')}")
        print(f"  Model: {manifest.get('model_name', 'N/A')}")
        print(f"  Base model: {manifest.get('base_model', 'N/A')}")
        print(f"  Rank: {manifest.get('rank', 'N/A')}")
        print(f"  Alpha: {manifest.get('alpha', 'N/A')}")

        meta = manifest.get("metadata", {})
        print(f"  Domain: {meta.get('domain', 'N/A')}")

        screening = manifest.get("screening", {})
        print(f"  QA verify: {screening.get('qa_verify_score', 'N/A')}")
        rvu = screening.get("rvu_v2_passed")
        print(f"  RVUv2: {'PASS' if rvu else ('FAIL' if rvu is False else 'N/A')}")

    return 0 if valid else 1


def cmd_export(args):
    """Export TGSP to plain LoRA format."""
    from tensafe.skills.tg_tinker.export import (
        export_tgsp_to_safetensors,
        export_tgsp_to_pytorch,
        export_tgsp_to_gguf,
    )

    tgsp_path = args.file
    fmt = args.format
    output_dir = getattr(args, "output_dir", None) or str(
        Path(tgsp_path).parent / "export"
    )

    if not Path(tgsp_path).exists():
        print(f"ERROR: File not found: {tgsp_path}")
        return 1

    print(f"Exporting {tgsp_path} to {fmt} format...")

    try:
        if fmt == "safetensors":
            result = export_tgsp_to_safetensors(tgsp_path, output_dir)
        elif fmt in ("pytorch", "pt"):
            out_path = str(Path(output_dir) / "adapter_model.pt")
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            result = export_tgsp_to_pytorch(tgsp_path, out_path)
        elif fmt == "gguf":
            out_path = str(Path(output_dir) / "adapter.gguf")
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            result = export_tgsp_to_gguf(tgsp_path, out_path)
        else:
            print(f"ERROR: Unsupported format: {fmt}")
            print("Supported formats: safetensors, pytorch, pt, gguf")
            return 1

        print()
        print(f"Export complete: {result}")
        return 0

    except Exception as e:
        print(f"ERROR: Export failed: {e}")
        return 1


def cmd_import(args):
    """Import existing LoRA to TGSP."""
    from tensafe.skills.tg_tinker.export import import_lora_to_tgsp

    lora_dir = args.lora_dir
    if not Path(lora_dir).exists():
        print(f"ERROR: LoRA directory not found: {lora_dir}")
        return 1

    signing_key = getattr(args, "signing_key", None) if getattr(args, "sign", False) else None

    print(f"Importing LoRA from {lora_dir}...")

    try:
        tgsp_path = import_lora_to_tgsp(
            lora_dir=lora_dir,
            name=args.name,
            domain=getattr(args, "domain", None) or "general",
            signing_key=signing_key,
            output_path=getattr(args, "output", None),
            creator=getattr(args, "creator", None) or "TG Tinker CLI",
            description=getattr(args, "description", None) or "",
            base_model=getattr(args, "base_model", None) or "",
            tags=args.tags.split(",") if getattr(args, "tags", None) else None,
        )

        print()
        print(f"TGSP created: {tgsp_path}")
        file_size = Path(tgsp_path).stat().st_size
        print(f"  Size: {file_size / 1024:.1f} KB")

        # Optionally run screening
        if getattr(args, "screen", False):
            from tensafe.skills.tg_tinker.tinker import _stub_screen_rvu
            print()
            print("Running RVUv2 screening...")
            screen_result = _stub_screen_rvu(lora_dir)
            print(f"  Screening: {'PASS' if screen_result.passed else 'FAIL'}")

        return 0

    except Exception as e:
        print(f"ERROR: Import failed: {e}")
        return 1


def cmd_validate(args):
    """Run TenSafe validation suite (for marketplace listing)."""
    from tensafe.skills.tg_tinker.tinker import TGTinker

    tgsp_path = args.file
    if not Path(tgsp_path).exists():
        print(f"ERROR: File not found: {tgsp_path}")
        return 1

    tinker = TGTinker()
    result = tinker.validate(tgsp_path)

    print()
    print(f"TenSafe Validation Report: {tgsp_path}")
    print("=" * 60)

    for check_name, check_result in result.get("checks", {}).items():
        passed = check_result.get("passed", check_result.get("signed", False))
        status = "PASS" if passed else "FAIL"
        details_str = ""
        for k, v in check_result.items():
            if k not in ("passed", "signed"):
                details_str += f" {k}={v}"
        print(f"  [{status}] {check_name}{details_str}")

    print()
    mkt = "YES" if result.get("marketplace_ready") else "NO"
    print(f"  Marketplace ready: {mkt}")

    if result.get("warnings"):
        print()
        print("  Warnings:")
        for w in result["warnings"]:
            print(f"    - {w}")

    if result.get("errors"):
        print()
        print("  Errors:")
        for e in result["errors"]:
            print(f"    - {e}")

    print("=" * 60)

    return 0 if result.get("valid") else 1


def cmd_publish(args):
    """Publish to marketplace."""
    from tensafe.skills.tg_tinker.tinker import TGTinker
    from tensafe.skills.tg_tinker.export import verify_tgsp

    tgsp_path = args.file
    if not Path(tgsp_path).exists():
        print(f"ERROR: File not found: {tgsp_path}")
        return 1

    # Validate first
    print(f"Validating {tgsp_path} for marketplace...")
    tinker = TGTinker()
    validation = tinker.validate(tgsp_path)

    if not validation.get("marketplace_ready"):
        print()
        print("Adapter is NOT marketplace ready.")
        for w in validation.get("warnings", []):
            print(f"  WARNING: {w}")
        for e in validation.get("errors", []):
            print(f"  ERROR: {e}")
        print()
        print("Fix the issues above and re-validate with: tensafe validate")
        return 1

    valid, details = verify_tgsp(tgsp_path)
    manifest = details.get("manifest", {})

    print()
    print("Publishing to marketplace...")
    print(f"  Adapter: {manifest.get('model_name', 'unknown')}")
    print(f"  Domain: {manifest.get('metadata', {}).get('domain', 'unknown')}")
    print(f"  Creator: {manifest.get('creator', 'unknown')}")

    if getattr(args, "marketplace", False):
        print(f"  Target: TenSafe Marketplace")
        print(f"  Marketplace fee: 0% (creators keep 100%)")
    else:
        print(f"  Target: local registry")

    # In production, this would upload to the marketplace API
    file_size = Path(tgsp_path).stat().st_size
    print()
    print(f"[STUB] Marketplace publish would upload {tgsp_path}")
    print(f"  File size: {file_size / 1024:.1f} KB")
    print(f"  Adapter ID: {manifest.get('adapter_id', 'N/A')}")

    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser():
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="tensafe",
        description="TenSafe CLI -- create, validate, and manage TGSP LoRA adapters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  tensafe init --domain finance --name my_adapter\n"
            "  tensafe train --data ./corpus --rank 30 --alpha 64\n"
            "  tensafe verify my_adapter.tgsp\n"
            "  tensafe export my_adapter.tgsp --format safetensors\n"
            "  tensafe import ./lora_dir --sign --name my_adapter\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # -- init --
    p_init = subparsers.add_parser("init", help="Scaffold a new adapter project")
    p_init.add_argument("--domain", required=True,
                        help="Domain (healthcare, finance, legal, engineering, general)")
    p_init.add_argument("--name", required=True, help="Adapter project name")
    p_init.add_argument("--output-dir", default=".",
                        help="Parent directory for project (default: .)")
    p_init.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B", help="Base model")
    p_init.add_argument("--rank", type=int, default=30, help="LoRA rank (default: 30)")
    p_init.add_argument("--alpha", type=int, default=64, help="LoRA alpha (default: 64)")
    p_init.add_argument("--creator", default=None, help="Creator name/email")

    # -- train --
    p_train = subparsers.add_parser("train", help="Train adapter from data")
    p_train.add_argument("--data", required=True, help="Path to data directory or file")
    p_train.add_argument("--domain", default=None,
                         help="Domain (auto-detected if omitted)")
    p_train.add_argument("--name", default=None,
                         help="Adapter name (auto-generated if omitted)")
    p_train.add_argument("--rank", type=int, default=30,
                         help="LoRA rank (default: 30)")
    p_train.add_argument("--alpha", type=int, default=64,
                         help="LoRA alpha (default: 64)")
    p_train.add_argument("--max-steps", type=int, default=2000,
                         help="Max training steps (default: 2000)")
    p_train.add_argument("--output-dir", default="./tg_tinker_output",
                         help="Output directory")
    p_train.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B",
                         help="Base model")
    p_train.add_argument("--signing-key", default=None,
                         help="Hex signing key for TGSP")
    p_train.add_argument("--export-format", default=None,
                         help="Export format: safetensors, pytorch, gguf")
    p_train.add_argument("--creator", default=None, help="Creator name/email")
    p_train.add_argument("--description", default=None, help="Adapter description")
    p_train.add_argument("--tags", default=None, help="Comma-separated tags")

    # -- screen --
    p_screen = subparsers.add_parser("screen", help="Run RVUv2 safety screening")
    p_screen.add_argument("path", help="Path to checkpoint directory or TGSP file")

    # -- pack --
    p_pack = subparsers.add_parser("pack", help="Package as signed TGSP")
    p_pack.add_argument("checkpoint_dir", nargs="?", default=".",
                        help="Checkpoint directory (default: .)")
    p_pack.add_argument("--sign", action="store_true",
                        help="Sign the TGSP package")
    p_pack.add_argument("--signing-key", default=None,
                        help="Hex signing key (auto-generates if --sign and omitted)")
    p_pack.add_argument("--creator", default=None,
                        help="Creator identity (Name <email>)")
    p_pack.add_argument("--skill-doc", default=None,
                        help="Path to SKILL.md file to embed")
    p_pack.add_argument("--name", default=None, help="Adapter name")
    p_pack.add_argument("--domain", default=None, help="Domain")
    p_pack.add_argument("--rank", type=int, default=30, help="LoRA rank")
    p_pack.add_argument("--alpha", type=int, default=64, help="LoRA alpha")
    p_pack.add_argument("--description", default=None, help="Description")
    p_pack.add_argument("--output", default=None, help="Output .tgsp path")

    # -- verify --
    p_verify = subparsers.add_parser("verify", help="Verify TGSP integrity")
    p_verify.add_argument("file", help="Path to .tgsp file")

    # -- export --
    p_export = subparsers.add_parser("export",
                                     help="Export TGSP to plain LoRA format")
    p_export.add_argument("file", help="Path to .tgsp file")
    p_export.add_argument("--format", required=True,
                          choices=["safetensors", "pytorch", "pt", "gguf"],
                          help="Export format")
    p_export.add_argument("--output-dir", default=None, help="Output directory")

    # -- import --
    p_import = subparsers.add_parser("import", help="Import LoRA to TGSP")
    p_import.add_argument("lora_dir", help="Path to LoRA adapter directory")
    p_import.add_argument("--sign", action="store_true",
                          help="Sign the TGSP package")
    p_import.add_argument("--signing-key", default=None,
                          help="Hex signing key")
    p_import.add_argument("--screen", action="store_true",
                          help="Run RVUv2 screening after import")
    p_import.add_argument("--name", required=True, help="Adapter name")
    p_import.add_argument("--domain", default=None, help="Domain")
    p_import.add_argument("--creator", default=None, help="Creator name/email")
    p_import.add_argument("--description", default=None, help="Description")
    p_import.add_argument("--base-model", default=None, help="Base model name")
    p_import.add_argument("--output", default=None, help="Output .tgsp path")
    p_import.add_argument("--tags", default=None, help="Comma-separated tags")

    # -- publish --
    p_publish = subparsers.add_parser("publish", help="Publish to marketplace")
    p_publish.add_argument("file", help="Path to .tgsp file")
    p_publish.add_argument("--marketplace", action="store_true",
                           help="Publish to TenSafe marketplace")

    # -- validate --
    p_validate = subparsers.add_parser("validate",
                                       help="Run TenSafe validation suite")
    p_validate.add_argument("file", help="Path to .tgsp file")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    commands = {
        "init": cmd_init,
        "train": cmd_train,
        "screen": cmd_screen,
        "pack": cmd_pack,
        "verify": cmd_verify,
        "export": cmd_export,
        "import": cmd_import,
        "publish": cmd_publish,
        "validate": cmd_validate,
    }

    handler = commands.get(args.command)
    if handler is None:
        print(f"Unknown command: {args.command}")
        parser.print_help()
        return 1

    try:
        return handler(args) or 0
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as e:
        logger.error(f"Command '{args.command}' failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
