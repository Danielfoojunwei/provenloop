"""TG Tinker -- Agentic LoRA Creator.

Builds production-quality LoRA adapters from any dataset. Supports three
modes of operation:

  1. One-Click Creation  -- end-to-end from data directory to signed TGSP
  2. Agent Self-Improvement -- refine an adapter from interaction feedback
  3. Continuous Improvement Loop -- background loop sampling live traffic

Heavy ML operations (model loading, SFT training, inference) are stubbed
with clear interfaces. Data analysis, curation, config generation, TGSP
packaging, and the full pipeline orchestration are real working code.
"""

import hashlib
import json
import logging
import os
import re
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from tensafe.skills.tg_tinker.data_analyzer import (
    DataAnalysis,
    LoraConfig,
    analyze_data,
    classify_domain,
    suggest_full_config,
)
from tensafe.skills.tg_tinker.export import (
    export_tgsp_to_gguf,
    export_tgsp_to_pytorch,
    export_tgsp_to_safetensors,
    read_tgsp,
    verify_tgsp,
    write_tgsp,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quality gate thresholds
# ---------------------------------------------------------------------------

QA_VERIFY_THRESHOLD = 0.80
DP_MAX_EPSILON = 8.0
DP_DELTA = 1e-5


# ---------------------------------------------------------------------------
# Data classes for pipeline state
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
    """Result of an SFT training run."""
    checkpoint_dir: str
    final_loss: float
    best_loss: float
    total_steps: int
    training_time_seconds: float
    dp_epsilon_spent: float
    dp_budget_ok: bool
    lora_config: dict = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Result of adapter evaluation."""
    qa_verify_score: float
    passed: bool
    test_examples: int
    correct: int
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScreeningResult:
    """Result of RVUv2 safety screening."""
    passed: bool
    toxicity_score: float = 0.0
    bias_score: float = 0.0
    ip_score: float = 0.0
    layer_results: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class AdapterManifest:
    """Complete metadata for a TGSP adapter."""
    adapter_id: str
    name: str
    domain: str
    base_model: str
    rank: int
    alpha: int
    creator: str
    created_at: str
    qa_verify_score: float
    rvu_passed: bool
    dp_epsilon: float
    dp_budget_ok: bool
    training_steps: int
    training_loss: float
    lora_config: dict
    description: str = ""
    tags: List[str] = field(default_factory=list)
    license: str = "apache-2.0"
    version: str = "1.0.0"
    format_version: str = "1.0"


@dataclass
class ImprovementPlan:
    """Plan for adapter self-improvement."""
    weak_areas: List[str]
    suggested_data: List[str]
    current_score: float
    target_score: float
    strategy: str


@dataclass
class ContinuousLoopConfig:
    """Configuration for continuous improvement loop."""
    check_interval_seconds: int = 3600
    min_samples_per_iteration: int = 100
    quality_threshold: float = 0.80
    max_iterations: int = 10
    auto_deploy: bool = False
    approval_mode: str = "human_in_loop"  # or "meta_agent"


# ---------------------------------------------------------------------------
# Dataset curation
# ---------------------------------------------------------------------------

@dataclass
class CuratedDataset:
    """A curated dataset ready for training."""
    train_path: str
    test_path: str
    train_examples: int
    test_examples: int
    format: str  # "instruction" | "conversation" | "qa"
    domain: str
    source_analysis: DataAnalysis


def curate_dataset(
    data_path: str,
    output_dir: str,
    analysis: Optional[DataAnalysis] = None,
    test_split: float = 0.2,
) -> CuratedDataset:
    """Clean, format, and split data for SFT training.

    Processes raw data files into instruction-following format, extracts
    Q&A pairs where possible, deduplicates, and creates train/test splits.

    Args:
        data_path: Path to raw data directory or file.
        output_dir: Directory for curated output.
        analysis: Pre-computed DataAnalysis (will analyze if None).
        test_split: Fraction of data for test set (default 0.2).

    Returns:
        CuratedDataset with paths to train and test JSONL files.
    """
    if analysis is None:
        analysis = analyze_data(data_path)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Collect all examples from source files
    examples = []
    for file_info in analysis.files:
        file_examples = _extract_examples(file_info)
        examples.extend(file_examples)

    if not examples:
        raise ValueError(f"No training examples could be extracted from {data_path}")

    # Deduplicate by content hash
    seen = set()
    unique_examples = []
    for ex in examples:
        h = hashlib.md5(json.dumps(ex, sort_keys=True).encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique_examples.append(ex)

    dedup_removed = len(examples) - len(unique_examples)
    if dedup_removed > 0:
        logger.info(f"Removed {dedup_removed} duplicate examples")

    examples = unique_examples

    # Determine format
    if analysis.has_conversation_format:
        fmt = "conversation"
    elif analysis.has_instruction_format:
        fmt = "instruction"
    elif analysis.has_qa_pairs:
        fmt = "qa"
    else:
        fmt = "instruction"
        examples = _convert_to_instruction_format(examples)

    # Shuffle deterministically
    import random
    rng = random.Random(42)
    rng.shuffle(examples)

    # Split
    split_idx = max(1, int(len(examples) * (1 - test_split)))
    train_examples = examples[:split_idx]
    test_examples = examples[split_idx:]

    # Write JSONL files
    train_path = out / "train.jsonl"
    test_path = out / "test.jsonl"

    _write_jsonl(train_examples, train_path)
    _write_jsonl(test_examples, test_path)

    logger.info(
        f"Curated dataset: {len(train_examples)} train, "
        f"{len(test_examples)} test ({fmt} format)"
    )

    return CuratedDataset(
        train_path=str(train_path),
        test_path=str(test_path),
        train_examples=len(train_examples),
        test_examples=len(test_examples),
        format=fmt,
        domain=analysis.detected_domain,
        source_analysis=analysis,
    )


def _extract_examples(file_info) -> List[dict]:
    """Extract training examples from a single file."""
    examples = []
    path = Path(file_info.path)

    try:
        if file_info.format == "jsonl":
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            examples.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

        elif file_info.format == "json":
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
            if isinstance(data, list):
                examples.extend(data)
            elif isinstance(data, dict):
                for key in ("data", "examples", "rows", "items", "records", "train"):
                    if key in data and isinstance(data[key], list):
                        examples.extend(data[key])
                        break
                else:
                    examples.append(data)

        elif file_info.format in ("csv", "tsv"):
            import csv
            delimiter = "\t" if file_info.format == "tsv" else ","
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                for row in reader:
                    examples.append(dict(row))

        elif file_info.format == "text":
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            # Split on double newlines into paragraphs
            paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
            for para in paragraphs:
                examples.append({"text": para})

        elif file_info.format == "parquet":
            try:
                import pyarrow.parquet as pq
                table = pq.read_table(path)
                for row in table.to_pylist():
                    examples.append(row)
            except ImportError:
                logger.warning(f"pyarrow not available, skipping {path.name}")

        elif file_info.format == "pdf":
            try:
                import fitz
                doc = fitz.open(str(path))
                for page in doc:
                    text = page.get_text().strip()
                    if text:
                        examples.append({"text": text})
                doc.close()
            except ImportError:
                logger.warning(f"PyMuPDF not available, skipping {path.name}")

    except Exception as e:
        logger.warning(f"Error extracting examples from {path}: {e}")

    return examples


def _convert_to_instruction_format(examples: List[dict]) -> List[dict]:
    """Convert generic examples to instruction-following format."""
    converted = []
    for ex in examples:
        # Try common field mappings
        instruction = (
            ex.get("instruction")
            or ex.get("question")
            or ex.get("prompt")
            or ex.get("input")
            or ex.get("user_prompt")
            or ex.get("query")
            or ""
        )
        response = (
            ex.get("response")
            or ex.get("answer")
            or ex.get("output")
            or ex.get("completion")
            or ex.get("text", "")
        )
        system = (
            ex.get("system")
            or ex.get("system_prompt")
            or ex.get("context")
            or ""
        )

        if instruction and response:
            text = ""
            if system:
                text += f"### System:\n{system}\n\n"
            text += f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
            converted.append({"text": text})
        elif response:
            converted.append({"text": response})
        elif "text" in ex:
            converted.append({"text": ex["text"]})

    return converted


def _write_jsonl(examples: List[dict], path: Path) -> None:
    """Write examples to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Stub training / evaluation / screening
# ---------------------------------------------------------------------------

def _stub_train_sft(
    dataset: CuratedDataset,
    lora_config: LoraConfig,
    output_dir: str,
    base_model: str = "Qwen/Qwen2.5-1.5B",
    max_steps: int = 2000,
    batch_size: int = 1,
    grad_accum: int = 8,
    learning_rate: float = 1e-4,
    dp_noise_multiplier: float = 1.0,
    dp_target_epsilon: float = 8.0,
    dp_target_delta: float = 1e-5,
) -> TrainingResult:
    """Train a LoRA adapter via SFT with differential privacy.

    STUB: In production, this calls the TenSafe orchestrator for real
    training with DP-SGD. The stub writes a placeholder checkpoint and
    returns simulated metrics.

    The real implementation would:
      1. Load base model and tokenizer
      2. Initialize LoRA layers per lora_config
      3. Run DP-SGD training loop with gradient accumulation
      4. Track epsilon spend via privacy accountant
      5. Save checkpoints with resume support
      6. Return actual training metrics
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"[STUB] Training SFT adapter: "
        f"{dataset.train_examples} examples, "
        f"rank={lora_config.rank}, alpha={lora_config.alpha}, "
        f"steps={max_steps}"
    )

    # Simulate training time based on dataset size
    simulated_steps = min(max_steps, dataset.train_examples * 3)
    simulated_loss = max(0.3, 2.0 - (simulated_steps / max_steps) * 1.5)
    simulated_epsilon = min(dp_target_epsilon, simulated_steps * 0.002)

    # Write placeholder checkpoint
    checkpoint_info = {
        "stub": True,
        "lora_config": lora_config.to_dict(),
        "base_model": base_model,
        "train_examples": dataset.train_examples,
        "total_steps": simulated_steps,
        "final_loss": simulated_loss,
    }
    ckpt_path = out / "adapter_final.json"
    with open(ckpt_path, "w") as f:
        json.dump(checkpoint_info, f, indent=2)

    # Write training metrics
    metrics = {
        "total_steps": simulated_steps,
        "final_loss": simulated_loss,
        "best_loss": simulated_loss * 0.9,
        "epsilon_spent": simulated_epsilon,
        "training_config": {
            "max_steps": max_steps,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "learning_rate": learning_rate,
            "dp_noise_multiplier": dp_noise_multiplier,
        },
    }
    with open(out / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return TrainingResult(
        checkpoint_dir=str(out),
        final_loss=simulated_loss,
        best_loss=simulated_loss * 0.9,
        total_steps=simulated_steps,
        training_time_seconds=simulated_steps * 0.5,
        dp_epsilon_spent=simulated_epsilon,
        dp_budget_ok=simulated_epsilon <= dp_target_epsilon,
        lora_config=lora_config.to_dict(),
    )


def _stub_evaluate(
    checkpoint_dir: str, test_path: str
) -> EvaluationResult:
    """Evaluate adapter quality on held-out test set.

    STUB: In production, this loads the adapter, runs inference on the
    test set, and computes qa_verify accuracy. The stub reads the test
    file to count examples and returns a simulated score.

    The real implementation would:
      1. Load base model + trained LoRA adapter
      2. Run inference on each test example
      3. Compare generated responses to reference answers
      4. Compute qa_verify score (exact match + semantic similarity)
      5. Return detailed evaluation metrics
    """
    test_path_obj = Path(test_path)
    test_count = 0
    if test_path_obj.exists():
        with open(test_path_obj) as f:
            test_count = sum(1 for line in f if line.strip())

    # Simulate a reasonable score based on dataset size
    simulated_score = min(0.92, 0.75 + (test_count / 1000) * 0.05)
    simulated_correct = int(test_count * simulated_score)

    logger.info(
        f"[STUB] Evaluation: {simulated_correct}/{test_count} correct, "
        f"score={simulated_score:.4f}"
    )

    return EvaluationResult(
        qa_verify_score=simulated_score,
        passed=simulated_score >= QA_VERIFY_THRESHOLD,
        test_examples=test_count,
        correct=simulated_correct,
        details={"stub": True, "method": "simulated"},
    )


def _stub_screen_rvu(checkpoint_dir: str) -> ScreeningResult:
    """Run RVUv2 3-layer safety screening on an adapter.

    STUB: In production, this runs the RVUv2 screening pipeline which
    includes toxicity detection, bias analysis, and IP/PII scanning.
    The stub returns a passing result.

    The real implementation would:
      Layer 1: Toxicity scan (run toxic prompts through adapter,
               score responses for harmful content)
      Layer 2: Bias analysis (test for demographic bias in outputs
               across protected categories)
      Layer 3: IP/PII scan (check training data and outputs for
               personal information or copyrighted material)
    """
    logger.info("[STUB] RVUv2 screening: 3-layer safety check")

    return ScreeningResult(
        passed=True,
        toxicity_score=0.02,
        bias_score=0.05,
        ip_score=0.01,
        layer_results={
            "layer_1_toxicity": {"passed": True, "score": 0.02, "threshold": 0.10},
            "layer_2_bias": {"passed": True, "score": 0.05, "threshold": 0.15},
            "layer_3_ip": {"passed": True, "score": 0.01, "threshold": 0.05},
        },
        warnings=[],
    )


def _stub_generate_weights_payload(
    checkpoint_dir: str, lora_config: LoraConfig
) -> bytes:
    """Generate the binary weights payload for TGSP packaging.

    STUB: In production, this loads the trained checkpoint and serializes
    the LoRA weight matrices (A, B for each target module). The stub
    generates deterministic placeholder bytes.

    The real implementation would:
      1. Load adapter_final.pt from checkpoint_dir
      2. Extract LoRA A and B matrices for each target module
      3. Serialize via torch.save or safetensors
      4. Return raw bytes
    """
    import struct

    # Generate deterministic placeholder payload
    # Simulate realistic sizes: rank 30 means A is [hidden, 30], B is [30, hidden]
    # For Qwen 2.5 1.5B, hidden_size = 1536
    hidden_size = 1536
    rank = lora_config.rank
    n_modules = len(lora_config.target_modules)

    # Each module has A: [rank, hidden] and B: [hidden, rank] in float16
    # Size per module: 2 * rank * hidden * 2 bytes
    bytes_per_module = 2 * rank * hidden_size * 2
    total_bytes = n_modules * bytes_per_module

    # Create a deterministic pseudo-random payload
    # In production this would be the actual weight tensors
    import hashlib
    seed = hashlib.sha256(checkpoint_dir.encode()).digest()
    payload = bytearray(total_bytes)

    # Fill with pattern based on seed for reproducibility
    for i in range(0, min(total_bytes, len(seed))):
        payload[i] = seed[i % len(seed)]

    # Prepend a small header indicating this is a stub
    header = json.dumps({
        "stub": True,
        "n_modules": n_modules,
        "rank": rank,
        "hidden_size": hidden_size,
        "target_modules": lora_config.target_modules,
    }).encode("utf-8")

    return struct.pack("<I", len(header)) + header + bytes(payload)


# ---------------------------------------------------------------------------
# TGSP packaging
# ---------------------------------------------------------------------------

def package_tgsp(
    checkpoint_dir: str,
    manifest_data: AdapterManifest,
    signing_key: Optional[str] = None,
    output_path: Optional[str] = None,
    weights_payload: Optional[bytes] = None,
) -> str:
    """Package a trained adapter as a signed TGSP file.

    Builds the TGSP manifest from AdapterManifest, serializes the
    weights payload, signs if a key is provided, and writes the
    binary TGSP file.

    Args:
        checkpoint_dir: Directory containing the trained checkpoint.
        manifest_data: Adapter metadata.
        signing_key: Optional hex signing key for TGSP signature.
        output_path: Output .tgsp path. Defaults to checkpoint_dir/<name>.tgsp.
        weights_payload: Pre-serialized weights bytes. If None, loads from checkpoint.

    Returns:
        Path to the created .tgsp file.
    """
    import hmac

    if output_path is None:
        output_path = str(
            Path(checkpoint_dir) / f"{manifest_data.name}.tgsp"
        )

    # Build manifest dict
    manifest = {
        "format_version": manifest_data.format_version,
        "adapter_id": manifest_data.adapter_id,
        "model_name": manifest_data.name,
        "model_version": manifest_data.version,
        "base_model": manifest_data.base_model,
        "rank": manifest_data.rank,
        "alpha": manifest_data.alpha,
        "creator": manifest_data.creator,
        "license": manifest_data.license,
        "created_at": manifest_data.created_at,
        "metadata": {
            "domain": manifest_data.domain,
            "description": manifest_data.description,
            "tags": manifest_data.tags,
            "expert_type": manifest_data.domain,
        },
        "lora_config": manifest_data.lora_config,
        "screening": {
            "rvu_v2_passed": manifest_data.rvu_passed,
            "qa_verify_score": manifest_data.qa_verify_score,
            "dp_epsilon": manifest_data.dp_epsilon,
            "dp_budget_verified": manifest_data.dp_budget_ok,
        },
        "training": {
            "total_steps": manifest_data.training_steps,
            "final_loss": manifest_data.training_loss,
        },
        "usage_metering": True,
        "price_per_1k_tokens": 0.0,
    }

    # Generate or use provided weights payload
    if weights_payload is None:
        lora_cfg = LoraConfig(**manifest_data.lora_config)
        weights_payload = _stub_generate_weights_payload(checkpoint_dir, lora_cfg)

    # Sign manifest
    if signing_key:
        signable = json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()
        try:
            key_bytes = bytes.fromhex(signing_key)
        except ValueError:
            key_bytes = signing_key.encode("utf-8")
        sig = hmac.new(key_bytes, signable, hashlib.sha256).hexdigest()
        manifest["signature"] = sig
        manifest["signed"] = True
    else:
        manifest["signed"] = False

    return write_tgsp(manifest, weights_payload, output_path)


# ---------------------------------------------------------------------------
# TGTinker class
# ---------------------------------------------------------------------------

class TGTinker:
    """Agentic LoRA creator. Builds production-quality adapters from any dataset.

    Three operational modes:
      1. create_adapter()     -- one-click end-to-end creation
      2. improve_adapter()    -- agent self-improvement from feedback
      3. continuous_improve() -- background improvement loop

    Quality gates enforced:
      - qa_verify >= 0.80
      - RVUv2 3-layer safety pass
      - DP budget check (epsilon <= target)
    """

    def __init__(
        self,
        base_model: str = "Qwen/Qwen2.5-1.5B",
        output_dir: str = "./tg_tinker_output",
        default_rank: int = 30,
        default_alpha: int = 64,
    ):
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.default_rank = default_rank
        self.default_alpha = default_alpha
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_adapter(
        self,
        data_path: str,
        domain: Optional[str] = None,
        name: Optional[str] = None,
        signing_key: Optional[str] = None,
        export_format: Optional[str] = None,
        rank: Optional[int] = None,
        alpha: Optional[int] = None,
        max_steps: int = 2000,
        creator: str = "TG Tinker",
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Mode 1: One-click creation from data directory.

        Executes the full 8-step pipeline:
          1. Analyze data (detect format, volume, domain)
          2. Classify domain (auto-detect if not specified)
          3. Curate dataset (clean, extract Q&A, split 80/20)
          4. Train SFT (with DP, rank 30, alpha 64)
          5. Evaluate (qa_verify score)
          6. Screen RVU (RVUv2 3-layer safety)
          7. Package TGSP (create signed .tgsp)
          8. Export (optional: safetensors/pytorch/gguf)

        Args:
            data_path: Path to data directory or single file.
            domain: Domain hint (auto-detected if None).
            name: Adapter name (auto-generated if None).
            signing_key: Hex signing key for TGSP signature.
            export_format: Optional export format: safetensors, pytorch, gguf.
            rank: LoRA rank (default: auto from data analysis or 30).
            alpha: LoRA alpha (default: auto from data analysis or 64).
            max_steps: Maximum training steps.
            creator: Creator name for manifest.
            description: Adapter description.
            tags: Tags for marketplace listing.

        Returns:
            Dict with pipeline results including paths to outputs.
        """
        result: Dict[str, Any] = {
            "success": False,
            "steps_completed": [],
            "errors": [],
        }

        # Generate run ID and output directory
        run_id = str(uuid.uuid4())[:8]
        adapter_name = name or f"adapter_{run_id}"
        run_dir = self.output_dir / adapter_name
        run_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"=== TG Tinker: Creating adapter '{adapter_name}' ===")
        logger.info(f"Run ID: {run_id}, Output: {run_dir}")

        # ---------------------------------------------------------------
        # Step 1: Analyze data
        # ---------------------------------------------------------------
        logger.info("Step 1/8: Analyzing data...")
        try:
            analysis = analyze_data(data_path)
            result["analysis"] = analysis.to_dict()
            result["steps_completed"].append("analyze_data")
            logger.info(
                f"  Found {analysis.total_files} files, "
                f"~{analysis.estimated_examples} examples, "
                f"formats: {analysis.formats_found}"
            )
        except Exception as e:
            result["errors"].append(f"Data analysis failed: {e}")
            return result

        # ---------------------------------------------------------------
        # Step 2: Classify domain
        # ---------------------------------------------------------------
        logger.info("Step 2/8: Classifying domain...")
        if domain is None:
            domain = analysis.detected_domain
        result["domain"] = domain
        result["steps_completed"].append("classify_domain")
        logger.info(f"  Domain: {domain}")

        if analysis.warnings:
            for w in analysis.warnings:
                logger.warning(f"  WARNING: {w}")
            result["warnings"] = analysis.warnings

        # ---------------------------------------------------------------
        # Step 3: Curate dataset
        # ---------------------------------------------------------------
        logger.info("Step 3/8: Curating dataset...")
        try:
            curated_dir = run_dir / "curated"
            dataset = curate_dataset(
                data_path, str(curated_dir), analysis=analysis
            )
            result["dataset"] = {
                "train_path": dataset.train_path,
                "test_path": dataset.test_path,
                "train_examples": dataset.train_examples,
                "test_examples": dataset.test_examples,
                "format": dataset.format,
            }
            result["steps_completed"].append("curate_dataset")
            logger.info(
                f"  {dataset.train_examples} train, "
                f"{dataset.test_examples} test"
            )
        except Exception as e:
            result["errors"].append(f"Dataset curation failed: {e}")
            return result

        # ---------------------------------------------------------------
        # Step 4: Generate LoRA config and train
        # ---------------------------------------------------------------
        logger.info("Step 4/8: Training SFT adapter...")
        lora_config = suggest_full_config(analysis)
        if rank is not None:
            lora_config.rank = rank
        if alpha is not None:
            lora_config.alpha = alpha

        result["lora_config"] = lora_config.to_dict()

        try:
            checkpoint_dir = str(run_dir / "checkpoint")
            training_result = _stub_train_sft(
                dataset=dataset,
                lora_config=lora_config,
                output_dir=checkpoint_dir,
                base_model=self.base_model,
                max_steps=max_steps,
            )
            result["training"] = {
                "checkpoint_dir": training_result.checkpoint_dir,
                "final_loss": training_result.final_loss,
                "best_loss": training_result.best_loss,
                "total_steps": training_result.total_steps,
                "dp_epsilon_spent": training_result.dp_epsilon_spent,
                "dp_budget_ok": training_result.dp_budget_ok,
            }
            result["steps_completed"].append("train_sft")
            logger.info(
                f"  Steps: {training_result.total_steps}, "
                f"Loss: {training_result.final_loss:.4f}, "
                f"Epsilon: {training_result.dp_epsilon_spent:.4f}"
            )
        except Exception as e:
            result["errors"].append(f"Training failed: {e}")
            return result

        # ---------------------------------------------------------------
        # Step 5: Evaluate
        # ---------------------------------------------------------------
        logger.info("Step 5/8: Evaluating adapter quality...")
        try:
            eval_result = _stub_evaluate(checkpoint_dir, dataset.test_path)
            result["evaluation"] = {
                "qa_verify_score": eval_result.qa_verify_score,
                "passed": eval_result.passed,
                "test_examples": eval_result.test_examples,
                "correct": eval_result.correct,
            }
            result["steps_completed"].append("evaluate")
            logger.info(
                f"  qa_verify: {eval_result.qa_verify_score:.4f} "
                f"({'PASS' if eval_result.passed else 'FAIL'})"
            )

            if not eval_result.passed:
                result["errors"].append(
                    f"Quality gate FAILED: qa_verify={eval_result.qa_verify_score:.4f} "
                    f"< {QA_VERIFY_THRESHOLD}. Consider more/better training data."
                )
                return result
        except Exception as e:
            result["errors"].append(f"Evaluation failed: {e}")
            return result

        # ---------------------------------------------------------------
        # Step 6: RVUv2 safety screening
        # ---------------------------------------------------------------
        logger.info("Step 6/8: Running RVUv2 safety screening...")
        try:
            screen_result = _stub_screen_rvu(checkpoint_dir)
            result["screening"] = {
                "passed": screen_result.passed,
                "toxicity": screen_result.toxicity_score,
                "bias": screen_result.bias_score,
                "ip": screen_result.ip_score,
                "warnings": screen_result.warnings,
            }
            result["steps_completed"].append("screen_rvu")
            logger.info(
                f"  RVUv2: {'PASS' if screen_result.passed else 'FAIL'} "
                f"(tox={screen_result.toxicity_score:.3f}, "
                f"bias={screen_result.bias_score:.3f}, "
                f"ip={screen_result.ip_score:.3f})"
            )

            if not screen_result.passed:
                result["errors"].append(
                    "Safety screening FAILED. "
                    "Review screening warnings and remediate before packaging."
                )
                for w in screen_result.warnings:
                    result["errors"].append(f"  RVUv2 warning: {w}")
                return result
        except Exception as e:
            result["errors"].append(f"RVUv2 screening failed: {e}")
            return result

        # ---------------------------------------------------------------
        # Step 7: Package as TGSP
        # ---------------------------------------------------------------
        logger.info("Step 7/8: Packaging as TGSP...")
        try:
            manifest_data = AdapterManifest(
                adapter_id=str(uuid.uuid4()),
                name=adapter_name,
                domain=domain,
                base_model=self.base_model,
                rank=lora_config.rank,
                alpha=lora_config.alpha,
                creator=creator,
                created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                qa_verify_score=eval_result.qa_verify_score,
                rvu_passed=screen_result.passed,
                dp_epsilon=training_result.dp_epsilon_spent,
                dp_budget_ok=training_result.dp_budget_ok,
                training_steps=training_result.total_steps,
                training_loss=training_result.final_loss,
                lora_config=lora_config.to_dict(),
                description=description or f"{domain} LoRA adapter",
                tags=tags or [domain, "lora", "tgsp"],
                license="apache-2.0",
            )

            tgsp_path = package_tgsp(
                checkpoint_dir=checkpoint_dir,
                manifest_data=manifest_data,
                signing_key=signing_key,
                output_path=str(run_dir / f"{adapter_name}.tgsp"),
            )
            result["tgsp_path"] = tgsp_path
            result["steps_completed"].append("package_tgsp")
            logger.info(f"  TGSP: {tgsp_path}")
        except Exception as e:
            result["errors"].append(f"TGSP packaging failed: {e}")
            return result

        # ---------------------------------------------------------------
        # Step 8: Optional export
        # ---------------------------------------------------------------
        if export_format:
            logger.info(f"Step 8/8: Exporting to {export_format}...")
            try:
                export_dir = str(run_dir / "export")
                export_path = self._export(tgsp_path, export_format, export_dir)
                result["export_path"] = export_path
                result["export_format"] = export_format
                result["steps_completed"].append("export")
                logger.info(f"  Exported: {export_path}")
            except Exception as e:
                result["errors"].append(f"Export failed: {e}")
                # Export failure is non-fatal -- TGSP was already created
                logger.warning(f"  Export failed (non-fatal): {e}")
        else:
            logger.info("Step 8/8: No export requested (TGSP is default)")
            result["steps_completed"].append("export_skipped")

        result["success"] = True
        result["adapter_name"] = adapter_name

        logger.info(
            f"=== TG Tinker: Adapter '{adapter_name}' created successfully ==="
        )
        logger.info(f"  TGSP: {tgsp_path}")
        logger.info(f"  Domain: {domain}")
        logger.info(f"  Quality: {eval_result.qa_verify_score:.4f}")
        logger.info(f"  Safety: PASSED")

        return result

    def improve_adapter(
        self,
        current_adapter: str,
        interaction_log: str,
        feedback: Optional[str] = None,
        signing_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Mode 2: Agent self-improvement.

        Analyzes the current adapter's performance from interaction logs,
        identifies weak areas, curates targeted training data from the
        logs, and produces an improved adapter version.

        Args:
            current_adapter: Path to current .tgsp adapter.
            interaction_log: Path to JSONL interaction log file.
            feedback: Optional path to feedback/rating file.
            signing_key: Optional signing key for new TGSP.

        Returns:
            Dict with improvement results.
        """
        result: Dict[str, Any] = {
            "success": False,
            "improvement_plan": None,
            "new_adapter": None,
        }

        logger.info("=== TG Tinker: Self-improvement mode ===")

        # Verify current adapter
        valid, details = verify_tgsp(current_adapter)
        if not valid:
            result["errors"] = [f"Current adapter invalid: {details.get('errors', [])}"]
            return result

        current_manifest = details.get("manifest", {})
        current_domain = current_manifest.get("metadata", {}).get("domain", "general")
        current_score = current_manifest.get("screening", {}).get(
            "qa_verify_score", 0.0
        )

        # Analyze interaction log
        logger.info("Analyzing interaction logs...")
        plan = self._analyze_interactions(interaction_log, feedback, current_score)
        result["improvement_plan"] = {
            "weak_areas": plan.weak_areas,
            "suggested_data": plan.suggested_data,
            "current_score": plan.current_score,
            "target_score": plan.target_score,
            "strategy": plan.strategy,
        }

        # Curate improvement data from interactions
        logger.info("Curating improvement dataset from interactions...")
        improvement_dir = self.output_dir / "improvement_data"
        improvement_dir.mkdir(parents=True, exist_ok=True)

        examples = self._extract_improvement_examples(interaction_log, feedback)

        if not examples:
            result["errors"] = [
                "No improvement examples could be extracted from interaction logs."
            ]
            return result

        # Write improvement data
        improvement_data_path = improvement_dir / "improvement.jsonl"
        _write_jsonl(examples, improvement_data_path)

        logger.info(f"Extracted {len(examples)} improvement examples")

        # Create improved adapter
        adapter_name = current_manifest.get("model_name", "adapter") + "_improved"
        improved = self.create_adapter(
            data_path=str(improvement_dir),
            domain=current_domain,
            name=adapter_name,
            signing_key=signing_key,
            description=f"Improved version: {plan.strategy}",
            tags=[current_domain, "improved", "self-improvement"],
        )

        result["new_adapter"] = improved
        result["success"] = improved.get("success", False)

        return result

    def continuous_improve(
        self,
        adapter_path: str,
        traffic_sample: str,
        loop_config: Optional[ContinuousLoopConfig] = None,
    ) -> Dict[str, Any]:
        """Mode 3: Continuous improvement loop.

        Runs a single iteration of the continuous improvement loop:
          1. Sample recent traffic
          2. Score responses
          3. Extract low-confidence examples
          4. Curate new training data
          5. Train improved adapter
          6. Compare with current adapter
          7. Report results (auto-deploy or await approval)

        In production, this would run as a background service. Here it
        executes a single iteration.

        Args:
            adapter_path: Path to current production .tgsp adapter.
            traffic_sample: Path to JSONL file of recent traffic logs.
            loop_config: Configuration for the improvement loop.

        Returns:
            Dict with iteration results.
        """
        if loop_config is None:
            loop_config = ContinuousLoopConfig()

        result: Dict[str, Any] = {
            "success": False,
            "iteration": 1,
            "samples_analyzed": 0,
            "low_confidence_found": 0,
            "improvement_created": False,
        }

        logger.info("=== TG Tinker: Continuous improvement iteration ===")

        # Verify current adapter
        valid, details = verify_tgsp(adapter_path)
        if not valid:
            result["errors"] = [f"Current adapter invalid: {details.get('errors', [])}"]
            return result

        current_manifest = details.get("manifest", {})

        # Analyze traffic sample
        logger.info("Analyzing traffic sample...")
        traffic_path = Path(traffic_sample)
        if not traffic_path.exists():
            result["errors"] = [f"Traffic sample not found: {traffic_sample}"]
            return result

        samples = []
        with open(traffic_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        result["samples_analyzed"] = len(samples)

        if len(samples) < loop_config.min_samples_per_iteration:
            result["errors"] = [
                f"Insufficient samples: {len(samples)} < "
                f"{loop_config.min_samples_per_iteration} minimum"
            ]
            return result

        # Score and extract low-confidence examples
        low_confidence = self._find_low_confidence(samples, loop_config)
        result["low_confidence_found"] = len(low_confidence)

        if not low_confidence:
            logger.info("No low-confidence examples found. Adapter is performing well.")
            result["success"] = True
            result["message"] = "No improvement needed this iteration."
            return result

        # Curate improvement data
        improvement_dir = self.output_dir / "continuous_improvement"
        improvement_dir.mkdir(parents=True, exist_ok=True)
        improvement_data_path = improvement_dir / "low_confidence.jsonl"
        _write_jsonl(low_confidence, improvement_data_path)

        # Create improved adapter
        domain = current_manifest.get("metadata", {}).get("domain", "general")
        adapter_name = current_manifest.get("model_name", "adapter") + "_v2"

        improved = self.create_adapter(
            data_path=str(improvement_dir),
            domain=domain,
            name=adapter_name,
            description="Continuous improvement iteration",
            tags=[domain, "continuous_improvement"],
        )

        result["new_adapter"] = improved
        result["improvement_created"] = improved.get("success", False)
        result["success"] = True
        result["approval_mode"] = loop_config.approval_mode

        if loop_config.auto_deploy and improved.get("success"):
            result["auto_deployed"] = True
            logger.info("Auto-deploy enabled: new adapter ready for deployment")
        else:
            result["auto_deployed"] = False
            logger.info(
                f"Approval mode: {loop_config.approval_mode}. "
                "New adapter awaiting approval."
            )

        return result

    def validate(self, tgsp_path: str) -> Dict[str, Any]:
        """Run TenSafe validation suite for marketplace listing.

        Performs comprehensive validation including:
          - TGSP format verification (magic, manifest, hash)
          - Manifest completeness check
          - Quality gate verification (qa_verify, RVUv2, DP)
          - Marketplace readiness check

        Args:
            tgsp_path: Path to .tgsp file.

        Returns:
            Dict with validation results.
        """
        result: Dict[str, Any] = {
            "valid": False,
            "checks": {},
            "marketplace_ready": False,
            "errors": [],
            "warnings": [],
        }

        logger.info(f"=== TenSafe Validation: {tgsp_path} ===")

        # 1. Format verification
        logger.info("Check 1: TGSP format verification...")
        valid, details = verify_tgsp(tgsp_path)
        result["checks"]["format"] = {
            "passed": valid,
            "magic_ok": details.get("magic_ok", False),
            "manifest_ok": details.get("manifest_ok", False),
            "hash_ok": details.get("hash_ok", False),
        }
        if not valid:
            result["errors"].extend(details.get("errors", []))
            return result

        manifest = details.get("manifest", {})

        # 2. Manifest completeness
        logger.info("Check 2: Manifest completeness...")
        required_fields = [
            "adapter_id", "model_name", "creator", "format_version",
        ]
        missing = [f for f in required_fields if f not in manifest]
        result["checks"]["manifest_completeness"] = {
            "passed": len(missing) == 0,
            "missing_fields": missing,
        }
        if missing:
            result["warnings"].append(f"Missing manifest fields: {missing}")

        # 3. Quality gates
        logger.info("Check 3: Quality gates...")
        screening = manifest.get("screening", {})

        qa_score = screening.get("qa_verify_score")
        qa_pass = qa_score is not None and qa_score >= QA_VERIFY_THRESHOLD
        result["checks"]["qa_verify"] = {
            "passed": qa_pass,
            "score": qa_score,
            "threshold": QA_VERIFY_THRESHOLD,
        }
        if not qa_pass:
            result["warnings"].append(
                f"qa_verify score {qa_score} < {QA_VERIFY_THRESHOLD}"
            )

        rvu_pass = screening.get("rvu_v2_passed", False)
        result["checks"]["rvu_v2"] = {"passed": rvu_pass}
        if not rvu_pass:
            result["warnings"].append("RVUv2 screening not passed")

        dp_ok = screening.get("dp_budget_verified", False)
        result["checks"]["dp_budget"] = {
            "passed": dp_ok,
            "epsilon": screening.get("dp_epsilon"),
        }
        if not dp_ok:
            result["warnings"].append("DP budget not verified")

        # 4. Signature check
        logger.info("Check 4: Signature verification...")
        is_signed = manifest.get("signed", False) or "signature" in manifest
        result["checks"]["signature"] = {"signed": is_signed}
        if not is_signed:
            result["warnings"].append("Adapter is not signed")

        # 5. Marketplace readiness
        marketplace_ready = (
            valid
            and len(missing) == 0
            and qa_pass
            and rvu_pass
            and dp_ok
            and is_signed
        )
        result["marketplace_ready"] = marketplace_ready
        result["valid"] = valid

        status = "MARKETPLACE READY" if marketplace_ready else "NOT READY"
        logger.info(f"=== Validation complete: {status} ===")

        return result

    # -------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------

    def _export(
        self, tgsp_path: str, format: str, output_dir: str
    ) -> str:
        """Export TGSP to the specified format."""
        fmt = format.lower().strip()
        if fmt == "safetensors":
            return export_tgsp_to_safetensors(tgsp_path, output_dir)
        elif fmt in ("pytorch", "pt", "bin"):
            out_path = str(Path(output_dir) / "adapter_model.pt")
            return export_tgsp_to_pytorch(tgsp_path, out_path)
        elif fmt == "gguf":
            out_path = str(Path(output_dir) / "adapter.gguf")
            return export_tgsp_to_gguf(tgsp_path, out_path)
        else:
            raise ValueError(
                f"Unsupported export format: {format}. "
                "Supported: safetensors, pytorch, gguf"
            )

    def _analyze_interactions(
        self,
        interaction_log: str,
        feedback: Optional[str],
        current_score: float,
    ) -> ImprovementPlan:
        """Analyze interaction logs to identify improvement areas."""
        weak_areas = []
        suggested_data = []

        # Parse interaction log
        log_path = Path(interaction_log)
        interactions = []
        if log_path.exists():
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            interactions.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

        # Parse feedback if provided
        ratings = []
        if feedback:
            fb_path = Path(feedback)
            if fb_path.exists():
                with open(fb_path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                ratings.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue

        # Analyze weak areas
        if interactions:
            # Look for patterns in low-rated interactions
            low_quality = [
                i for i in interactions
                if i.get("score", 1.0) < 0.5 or i.get("rating", 5) < 3
            ]
            if low_quality:
                weak_areas.append(
                    f"{len(low_quality)}/{len(interactions)} interactions rated low"
                )
                suggested_data.append("Curate corrections for low-rated responses")

            # Check for unanswered queries
            unanswered = [
                i for i in interactions
                if not i.get("response") or len(str(i.get("response", ""))) < 10
            ]
            if unanswered:
                weak_areas.append(f"{len(unanswered)} queries without adequate response")
                suggested_data.append("Add training data for unanswered query types")

        strategy = (
            "Targeted improvement on weak areas"
            if weak_areas
            else "General quality enhancement"
        )

        return ImprovementPlan(
            weak_areas=weak_areas or ["No specific weak areas identified"],
            suggested_data=suggested_data or ["Expand with domain-specific examples"],
            current_score=current_score,
            target_score=min(1.0, current_score + 0.05),
            strategy=strategy,
        )

    def _extract_improvement_examples(
        self,
        interaction_log: str,
        feedback: Optional[str],
    ) -> List[dict]:
        """Extract training examples from interaction logs and feedback."""
        examples = []
        log_path = Path(interaction_log)

        if log_path.exists():
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        interaction = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Extract high-quality interactions as positive examples
                    score = interaction.get("score", 0.5)
                    rating = interaction.get("rating", 3)
                    if score >= 0.8 or rating >= 4:
                        query = interaction.get("query", interaction.get("prompt", ""))
                        response = interaction.get(
                            "response", interaction.get("completion", "")
                        )
                        if query and response:
                            examples.append({
                                "text": (
                                    f"### Instruction:\n{query}\n\n"
                                    f"### Response:\n{response}"
                                ),
                            })

                    # Use corrected versions from feedback
                    correction = interaction.get("correction")
                    if correction:
                        query = interaction.get("query", interaction.get("prompt", ""))
                        if query:
                            examples.append({
                                "text": (
                                    f"### Instruction:\n{query}\n\n"
                                    f"### Response:\n{correction}"
                                ),
                            })

        # Also parse feedback file if provided
        if feedback:
            fb_path = Path(feedback)
            if fb_path.exists():
                with open(fb_path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            fb = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if fb.get("corrected_response"):
                            examples.append({
                                "text": (
                                    f"### Instruction:\n{fb.get('query', '')}\n\n"
                                    f"### Response:\n{fb['corrected_response']}"
                                ),
                            })

        return examples

    def _find_low_confidence(
        self,
        samples: List[dict],
        config: ContinuousLoopConfig,
    ) -> List[dict]:
        """Find low-confidence examples from traffic samples.

        In production, this would run the adapter on each sample and
        check the model's confidence (logit entropy, perplexity, etc.).
        The stub uses score/confidence fields if present.
        """
        low_confidence = []
        for sample in samples:
            confidence = sample.get(
                "confidence",
                sample.get("score", sample.get("perplexity", 1.0))
            )
            # Treat perplexity > 10 or confidence < 0.5 as low confidence
            if (
                (isinstance(confidence, (int, float)) and confidence < 0.5)
                or sample.get("perplexity", 0) > 10
                or sample.get("error", False)
            ):
                low_confidence.append(sample)

        logger.info(
            f"Found {len(low_confidence)}/{len(samples)} "
            "low-confidence examples"
        )
        return low_confidence
