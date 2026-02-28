"""Data analysis module for TG Tinker.

Detects file formats, counts examples, estimates training time,
auto-classifies domain, and generates suggested LoRA configurations
based on data characteristics.
"""

import csv
import io
import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain classification keywords
# ---------------------------------------------------------------------------

DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "healthcare": [
        "patient", "diagnosis", "treatment", "clinical", "medical",
        "symptom", "prescription", "hospital", "surgery", "therapy",
        "icd", "cpt", "ehr", "fhir", "hl7", "dosage", "pathology",
        "oncology", "cardiology", "radiology", "pharmaceutical",
        "drug", "disease", "chronic", "acute", "prognosis",
        "vitals", "blood pressure", "lab results", "biopsy",
    ],
    "finance": [
        "bank", "investment", "portfolio", "stock", "bond",
        "loan", "mortgage", "interest rate", "dividend", "equity",
        "asset", "liability", "revenue", "profit", "margin",
        "hedge", "derivative", "etf", "mutual fund", "credit",
        "fdic", "sec", "finra", "compliance", "audit",
        "risk", "volatility", "yield", "amortization", "capital",
    ],
    "legal": [
        "court", "jurisdiction", "statute", "plaintiff", "defendant",
        "contract", "clause", "tort", "liability", "damages",
        "attorney", "counsel", "deposition", "discovery", "brief",
        "precedent", "appeal", "verdict", "settlement", "arbitration",
        "regulation", "compliance", "fiduciary", "indemnity", "waiver",
        "intellectual property", "patent", "trademark", "copyright",
    ],
    "engineering": [
        "algorithm", "function", "class", "module", "api",
        "database", "query", "server", "deploy", "container",
        "kubernetes", "docker", "ci/cd", "pipeline", "test",
        "bug", "debug", "refactor", "architecture", "microservice",
        "latency", "throughput", "cache", "index", "schema",
        "compiler", "runtime", "memory", "thread", "process",
    ],
}

# Supported file extensions and their types
FILE_TYPES = {
    ".jsonl": "jsonl",
    ".json": "json",
    ".csv": "csv",
    ".tsv": "tsv",
    ".txt": "text",
    ".md": "text",
    ".pdf": "pdf",
    ".parquet": "parquet",
    ".pq": "parquet",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FileInfo:
    """Metadata about a single data file."""
    path: str
    name: str
    format: str
    size_bytes: int
    estimated_examples: int
    sample_text: str = ""


@dataclass
class DataAnalysis:
    """Complete analysis of a data directory or file."""
    data_path: str
    total_files: int = 0
    total_size_bytes: int = 0
    formats_found: Dict[str, int] = field(default_factory=dict)
    estimated_examples: int = 0
    files: List[FileInfo] = field(default_factory=list)
    detected_domain: str = "general"
    domain_scores: Dict[str, float] = field(default_factory=dict)
    has_qa_pairs: bool = False
    has_instruction_format: bool = False
    has_conversation_format: bool = False
    sample_fields: List[str] = field(default_factory=list)
    estimated_training_time_minutes: float = 0.0
    suggested_rank: int = 30
    suggested_alpha: int = 64
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "data_path": self.data_path,
            "total_files": self.total_files,
            "total_size_bytes": self.total_size_bytes,
            "total_size_mb": round(self.total_size_bytes / (1024 * 1024), 2),
            "formats_found": self.formats_found,
            "estimated_examples": self.estimated_examples,
            "detected_domain": self.detected_domain,
            "domain_scores": {k: round(v, 4) for k, v in self.domain_scores.items()},
            "has_qa_pairs": self.has_qa_pairs,
            "has_instruction_format": self.has_instruction_format,
            "has_conversation_format": self.has_conversation_format,
            "sample_fields": self.sample_fields,
            "estimated_training_time_minutes": round(self.estimated_training_time_minutes, 1),
            "suggested_rank": self.suggested_rank,
            "suggested_alpha": self.suggested_alpha,
            "warnings": self.warnings,
        }


@dataclass
class LoraConfig:
    """LoRA configuration for adapter training."""
    rank: int = 30
    alpha: int = 64
    dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
            "task_type": self.task_type,
        }


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------

def analyze_data(data_path: str) -> DataAnalysis:
    """Analyze a data directory or single file.

    Detects file formats, counts examples, samples text for domain
    classification, and returns a complete DataAnalysis.

    Args:
        data_path: Path to a directory of data files or a single file.

    Returns:
        DataAnalysis with detected formats, example counts, and domain.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")

    analysis = DataAnalysis(data_path=str(path.resolve()))

    # Collect files
    if path.is_file():
        data_files = [path]
    else:
        data_files = []
        for ext in FILE_TYPES:
            data_files.extend(path.rglob(f"*{ext}"))
        data_files.sort(key=lambda p: p.name)

    if not data_files:
        analysis.warnings.append("No supported data files found.")
        return analysis

    analysis.total_files = len(data_files)
    all_sample_text = []

    for fpath in data_files:
        file_info = _analyze_file(fpath)
        analysis.files.append(file_info)
        analysis.total_size_bytes += file_info.size_bytes
        analysis.estimated_examples += file_info.estimated_examples

        # Track format counts
        fmt = file_info.format
        analysis.formats_found[fmt] = analysis.formats_found.get(fmt, 0) + 1

        if file_info.sample_text:
            all_sample_text.append(file_info.sample_text)

    # Detect data structure from samples
    analysis.has_qa_pairs = _detect_qa_pairs(analysis.files)
    analysis.has_instruction_format = _detect_instruction_format(analysis.files)
    analysis.has_conversation_format = _detect_conversation_format(analysis.files)
    analysis.sample_fields = _detect_fields(analysis.files)

    # Domain classification
    combined_text = " ".join(all_sample_text)
    analysis.domain_scores = classify_domain_scores(combined_text)
    analysis.detected_domain = max(
        analysis.domain_scores, key=analysis.domain_scores.get
    ) if analysis.domain_scores else "general"

    # If the top score is too low, fall back to general
    top_score = analysis.domain_scores.get(analysis.detected_domain, 0.0)
    if top_score < 0.05:
        analysis.detected_domain = "general"

    # Suggest LoRA config based on data size
    analysis.suggested_rank, analysis.suggested_alpha = suggest_lora_config(
        analysis.estimated_examples, analysis.total_size_bytes
    )

    # Estimate training time (rough: ~0.5s per example at rank 30 on A100)
    analysis.estimated_training_time_minutes = (
        analysis.estimated_examples * 0.5 / 60.0
    )

    # Warnings
    if analysis.estimated_examples < 100:
        analysis.warnings.append(
            f"Only {analysis.estimated_examples} examples detected. "
            "Minimum 100 recommended for quality LoRA training."
        )
    if analysis.estimated_examples > 500_000:
        analysis.warnings.append(
            f"{analysis.estimated_examples} examples is very large. "
            "Consider filtering to the most relevant subset."
        )
    if analysis.total_size_bytes > 10 * 1024 * 1024 * 1024:
        analysis.warnings.append(
            "Dataset exceeds 10 GB. Consider pre-filtering or streaming."
        )

    return analysis


def _analyze_file(fpath: Path) -> FileInfo:
    """Analyze a single data file."""
    ext = fpath.suffix.lower()
    fmt = FILE_TYPES.get(ext, "unknown")
    size = fpath.stat().st_size
    sample_text = ""
    estimated_examples = 0

    try:
        if fmt == "jsonl":
            estimated_examples, sample_text = _analyze_jsonl(fpath)
        elif fmt == "json":
            estimated_examples, sample_text = _analyze_json(fpath)
        elif fmt in ("csv", "tsv"):
            estimated_examples, sample_text = _analyze_csv(fpath, fmt)
        elif fmt == "text":
            estimated_examples, sample_text = _analyze_text(fpath)
        elif fmt == "parquet":
            estimated_examples, sample_text = _analyze_parquet(fpath)
        elif fmt == "pdf":
            estimated_examples, sample_text = _analyze_pdf(fpath)
        else:
            # Unknown format: rough estimate from file size
            estimated_examples = max(1, size // 1000)
    except Exception as e:
        logger.warning(f"Error analyzing {fpath}: {e}")
        estimated_examples = max(1, size // 1000)

    return FileInfo(
        path=str(fpath),
        name=fpath.name,
        format=fmt,
        size_bytes=size,
        estimated_examples=estimated_examples,
        sample_text=sample_text[:5000],  # cap sample text
    )


def _analyze_jsonl(fpath: Path) -> Tuple[int, str]:
    """Analyze a JSONL file."""
    count = 0
    samples = []
    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            count += 1
            if i < 10:
                try:
                    obj = json.loads(line)
                    samples.append(json.dumps(obj, ensure_ascii=False)[:500])
                except json.JSONDecodeError:
                    samples.append(line[:500])
    return count, " ".join(samples)


def _analyze_json(fpath: Path) -> Tuple[int, str]:
    """Analyze a JSON file (array or single object)."""
    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)

    if isinstance(data, list):
        count = len(data)
        sample_items = data[:10]
        sample_text = " ".join(
            json.dumps(item, ensure_ascii=False)[:500] for item in sample_items
        )
    elif isinstance(data, dict):
        # Could be a single example or a dataset wrapper
        for key in ("data", "examples", "rows", "items", "records", "train"):
            if key in data and isinstance(data[key], list):
                count = len(data[key])
                sample_items = data[key][:10]
                sample_text = " ".join(
                    json.dumps(item, ensure_ascii=False)[:500]
                    for item in sample_items
                )
                return count, sample_text
        count = 1
        sample_text = json.dumps(data, ensure_ascii=False)[:2000]
    else:
        count = 1
        sample_text = str(data)[:2000]

    return count, sample_text


def _analyze_csv(fpath: Path, fmt: str) -> Tuple[int, str]:
    """Analyze a CSV or TSV file."""
    delimiter = "\t" if fmt == "tsv" else ","
    count = 0
    samples = []
    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter=delimiter)
        header = None
        for i, row in enumerate(reader):
            if i == 0:
                header = row
                continue
            count += 1
            if i <= 10:
                if header:
                    row_dict = dict(zip(header, row))
                    samples.append(json.dumps(row_dict, ensure_ascii=False)[:500])
                else:
                    samples.append(" ".join(row)[:500])
    return count, " ".join(samples)


def _analyze_text(fpath: Path) -> Tuple[int, str]:
    """Analyze a plain text or markdown file."""
    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    # Estimate examples by paragraph breaks (double newline) or document markers
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    count = max(1, len(paragraphs))

    # Sample the first ~5000 chars
    sample_text = text[:5000]
    return count, sample_text


def _analyze_parquet(fpath: Path) -> Tuple[int, str]:
    """Analyze a Parquet file."""
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(fpath)
        count = table.num_rows
        # Sample first 10 rows as text
        df_head = table.slice(0, min(10, count)).to_pydict()
        sample_text = json.dumps(df_head, ensure_ascii=False, default=str)[:5000]
        return count, sample_text
    except ImportError:
        logger.warning(
            "pyarrow not installed. Estimating parquet file from size. "
            "Install pyarrow for accurate analysis: pip install pyarrow"
        )
        size = fpath.stat().st_size
        # Rough estimate: ~200 bytes per row in parquet
        return max(1, size // 200), ""


def _analyze_pdf(fpath: Path) -> Tuple[int, str]:
    """Analyze a PDF file."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(fpath))
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        # Estimate: one example per page-worth of text
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        count = max(1, len(paragraphs))
        return count, text[:5000]
    except ImportError:
        logger.warning(
            "PyMuPDF not installed. Cannot analyze PDF content. "
            "Install via: pip install PyMuPDF"
        )
        size = fpath.stat().st_size
        # Rough estimate: 3KB per page, 3 examples per page
        pages = max(1, size // 3000)
        return pages * 3, ""


# ---------------------------------------------------------------------------
# Domain classification
# ---------------------------------------------------------------------------

def classify_domain_scores(text: str) -> Dict[str, float]:
    """Score text against each domain's keyword list.

    Returns a dict of domain -> score (0.0 to 1.0) based on keyword
    density. Higher score means more keywords from that domain appear
    in the text.

    Args:
        text: Combined sample text from the dataset.

    Returns:
        Dict mapping domain name to relevance score.
    """
    if not text:
        return {d: 0.0 for d in DOMAIN_KEYWORDS}

    text_lower = text.lower()
    words = set(re.findall(r"\b[a-z][a-z/ ]{2,}\b", text_lower))
    total_words = max(1, len(text_lower.split()))

    scores: Dict[str, float] = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        hits = 0
        for kw in keywords:
            # Count occurrences in the text
            occurrences = text_lower.count(kw.lower())
            if occurrences > 0:
                hits += min(occurrences, 10)  # cap per-keyword contribution
        # Normalize: hits per 1000 words, then scale to 0-1
        density = hits / (total_words / 1000.0)
        scores[domain] = min(1.0, density / 20.0)

    return scores


def classify_domain(text: str) -> str:
    """Classify text into a domain.

    Args:
        text: Combined sample text from the dataset.

    Returns:
        Domain name string: healthcare, finance, legal, engineering, or general.
    """
    scores = classify_domain_scores(text)
    if not scores:
        return "general"
    best = max(scores, key=scores.get)
    if scores[best] < 0.05:
        return "general"
    return best


# ---------------------------------------------------------------------------
# Structure detection helpers
# ---------------------------------------------------------------------------

def _detect_qa_pairs(files: List[FileInfo]) -> bool:
    """Check if data contains question-answer pairs."""
    qa_indicators = [
        '"question"', '"answer"', '"Q"', '"A"',
        '"input"', '"output"', '"prompt"', '"response"',
        '"instruction"', '"completion"',
    ]
    for fi in files:
        text = fi.sample_text.lower()
        if any(ind.lower() in text for ind in qa_indicators):
            return True
    return False


def _detect_instruction_format(files: List[FileInfo]) -> bool:
    """Check if data uses instruction-following format."""
    markers = [
        "### instruction", "### input", "### response",
        "### system", "<|im_start|>", "<|im_end|>",
        '"instruction"', '"system_prompt"', '"user_prompt"',
    ]
    for fi in files:
        text = fi.sample_text.lower()
        if any(m.lower() in text for m in markers):
            return True
    return False


def _detect_conversation_format(files: List[FileInfo]) -> bool:
    """Check if data contains multi-turn conversations."""
    conv_markers = [
        '"conversations"', '"messages"', '"turns"',
        '"role"', '"content"', '"human"', '"assistant"',
        '"user":', '"bot":', '"system":',
    ]
    for fi in files:
        text = fi.sample_text.lower()
        matches = sum(1 for m in conv_markers if m.lower() in text)
        if matches >= 2:
            return True
    return False


def _detect_fields(files: List[FileInfo]) -> List[str]:
    """Detect JSON field names in the data."""
    field_counts: Dict[str, int] = {}
    for fi in files:
        if fi.format in ("jsonl", "json"):
            # Extract field names from sample text
            found = re.findall(r'"([a-zA-Z_][a-zA-Z0-9_]*)":', fi.sample_text)
            for name in found:
                field_counts[name] = field_counts.get(name, 0) + 1
        elif fi.format in ("csv", "tsv"):
            # First row of sample often contains headers
            found = re.findall(r'"([a-zA-Z_][a-zA-Z0-9_]*)"', fi.sample_text[:500])
            for name in found:
                field_counts[name] = field_counts.get(name, 0) + 1

    # Return most common fields
    sorted_fields = sorted(field_counts.items(), key=lambda x: -x[1])
    return [name for name, _ in sorted_fields[:20]]


# ---------------------------------------------------------------------------
# Config suggestion
# ---------------------------------------------------------------------------

def suggest_lora_config(
    num_examples: int, total_size_bytes: int
) -> Tuple[int, int]:
    """Suggest LoRA rank and alpha based on dataset characteristics.

    Args:
        num_examples: Number of training examples.
        total_size_bytes: Total dataset size in bytes.

    Returns:
        Tuple of (suggested_rank, suggested_alpha).
    """
    # Small datasets: lower rank to avoid overfitting
    if num_examples < 500:
        rank = 16
        alpha = 32
    elif num_examples < 5000:
        rank = 24
        alpha = 48
    elif num_examples < 50000:
        rank = 30
        alpha = 64
    elif num_examples < 200000:
        rank = 48
        alpha = 96
    else:
        # Very large datasets can support higher rank
        rank = 64
        alpha = 128

    return rank, alpha


def suggest_full_config(analysis: DataAnalysis) -> LoraConfig:
    """Generate a complete LoraConfig from a DataAnalysis.

    Args:
        analysis: DataAnalysis from analyze_data().

    Returns:
        LoraConfig with suggested parameters.
    """
    rank, alpha = suggest_lora_config(
        analysis.estimated_examples, analysis.total_size_bytes
    )

    # Default target modules for most architectures
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    # For very small datasets, reduce target modules to prevent overfitting
    if analysis.estimated_examples < 500:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    return LoraConfig(
        rank=rank,
        alpha=alpha,
        dropout=0.05 if analysis.estimated_examples < 1000 else 0.0,
        target_modules=target_modules,
    )
