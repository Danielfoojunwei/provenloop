"""Finance dataset preparation for SFT and RL training.

Downloads and prepares 3 filtered finance datasets from HuggingFace:
- Banking: filtered from Sujet-Finance-Instruct-177k (QA + conversation tasks)
- Investment: filtered from Sujet + financial_phrasebank
- Combined: full Sujet dataset for shared attention LoRA

Sujet dataset columns: inputs, answer, user_prompt, system_prompt,
                       task_type, dataset, index_level, conversation_id
"""

import logging
from typing import Dict, List

from datasets import Dataset, concatenate_datasets, load_dataset

logger = logging.getLogger(__name__)

# ── Column names in sujet-ai/Sujet-Finance-Instruct-177k ──
_COL_INPUT = "inputs"
_COL_ANSWER = "answer"
_COL_USER = "user_prompt"
_COL_SYSTEM = "system_prompt"
_COL_TASK = "task_type"
_COL_DATASET = "dataset"


def load_banking_dataset(max_samples: int = 30000) -> Dataset:
    """Load and filter Sujet Finance Instruct for banking topics."""
    ds = load_dataset("sujet-ai/Sujet-Finance-Instruct-177k", split="train")

    banking_keywords = [
        "bank", "deposit", "loan", "mortgage", "credit",
        "savings", "checking", "interest rate", "fdic",
        "branch", "atm", "wire transfer", "overdraft",
        "refinanc", "amortiz", "collateral", "underwriting",
        "escrow", "lien", "principal", "maturity",
        "lending", "borrower", "treasury", "central bank",
        "commercial bank", "retail bank", "liquidity",
    ]

    def is_banking(example):
        text = (
            str(example.get(_COL_USER, "")) + " "
            + str(example.get(_COL_ANSWER, "")) + " "
            + str(example.get(_COL_INPUT, ""))
        ).lower()
        return any(kw in text for kw in banking_keywords)

    filtered = ds.filter(is_banking)
    logger.info(f"Banking dataset: {len(filtered)} samples (from {len(ds)} total)")

    if len(filtered) == 0:
        logger.warning("Banking filter matched 0 samples — using QA subset instead")
        filtered = ds.filter(lambda x: x.get(_COL_TASK) in ("qa", "qa_with_context"))
        filtered = filtered.shuffle(seed=42).select(range(min(max_samples, len(filtered))))

    if len(filtered) > max_samples:
        filtered = filtered.shuffle(seed=42).select(range(max_samples))

    return filtered.map(_format_sujet, remove_columns=filtered.column_names)


def load_investment_dataset(max_samples: int = 20000) -> Dataset:
    """Load financial_phrasebank + Sujet filtered for investment topics."""
    parts = []

    # financial_phrasebank — sentiment with investment context
    try:
        fpb = load_dataset(
            "financial_phrasebank", "sentences_allagree", split="train"
        )
        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        fpb = fpb.map(
            lambda x: {
                "text": (
                    f"### Instruction:\n"
                    f"Analyze the following financial statement and provide "
                    f"sentiment analysis with investment implications.\n\n"
                    f"### Input:\n{x['sentence']}\n\n"
                    f"### Response:\n"
                    f"Sentiment: {label_map.get(x['label'], 'neutral')}. "
                    + _investment_context(x["sentence"])
                ),
            },
            remove_columns=fpb.column_names,
        )
        parts.append(fpb)
        logger.info(f"financial_phrasebank: {len(fpb)} samples")
    except Exception as e:
        logger.warning(f"Could not load financial_phrasebank: {e}")

    # Sujet finance filtered for investment keywords
    ds = load_dataset("sujet-ai/Sujet-Finance-Instruct-177k", split="train")
    investment_keywords = [
        "invest", "portfolio", "stock", "bond", "equity",
        "dividend", "etf", "mutual fund", "asset allocation",
        "market", "hedge", "derivative", "option", "futures",
        "yield", "return", "risk", "diversif", "valuation",
        "rebalancing", "bull", "bear", "capitalization",
        "share price", "earnings", "ipo", "securities",
    ]

    def is_investment(example):
        text = (
            str(example.get(_COL_USER, "")) + " "
            + str(example.get(_COL_ANSWER, "")) + " "
            + str(example.get(_COL_INPUT, ""))
        ).lower()
        return any(kw in text for kw in investment_keywords)

    inv = ds.filter(is_investment)
    inv = inv.map(_format_sujet, remove_columns=inv.column_names)
    parts.append(inv)
    logger.info(f"Sujet investment filter: {len(inv)} samples")

    combined = concatenate_datasets(parts)
    logger.info(f"Investment dataset total: {len(combined)} samples")

    if len(combined) > max_samples:
        combined = combined.shuffle(seed=42).select(range(max_samples))

    return combined


def load_combined_finance_dataset(max_samples: int = 50000) -> Dataset:
    """Load full combined finance dataset for shared attention LoRA."""
    ds = load_dataset("sujet-ai/Sujet-Finance-Instruct-177k", split="train")
    formatted = ds.map(_format_sujet, remove_columns=ds.column_names)

    if len(formatted) > max_samples:
        formatted = formatted.shuffle(seed=42).select(range(max_samples))

    logger.info(f"Combined finance dataset: {len(formatted)} samples")
    return formatted


def load_rl_prompts(n_prompts: int = 2000) -> List[str]:
    """Load diverse finance prompts for RL training phase."""
    ds = load_dataset("sujet-ai/Sujet-Finance-Instruct-177k", split="train")
    ds = ds.shuffle(seed=123).select(range(min(n_prompts, len(ds))))

    prompts = []
    for row in ds:
        system = str(row.get(_COL_SYSTEM, "") or "")
        user = str(row.get(_COL_USER, "") or "")
        prompt = ""
        if system:
            prompt += f"### System:\n{system}\n\n"
        prompt += f"### Instruction:\n{user}\n\n### Response:\n"
        prompts.append(prompt)

    logger.info(f"Loaded {len(prompts)} RL prompts")
    return prompts


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _format_sujet(example) -> Dict[str, str]:
    """Format a Sujet Finance example into instruction-following text."""
    system = str(example.get(_COL_SYSTEM, "") or "")
    user = str(example.get(_COL_USER, "") or "")
    answer = str(example.get(_COL_ANSWER, "") or "")

    text = ""
    if system:
        text += f"### System:\n{system}\n\n"
    text += f"### Instruction:\n{user}\n\n### Response:\n{answer}"

    return {"text": text}


def _investment_context(sentence: str) -> str:
    """Generate brief investment analysis context from a sentence."""
    s = sentence.lower()
    if any(w in s for w in ["profit", "growth", "increase", "positive", "rose", "gain"]):
        return "This indicates positive market sentiment with potential upside for investors."
    if any(w in s for w in ["loss", "decline", "decrease", "negative", "fell", "drop"]):
        return "This signals bearish conditions; investors should consider risk mitigation."
    return "Neutral outlook; recommend monitoring for directional signals."
