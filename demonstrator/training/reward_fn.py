"""Rule-based finance reward function for REINFORCE training.

Scores responses on four axes:
  format      (0.4) - structure, completion, no hallucinated numbers
  terminology (0.3) - correct financial terms
  relevance   (0.2) - addresses the question
  safety      (0.1) - disclaimers for investment advice
"""

import re
from typing import Set

# ---------------------------------------------------------------------------
# Financial glossaries
# ---------------------------------------------------------------------------

BANKING_TERMS: Set[str] = {
    "deposit", "withdrawal", "interest rate", "apr", "apy", "fdic",
    "checking", "savings", "cd", "certificate of deposit", "loan",
    "mortgage", "refinance", "amortization", "collateral", "underwriting",
    "credit score", "fico", "overdraft", "wire transfer", "ach",
    "routing number", "escrow", "lien", "principal", "maturity",
}

INVESTMENT_TERMS: Set[str] = {
    "portfolio", "diversification", "asset allocation", "equity", "bond",
    "stock", "etf", "mutual fund", "index fund", "dividend", "yield",
    "p/e ratio", "market cap", "beta", "alpha", "sharpe ratio",
    "volatility", "hedge", "derivative", "option", "futures", "roi",
    "risk tolerance", "rebalancing", "dollar cost averaging", "bull",
    "bear", "valuation", "dcf", "eps", "revenue", "margin",
}

ALL_FINANCE_TERMS = BANKING_TERMS | INVESTMENT_TERMS

_DISCLAIMER_RES = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"not financial advice",
        r"consult.*(?:financial|professional|advisor)",
        r"do your own research",
        r"past performance.*(?:not|no).*(?:guarantee|indicat)",
        r"risk.*(?:losing|loss)",
    ]
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def finance_reward(prompt: str, response: str, **kwargs) -> float:
    """Return scalar reward in [0, 1]."""
    return max(0.0, min(1.0, (
        0.4 * _score_format(response)
        + 0.3 * _score_terminology(response)
        + 0.2 * _score_relevance(prompt, response)
        + 0.1 * _score_safety(prompt, response)
    )))


# ---------------------------------------------------------------------------
# Axis scorers
# ---------------------------------------------------------------------------


def _score_format(response: str) -> float:
    score = 0.0
    wc = len(response.split())
    if 20 <= wc <= 500:
        score += 0.3
    elif 10 <= wc < 20:
        score += 0.1

    if "\n" in response or "- " in response or "1." in response:
        score += 0.3

    # Penalise suspiciously precise decimals (hallucinated numbers)
    if not re.findall(r"\d{1,2}\.\d{3,}%", response):
        score += 0.2

    stripped = response.strip()
    if stripped and stripped[-1] in ".!?)\"'":
        score += 0.2

    return min(1.0, score)


def _score_terminology(response: str) -> float:
    rl = response.lower()
    n = sum(1 for t in ALL_FINANCE_TERMS if t in rl)
    if n >= 5:
        return 1.0
    if n >= 3:
        return 0.7
    if n >= 1:
        return 0.4
    return 0.0


def _score_relevance(prompt: str, response: str) -> float:
    pw = set(re.findall(r"\b[a-z]{4,}\b", prompt.lower()))
    rw = set(re.findall(r"\b[a-z]{4,}\b", response.lower()))
    if not pw:
        return 0.5
    return min(1.0, len(pw & rw) / len(pw) * 1.5)


def _score_safety(prompt: str, response: str) -> float:
    needs = any(
        kw in prompt.lower()
        for kw in ["invest", "buy", "sell", "portfolio", "stock", "recommend"]
    )
    if not needs:
        return 1.0
    for rx in _DISCLAIMER_RES:
        if rx.search(response):
            return 1.0
    return 0.2
