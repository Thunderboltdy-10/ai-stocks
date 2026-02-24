"""Predefined symbol universes for robust train/eval workflows.

Using fixed, diversified sets helps avoid accidental tuning to a tiny basket.
"""

from __future__ import annotations

from typing import Iterable, List

# 5-symbol quick iteration set (kept for fast local loops).
CORE_5 = ["AAPL", "TSLA", "XOM", "JPM", "KO"]

# 5-symbol out-of-set holdout used as robustness gate.
HOLDOUT_5 = ["AMZN", "MSFT", "NVDA", "PFE", "WMT"]

# 10-symbol intraday-compatible set currently used in the project.
INTRADAY_10 = CORE_5 + HOLDOUT_5

# 15-symbol diversified daily set spanning tech, energy, financials, healthcare, ETF.
DAILY_15 = INTRADAY_10 + ["BAC", "CVX", "DIS", "GOOGL", "SPY"]

# 15-symbol split with explicit train-vs-holdout grouping.
CORE_10 = ["AAPL", "TSLA", "XOM", "JPM", "KO", "BAC", "CVX", "DIS", "GOOGL", "SPY"]
HOLDOUT_5_EXT = HOLDOUT_5

SYMBOL_SETS = {
    "core5": CORE_5,
    "holdout5": HOLDOUT_5,
    "intraday10": INTRADAY_10,
    "daily15": DAILY_15,
    "core10": CORE_10,
    "holdout5_ext": HOLDOUT_5_EXT,
}

DEFAULT_CORE_SET = "core5"
DEFAULT_HOLDOUT_SET = "holdout5"
DEFAULT_INTRADAY_SET = "intraday10"
DEFAULT_DAILY_SET = "daily15"


def dedupe_symbols(symbols: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        s = str(symbol).strip().upper()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def parse_symbols_csv(symbols_csv: str | None) -> List[str]:
    if not symbols_csv:
        return []
    return dedupe_symbols(symbols_csv.split(","))


def get_symbol_set(name: str | None) -> List[str]:
    key = str(name or "").strip().lower()
    if not key:
        return []
    return list(SYMBOL_SETS.get(key, []))


def resolve_symbols(
    symbols_csv: str | None = None,
    *,
    symbol_set: str | None = None,
    fallback_set: str = DEFAULT_CORE_SET,
) -> List[str]:
    explicit = parse_symbols_csv(symbols_csv)
    if explicit:
        return explicit
    from_set = get_symbol_set(symbol_set)
    if from_set:
        return from_set
    fallback = get_symbol_set(fallback_set)
    return fallback if fallback else list(CORE_5)


def available_symbol_sets() -> List[str]:
    return sorted(SYMBOL_SETS.keys())
