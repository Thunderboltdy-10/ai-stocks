"""Data quality and leakage validation for training datasets."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from statsmodels.tsa.stattools import adfuller

    HAS_ADF = True
except Exception:  # pragma: no cover
    HAS_ADF = False


@dataclass
class ValidationResult:
    passed: bool
    details: Dict


class DataValidator:
    def __init__(self, target_col: str = "target_1d") -> None:
        self.target_col = target_col

    def validate_no_leakage(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        corr_threshold: float = 0.95,
    ) -> ValidationResult:
        if self.target_col not in df.columns:
            return ValidationResult(False, {"error": f"Missing target column {self.target_col}"})

        corr = df[feature_cols + [self.target_col]].corr(numeric_only=True)[self.target_col].drop(self.target_col)
        flagged = corr[corr.abs() > corr_threshold].sort_values(key=np.abs, ascending=False)

        return ValidationResult(
            passed=len(flagged) == 0,
            details={
                "threshold": corr_threshold,
                "max_abs_corr": float(corr.abs().max()) if len(corr) else 0.0,
                "flagged_features": flagged.to_dict(),
            },
        )

    def validate_no_nans(self, df: pd.DataFrame, cols: List[str]) -> ValidationResult:
        nan_counts = df[cols].isna().sum()
        bad = nan_counts[nan_counts > 0]
        return ValidationResult(
            passed=len(bad) == 0,
            details={"nan_columns": bad.to_dict(), "total_nan": int(nan_counts.sum())},
        )

    def validate_feature_stationarity(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        max_features: int = 12,
        p_value_threshold: float = 0.05,
    ) -> ValidationResult:
        if not HAS_ADF:
            return ValidationResult(True, {"skipped": True, "reason": "statsmodels unavailable"})

        sampled = feature_cols[:max_features]
        p_values: Dict[str, float] = {}

        for col in sampled:
            series = df[col].dropna()
            if len(series) < 60:
                continue
            try:
                p_values[col] = float(adfuller(series, autolag="AIC")[1])
            except Exception:
                continue

        non_stationary = {k: v for k, v in p_values.items() if v > p_value_threshold}
        pass_rate = (len(p_values) - len(non_stationary)) / max(len(p_values), 1)

        return ValidationResult(
            passed=pass_rate >= 0.5,
            details={
                "tested_features": len(p_values),
                "pass_rate": float(pass_rate),
                "non_stationary": non_stationary,
            },
        )

    def validate_target_distribution(
        self,
        df: pd.DataFrame,
        min_positive_pct: float = 0.30,
        max_positive_pct: float = 0.70,
    ) -> ValidationResult:
        if self.target_col not in df.columns:
            return ValidationResult(False, {"error": f"Missing target column {self.target_col}"})

        y = df[self.target_col].dropna()
        if len(y) == 0:
            return ValidationResult(False, {"error": "Empty target series"})

        positive_pct = float((y > 0).mean())
        passed = min_positive_pct <= positive_pct <= max_positive_pct
        return ValidationResult(
            passed=passed,
            details={
                "positive_pct": positive_pct,
                "negative_pct": float((y < 0).mean()),
                "mean": float(y.mean()),
                "std": float(y.std()),
            },
        )

    def run_all_validations(self, df: pd.DataFrame, feature_cols: List[str]) -> Dict:
        checks = {
            "no_leakage": self.validate_no_leakage(df, feature_cols),
            "no_nans": self.validate_no_nans(df, feature_cols + [self.target_col]),
            "stationarity": self.validate_feature_stationarity(df, feature_cols),
            "target_distribution": self.validate_target_distribution(df),
        }

        report = {
            "passed": all(v.passed for v in checks.values()),
            "checks": {k: {"passed": v.passed, "details": v.details} for k, v in checks.items()},
        }

        if not report["passed"]:
            logger.warning("Data validation failed for one or more checks")
        return report
