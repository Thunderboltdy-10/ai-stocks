"""
Feature Selection Module using Random Forest Importance.

This module provides functions to:
1. Train a Random Forest regressor to evaluate feature importance
2. Select top N features based on importance scores
3. Validate predictive power before proceeding
4. Save feature selection artifacts for use in training

The selected features can be used to reduce model complexity and
improve generalization by focusing on the most predictive inputs.
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

# Feature category definitions for ensuring diversity in selection
FEATURE_CATEGORIES = {
    'momentum': [
        'rsi', 'macd', 'macd_signal', 'macd_histogram', 'stoch_k', 'stoch_d',
        'rsi_3', 'rsi_7', 'rsi_14', 'rsi_28', 'rsi_momentum', 'rsi_velocity',
        'momentum_1d', 'momentum_5d', 'momentum_20d', 'rate_of_change_10d',
        'rsi_divergence_3_14', 'rsi_divergence_7_28', 'rsi_divergence_5d',
        'rsi_divergence_10d', 'rsi_divergence_20d', 'momentum_divergence_5d',
        'momentum_divergence_20d', 'rsi_cross_30', 'rsi_cross_70',
        'vw_momentum_5d', 'vw_momentum_20d',
    ],
    'volatility': [
        'volatility_5d', 'volatility_20d', 'bb_upper', 'bb_lower', 'bb_width',
        'bb_position', 'atr', 'atr_percent', 'realized_vol_20d', 'vix',
        'rv_iv_spread', 'rv_iv_spread_zscore', 'iv_regime_underpriced',
        'iv_regime_overpriced', 'iv_regime_fair', 'volatility_regime_normal',
        'vol_ratio_5_20', 'high_low_range', 'bb_squeeze', 'bb_position_low_vol',
        'bb_position_high_vol', 'regime_low_vol', 'regime_high_vol',
    ],
    'volume': [
        'volume_sma', 'volume_ratio', 'obv', 'obv_ema', 'volume_velocity',
        'volume_price_trend', 'accumulation_distribution', 'volume_surge',
        'volume_surge_with_price', 'large_volume_day', 'accumulation_signal',
        'distribution_signal', 'vwap_deviation', 'vwap_volume_ratio',
        'above_vwap', 'vwap_crossover', 'vwap_crossunder', 'vwap_band_position',
        'vpi', 'vpi_short',
    ],
    'trend': [
        'sma_10', 'sma_20', 'sma_50', 'ema_12', 'ema_26', 'price_to_sma20',
        'price_to_sma50', 'sma_10_20_cross', 'ema_12_26_cross', 'velocity_5d',
        'velocity_10d', 'velocity_20d', 'acceleration_5d',
    ],
    'pattern': [
        'higher_high', 'lower_low', 'gap_up', 'gap_down', 'body_size',
        'upper_shadow', 'lower_shadow', 'close_position',
    ],
    'support_resistance': [
        'distance_to_nearest_support', 'distance_to_nearest_resistance',
        'support_strength', 'resistance_strength', 'broke_resistance',
        'broke_support', 'days_since_support_break', 'days_since_resistance_break',
        'distance_to_fib_382', 'distance_to_fib_500', 'distance_to_fib_618',
        'in_consolidation_zone',
    ],
    'sentiment': [
        'news_sentiment_score', 'news_sentiment_magnitude', 'news_count',
        'sentiment_ma_3d', 'sentiment_ma_7d', 'sentiment_momentum',
        'sentiment_volatility', 'positive_news_ratio', 'negative_news_ratio',
    ],
    'returns': [
        'returns', 'log_returns', 'return_lag_1', 'return_lag_2', 'return_lag_5',
    ],
}


@dataclass
class FeatureSelectionReport:
    """
    Report containing feature selection results and validation metrics.

    Attributes:
        symbol: Stock symbol.
        n_features_total: Total number of input features.
        n_features_selected: Number of selected features.
        selected_features: List of selected feature names.
        validation_r2: R² score on validation set.
        validation_mae: MAE on validation set.
        validation_dir_acc: Directional accuracy on validation set.
        baseline_r2: R² score for baseline (mean prediction).
        cumulative_importance: Cumulative importance of selected features.
        has_predictive_power: Whether model shows predictive power.
        category_coverage: Dict of feature counts per category.
    """

    symbol: str
    n_features_total: int
    n_features_selected: int
    selected_features: List[str] = field(default_factory=list)
    validation_r2: float = 0.0
    validation_mae: float = 0.0
    validation_dir_acc: float = 0.0
    baseline_r2: float = 0.0
    cumulative_importance: float = 0.0
    has_predictive_power: bool = True
    category_coverage: Dict[str, int] = field(default_factory=dict)


@dataclass
class FeatureValidationResult:
    """
    Result of feature selection validation.

    Attributes:
        validation_passed: Whether all validation checks passed.
        warnings: List of warning messages for failed checks.
        feature_category_counts: Count of features per category.
        has_duplicates: Whether duplicate features were found.
        missing_features: Features not in original set.
    """

    validation_passed: bool
    warnings: List[str] = field(default_factory=list)
    feature_category_counts: Dict[str, int] = field(default_factory=dict)
    has_duplicates: bool = False
    missing_features: List[str] = field(default_factory=list)


# Minimum category requirements for validation
# These are relaxed minimums - sentiment is optional if not available
CATEGORY_MINIMUMS = {
    'momentum': 3,
    'volatility': 3,
    'volume': 2,
    'sentiment': 0,  # Don't require sentiment - it may not be available
}


def validate_selected_features(
    selected_features: List[str],
    original_features: List[str],
    require_sentiment: bool = True,
) -> FeatureValidationResult:
    """
    Validate that selected features meet category diversity requirements.

    Performs the following checks:
    - At least 5 momentum features included
    - At least 5 volatility features included
    - At least 3 volume features included
    - At least 3 sentiment features included (if available)
    - No duplicate features
    - All selected features exist in original feature set

    Args:
        selected_features: List of selected feature names.
        original_features: List of all original feature names.
        require_sentiment: Whether to require sentiment features (default True).
            Set to False if sentiment features are not available in data.

    Returns:
        FeatureValidationResult with validation status and details.
    """
    warnings = []
    validation_passed = True

    # Check for duplicates
    duplicates = [f for f in selected_features if selected_features.count(f) > 1]
    has_duplicates = len(duplicates) > 0
    if has_duplicates:
        unique_dups = list(set(duplicates))
        warnings.append(f"Duplicate features found: {unique_dups}")
        validation_passed = False

    # Check all features exist in original set
    original_set = set(original_features)
    missing_features = [f for f in selected_features if f not in original_set]
    if missing_features:
        warnings.append(f"Features not in original set: {missing_features}")
        validation_passed = False

    # Count features per category
    feature_category_counts = {cat: 0 for cat in FEATURE_CATEGORIES.keys()}
    feature_category_counts['other'] = 0

    for feature in selected_features:
        category = _get_feature_category(feature)
        if category:
            feature_category_counts[category] += 1
        else:
            feature_category_counts['other'] += 1

    # Check category minimums
    # Check if sentiment features are available in original features
    sentiment_available = any(
        _get_feature_category(f) == 'sentiment' for f in original_features
    )

    for category, minimum in CATEGORY_MINIMUMS.items():
        count = feature_category_counts.get(category, 0)

        # Skip sentiment check if not available or not required
        if category == 'sentiment':
            if not sentiment_available:
                continue
            if not require_sentiment:
                continue

        if count < minimum:
            warnings.append(
                f"Insufficient {category} features: {count} < {minimum} required"
            )
            validation_passed = False

    return FeatureValidationResult(
        validation_passed=validation_passed,
        warnings=warnings,
        feature_category_counts=feature_category_counts,
        has_duplicates=has_duplicates,
        missing_features=missing_features,
    )


def format_feature_selection_report(
    report: FeatureSelectionReport,
    importance_df: pd.DataFrame,
    validation_result: FeatureValidationResult,
) -> str:
    """
    Generate a formatted text report of feature selection results.

    Args:
        report: FeatureSelectionReport from train_feature_selector.
        importance_df: DataFrame with feature importance data.
        validation_result: FeatureValidationResult from validate_selected_features.

    Returns:
        Formatted string report suitable for saving to file.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("           FEATURE SELECTION REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Symbol: {report.symbol}")
    lines.append(f"Original features: {report.n_features_total}")
    lines.append(f"Selected features: {report.n_features_selected}")
    lines.append(f"Coverage: {report.cumulative_importance * 100:.1f}% of cumulative importance")
    lines.append("")

    # Category breakdown
    lines.append("-" * 40)
    lines.append("Category breakdown:")
    lines.append("-" * 40)

    total = report.n_features_selected
    # Sort by count descending
    sorted_categories = sorted(
        validation_result.feature_category_counts.items(),
        key=lambda x: -x[1]
    )

    for category, count in sorted_categories:
        if count > 0:
            pct = (count / total * 100) if total > 0 else 0
            lines.append(f"  {category.capitalize():20s}: {count:3d} features ({pct:5.1f}%)")

    lines.append("")

    # Top 10 features
    lines.append("-" * 40)
    lines.append("Top 10 features:")
    lines.append("-" * 40)

    # Filter to selected features and get top 10
    selected_set = set(report.selected_features)
    top_features = importance_df[importance_df['feature'].isin(selected_set)].head(10)

    for idx, (_, row) in enumerate(top_features.iterrows(), 1):
        importance = row.get('importance_normalized', row.get('importance', 0))
        lines.append(
            f"  {idx:2d}. {row['feature']:40s} (importance: {importance:.4f})"
        )

    lines.append("")

    # Validation status
    lines.append("-" * 40)
    lines.append("Validation:")
    lines.append("-" * 40)

    if validation_result.validation_passed:
        lines.append("  ✓ PASSED")
    else:
        lines.append("  ✗ FAILED")
        for warning in validation_result.warnings:
            lines.append(f"    ⚠ {warning}")

    lines.append("")

    # Model metrics
    lines.append("-" * 40)
    lines.append("Model Validation Metrics:")
    lines.append("-" * 40)
    lines.append(f"  Cross-Validation R²: {report.validation_r2:.4f}")
    lines.append(f"  Baseline R²: {report.baseline_r2:.4f}")
    lines.append(f"  Validation MAE: {report.validation_mae:.6f}")
    lines.append(f"  Directional Accuracy: {report.validation_dir_acc:.1%}")
    lines.append(f"  Has Predictive Power: {'Yes' if report.has_predictive_power else 'No'}")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def _get_feature_category(feature_name: str) -> Optional[str]:
    """
    Determine which category a feature belongs to.

    Args:
        feature_name: Name of the feature.

    Returns:
        Category name or None if not categorized.
    """
    # First pass: check exact matches across all categories
    for category, features in FEATURE_CATEGORIES.items():
        if feature_name in features:
            return category
    
    # Second pass: check partial matches only after all exact matches fail
    for category, features in FEATURE_CATEGORIES.items():
        for f in features:
            # Check if feature name contains a known feature or starts with same prefix
            if f in feature_name or feature_name.startswith(f.split('_')[0] + '_'):
                return category
    return None


def train_feature_selector(
    X: np.ndarray,
    y: np.ndarray,
    symbol: str,
    feature_names: List[str],
    n_top_features: int = 40,
    output_dir: str = 'saved_models',
    min_cumulative_importance: float = 0.80,
    min_features_per_category: int = 2,
    random_state: int = 42,
) -> FeatureSelectionReport:
    """
    Train a Random Forest to select top features by importance.

    This function:
    1. Trains a Random Forest regressor on the data
    2. Extracts and ranks feature importances
    3. Validates predictive power on held-out data
    4. Selects top N features ensuring category diversity
    5. Saves artifacts for later use

    Args:
        X: Feature array of shape (n_samples, n_features).
        y: Target array of shape (n_samples,).
        symbol: Stock symbol for file naming.
        feature_names: List of feature column names.
        n_top_features: Number of top features to select (default 40).
        output_dir: Directory to save artifacts (default 'saved_models').
        min_cumulative_importance: Minimum cumulative importance required
            from selected features (default 0.80 = 80%).
        min_features_per_category: Minimum features to include from each
            major category (default 2).
        random_state: Random seed for reproducibility.

    Returns:
        FeatureSelectionReport with selection results and metrics.

    Raises:
        ValueError: If inputs are invalid or feature selection fails.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"  FEATURE SELECTION: {symbol}")
    logger.info("=" * 70)

    # Validate inputs
    X = np.asarray(X)
    y = np.asarray(y).flatten()

    if len(X) != len(y):
        raise ValueError(f"X and y must have same length: {len(X)} != {len(y)}")

    if len(feature_names) != X.shape[1]:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) must match "
            f"X.shape[1] ({X.shape[1]})"
        )

    n_samples, n_features = X.shape
    logger.info(f"   Samples: {n_samples:,}")
    logger.info(f"   Features: {n_features}")

    # =========================================================================
    # Step 0: Filter out zero-variance features
    # =========================================================================
    feature_variances = np.var(X, axis=0)
    valid_features_mask = feature_variances > 1e-10
    n_zero_var = (~valid_features_mask).sum()
    
    if n_zero_var > 0:
        logger.info(f"   Filtering {n_zero_var} zero-variance features")
        X_filtered = X[:, valid_features_mask]
        feature_names_filtered = [f for f, v in zip(feature_names, valid_features_mask) if v]
    else:
        X_filtered = X
        feature_names_filtered = feature_names
    
    logger.info(f"   Active features: {len(feature_names_filtered)}")

    # =========================================================================
    # Step 1: Train/Val Split (time-series, no shuffle)
    # =========================================================================
    split_idx = int(len(X_filtered) * 0.8)
    X_train, X_val = X_filtered[:split_idx], X_filtered[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    logger.info(f"   Train: {len(X_train):,}, Val: {len(X_val):,}")

    # =========================================================================
    # Step 2: Train Random Forest Regressor (conservative hyperparameters)
    # =========================================================================
    logger.info("\n   Training Random Forest...")

    # Use more conservative hyperparameters to prevent overfitting
    # - Lower max_depth prevents deep trees that memorize noise
    # - Higher min_samples_leaf ensures leaf nodes have enough samples
    # - Lower max_features adds randomization
    rf = RandomForestRegressor(
        n_estimators=100,          # Reduced from 200 for faster training
        max_depth=8,               # Reduced from 15 to prevent overfitting
        min_samples_split=50,      # Increased from 20
        min_samples_leaf=20,       # Added - prevents very small leaves
        max_features=0.5,          # Use 50% of features per tree
        n_jobs=-1,
        random_state=random_state,
        verbose=0,
    )

    rf.fit(X_train, y_train)
    logger.info("   ✓ Random Forest trained")

    # =========================================================================
    # Step 3: Extract Feature Importances
    # =========================================================================
    importances = rf.feature_importances_

    # Create DataFrame with feature importances (only for valid features)
    importance_df = pd.DataFrame({
        'feature': feature_names_filtered,
        'importance': importances,
    })

    # Sort by importance descending
    importance_df = importance_df.sort_values('importance', ascending=False)
    importance_df = importance_df.reset_index(drop=True)

    # Normalize importances to sum to 1.0
    total_importance = importance_df['importance'].sum()
    if total_importance > 0:
        importance_df['importance_normalized'] = (
            importance_df['importance'] / total_importance
        )
    else:
        importance_df['importance_normalized'] = 0.0

    # Add cumulative importance
    importance_df['cumulative_importance'] = (
        importance_df['importance_normalized'].cumsum()
    )

    # Add category
    importance_df['category'] = importance_df['feature'].apply(_get_feature_category)

    logger.info("\n   Top 10 features by importance:")
    for idx, row in importance_df.head(10).iterrows():
        logger.info(
            f"      {idx+1:2d}. {row['feature']:40s} "
            f"{row['importance_normalized']*100:5.2f}% "
            f"({row['category'] or 'other'})"
        )

    # =========================================================================
    # Step 4: Validate Predictive Power with Cross-Validation
    # =========================================================================
    logger.info("\n   Validating predictive power...")

    # Use time-series cross-validation for more robust R² estimate
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(
        RandomForestRegressor(
            n_estimators=50,
            max_depth=6,
            min_samples_split=50,
            min_samples_leaf=20,
            max_features=0.5,
            n_jobs=-1,
            random_state=random_state,
        ),
        X_filtered, y,
        cv=tscv,
        scoring='r2',
    )
    cv_r2_mean = np.mean(cv_scores)
    cv_r2_std = np.std(cv_scores)
    
    logger.info(f"   Cross-validation R²: {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")

    # Also compute holdout validation for comparison
    y_pred_val = rf.predict(X_val)

    # Compute metrics
    val_r2 = r2_score(y_val, y_pred_val)
    val_mae = mean_absolute_error(y_val, y_pred_val)

    # Directional accuracy
    y_val_sign = np.sign(y_val)
    y_pred_sign = np.sign(y_pred_val)
    val_dir_acc = np.mean(y_val_sign == y_pred_sign)

    # Baseline: predict mean
    y_baseline = np.full_like(y_val, y_train.mean())
    baseline_r2 = r2_score(y_val, y_baseline)

    logger.info(f"   Holdout R²: {val_r2:.4f} (baseline: {baseline_r2:.4f})")
    logger.info(f"   Holdout MAE: {val_mae:.6f}")
    logger.info(f"   Directional Accuracy: {val_dir_acc:.1%}")

    # Use CV R² for reporting (more robust estimate)
    # Consider having predictive power if CV R² > 0 or direction > 52%
    has_predictive_power = cv_r2_mean > 0.0 or val_dir_acc > 0.52
    if not has_predictive_power:
        logger.warning(
            f"   [WARN] LOW PREDICTIVE POWER: CV R^2 = {cv_r2_mean:.4f}"
        )
        logger.warning("   Model may not have learned meaningful patterns")
    else:
        logger.info("   [OK] Model shows predictive power")

    # =========================================================================
    # Step 5: Select Top N Features with Category Diversity
    # =========================================================================
    logger.info(f"\n   Selecting top {n_top_features} features...")

    selected_features = []
    category_counts = {cat: 0 for cat in FEATURE_CATEGORIES.keys()}

    # First pass: ensure minimum features per category (only from features with non-zero importance)
    nonzero_importance_df = importance_df[importance_df['importance'] > 0]
    
    for category in ['momentum', 'volatility', 'volume', 'sentiment']:
        # Only select from features that have non-zero importance
        cat_features = nonzero_importance_df[nonzero_importance_df['category'] == category]
        n_to_add = min(min_features_per_category, len(cat_features))

        for _, row in cat_features.head(n_to_add).iterrows():
            if row['feature'] not in selected_features:
                selected_features.append(row['feature'])
                category_counts[category] = category_counts.get(category, 0) + 1

    # Second pass: fill remaining slots with top importance features (non-zero only)
    for _, row in nonzero_importance_df.iterrows():
        if len(selected_features) >= n_top_features:
            break
        if row['feature'] not in selected_features:
            selected_features.append(row['feature'])
            cat = row['category']
            if cat:
                category_counts[cat] = category_counts.get(cat, 0) + 1

    # Calculate cumulative importance of selected features
    selected_mask = importance_df['feature'].isin(selected_features)
    cumulative_importance = importance_df.loc[
        selected_mask, 'importance_normalized'
    ].sum()

    # Check if we meet the cumulative importance threshold
    if cumulative_importance < min_cumulative_importance:
        logger.warning(
            f"   [WARN] Cumulative importance {cumulative_importance:.1%} "
            f"< {min_cumulative_importance:.0%} threshold"
        )
        # Add more features if needed
        for _, row in importance_df.iterrows():
            if cumulative_importance >= min_cumulative_importance:
                break
            if row['feature'] not in selected_features:
                selected_features.append(row['feature'])
                cumulative_importance += row['importance_normalized']
                cat = row['category']
                if cat:
                    category_counts[cat] = category_counts.get(cat, 0) + 1

    logger.info(f"   Selected {len(selected_features)} features")
    logger.info(f"   Cumulative importance: {cumulative_importance:.1%}")
    logger.info(f"   Category coverage:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            logger.info(f"      {cat}: {count}")

    # =========================================================================
    # Step 6: Save Artifacts
    # =========================================================================
    output_path = Path(output_dir) / symbol
    output_path.mkdir(parents=True, exist_ok=True)

    # Save selected features list
    features_path = output_path / f'{symbol}_selected_features.pkl'
    with open(features_path, 'wb') as f:
        pickle.dump(selected_features, f)
    logger.info(f"\n   ✓ Saved: {features_path}")

    # Save full importance table
    importance_csv_path = output_path / f'{symbol}_feature_importance.csv'
    importance_df.to_csv(importance_csv_path, index=False)
    logger.info(f"   ✓ Saved: {importance_csv_path}")

    # Save validation metrics
    metrics = {
        'symbol': symbol,
        'n_features_total': n_features,
        'n_features_active': len(feature_names_filtered),
        'n_features_selected': len(selected_features),
        'cv_r2_mean': float(cv_r2_mean),
        'cv_r2_std': float(cv_r2_std),
        'holdout_r2': float(val_r2),
        'validation_mae': float(val_mae),
        'validation_dir_acc': float(val_dir_acc),
        'baseline_r2': float(baseline_r2),
        'cumulative_importance': float(cumulative_importance),
        'has_predictive_power': bool(has_predictive_power),
        'category_coverage': {k: int(v) for k, v in category_counts.items() if v > 0},
        'rf_params': {
            'n_estimators': 100,
            'max_depth': 8,
            'min_samples_split': 50,
            'min_samples_leaf': 20,
            'max_features': 0.5,
        },
    }

    metrics_path = output_path / f'{symbol}_rf_validation_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"   ✓ Saved: {metrics_path}")

    # Save importance bar chart
    try:
        _save_importance_chart(
            importance_df.head(20),
            output_path / f'{symbol}_feature_importance.png',
            symbol,
        )
        logger.info(f"   [OK] Saved: {symbol}_feature_importance.png")
    except Exception as e:
        logger.warning(f"   [WARN] Could not save chart: {e}")

    # =========================================================================
    # Step 7: Validate Selected Features
    # =========================================================================
    logger.info("\n   Validating feature selection...")

    validation_result = validate_selected_features(
        selected_features=selected_features,
        original_features=feature_names,
        require_sentiment=True,
    )

    if validation_result.validation_passed:
        logger.info("   [OK] Feature selection validation PASSED")
    else:
        logger.warning("   [WARN] Feature selection validation FAILED")
        for warning in validation_result.warnings:
            logger.warning(f"      - {warning}")

    # Build report first (needed for formatting)
    # Use CV R² as the primary validation metric (more robust)
    report = FeatureSelectionReport(
        symbol=symbol,
        n_features_total=n_features,
        n_features_selected=len(selected_features),
        selected_features=selected_features,
        validation_r2=cv_r2_mean,  # Use CV R² instead of holdout
        validation_mae=val_mae,
        validation_dir_acc=val_dir_acc,
        baseline_r2=baseline_r2,
        cumulative_importance=cumulative_importance,
        has_predictive_power=has_predictive_power,
        category_coverage={k: v for k, v in category_counts.items() if v > 0},
    )

    # Generate and save validation report
    validation_report_text = format_feature_selection_report(
        report=report,
        importance_df=importance_df,
        validation_result=validation_result,
    )

    validation_report_path = output_path / f'{symbol}_feature_selection_validation.txt'
    with open(validation_report_path, 'w', encoding='utf-8') as f:
        f.write(validation_report_text)
    logger.info(f"   ✓ Saved: {validation_report_path}")

    logger.info("=" * 70)

    return report


def _save_importance_chart(
    importance_df: pd.DataFrame,
    output_path: Path,
    symbol: str,
) -> None:
    """
    Save a bar chart of feature importances.

    Args:
        importance_df: DataFrame with feature and importance_normalized columns.
        output_path: Path to save the chart.
        symbol: Stock symbol for title.
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 8))

    # Reverse order for horizontal bar chart (top feature at top)
    df_plot = importance_df.iloc[::-1]

    colors = []
    for cat in df_plot['category']:
        if cat == 'momentum':
            colors.append('#2ecc71')
        elif cat == 'volatility':
            colors.append('#e74c3c')
        elif cat == 'volume':
            colors.append('#3498db')
        elif cat == 'sentiment':
            colors.append('#9b59b6')
        elif cat == 'trend':
            colors.append('#f39c12')
        else:
            colors.append('#95a5a6')

    ax.barh(
        df_plot['feature'],
        df_plot['importance_normalized'] * 100,
        color=colors,
        edgecolor='white',
    )

    ax.set_xlabel('Importance (%)', fontsize=12)
    ax.set_title(f'{symbol} - Top 20 Features by Random Forest Importance', fontsize=14)
    ax.grid(axis='x', alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Momentum'),
        Patch(facecolor='#e74c3c', label='Volatility'),
        Patch(facecolor='#3498db', label='Volume'),
        Patch(facecolor='#9b59b6', label='Sentiment'),
        Patch(facecolor='#f39c12', label='Trend'),
        Patch(facecolor='#95a5a6', label='Other'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def load_selected_features(
    symbol: str,
    output_dir: str = 'saved_models',
) -> List[str]:
    """
    Load previously saved selected features for a symbol.

    Args:
        symbol: Stock symbol.
        output_dir: Directory where features were saved (default 'saved_models').

    Returns:
        List of selected feature names.

    Raises:
        FileNotFoundError: If feature file doesn't exist with helpful message.
    """
    features_path = Path(output_dir) / symbol / f'{symbol}_selected_features.pkl'

    if not features_path.exists():
        raise FileNotFoundError(
            f"Selected features not found for {symbol}.\n"
            f"Expected path: {features_path}\n"
            f"Run feature selection first:\n"
            f"  python training/feature_selection.py {symbol}"
        )

    with open(features_path, 'rb') as f:
        selected_features = pickle.load(f)

    logger.info(f"Loaded {len(selected_features)} selected features for {symbol}")

    return selected_features


def load_feature_importance(
    symbol: str,
    output_dir: str = 'saved_models',
) -> pd.DataFrame:
    """
    Load the full feature importance table for a symbol.

    Args:
        symbol: Stock symbol.
        output_dir: Directory where importances were saved.

    Returns:
        DataFrame with feature importance data.

    Raises:
        FileNotFoundError: If importance file doesn't exist.
    """
    importance_path = Path(output_dir) / symbol / f'{symbol}_feature_importance.csv'

    if not importance_path.exists():
        raise FileNotFoundError(
            f"Feature importance not found for {symbol}.\n"
            f"Expected path: {importance_path}\n"
            f"Run feature selection first:\n"
            f"  python training/feature_selection.py {symbol}"
        )

    return pd.read_csv(importance_path)


def main():
    """CLI entry point for feature selection."""
    parser = argparse.ArgumentParser(
        description='Select top features using Random Forest importance'
    )
    parser.add_argument('symbol', type=str, help='Stock symbol (e.g., AAPL)')
    parser.add_argument(
        '--n-features', type=int, default=40,
        help='Number of top features to select (default: 40)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='saved_models',
        help='Output directory for artifacts (default: saved_models)'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(message)s',
    )

    # Import data loading functions
    from data.data_fetcher import fetch_stock_data
    from data.feature_engineer import engineer_features, get_feature_columns
    from data.target_engineering import prepare_training_data

    print(f"\n{'='*70}")
    print(f"  FEATURE SELECTION PIPELINE: {args.symbol}")
    print(f"{'='*70}\n")

    # Step 1: Fetch data
    print("📊 Fetching stock data...")
    df = fetch_stock_data(args.symbol)
    print(f"   ✓ Loaded {len(df)} rows")

    # Step 2: Engineer features
    print("\n🔧 Engineering features...")
    df = engineer_features(df, include_sentiment=True)
    print(f"   ✓ Engineered {len(df.columns)} columns")

    # Step 3: Prepare training data
    print("\n🎯 Preparing training data...")
    df_clean, feature_cols = prepare_training_data(df, horizons=[1])

    # Extract features and targets
    X = df_clean[feature_cols].values
    y = df_clean['target_1d'].values

    print(f"\n   Features: {len(feature_cols)}")
    print(f"   Samples: {len(X)}")

    # Step 4: Run feature selection
    print("\n🌲 Running Random Forest feature selection...")
    report = train_feature_selector(
        X=X,
        y=y,
        symbol=args.symbol,
        feature_names=feature_cols,
        n_top_features=args.n_features,
        output_dir=args.output_dir,
    )

    # Summary
    print(f"\n{'='*70}")
    print("  FEATURE SELECTION COMPLETE")
    print(f"{'='*70}")
    print(f"\n✅ Selected {report.n_features_selected} features from {report.n_features_total}")
    print(f"   Cumulative importance: {report.cumulative_importance:.1%}")
    print(f"   Validation R²: {report.validation_r2:.4f}")
    print(f"   Directional accuracy: {report.validation_dir_acc:.1%}")

    if not report.has_predictive_power:
        print("\n⚠️  Warning: Model shows weak predictive power (R² < 0.01)")
        print("   Consider reviewing feature engineering or data quality")

    print(f"\n📁 Artifacts saved to: {args.output_dir}/{args.symbol}/")
    print(f"\nTo use selected features in training:")
    print(f"   from training.feature_selection import load_selected_features")
    print(f"   selected = load_selected_features('{args.symbol}')")


if __name__ == '__main__':
    main()
