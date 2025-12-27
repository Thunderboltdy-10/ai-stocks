"""Utility helpers for feature selection and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SequentialFeatureSelector, mutual_info_regression
from sklearn.metrics import get_scorer
from sklearn.model_selection import TimeSeriesSplit


@dataclass
class FeatureSelectionResult:
    """Aggregate output from the ensemble feature selection pipeline."""

    selected_features: List[str]
    method_rankings: Dict[str, List[str]]
    correlation_matrix: pd.DataFrame
    correlation_plot: Optional[plt.Figure]
    random_forest_importances: pd.DataFrame
    random_forest_plot: Optional[plt.Figure]
    mutual_information_scores: pd.DataFrame
    mutual_information_plot: Optional[plt.Figure]
    sequential_results: pd.DataFrame
    sequential_plot: Optional[plt.Figure]
    ensemble_votes: pd.DataFrame
    venn_plot: Optional[plt.Figure]
    selection_matrix: pd.DataFrame


def _sanitize_matrix(matrix: np.ndarray) -> np.ndarray:
    """Replace NaNs/Infs to keep sklearn estimators happy."""

    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)


def _create_venn_figure(method_rankings: Dict[str, List[str]]) -> plt.Figure:
    """Create a simple three-set Venn diagram summarising method overlap."""

    methods = ["random_forest", "mutual_information", "sequential"]
    sets = [set(method_rankings.get(method, [])) for method in methods]

    if sum(len(s) for s in sets) == 0:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
        ax.text(0.5, 0.5, "No features selected", ha="center", va="center")
        ax.axis("off")
        return fig

    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    circle_positions = [(0.4, 0.5), (0.6, 0.5), (0.5, 0.7)]
    radius = 0.3
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for (x, y), color in zip(circle_positions, colors):
        ax.add_patch(Circle((x, y), radius, alpha=0.25, color=color, linewidth=2))

    labels = {
        "100": len(sets[0] - sets[1] - sets[2]),
        "010": len(sets[1] - sets[0] - sets[2]),
        "001": len(sets[2] - sets[0] - sets[1]),
        "110": len((sets[0] & sets[1]) - sets[2]),
        "011": len((sets[1] & sets[2]) - sets[0]),
        "101": len((sets[0] & sets[2]) - sets[1]),
        "111": len(sets[0] & sets[1] & sets[2]),
    }

    ax.text(0.25, 0.5, str(labels["100"]), ha="center", va="center", fontsize=12)
    ax.text(0.75, 0.5, str(labels["010"]), ha="center", va="center", fontsize=12)
    ax.text(0.5, 0.88, str(labels["001"]), ha="center", va="center", fontsize=12)
    ax.text(0.5, 0.45, str(labels["110"]), ha="center", va="center", fontsize=12)
    ax.text(0.6, 0.68, str(labels["011"]), ha="center", va="center", fontsize=12)
    ax.text(0.4, 0.68, str(labels["101"]), ha="center", va="center", fontsize=12)
    ax.text(0.5, 0.6, str(labels["111"]), ha="center", va="center", fontsize=12, fontweight="bold")

    ax.text(0.2, 0.2, "Random Forest", color=colors[0], fontsize=12)
    ax.text(0.8, 0.2, "Mutual Information", color=colors[1], fontsize=12, ha="right")
    ax.text(0.5, 0.95, "Sequential", color=colors[2], fontsize=12, ha="center")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Feature Selection Overlap")

    return fig


def analyze_feature_correlation(
    features_df: pd.DataFrame,
    threshold: float = 0.95,
    annotate: bool = False,
) -> Tuple[pd.DataFrame, List[str], List[str], plt.Figure]:
    """Identify and visualise highly correlated feature pairs."""

    corr_matrix = features_df.corr(method="pearson").fillna(0.0)
    upper_mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    upper_tri = corr_matrix.where(upper_mask)

    to_drop = [
        column
        for column in upper_tri.columns
        if (upper_tri[column].abs() > threshold).any()
    ]
    to_keep = [feature for feature in features_df.columns if feature not in to_drop]

    fig, ax = plt.subplots(figsize=(10, 8), dpi=120)
    im = ax.imshow(corr_matrix.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_matrix)))
    ax.set_xticklabels(corr_matrix.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr_matrix)))
    ax.set_yticklabels(corr_matrix.index, fontsize=8)
    ax.set_title("Feature Correlation Heatmap (|r|)")
    fig.colorbar(im, ax=ax, fraction=0.018, pad=0.05, label="Correlation")

    if annotate:
        for i in range(corr_matrix.shape[0]):
            for j in range(corr_matrix.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{corr_matrix.values[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="black",
                )

    fig.tight_layout()
    return corr_matrix, to_drop, to_keep, fig


def compute_feature_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    top_n: int = 30,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, List[str], plt.Figure]:
    """Rank features using a RandomForestRegressor ensemble."""

    estimator = RandomForestRegressor(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
        max_depth=None,
    )

    X_clean = _sanitize_matrix(np.asarray(X))
    y_clean = np.nan_to_num(np.asarray(y), nan=0.0)

    estimator.fit(X_clean, y_clean)
    importances = estimator.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    top_features = importance_df.head(top_n)["feature"].tolist()

    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    subset = importance_df.head(top_n)
    ax.barh(subset["feature"][::-1], subset["importance"][::-1])
    ax.set_xlabel("Importance Score")
    ax.set_title("Random Forest Feature Importance (Top N)")
    plt.tight_layout()

    return importance_df, top_features, fig


def compute_mutual_information(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    top_n: int = 30,
    random_state: Optional[int] = 42,
) -> Tuple[pd.DataFrame, List[str], plt.Figure]:
    """Score features using mutual information against the target."""

    X_clean = _sanitize_matrix(np.asarray(X))
    y_clean = np.nan_to_num(np.asarray(y), nan=0.0)

    mi_scores = mutual_info_regression(
        X_clean,
        y_clean,
        random_state=random_state,
    )

    mi_df = pd.DataFrame({"feature": feature_names, "mi_score": mi_scores}).sort_values(
        "mi_score", ascending=False
    )

    top_features = mi_df.head(top_n)["feature"].tolist()

    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    subset = mi_df.head(top_n)
    ax.barh(subset["feature"][::-1], subset["mi_score"][::-1], color="#1f77b4")
    ax.set_xlabel("Mutual Information")
    ax.set_title("Mutual Information Scores (Top N)")
    plt.tight_layout()

    return mi_df, top_features, fig


def sequential_feature_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    estimator=None,
    n_features: int = 25,
    feature_steps: Optional[Sequence[int]] = None,
    direction: str = "forward",
    scoring: str = "r2",
    n_splits: int = 5,
) -> Tuple[pd.DataFrame, List[str], plt.Figure]:
    """Run sequential feature selection with time-series aware splits."""

    if estimator is None:
        estimator = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)

    if feature_steps is None:
        feature_steps = [10, 15, 20, 25, 30]

    X_clean = _sanitize_matrix(np.asarray(X))
    y_clean = np.nan_to_num(np.asarray(y), nan=0.0)

    max_splits = min(n_splits, len(X_clean) - 1)
    if max_splits < 2:
        raise ValueError("Not enough samples to run TimeSeriesSplit for sequential selection.")

    tscv = TimeSeriesSplit(n_splits=max_splits)
    results = []
    best_features: List[str] = []
    best_score = -np.inf

    scorer = get_scorer(scoring)

    for k_features in feature_steps:
        if k_features <= 0 or k_features > len(feature_names):
            continue

        selector = SequentialFeatureSelector(
            estimator,
            n_features_to_select=k_features,
            direction=direction,
            scoring=scoring,
            cv=tscv,
            n_jobs=-1,
        )
        selector.fit(X_clean, y_clean)

        selected_idx = selector.get_support(indices=True)
        selected_features = [feature_names[idx] for idx in selected_idx]

        fold_scores: List[float] = []
        for train_idx, test_idx in tscv.split(X_clean):
            fold_estimator = clone(estimator)
            fold_estimator.fit(X_clean[train_idx][:, selected_idx], y_clean[train_idx])
            fold_score = scorer(
                fold_estimator,
                X_clean[test_idx][:, selected_idx],
                y_clean[test_idx],
            )
            fold_scores.append(float(fold_score))

        mean_score = float(np.mean(fold_scores))
        results.append(
            {
                "n_features": k_features,
                "selected_features": selected_features,
                "score": mean_score,
            }
        )

        if mean_score > best_score:
            best_score = mean_score
            best_features = selected_features

    if not results:
        raise ValueError("Sequential feature selection did not evaluate any feature counts.")

    results_df = pd.DataFrame(results).sort_values("n_features")

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    ax.plot(results_df["n_features"], results_df["score"], marker="o")
    ax.set_xlabel("Number of Features")
    ax.set_ylabel(f"CV {scoring.upper()} Score")
    ax.set_title("Sequential Feature Selection Performance")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if n_features not in [row["n_features"] for row in results]:
        closest_idx = int(np.argmin(np.abs(results_df["n_features"] - n_features)))
        best_features = results_df.iloc[closest_idx]["selected_features"]

    return results_df, best_features, fig


def select_optimal_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    method: str = "ensemble",
    correlation_threshold: float = 0.95,
    top_n: int = 30,
    n_features: int = 25,
    estimator=None,
    min_votes: int = 2,
    enable_sequential: bool = True,
    sequential_kwargs: Optional[Dict] = None,
) -> FeatureSelectionResult:
    """Combine multiple feature selection techniques into a single recommendation."""

    features_df = pd.DataFrame(X, columns=feature_names)

    print("   [FS] Computing correlation filter…")
    corr_matrix, _, corr_keep, corr_fig = analyze_feature_correlation(
        features_df, threshold=correlation_threshold
    )

    print("   [FS] Running random forest importance…")
    rf_importances, rf_top, rf_fig = compute_feature_importance(
        X, y, feature_names, top_n=top_n
    )

    print("   [FS] Calculating mutual information…")
    mi_scores, mi_top, mi_fig = compute_mutual_information(
        X, y, feature_names, top_n=top_n
    )

    if enable_sequential:
        print("   [FS] Sequential feature selection enabled…")
        seq_kwargs = sequential_kwargs or {}
        seq_results_df, seq_top, seq_fig = sequential_feature_selection(
            X,
            y,
            feature_names,
            estimator=estimator,
            n_features=n_features,
            **seq_kwargs,
        )
    else:
        print("   [FS] Skipping sequential feature selection (disabled).")
        seq_results_df = pd.DataFrame(
            columns=["n_features", "selected_features", "score"], dtype=object
        )
        seq_top: List[str] = []
        seq_fig = None

    method_rankings = {
        "correlation": corr_keep,
        "random_forest": rf_top,
        "mutual_information": mi_top,
        "sequential": seq_top,
    }

    votes = pd.DataFrame({"feature": feature_names})
    votes["votes"] = 0

    for method_features in method_rankings.values():
        votes.loc[votes["feature"].isin(method_features), "votes"] += 1

    selection_matrix = votes.copy()
    for method_name, features in method_rankings.items():
        selection_matrix[method_name] = selection_matrix["feature"].isin(features)

    votes.sort_values(["votes", "feature"], ascending=[False, True], inplace=True)

    if method == "correlation":
        selected = corr_keep
    elif method == "rf":
        selected = rf_top
    elif method == "mi":
        selected = mi_top
    elif method == "sequential":
        selected = seq_top
    else:
        selected = votes.loc[votes["votes"] >= min_votes, "feature"].tolist()
        if not selected:
            selected = rf_top

    if n_features > 0:
        selected = selected[:n_features]

    venn_fig = _create_venn_figure(method_rankings)

    return FeatureSelectionResult(
        selected_features=selected,
        method_rankings=method_rankings,
        correlation_matrix=corr_matrix.abs(),
        correlation_plot=corr_fig,
        random_forest_importances=rf_importances,
        random_forest_plot=rf_fig,
        mutual_information_scores=mi_scores,
        mutual_information_plot=mi_fig,
        sequential_results=seq_results_df,
        sequential_plot=seq_fig,
        ensemble_votes=votes,
        venn_plot=venn_fig,
        selection_matrix=selection_matrix,
    )
