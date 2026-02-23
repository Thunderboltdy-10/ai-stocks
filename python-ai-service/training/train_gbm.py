"""GBM-first trainer with Optuna tuning, SHAP feature selection, and purged CV."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import optuna
from sklearn.preprocessing import RobustScaler

from data.cache_manager import DataCacheManager
from data.data_splitter import PurgedTimeSeriesSplit, create_train_val_test_split
from data.data_validator import DataValidator
from validation.walk_forward import evaluate_predictions, run_walk_forward_validation

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

try:
    import xgboost as xgb

    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import lightgbm as lgb

    HAS_LGB = True
except Exception:
    HAS_LGB = False

try:
    import shap

    HAS_SHAP = True
except Exception:
    HAS_SHAP = False


def _probe_xgb_gpu() -> bool:
    if not HAS_XGB:
        return False
    X = np.random.randn(256, 8).astype(np.float32)
    y = np.random.randn(256).astype(np.float32)
    try:
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=16,
            max_depth=3,
            learning_rate=0.1,
            tree_method="hist",
            device="cuda",
            eval_metric="rmse",
        )
        model.fit(X, y, verbose=False)
        _ = model.predict(X[:16])
        return True
    except Exception:
        return False


def _probe_lgb_gpu() -> bool:
    if not HAS_LGB:
        return False
    X = np.random.randn(256, 8).astype(np.float32)
    y = np.random.randn(256).astype(np.float32)
    try:
        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=16,
            max_depth=3,
            learning_rate=0.1,
            device_type="gpu",
            verbosity=-1,
        )
        model.fit(X, y)
        _ = model.predict(X[:16])
        return True
    except Exception:
        return False


def compute_regression_sample_weights(y: np.ndarray, max_weight: float = 2.0) -> np.ndarray:
    y = np.asarray(y).reshape(-1)
    pos = np.sum(y > 0)
    neg = np.sum(y < 0)

    weights = np.ones_like(y, dtype=float)
    if pos > 0 and neg > 0:
        if pos > neg:
            weights[y < 0] = min(pos / neg, max_weight)
        else:
            weights[y > 0] = min(neg / pos, max_weight)

    return weights / np.mean(weights)


def _ensure_json_serializable(data: Dict) -> Dict:
    return json.loads(json.dumps(data, default=str))


@dataclass
class DatasetBundle:
    feature_columns: List[str]
    selected_columns: List[str]
    X_train: np.ndarray
    y_train: np.ndarray
    X_holdout: np.ndarray
    y_holdout: np.ndarray


class GBMTrainer:
    def __init__(
        self,
        symbol: str,
        n_trials: int = 30,
        include_sentiment: bool = False,
        use_cache: bool = True,
        force_refresh: bool = False,
        overwrite: bool = False,
        max_features: int = 45,
        seed: int = 42,
        allow_cpu_fallback: bool = False,
        target_horizon: int = 1,
        use_lgb: bool = True,
    ) -> None:
        if not HAS_XGB:
            raise ImportError("xgboost is required for training/train_gbm.py")
        if not HAS_LGB:
            raise ImportError("lightgbm is required for training/train_gbm.py")

        self.symbol = symbol.upper()
        self.n_trials = n_trials
        self.include_sentiment = include_sentiment
        self.use_cache = use_cache
        self.force_refresh = force_refresh
        self.overwrite = overwrite
        self.max_features = max_features
        self.seed = seed
        self.allow_cpu_fallback = allow_cpu_fallback
        self.target_horizon = max(1, int(target_horizon))

        self.xgb_gpu_enabled = _probe_xgb_gpu()
        if not self.xgb_gpu_enabled and not self.allow_cpu_fallback:
            raise RuntimeError(
                "XGBoost CUDA training is required but unavailable. "
                "Install/enable GPU support or pass --allow-cpu-fallback."
            )

        self.lgb_gpu_enabled = _probe_lgb_gpu()
        self.use_lgb = bool(use_lgb)
        if self.use_lgb and not self.lgb_gpu_enabled and not self.allow_cpu_fallback:
            logger.warning("LightGBM GPU backend unavailable; disabling LightGBM to keep GPU-only training")
            self.use_lgb = False
        elif self.use_lgb and not self.lgb_gpu_enabled and self.allow_cpu_fallback:
            logger.warning("LightGBM GPU backend unavailable; LightGBM will run on CPU due --allow-cpu-fallback")

        self.save_dir = Path("saved_models") / self.symbol / "gbm"
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _load_dataset(self) -> DatasetBundle:
        cache = DataCacheManager(cache_version="gbm_v2")
        _, _, prepared_df, feature_cols = cache.get_or_fetch_data(
            symbol=self.symbol,
            include_sentiment=self.include_sentiment,
            force_refresh=self.force_refresh or not self.use_cache,
            period="max",
            horizons=[self.target_horizon],
        )

        target_col = f"target_{self.target_horizon}d"
        if target_col not in prepared_df.columns:
            raise ValueError(f"{target_col} missing from prepared data")

        # Mild clipping improves robustness around crash outliers.
        clip_val = float(0.15 * np.sqrt(self.target_horizon))
        prepared_df[target_col] = prepared_df[target_col].clip(-clip_val, clip_val)

        validator = DataValidator(target_col=target_col)
        report = validator.run_all_validations(prepared_df, feature_cols)
        if not report["passed"]:
            logger.warning("Validation report has failing checks: %s", _ensure_json_serializable(report))

        X = prepared_df[feature_cols].to_numpy(dtype=np.float32)
        y = prepared_df[target_col].to_numpy(dtype=np.float32)

        train_idx, val_idx, holdout_idx = create_train_val_test_split(
            n_samples=len(prepared_df),
            train_pct=0.70,
            val_pct=0.15,
            embargo_days=5,
        )

        trainval_idx = np.concatenate([train_idx, val_idx])

        X_train = X[trainval_idx]
        y_train = y[trainval_idx]
        X_holdout = X[holdout_idx]
        y_holdout = y[holdout_idx]

        logger.info(
            "Dataset | total=%d train+val=%d holdout=%d features=%d",
            len(X),
            len(X_train),
            len(X_holdout),
            len(feature_cols),
        )

        if len(X_holdout) < 100:
            raise ValueError("Holdout set too small for trustworthy evaluation")

        return DatasetBundle(
            feature_columns=feature_cols,
            selected_columns=feature_cols.copy(),
            X_train=X_train,
            y_train=y_train,
            X_holdout=X_holdout,
            y_holdout=y_holdout,
        )

    @staticmethod
    def _objective_score(metrics: Dict[str, float]) -> float:
        # Balanced score: predictive direction + tradability + non-collapsed variance.
        return (
            float(metrics["sharpe"])
            + 8.0 * (float(metrics["dir_acc"]) - 0.5)
            + 40.0 * max(float(metrics["pred_std"]) - 0.003, 0.0)
            - 4.0 * abs(float(metrics["positive_pct"]) - 0.5)
        )

    def _splitter_for_tuning(self, n_samples: int) -> PurgedTimeSeriesSplit:
        test_size = min(252, max(90, n_samples // 8))
        min_train = max(500, int(n_samples * 0.45))
        return PurgedTimeSeriesSplit(
            n_splits=5,
            test_size=test_size,
            embargo_days=5,
            purge_days=5,
            min_train_size=min_train,
        )

    def _xgb_base_params(self) -> Dict:
        return {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "device": "cuda" if self.xgb_gpu_enabled else "cpu",
            "eval_metric": "rmse",
            "n_jobs": -1,
            "random_state": self.seed,
        }

    def _lgb_base_params(self) -> Dict:
        return {
            "objective": "regression",
            "metric": "rmse",
            "n_jobs": -1,
            "random_state": self.seed,
            "verbosity": -1,
            "device_type": "gpu" if self.lgb_gpu_enabled else "cpu",
        }

    def _objective_xgb(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        splitter = self._splitter_for_tuning(len(X))

        params = self._xgb_base_params() | {
            "n_estimators": trial.suggest_int("n_estimators", 250, 1800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 12.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
        }

        fold_scores = []
        for train_idx, test_idx in splitter.split(X):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            scaler = RobustScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            weights = compute_regression_sample_weights(y_train)
            model = xgb.XGBRegressor(**params)
            try:
                model.fit(X_train_s, y_train, sample_weight=weights, verbose=False)
            except xgb.core.XGBoostError:
                if not self.allow_cpu_fallback:
                    raise
                cpu_params = params | {"device": "cpu"}
                model = xgb.XGBRegressor(**cpu_params)
                model.fit(X_train_s, y_train, sample_weight=weights, verbose=False)

            pred = model.predict(X_test_s)
            metrics = evaluate_predictions(y_test, pred)
            score = self._objective_score(metrics)
            fold_scores.append(score)

        return float(np.mean(fold_scores))

    def _objective_lgb(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        splitter = self._splitter_for_tuning(len(X))

        params = self._lgb_base_params() | {
            "n_estimators": trial.suggest_int("n_estimators", 250, 1800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
        }

        fold_scores = []
        for train_idx, test_idx in splitter.split(X):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            scaler = RobustScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            weights = compute_regression_sample_weights(y_train)
            model = lgb.LGBMRegressor(**params)
            try:
                model.fit(X_train_s, y_train, sample_weight=weights)
            except Exception:
                cpu_params = params | {"device_type": "cpu"}
                model = lgb.LGBMRegressor(**cpu_params)
                model.fit(X_train_s, y_train, sample_weight=weights)

            pred = model.predict(X_test_s)
            metrics = evaluate_predictions(y_test, pred)
            score = self._objective_score(metrics)
            fold_scores.append(score)

        return float(np.mean(fold_scores))

    def _run_optuna(self, X: np.ndarray, y: np.ndarray) -> Tuple[Dict, Optional[Dict]]:
        logger.info("Running Optuna tuning | trials=%d", self.n_trials)

        xgb_study = optuna.create_study(direction="maximize")
        xgb_study.optimize(lambda t: self._objective_xgb(t, X, y), n_trials=self.n_trials)
        xgb_best = self._xgb_base_params() | xgb_study.best_params

        lgb_best: Optional[Dict] = None
        lgb_value = None
        if self.use_lgb:
            lgb_study = optuna.create_study(direction="maximize")
            lgb_study.optimize(lambda t: self._objective_lgb(t, X, y), n_trials=self.n_trials)
            lgb_best = self._lgb_base_params() | lgb_study.best_params
            lgb_value = lgb_study.best_value

        logger.info("Best XGB objective: %.4f", xgb_study.best_value)
        if lgb_value is not None:
            logger.info("Best LGB objective: %.4f", lgb_value)
        return xgb_best, lgb_best

    def _select_features_with_shap(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_columns: List[str],
        xgb_params: Dict,
    ) -> Tuple[List[str], Dict[str, float]]:
        scaler = RobustScaler()
        Xs = scaler.fit_transform(X_train)
        weights = compute_regression_sample_weights(y_train)

        model = xgb.XGBRegressor(**(xgb_params | {"n_estimators": min(700, int(xgb_params.get("n_estimators", 700)))}))
        try:
            model.fit(Xs, y_train, sample_weight=weights, verbose=False)
        except xgb.core.XGBoostError:
            model = xgb.XGBRegressor(**(xgb_params | {"device": "cpu", "n_estimators": min(700, int(xgb_params.get("n_estimators", 700)))}))
            model.fit(Xs, y_train, sample_weight=weights, verbose=False)

        importance: Dict[str, float]

        if HAS_SHAP:
            try:
                sample_n = min(2500, len(Xs))
                sample_idx = np.linspace(0, len(Xs) - 1, sample_n, dtype=int)
                X_sample = Xs[sample_idx]
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                scores = np.mean(np.abs(shap_values), axis=0)
                importance = {feature_columns[i]: float(scores[i]) for i in range(len(feature_columns))}
            except Exception as exc:
                logger.warning("SHAP failed (%s); using model feature_importances_ fallback", exc)
                raw = model.feature_importances_
                importance = {feature_columns[i]: float(raw[i]) for i in range(len(feature_columns))}
        else:
            raw = model.feature_importances_
            importance = {feature_columns[i]: float(raw[i]) for i in range(len(feature_columns))}

        ranked = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
        keep_n = min(self.max_features, len(ranked))
        keep_n = max(30, keep_n)
        selected = [name for name, _ in ranked[:keep_n]]

        logger.info("Feature selection | selected=%d/%d", len(selected), len(feature_columns))
        return selected, dict(ranked)

    def _fit_xgb(self, X: np.ndarray, y: np.ndarray, params: Dict) -> Tuple[object, RobustScaler]:
        scaler = RobustScaler()
        Xs = scaler.fit_transform(X)
        weights = compute_regression_sample_weights(y)

        model = xgb.XGBRegressor(**params)
        try:
            model.fit(Xs, y, sample_weight=weights, verbose=False)
        except xgb.core.XGBoostError:
            if not self.allow_cpu_fallback:
                raise
            model = xgb.XGBRegressor(**(params | {"device": "cpu"}))
            model.fit(Xs, y, sample_weight=weights, verbose=False)
        return model, scaler

    def _fit_lgb(self, X: np.ndarray, y: np.ndarray, params: Dict) -> Tuple[object, RobustScaler]:
        scaler = RobustScaler()
        Xs = scaler.fit_transform(X)
        weights = compute_regression_sample_weights(y)

        model = lgb.LGBMRegressor(**params)
        try:
            model.fit(Xs, y, sample_weight=weights)
        except Exception:
            model = lgb.LGBMRegressor(**(params | {"device_type": "cpu"}))
            model.fit(Xs, y, sample_weight=weights)
        return model, scaler

    def _walk_forward_metrics(self, X: np.ndarray, y: np.ndarray, params: Dict, model_type: str) -> Dict:
        splitter = self._splitter_for_tuning(len(X))

        def fit_predict_fn(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
            if model_type == "xgb":
                model, scaler = self._fit_xgb(X_train, y_train, params)
                pred = model.predict(scaler.transform(X_test))
                importance = None
                if hasattr(model, "feature_importances_"):
                    importance = np.asarray(model.feature_importances_)
            else:
                model, scaler = self._fit_lgb(X_train, y_train, params)
                pred = model.predict(scaler.transform(X_test))
                importance = None
                if hasattr(model, "feature_importances_"):
                    importance = np.asarray(model.feature_importances_)

            imp_series = None
            if importance is not None:
                imp_series = None
                # caller does not need named stability in this context
            return pred, imp_series

        return run_walk_forward_validation(X, y, fit_predict_fn=fit_predict_fn, splitter=splitter)

    def _derive_ensemble_weights(self, wf_xgb: Dict, wf_lgb: Optional[Dict]) -> Dict[str, float]:
        if not wf_lgb:
            return {"xgb": 1.0, "lgb": 0.0}
        xgb_agg = wf_xgb.get("aggregate", {})
        lgb_agg = wf_lgb.get("aggregate", {})

        xgb_score = (
            float(xgb_agg.get("sharpe_mean", 0.0))
            + 8.0 * (float(xgb_agg.get("dir_acc_mean", 0.5)) - 0.5)
            + 30.0 * max(float(xgb_agg.get("pred_std_mean", 0.0)) - 0.003, 0.0)
            - 3.0 * abs(float(xgb_agg.get("positive_pct_mean", 0.5)) - 0.5)
        )
        lgb_score = (
            float(lgb_agg.get("sharpe_mean", 0.0))
            + 8.0 * (float(lgb_agg.get("dir_acc_mean", 0.5)) - 0.5)
            + 30.0 * max(float(lgb_agg.get("pred_std_mean", 0.0)) - 0.003, 0.0)
            - 3.0 * abs(float(lgb_agg.get("positive_pct_mean", 0.5)) - 0.5)
        )

        x = max(0.0, xgb_score)
        l = max(0.0, lgb_score)
        if x + l <= 1e-9:
            return {"xgb": 0.5, "lgb": 0.5}
        return {"xgb": float(x / (x + l)), "lgb": float(l / (x + l))}

    def train(self) -> Dict:
        bundle = self._load_dataset()

        # Overwrite gate
        target_meta = self.save_dir / "training_metadata.json"
        if target_meta.exists() and not self.overwrite:
            logger.info("Models already exist for %s; skipping (use --overwrite)", self.symbol)
            with target_meta.open("r", encoding="utf-8") as fh:
                return json.load(fh)

        xgb_params, lgb_params = self._run_optuna(bundle.X_train, bundle.y_train)

        selected_cols, shap_ranked = self._select_features_with_shap(
            bundle.X_train,
            bundle.y_train,
            bundle.feature_columns,
            xgb_params,
        )
        bundle.selected_columns = selected_cols

        selected_idx = [bundle.feature_columns.index(c) for c in selected_cols]
        X_train_sel = bundle.X_train[:, selected_idx]
        X_holdout_sel = bundle.X_holdout[:, selected_idx]

        wf_xgb = self._walk_forward_metrics(X_train_sel, bundle.y_train, xgb_params, model_type="xgb")
        wf_lgb = None
        if self.use_lgb and lgb_params is not None:
            wf_lgb = self._walk_forward_metrics(X_train_sel, bundle.y_train, lgb_params, model_type="lgb")
        ensemble_weights = self._derive_ensemble_weights(wf_xgb, wf_lgb)

        xgb_model, xgb_scaler = self._fit_xgb(X_train_sel, bundle.y_train, xgb_params)
        lgb_model = None
        lgb_scaler = None
        if self.use_lgb and lgb_params is not None:
            lgb_model, lgb_scaler = self._fit_lgb(X_train_sel, bundle.y_train, lgb_params)

        xgb_holdout_pred = xgb_model.predict(xgb_scaler.transform(X_holdout_sel))
        if lgb_model is not None and lgb_scaler is not None:
            lgb_holdout_pred = lgb_model.predict(lgb_scaler.transform(X_holdout_sel))
            ensemble_holdout_pred = (
                ensemble_weights["xgb"] * xgb_holdout_pred + ensemble_weights["lgb"] * lgb_holdout_pred
            )
        else:
            lgb_holdout_pred = None
            ensemble_holdout_pred = xgb_holdout_pred

        xgb_holdout = evaluate_predictions(bundle.y_holdout, xgb_holdout_pred)
        lgb_holdout = evaluate_predictions(bundle.y_holdout, lgb_holdout_pred) if lgb_holdout_pred is not None else None
        ens_holdout = evaluate_predictions(bundle.y_holdout, ensemble_holdout_pred)

        val_dir = float(wf_xgb["aggregate"]["dir_acc_mean"])
        if wf_lgb is not None:
            val_dir = max(val_dir, float(wf_lgb["aggregate"]["dir_acc_mean"]))
        test_dir = ens_holdout["dir_acc"]
        if val_dir > 0.5:
            wfe = ((test_dir - 0.5) / (val_dir - 0.5)) * 100.0
        else:
            wfe = 0.0

        model_artifacts = {
            "xgb_model_path": str(self.save_dir / "xgb_model.joblib"),
            "lgb_model_path": str(self.save_dir / "lgb_model.joblib") if lgb_model is not None else None,
            "xgb_scaler_path": str(self.save_dir / "xgb_scaler.joblib"),
            "lgb_scaler_path": str(self.save_dir / "lgb_scaler.joblib") if lgb_scaler is not None else None,
        }

        # Persist artifacts
        joblib.dump(xgb_model, self.save_dir / "xgb_model.joblib")
        joblib.dump(xgb_scaler, self.save_dir / "xgb_scaler.joblib")
        if lgb_model is not None and lgb_scaler is not None:
            joblib.dump(lgb_model, self.save_dir / "lgb_model.joblib")
            joblib.dump(lgb_scaler, self.save_dir / "lgb_scaler.joblib")
        else:
            for stale in [
                self.save_dir / "lgb_model.joblib",
                self.save_dir / "lgb_scaler.joblib",
                self.save_dir / "lgb_reg.joblib",
            ]:
                if stale.exists():
                    stale.unlink(missing_ok=True)

        with (self.save_dir / "feature_columns.pkl").open("wb") as fh:
            pickle.dump(selected_cols, fh)

        with (self.save_dir / "feature_importance.json").open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "shap_ranked": shap_ranked,
                    "selected_columns": selected_cols,
                },
                fh,
                indent=2,
            )

        with (self.save_dir / "hyperparameters.json").open("w", encoding="utf-8") as fh:
            json.dump({"xgb": xgb_params, "lgb": lgb_params}, fh, indent=2)

        metadata = {
            "symbol": self.symbol,
            "runtime": {
                "xgb_gpu_enabled": self.xgb_gpu_enabled,
                "lgb_gpu_enabled": self.lgb_gpu_enabled,
                "lgb_enabled": self.use_lgb,
                "allow_cpu_fallback": self.allow_cpu_fallback,
            },
            "target_horizon_days": self.target_horizon,
            "target_distribution": {
                "train_positive_pct": float((bundle.y_train > 0).mean()),
                "train_mean": float(np.mean(bundle.y_train)),
                "train_std": float(np.std(bundle.y_train)),
                "holdout_positive_pct": float((bundle.y_holdout > 0).mean()),
                "holdout_mean": float(np.mean(bundle.y_holdout)),
                "holdout_std": float(np.std(bundle.y_holdout)),
            },
            "ensemble_weights": ensemble_weights,
            "n_features_original": len(bundle.feature_columns),
            "n_features_selected": len(selected_cols),
            "selected_columns": selected_cols,
            "walk_forward": {
                "xgb": wf_xgb,
                "lgb": wf_lgb,
            },
            "holdout": {
                "xgb": xgb_holdout,
                "lgb": lgb_holdout,
                "ensemble": ens_holdout,
            },
            "wfe": float(wfe),
            "quality_gate": {
                "pred_std_gt_0_005": ens_holdout["pred_std"] > 0.005,
                "positive_pct_in_range": 0.30 <= ens_holdout["positive_pct"] <= 0.70,
                "wfe_gt_40": wfe > 40.0,
                "dir_acc_gt_51": ens_holdout["dir_acc"] > 0.51,
            },
            "artifacts": model_artifacts,
        }

        with (self.save_dir / "training_metadata.json").open("w", encoding="utf-8") as fh:
            json.dump(_ensure_json_serializable(metadata), fh, indent=2)

        logger.info(
            "Holdout ensemble | dir_acc=%.4f sharpe=%.4f pred_std=%.6f positive_pct=%.3f wfe=%.1f%%",
            ens_holdout["dir_acc"],
            ens_holdout["sharpe"],
            ens_holdout["pred_std"],
            ens_holdout["positive_pct"],
            wfe,
        )

        return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GBM models with Optuna + SHAP + purged CV")
    parser.add_argument("symbol", type=str, help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing model artifacts")
    parser.add_argument("--tune", action="store_true", help="Compatibility flag (tuning is always on)")
    parser.add_argument("--n-trials", type=int, default=30, help="Optuna trials per model")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache use")
    parser.add_argument("--force-refresh", action="store_true", help="Force cache refresh")
    parser.add_argument("--include-sentiment", action="store_true", help="Include sentiment features")
    parser.add_argument("--max-features", type=int, default=45, help="Max SHAP-selected features")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--target-horizon", type=int, default=1, help="Forward target horizon in trading days")
    parser.add_argument("--no-lgb", action="store_true", help="Disable LightGBM and train GPU XGBoost only")
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="Allow CPU fallback if GPU backend is unavailable",
    )

    args = parser.parse_args()

    trainer = GBMTrainer(
        symbol=args.symbol,
        n_trials=args.n_trials,
        include_sentiment=args.include_sentiment,
        use_cache=not args.no_cache,
        force_refresh=args.force_refresh,
        overwrite=args.overwrite,
        max_features=args.max_features,
        seed=args.seed,
        allow_cpu_fallback=args.allow_cpu_fallback,
        target_horizon=args.target_horizon,
        use_lgb=not args.no_lgb,
    )

    result = trainer.train()
    print(json.dumps(_ensure_json_serializable(result), indent=2))


if __name__ == "__main__":
    main()
