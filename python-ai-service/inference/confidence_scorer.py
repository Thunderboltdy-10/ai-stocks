import math
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# P0.2 FIX: Calibration Metrics Computation
# ============================================================================

def compute_calibration_metrics(
    predictions: np.ndarray,
    actuals: np.ndarray,
    n_bins: int = 10
) -> dict:
    """
    Compute calibration metrics using Platt scaling approach.
    
    P0.2 FIX: This function enables calibration analysis which was previously
    returning nulls. Without it, cannot assess prediction reliability.
    
    Args:
        predictions: Model predictions (continuous returns)
        actuals: Actual outcomes (continuous returns)
        n_bins: Number of calibration bins
        
    Returns:
        dict with brier_score, calibration_error, calibration_points
    """
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss
    
    predictions = np.asarray(predictions, dtype=float)
    actuals = np.asarray(actuals, dtype=float)
    
    # Remove NaNs
    valid_mask = ~(np.isnan(predictions) | np.isnan(actuals))
    predictions = predictions[valid_mask]
    actuals = actuals[valid_mask]
    
    if len(predictions) < 30:
        logger.warning(f"Insufficient samples for calibration: {len(predictions)}")
        return {
            'brier_score': None,
            'calibration_error': None,
            'calibration_error_per_bin': {},
            'calibration_points': {},
            'n_samples': len(predictions),
            'n_bins': n_bins
        }
    
    # Convert to binary classification for calibration
    # (direction prediction: positive vs negative)
    pred_probs = 1 / (1 + np.exp(-predictions * 10))  # Sigmoid with scaling
    actual_labels = (actuals > 0).astype(int)
    
    # Compute Brier score
    brier = None
    try:
        brier = brier_score_loss(actual_labels, pred_probs)
    except Exception as e:
        logger.warning(f"Brier score computation failed: {e}")
    
    # Compute calibration curve
    calib_error = None
    calib_error_per_bin = {}
    calib_points = {}
    
    try:
        # Adjust n_bins if not enough data
        effective_bins = min(n_bins, len(predictions) // 10)
        effective_bins = max(2, effective_bins)
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            actual_labels, 
            pred_probs, 
            n_bins=effective_bins,
            strategy='quantile'  # Equal-sized bins
        )
        
        # Calibration error = mean absolute deviation from diagonal
        calib_error = float(np.mean(np.abs(fraction_of_positives - mean_predicted_value)))
        
        # Store per-bin errors
        calib_error_per_bin = {
            f"bin_{i}": float(abs(fraction_of_positives[i] - mean_predicted_value[i]))
            for i in range(len(fraction_of_positives))
        }
        
        # Store calibration points for plotting
        calib_points = {
            'predicted': mean_predicted_value.tolist(),
            'observed': fraction_of_positives.tolist()
        }
        
    except Exception as e:
        logger.warning(f"Calibration curve computation failed: {e}")
    
    return {
        'brier_score': float(brier) if brier is not None else None,
        'calibration_error': calib_error,
        'calibration_error_per_bin': calib_error_per_bin,
        'calibration_points': calib_points,
        'n_samples': len(predictions),
        'n_bins': n_bins
    }


class ConfidenceScorer:
    def __init__(self, model_weights=None):
        self.model_weights = model_weights or {
            'classifier': 0.35,
            'regressor': 0.30,
            'quantile': 0.20,
            'tft': 0.15
        }
        # normalize weights to sum to 1.0
        total = sum(self.model_weights.values())
        if total <= 0:
            total = 1.0
        for k in list(self.model_weights.keys()):
            self.model_weights[k] = float(self.model_weights[k]) / float(total)

    def compute_classifier_confidence(self, buy_prob, sell_prob, signal):
        # defensively handle None
        b = 0.0 if buy_prob is None else float(buy_prob)
        s = 0.0 if sell_prob is None else float(sell_prob)
        if signal == 1:
            base = b
        elif signal == -1:
            base = s
        else:
            base = 1.0 - max(b, s)

        # softmax-like normalization across three values (buy, sell, hold) for stability
        vals = np.array([b, s, 1.0 - max(b, s)])
        # prevent overflow
        ex = np.exp(vals - np.max(vals))
        sm = ex / (np.sum(ex) + 1e-9)
        # Map chosen base to its softmax value for consistency
        if signal == 1:
            conf = float(sm[0])
        elif signal == -1:
            conf = float(sm[1])
        else:
            conf = float(sm[2])

        return float(np.clip(conf, 0.0, 1.0))

    def compute_regressor_confidence(self, pred_return, historical_volatility):
        if pred_return is None:
            return None
        eps = 1e-8
        ratio = abs(float(pred_return)) / (float(historical_volatility) + eps)
        k = 10.0
        # sigmoid
        conf = 1.0 / (1.0 + math.exp(-k * (ratio - 0.5)))
        return float(np.clip(conf, 0.0, 1.0))

    def compute_quantile_confidence(self, q10, q50, q90):
        if q10 is None or q90 is None:
            return None
        uncertainty = float(q90) - float(q10)
        # avoid division by zero
        conf = 1.0 / (1.0 + max(0.0, uncertainty))
        # normalize to [0,1]
        return float(np.clip(conf, 0.0, 1.0))

    def compute_tft_confidence(self, horizon_forecasts):
        if not horizon_forecasts:
            return None
        # extract medians
        medians = []
        for h, vals in sorted(horizon_forecasts.items()):
            v = vals.get('q50') if isinstance(vals, dict) else None
            if v is None:
                continue
            medians.append(float(v))
        if not medians:
            return None
        # direction consistency
        signs = [np.sign(m) for m in medians]
        direction_score = 1.0 if all(s > 0 for s in signs) or all(s < 0 for s in signs) else 0.0
        # magnitude consistency: low std relative to mean magnitude
        std = float(np.std(medians))
        mean_abs = float(np.mean(np.abs(medians)))
        denom = mean_abs if mean_abs > 1e-8 else 1.0
        magnitude_score = 1.0 - np.clip(std / denom, 0.0, 1.0)
        conf = direction_score * 0.6 + magnitude_score * 0.4
        return float(np.clip(conf, 0.0, 1.0))

    def compute_regime_alignment_multiplier(self, signal, regime):
        # default neutral
        if regime is None:
            return 1.0
        try:
            r = int(regime)
        except Exception:
            return 1.0
        if signal == 0:
            return 1.0
        if r == 1:
            # Bullish
            return 1.2 if signal == 1 else 0.7
        if r == -1:
            # Bearish
            return 1.2 if signal == -1 else 0.7
        # Sideways
        return 0.9

    def compute_unified_confidence(
        self,
        signal,
        buy_prob=None,
        sell_prob=None,
        regressor_pred=None,
        historical_vol=None,
        quantile_preds=None,
        tft_forecasts=None,
        regime=None,
        model_availability=None
    ):
        component_conf = {'classifier': None, 'regressor': None, 'quantile': None, 'tft': None}

        avail = model_availability or {}
        weights = {k: float(self.model_weights.get(k, 0.0)) for k in component_conf.keys()}
        # normalize weights to available models only
        avail_weights = {k: v for k, v in weights.items() if avail.get(k, False)}
        total_avail = sum(avail_weights.values())
        if total_avail <= 0:
            # fallback: use classifier if present else regressor
            if avail.get('classifier'):
                avail_weights = {'classifier': 1.0}
            elif avail.get('regressor'):
                avail_weights = {'regressor': 1.0}
            else:
                avail_weights = {'classifier': 1.0}
            total_avail = sum(avail_weights.values())

        # compute per-model confidences
        if avail.get('classifier'):
            component_conf['classifier'] = self.compute_classifier_confidence(buy_prob, sell_prob, signal)
        if avail.get('regressor'):
            component_conf['regressor'] = self.compute_regressor_confidence(regressor_pred, historical_vol)
        if avail.get('quantile'):
            if quantile_preds is not None:
                q10 = quantile_preds.get('q10') if isinstance(quantile_preds, dict) else None
                q50 = quantile_preds.get('q50') if isinstance(quantile_preds, dict) else None
                q90 = quantile_preds.get('q90') if isinstance(quantile_preds, dict) else None
                component_conf['quantile'] = self.compute_quantile_confidence(q10, q50, q90)
        if avail.get('tft'):
            component_conf['tft'] = self.compute_tft_confidence(tft_forecasts)

        # Weighted average across available components
        weighted_sum = 0.0
        total_weight = 0.0
        for k, conf in component_conf.items():
            if conf is None:
                continue
            w = float(weights.get(k, 0.0))
            weighted_sum += conf * w
            total_weight += w

        if total_weight <= 0:
            unified = 0.0
        else:
            unified = weighted_sum / (total_weight + 1e-12)

        # apply regime multiplier
        regime_mult = self.compute_regime_alignment_multiplier(signal, regime) if regime is not None else 1.0
        unified = unified * regime_mult
        unified = float(np.clip(unified, 0.0, 1.0))

        # attribution: which model contributed most (component_conf * weight)
        contributions = {}
        best_model = None
        best_val = -1.0
        for k, conf in component_conf.items():
            if conf is None:
                contributions[k] = 0.0
                continue
            contrib = conf * float(weights.get(k, 0.0))
            contributions[k] = float(contrib)
            if contrib > best_val:
                best_val = contrib
                best_model = k

        attribution = {
            'top_model': best_model,
            'contributions': contributions
        }

        return {
            'unified_confidence': unified,
            'component_confidences': component_conf,
            'regime_multiplier': float(regime_mult),
            'attribution': attribution
        }

    def get_confidence_tier(self, unified_confidence):
        if unified_confidence is None:
            return 'VERY_LOW'
        u = float(unified_confidence)
        if u >= 0.85:
            return 'VERY_HIGH'
        if u >= 0.70:
            return 'HIGH'
        if u >= 0.50:
            return 'MEDIUM'
        if u >= 0.30:
            return 'LOW'
        return 'VERY_LOW'
