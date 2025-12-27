import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def evaluate_quantile_predictions(y_true, q10, q50, q90):
    """
    Evaluate quantile regression calibration

    Parameters
    - y_true: array-like of true values
    - q10: array-like of predicted 10th percentile
    - q50: array-like of predicted 50th percentile (median)
    - q90: array-like of predicted 90th percentile

    Returns
    - metrics: dict with coverage and q50 MAE
    """
    y_true = np.asarray(y_true)
    q10 = np.asarray(q10)
    q50 = np.asarray(q50)
    q90 = np.asarray(q90)

    metrics = {}

    # 1. Coverage checks
    metrics['q10_coverage'] = float(np.mean(y_true <= q10))  # Should be ~10%
    metrics['q90_coverage'] = float(np.mean(y_true <= q90))  # Should be ~90%
    metrics['q50_mae'] = float(np.mean(np.abs(y_true - q50)))

    # 2. Calibration plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Coverage plot
    ax[0].plot(y_true, label='Actual', alpha=0.6)
    ax[0].fill_between(range(len(y_true)), q10, q90, alpha=0.3, label='80% CI')
    ax[0].plot(q50, label='Median', color='red', linewidth=2)
    ax[0].set_title('Quantile Predictions vs Actual')
    ax[0].legend()

    # Calibration histogram (coverage within 10-90 quantiles)
    coverage = (y_true >= q10) & (y_true <= q90)
    ax[1].hist(coverage.astype(float), bins=2)
    ax[1].set_title(f'Coverage: {coverage.mean():.2%}')
    ax[1].set_xticks([0, 1])
    ax[1].set_xticklabels(['Outside', 'Inside'])

    plt.tight_layout()
    plt.savefig('quantile_calibration.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    return metrics
