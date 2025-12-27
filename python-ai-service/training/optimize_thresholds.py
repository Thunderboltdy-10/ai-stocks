"""
Create a comprehensive threshold optimization script for binary classifiers in a trading system.

See module docstring in the repo for detailed requirements.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, f1_score,
    precision_score, recall_score, matthews_corrcoef, cohen_kappa_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
from tensorflow import keras
import pickle
import json
import argparse
from pathlib import Path
from tabulate import tabulate
from colorama import Fore, Style, init

init(autoreset=True)


def load_model_and_data(symbol, model_dir, validation_file):
    model_dir = Path(model_dir) / symbol
    models = {}
    for side in ('buy', 'sell'):
        model_path = model_dir / f'classifier_{side}.h5'
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        try:
            models[side] = keras.models.load_model(str(model_path))
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_path}: {e}")

    # Load validation data
    val_path = Path(validation_file) if validation_file else model_dir / 'validation.pkl'
    if not val_path.exists():
        raise FileNotFoundError(f"Validation file not found: {val_path}")

    with open(val_path, 'rb') as f:
        data = pickle.load(f)

    # Expect keys 'X_val', 'y_val_buy', 'y_val_sell' or similar
    X_val = data.get('X_val') or data.get('X') or data.get('X_val_raw')
    y_buy = data.get('y_val_buy') or data.get('y_buy') or data.get('y_buy_val')
    y_sell = data.get('y_val_sell') or data.get('y_sell') or data.get('y_sell_val')

    if X_val is None or y_buy is None or y_sell is None:
        raise KeyError("Validation pickle must contain keys: 'X_val', 'y_val_buy', 'y_val_sell'")

    return models, np.asarray(X_val), np.asarray(y_buy).astype(int), np.asarray(y_sell).astype(int)


def optimize_threshold_youden(y_true, y_probs):
    # Use ROC curve to find threshold maximizing TPR - FPR
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    j_stat = tpr - fpr
    idx = np.nanargmax(j_stat)
    best_threshold = float(thresholds[idx])
    return best_threshold


def optimize_threshold_f1(y_true, y_probs):
    thresholds = np.linspace(0.01, 0.99, 99)
    best_f1 = -1.0
    best_t = 0.5
    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        try:
            f1 = f1_score(y_true, preds, zero_division=0)
        except Exception:
            f1 = 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def optimize_threshold_precision(y_true, y_probs, min_precision=0.50):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    # thresholds corresponds to precision[1:]
    if thresholds.size == 0:
        # degenerate case: no thresholds, return 0.5
        return 0.5

    # Map thresholds to precision values
    precisions_for_thresholds = precision[1:]
    candidates = [(thr, p) for thr, p in zip(thresholds, precisions_for_thresholds) if p >= min_precision]
    if candidates:
        # pick the highest threshold that still meets precision requirement (conservative)
        best_thr = float(max(candidates, key=lambda x: x[0])[0])
        return best_thr
    # fallback: return threshold that gives max precision
    best_idx = int(np.argmax(precisions_for_thresholds))
    return float(thresholds[best_idx])


def compute_all_metrics(y_true, y_pred, y_probs):
    # Ensure arrays
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        mcc = matthews_corrcoef(y_true, y_pred)
        if np.isnan(mcc):
            mcc = 0.0
    except Exception:
        mcc = 0.0
    try:
        kappa = cohen_kappa_score(y_true, y_pred)
        if np.isnan(kappa):
            kappa = 0.0
    except Exception:
        kappa = 0.0
    try:
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = float(auc(fpr, tpr))
    except Exception:
        roc_auc = 0.0
    try:
        precision_curve, recall_curve_vals, _ = precision_recall_curve(y_true, y_probs)
        pr_auc = float(auc(recall_curve_vals, precision_curve))
    except Exception:
        pr_auc = 0.0

    metrics = {
        'confusion_matrix': cm.tolist(),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'mcc': float(mcc),
        'kappa': float(kappa),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
    }
    return metrics


def plot_optimization_results(y_true, y_probs, thresholds_dict, save_path):
    """Plot ROC, PR, F1 vs threshold, and MCC bar chart for a single binary problem.

    `thresholds_dict` should be mapping method_name -> threshold
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    methods = list(thresholds_dict.keys())
    colors = ['C0', 'C1', 'C2']

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc_val = auc(fpr, tpr)

    # PR
    precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_true, y_probs)
    # F1 vs threshold
    thresholds = np.linspace(0.01, 0.99, 99)
    f1_scores = []
    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        f1_scores.append(f1_score(y_true, preds, zero_division=0))

    # MCC for each method
    mccs = []
    method_points = {}
    for m in methods:
        t = thresholds_dict[m]
        preds = (y_probs >= t).astype(int)
        metrics = compute_all_metrics(y_true, preds, y_probs)
        mccs.append(metrics['mcc'])
        method_points[m] = {
            't': t,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'mcc': metrics['mcc']
        }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax_roc = axes[0, 0]
    ax_pr = axes[0, 1]
    ax_f1 = axes[1, 0]
    ax_bar = axes[1, 1]

    ax_roc.plot(fpr, tpr, label=f'ROC (AUC={roc_auc_val:.3f})')
    for i, m in enumerate(methods):
        pt = method_points[m]
        # compute fpr/tpr for the specific threshold
        preds = (y_probs >= pt['t']).astype(int)
        m_tpr = pt['recall']
        # compute fpr manually
        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        m_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        ax_roc.scatter(m_fpr, m_tpr, color=colors[i], label=f"{m} t={pt['t']:.3f}")

    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.legend()
    ax_roc.grid(True)

    # PR curve
    precision_curve_vals, recall_curve_vals, _ = precision_recall_curve(y_true, y_probs)
    pr_auc_val = auc(recall_curve_vals, precision_curve_vals)
    ax_pr.plot(recall_curve_vals, precision_curve_vals, label=f'PR (AUC={pr_auc_val:.3f})')
    for i, m in enumerate(methods):
        pt = method_points[m]
        ax_pr.scatter(pt['recall'], pt['precision'], color=colors[i], label=f"{m} t={pt['t']:.3f}")
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.legend()
    ax_pr.grid(True)

    # F1 vs threshold
    ax_f1.plot(thresholds, f1_scores, label='F1')
    for i, m in enumerate(methods):
        t = thresholds_dict[m]
        # find nearest
        idx = int(np.argmin(np.abs(thresholds - t)))
        ax_f1.scatter(t, f1_scores[idx], color=colors[i], label=f"{m} t={t:.3f}")
    ax_f1.set_xlabel('Threshold')
    ax_f1.set_ylabel('F1 Score')
    ax_f1.legend()
    ax_f1.grid(True)

    # MCC bar chart
    ax_bar.bar(methods, mccs, color=colors[: len(methods)])
    ax_bar.set_ylabel('MCC')
    ax_bar.set_title('MCC comparison')
    for i, v in enumerate(mccs):
        ax_bar.text(i, v + 0.01, f"{v:.3f}", ha='center')

    plt.tight_layout()
    fig_path = str(save_path)
    fig.savefig(fig_path)
    plt.close(fig)
    return fig_path


def main():
    parser = argparse.ArgumentParser(description='Optimize classifier thresholds for buy/sell models')
    parser.add_argument('--symbol', required=True, help='Ticker symbol, e.g. AAPL')
    parser.add_argument('--model-dir', default='saved_models', help='Directory containing saved models')
    parser.add_argument('--validation-file', default=None, help='Path to validation pickle file')
    parser.add_argument('--output-dir', default=None, help='Where to save results (defaults to model dir/symbol)')
    parser.add_argument('--default-buy', type=float, default=0.30, help='Default buy threshold to compare')
    parser.add_argument('--default-sell', type=float, default=0.45, help='Default sell threshold to compare')
    args = parser.parse_args()

    symbol = args.symbol
    model_dir = args.model_dir
    validation_file = args.validation_file
    out_dir = Path(args.output_dir) if args.output_dir else Path(model_dir) / symbol
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Threshold Optimization for {symbol} ===")
    print(f"Current (Default): BUY={args.default_buy:.2f}, SELL={args.default_sell:.2f}\n")

    models, X_val, y_buy, y_sell = load_model_and_data(symbol, model_dir, validation_file)

    results = {'buy': {}, 'sell': {}}

    for side, y_true in (('buy', y_buy), ('sell', y_sell)):
        model = models[side]
        try:
            probs = model.predict(X_val, verbose=0).ravel()
        except Exception:
            # try wrapping input as numpy
            probs = model.predict(np.asarray(X_val), verbose=0).ravel()

        # compute thresholds for three methods
        thr_youden = optimize_threshold_youden(y_true, probs)
        thr_f1 = optimize_threshold_f1(y_true, probs)
        thr_prec = optimize_threshold_precision(y_true, probs, min_precision=0.50)

        methods = {
            "Youden's J": thr_youden,
            'F1 Maximization': thr_f1,
            'Precision@50%': thr_prec
        }

        # compute metrics for all methods
        table_rows = []
        method_metrics = {}
        for mname, thr in methods.items():
            preds = (probs >= thr).astype(int)
            metrics = compute_all_metrics(y_true, preds, probs)
            method_metrics[mname] = {'threshold': float(thr), 'metrics': metrics}
            table_rows.append([
                mname,
                f"{thr:.3f}",
                f"{metrics['precision']:.3f}",
                f"{metrics['recall']:.3f}",
                f"{metrics['f1']:.3f}",
                f"{metrics['mcc']:.3f}",
                f"{metrics['kappa']:.3f}",
                f"{metrics['roc_auc']:.3f}",
                f"{metrics['pr_auc']:.3f}"
            ])

        # choose best method by MCC
        best_method = max(method_metrics.items(), key=lambda kv: kv[1]['metrics']['mcc'])[0]
        best_info = method_metrics[best_method]

        # compute default baseline metrics
        default_thr = args.default_buy if side == 'buy' else args.default_sell
        default_preds = (probs >= default_thr).astype(int)
        default_metrics = compute_all_metrics(y_true, default_preds, probs)

        # print summary for side
        print(f"Method summary for {side.upper()}:")
        for row in table_rows:
            mname = row[0]
            prefix = '  '
            selected = ' ← SELECTED' if mname == best_method else ''
            print(f"{prefix}{mname}{selected}")
            print(f"    threshold: {row[1]} | F1={row[4]}, MCC={row[5]}, Kappa={row[6]}")

        print("")

        # Save results
        results[side]['methods'] = method_metrics
        results[side]['selected_method'] = best_method
        results[side]['selected_threshold'] = float(best_info['threshold'])
        results[side]['selected_metrics'] = best_info['metrics']
        results[side]['default_threshold'] = float(default_thr)
        results[side]['default_metrics'] = default_metrics

        # Plot and save
        plot_path = out_dir / f'{symbol}_{side}_thresholds.png'
        plot_optimization_results(y_true, probs, {k: v['threshold'] for k, v in method_metrics.items()}, str(plot_path))

    # Consolidate a single-method label: if both sides selected same method, use it; else 'per-class'
    if results['buy']['selected_method'] == results['sell']['selected_method']:
        overall_method = results['buy']['selected_method']
    else:
        overall_method = 'per-class'

    out_json = {
        'buy': results['buy']['selected_threshold'],
        'sell': results['sell']['selected_threshold'],
        'method': overall_method,
        'metrics': {
            'buy': results['buy']['selected_metrics'],
            'sell': results['sell']['selected_metrics']
        }
    }

    json_path = Path(model_dir) / symbol / 'optimal_thresholds.json'
    with open(json_path, 'w') as f:
        json.dump(out_json, f, indent=2)

    print(Fore.GREEN + f"Saved optimal thresholds to: {json_path}")

    # Build tabulated comparison
    headers = ['Side', 'Method', 'Threshold', 'Precision', 'Recall', 'F1', 'MCC', 'Kappa', 'ROC-AUC', 'PR-AUC']
    rows = []
    for side in ('buy', 'sell'):
        for mname, mdata in results[side]['methods'].items():
            m = mdata['metrics']
            rows.append([
                side.upper(),
                mname,
                f"{mdata['threshold']:.3f}",
                f"{m['precision']:.3f}",
                f"{m['recall']:.3f}",
                f"{m['f1']:.3f}",
                f"{m['mcc']:.3f}",
                f"{m['kappa']:.3f}",
                f"{m['roc_auc']:.3f}",
                f"{m['pr_auc']:.3f}"
            ])

    print('\nDetailed comparison:')
    print(tabulate(rows, headers=headers, tablefmt='github'))


if __name__ == '__main__':
    main()
