"""
Diagnostic Utilities for Model Training

Provides callbacks and analysis tools for monitoring training quality,
including confusion matrices, macro-F1 scores, and synthetic data validation.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import matplotlib.pyplot as plt
from scipy import stats


class ConfusionMatrixCallback(Callback):
    """
    Print confusion matrix and per-class recall at end of each epoch
    """
    
    def __init__(self, X_val, y_val, class_names=None):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.class_names = class_names or ['SELL', 'HOLD', 'BUY']
    
    def on_epoch_end(self, epoch, logs=None):
        """Called at end of each epoch"""
        # Get predictions
        y_pred_probs = self.model.predict(self.X_val, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Convert y_val to integers if it's one-hot encoded
        if len(self.y_val.shape) > 1 and self.y_val.shape[1] > 1:
            y_true = np.argmax(self.y_val, axis=1)
        else:
            y_true = self.y_val
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Compute per-class recall
        recalls = []
        for i in range(len(self.class_names)):
            if cm[i].sum() > 0:
                recall = cm[i, i] / cm[i].sum()
            else:
                recall = 0.0
            recalls.append(recall)
        
        # Print
        print(f"\n📊 Epoch {epoch + 1} - Confusion Matrix:")
        print(cm)
        print(f"   Per-class recall:")
        for i, name in enumerate(self.class_names):
            print(f"      {name}: {recalls[i]*100:.1f}%")


class MacroF1Callback(Callback):
    """
    Compute and log macro-F1 score for use with ReduceLROnPlateau
    """
    
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.best_f1 = 0.0
    
    def on_epoch_end(self, epoch, logs=None):
        """Called at end of each epoch"""
        logs = logs or {}
        
        # Get predictions
        y_pred_probs = self.model.predict(self.X_val, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Convert y_val to integers if it's one-hot encoded
        if len(self.y_val.shape) > 1 and self.y_val.shape[1] > 1:
            y_true = np.argmax(self.y_val, axis=1)
        else:
            y_true = self.y_val
        
        # Compute macro F1
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Add to logs for other callbacks
        logs['val_macro_f1'] = macro_f1
        
        # Track best
        if macro_f1 > self.best_f1:
            self.best_f1 = macro_f1
            print(f"\n✨ New best macro-F1: {macro_f1:.4f}")


class R2ScoreCallback(Callback):
    """
    Monitor R² score during regression training
    
    R² measures how well model predictions fit actual values compared to mean predictor.
    R² > 0: Better than mean
    R² = 0: Same as mean
    R² < 0: Worse than mean (predicting ~0 for all inputs)
    """
    
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.best_r2 = -float('inf')
    
    def on_epoch_end(self, epoch, logs=None):
        """Called at end of each epoch"""
        logs = logs or {}
        
        # Get predictions
        y_pred = self.model.predict(self.X_val, verbose=0).flatten()
        
        # Compute R²
        ss_res = np.sum((self.y_val - y_pred) ** 2)
        ss_tot = np.sum((self.y_val - np.mean(self.y_val)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Add to logs
        logs['val_r2'] = r2
        
        # Print update
        if epoch == 0 or r2 > self.best_r2:
            self.best_r2 = r2
            status = "✅" if r2 > 0.1 else "⚠️" if r2 > 0 else "❌"
            print(f"\n{status} Epoch {epoch + 1} R²: {r2:.4f} (best: {self.best_r2:.4f})")
        
        # Warning for negative R²
        if r2 < 0 and epoch > 5:
            print(f"   ⚠️  WARNING: Negative R² means model is worse than mean predictor!")
            print(f"   Model may be predicting near-zero values for all inputs.")


def compare_real_vs_synth(X_real, X_synth, feature_names=None, n_samples=200):
    """
    Compare real vs synthetic samples for quality validation
    
    Args:
        X_real: Real samples of shape (n_samples, timesteps, features)
        X_synth: Synthetic samples of same shape
        feature_names: Optional list of feature names
        n_samples: Number of samples to use for comparison
    
    Returns:
        Dict with diagnostic metrics
    """
    # Sample if too many
    if len(X_real) > n_samples:
        idx_real = np.random.choice(len(X_real), n_samples, replace=False)
        X_real = X_real[idx_real]
    
    if len(X_synth) > n_samples:
        idx_synth = np.random.choice(len(X_synth), n_samples, replace=False)
        X_synth = X_synth[idx_synth]
    
    # Flatten to (n_samples, timesteps * features)
    X_real_flat = X_real.reshape(len(X_real), -1)
    X_synth_flat = X_synth.reshape(len(X_synth), -1)
    
    # Compute statistics
    real_mean = np.mean(X_real_flat, axis=0)
    real_std = np.std(X_real_flat, axis=0)
    synth_mean = np.mean(X_synth_flat, axis=0)
    synth_std = np.std(X_synth_flat, axis=0)
    
    # Mean absolute difference
    mean_diff = np.abs(real_mean - synth_mean)
    std_diff = np.abs(real_std - synth_std)
    
    mean_diff_pct = np.mean(mean_diff / (np.abs(real_mean) + 1e-7)) * 100
    std_diff_pct = np.mean(std_diff / (real_std + 1e-7)) * 100
    
    # KL divergence (approximate, per feature)
    kl_divs = []
    for i in range(min(10, X_real_flat.shape[1])):  # Check first 10 features
        try:
            # Bin the data
            bins = np.histogram_bin_edges(
                np.concatenate([X_real_flat[:, i], X_synth_flat[:, i]]),
                bins=20
            )
            real_hist, _ = np.histogram(X_real_flat[:, i], bins=bins, density=True)
            synth_hist, _ = np.histogram(X_synth_flat[:, i], bins=bins, density=True)
            
            # Add small constant to avoid log(0)
            real_hist = real_hist + 1e-10
            synth_hist = synth_hist + 1e-10
            
            # Normalize
            real_hist = real_hist / real_hist.sum()
            synth_hist = synth_hist / synth_hist.sum()
            
            # KL divergence
            kl_div = np.sum(real_hist * np.log(real_hist / synth_hist))
            kl_divs.append(kl_div)
        except:
            kl_divs.append(np.nan)
    
    avg_kl_div = np.nanmean(kl_divs)
    
    # Statistical tests
    # Kolmogorov-Smirnov test for distribution similarity
    ks_stats = []
    for i in range(min(10, X_real_flat.shape[1])):
        ks_stat, _ = stats.ks_2samp(X_real_flat[:, i], X_synth_flat[:, i])
        ks_stats.append(ks_stat)
    
    avg_ks_stat = np.mean(ks_stats)
    
    # Determine quality
    is_valid = (std_diff_pct < 15.0) and (avg_kl_div < 0.1) and (avg_ks_stat < 0.2)
    
    diagnostics = {
        'mean_diff_pct': mean_diff_pct,
        'std_diff_pct': std_diff_pct,
        'avg_kl_divergence': avg_kl_div,
        'avg_ks_statistic': avg_ks_stat,
        'is_valid': is_valid,
        'n_real': len(X_real),
        'n_synth': len(X_synth)
    }
    
    print(f"\n🔬 Synthetic Data Quality Analysis:")
    print(f"   Real samples: {len(X_real)}")
    print(f"   Synthetic samples: {len(X_synth)}")
    print(f"   Mean difference: {mean_diff_pct:.2f}%")
    print(f"   Std difference: {std_diff_pct:.2f}%")
    print(f"   Avg KL divergence: {avg_kl_div:.4f}")
    print(f"   Avg KS statistic: {avg_ks_stat:.4f}")
    print(f"   Quality: {'✅ Valid' if is_valid else '⚠️  Warning: Synthetic data may be unrealistic'}")
    
    return diagnostics


def plot_class_distribution(y, title="Class Distribution", save_path=None):
    """
    Plot class distribution bar chart
    
    Args:
        y: Labels array
        title: Plot title
        save_path: Optional path to save plot
    """
    unique, counts = np.unique(y, return_counts=True)
    class_names = ['SELL', 'HOLD', 'BUY']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(unique)), counts)
    
    # Color bars
    colors = ['#e74c3c', '#95a5a6', '#2ecc71']  # Red, Gray, Green
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(unique)))
    ax.set_xticklabels([class_names[int(c)] for c in unique])
    
    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        pct = (count / len(y)) * 100
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=10)
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   📊 Saved plot to {save_path}")
    
    plt.close()


def print_training_summary(history, y_val, y_pred):
    """
    Print comprehensive training summary
    
    Args:
        history: Keras history object
        y_val: True validation labels
        y_pred: Predicted validation labels
    """
    print(f"\n{'='*70}")
    print(f"📈 TRAINING SUMMARY")
    print(f"{'='*70}\n")
    
    # Training metrics
    print(f"Final training loss: {history.history['loss'][-1]:.4f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
    
    if 'accuracy' in history.history:
        print(f"Final training accuracy: {history.history['accuracy'][-1]*100:.2f}%")
        print(f"Final validation accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")
    
    # Classification metrics
    print(f"\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=['SELL', 'HOLD', 'BUY'], zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    print(f"Confusion Matrix:")
    print(cm)
    print(f"            SELL  HOLD  BUY")
    
    # Per-class metrics
    print(f"\nPer-class Recall:")
    for i, name in enumerate(['SELL', 'HOLD', 'BUY']):
        if cm[i].sum() > 0:
            recall = cm[i, i] / cm[i].sum()
        else:
            recall = 0.0
        print(f"   {name}: {recall*100:.1f}%")
    
    # Macro F1
    macro_f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
    print(f"\nMacro F1 Score: {macro_f1:.4f}")
    
    print(f"\n{'='*70}\n")
