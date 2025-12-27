"""
Balanced Batch Generator for Class-Imbalanced Time Series

Generates batches with equal representation from each class by sampling
per-class indices. Preserves temporal order within sequences.
"""

import numpy as np
from tensorflow.keras.utils import Sequence


class BalancedSequence(Sequence):
    """
    Keras Sequence that yields balanced batches
    
    Each batch contains approximately equal numbers of samples from each class.
    Samples with replacement if a class has fewer samples than needed.
    """
    
    def __init__(self, X, y, batch_size=32, shuffle=True):
        """
        Initialize balanced sequence generator
        
        Args:
            X: Input sequences of shape (n_samples, timesteps, features)
            y: Labels of shape (n_samples,)
            batch_size: Batch size (should be divisible by number of classes)
            shuffle: Whether to shuffle batches between epochs
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Get class indices
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.idx_by_class = {}
        
        for c in self.classes:
            self.idx_by_class[c] = np.where(y == c)[0]
        
        # Samples per class per batch
        self.n_per_class = batch_size // self.n_classes
        
        # Calculate number of batches
        # Use minimum class count to avoid over-sampling too much
        min_class_size = min(len(idxs) for idxs in self.idx_by_class.values())
        self.n_batches = max(1, min_class_size // self.n_per_class)
        
        print(f"\n🎲 BalancedSequence initialized:")
        print(f"   Batch size: {batch_size}")
        print(f"   Classes: {self.n_classes}")
        print(f"   Samples per class per batch: {self.n_per_class}")
        print(f"   Batches per epoch: {self.n_batches}")
        for c in self.classes:
            print(f"   Class {c}: {len(self.idx_by_class[c])} samples")
        
        self.on_epoch_end()
    
    def __len__(self):
        """Number of batches per epoch"""
        return self.n_batches
    
    def __getitem__(self, idx):
        """Generate one batch"""
        batch_idx = []
        
        # Sample from each class
        for c in self.classes:
            class_indices = self.idx_by_class[c]
            
            # Sample with replacement if class is smaller than needed
            replace = len(class_indices) < self.n_per_class
            
            chosen = np.random.choice(
                class_indices, 
                size=self.n_per_class, 
                replace=replace
            )
            batch_idx.extend(chosen.tolist())
        
        # Fill remaining slots (if batch_size not perfectly divisible)
        remaining = self.batch_size - len(batch_idx)
        if remaining > 0:
            extra_idx = np.random.choice(len(self.X), size=remaining, replace=False)
            batch_idx.extend(extra_idx.tolist())
        
        # Shuffle within batch (temporal order preserved within each sequence)
        if self.shuffle:
            np.random.shuffle(batch_idx)
        
        batch_idx = batch_idx[:self.batch_size]
        
        return self.X[batch_idx], self.y[batch_idx]
    
    def on_epoch_end(self):
        """Called at end of each epoch"""
        if self.shuffle:
            # Shuffle indices within each class
            for c in self.classes:
                np.random.shuffle(self.idx_by_class[c])


class WeightedBatchGenerator:
    """
    Alternative generator that uses class weights instead of balanced sampling
    
    Useful when you want more control over class distribution without
    exact equal representation.
    """
    
    def __init__(self, X, y, batch_size=32, class_weights=None):
        """
        Initialize weighted batch generator
        
        Args:
            X: Input sequences
            y: Labels
            batch_size: Batch size
            class_weights: Dict mapping class -> weight (higher = more samples)
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        
        # Compute sampling probabilities
        if class_weights is None:
            # Balanced by default
            unique, counts = np.unique(y, return_counts=True)
            class_weights = {c: 1.0 / count for c, count in zip(unique, counts)}
        
        # Normalize weights to probabilities
        sample_weights = np.array([class_weights[label] for label in y])
        self.sample_probs = sample_weights / sample_weights.sum()
        
        self.n_batches = len(X) // batch_size
        
        print(f"\n⚖️  WeightedBatchGenerator initialized:")
        print(f"   Batch size: {batch_size}")
        print(f"   Batches per epoch: {self.n_batches}")
        print(f"   Class weights: {class_weights}")
    
    def __call__(self):
        """Generator function for tf.data.Dataset.from_generator"""
        while True:
            # Sample indices according to class weights
            batch_idx = np.random.choice(
                len(self.X),
                size=self.batch_size,
                replace=False,
                p=self.sample_probs
            )
            
            yield self.X[batch_idx], self.y[batch_idx]
    
    def __len__(self):
        return self.n_batches


def create_balanced_dataset(X, y, batch_size=32, method='balanced'):
    """
    Factory function to create balanced data generator
    
    Args:
        X: Input sequences
        y: Labels
        batch_size: Batch size
        method: 'balanced' or 'weighted'
    
    Returns:
        Appropriate generator for model.fit()
    """
    if method == 'balanced':
        return BalancedSequence(X, y, batch_size=batch_size, shuffle=True)
    elif method == 'weighted':
        # Compute balanced class weights
        unique, counts = np.unique(y, return_counts=True)
        total = sum(counts)
        class_weights = {c: total / (len(unique) * count) for c, count in zip(unique, counts)}
        return WeightedBatchGenerator(X, y, batch_size=batch_size, class_weights=class_weights)
    else:
        raise ValueError(f"Unknown method: {method}")
