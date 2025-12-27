"""
FinBERT-based Financial Sentiment Analyzer

This module provides sentiment analysis for financial text using FinBERT,
a BERT model pre-trained on financial text from Hugging Face.

Model: ProsusAI/finbert
- Pre-trained on financial news and earnings call transcripts
- Outputs: positive, negative, neutral probabilities
- Optimized for financial domain (better than general BERT)

Usage:
    from data.sentiment_analyzer import SentimentAnalyzer
    
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze_sentiment("Stock surges on strong earnings")
    print(result['sentiment_score'])  # 0.85 (bullish)
    
    # Batch processing
    headlines = ["Revenue beats estimates", "CEO resigns amid scandal"]
    results = analyzer.analyze_batch(headlines)
"""

import torch
import numpy as np
from typing import Dict, List, Union, Optional
from functools import lru_cache
import hashlib
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import transformers, provide helpful error if missing
try:
    from transformers import BertTokenizer, BertForSequenceClassification
except ImportError:
    logger.error(
        "transformers library not found. Install with:\n"
        "pip install transformers>=4.30.0 torch>=2.0.0"
    )
    raise


class SentimentAnalyzer:
    """
    FinBERT-based sentiment analyzer for financial text.
    
    This class wraps the ProsusAI/finbert model from HuggingFace to provide
    sentiment analysis optimized for financial news and reports.
    
    Features:
    - Continuous sentiment scores (-1 to +1)
    - Confidence metrics
    - Batch processing for efficiency
    - Caching to avoid redundant computation
    - GPU support (if available)
    
    Attributes:
        model: Pre-trained FinBERT model
        tokenizer: BERT tokenizer
        device: 'cuda' if GPU available, else 'cpu'
        cache_enabled: Whether to cache results
    """
    
    def __init__(
        self, 
        model_name: str = 'ProsusAI/finbert',
        use_cache: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize sentiment analyzer with FinBERT model.
        
        Args:
            model_name: HuggingFace model identifier (default: ProsusAI/finbert)
            use_cache: Enable LRU caching for repeated text (default: True)
            device: Force device ('cuda' or 'cpu'), auto-detect if None
        """
        logger.info("Initializing FinBERT Sentiment Analyzer...")
        
        # Determine device
        if device is not None:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and model
        try:
            logger.info(f"Loading model: {model_name}")
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertForSequenceClassification.from_pretrained(model_name)
            
            # Move model to device
            self.model = self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            logger.info("✓ Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        self.cache_enabled = use_cache
        self.model_name = model_name
        
        # Cache statistics
        self._cache_hits = 0
        self._cache_misses = 0
    
    def analyze_sentiment(self, text: str) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Financial text to analyze (headline, sentence, paragraph)
        
        Returns:
            Dictionary with:
            - sentiment_score: Continuous score from -1 (bearish) to +1 (bullish)
            - confidence: Maximum probability (0-1)
            - probabilities: Dict with positive, negative, neutral probs
            - label: String label ('positive', 'negative', 'neutral')
        
        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> result = analyzer.analyze_sentiment("Revenue exceeds expectations")
            >>> print(result)
            {
                'sentiment_score': 0.78,
                'confidence': 0.89,
                'label': 'positive',
                'probabilities': {
                    'positive': 0.89,
                    'negative': 0.03,
                    'neutral': 0.08
                }
            }
        """
        if not text or not text.strip():
            return self._empty_result()
        
        # Check cache if enabled
        if self.cache_enabled:
            cache_key = self._get_cache_key(text)
            cached_result = self._check_cache(cache_key)
            if cached_result is not None:
                self._cache_hits += 1
                return cached_result
            self._cache_misses += 1
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Extract probabilities
        # FinBERT output order: [positive, negative, neutral]
        pos_prob = probs[0][0].item()
        neg_prob = probs[0][1].item()
        neu_prob = probs[0][2].item()
        
        # Calculate continuous sentiment score
        # Range: -1 (fully negative) to +1 (fully positive)
        sentiment_score = pos_prob - neg_prob
        
        # Calculate confidence (max probability)
        confidence = max(pos_prob, neg_prob, neu_prob)
        
        # Determine label
        if pos_prob > neg_prob and pos_prob > neu_prob:
            label = 'positive'
        elif neg_prob > pos_prob and neg_prob > neu_prob:
            label = 'negative'
        else:
            label = 'neutral'
        
        result = {
            'sentiment_score': float(sentiment_score),
            'confidence': float(confidence),
            'label': label,
            'probabilities': {
                'positive': float(pos_prob),
                'negative': float(neg_prob),
                'neutral': float(neu_prob)
            }
        }
        
        # Cache result
        if self.cache_enabled:
            self._store_cache(cache_key, result)
        
        return result
    
    def analyze_batch(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        show_progress: bool = False
    ) -> List[Dict[str, Union[float, Dict[str, float]]]]:
        """
        Analyze sentiment for multiple texts efficiently.
        
        Processes texts in batches for better GPU utilization and speed.
        Critical for analyzing 50+ daily headlines.
        
        Args:
            texts: List of texts to analyze
            batch_size: Number of texts per batch (default: 32)
            show_progress: Print progress (default: False)
        
        Returns:
            List of sentiment dictionaries (same format as analyze_sentiment)
        
        Example:
            >>> headlines = [
            ...     "Stock rallies on earnings beat",
            ...     "Company faces regulatory scrutiny",
            ...     "Market remains stable"
            ... ]
            >>> results = analyzer.analyze_batch(headlines)
            >>> avg_sentiment = np.mean([r['sentiment_score'] for r in results])
        """
        if not texts:
            return []
        
        results = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(texts), batch_size):
            batch_texts = texts[batch_idx:batch_idx + batch_size]
            
            if show_progress:
                current_batch = batch_idx // batch_size + 1
                logger.info(f"Processing batch {current_batch}/{total_batches}")
            
            # Check cache for each text
            batch_results = []
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(batch_texts):
                if not text or not text.strip():
                    batch_results.append(self._empty_result())
                    continue
                
                if self.cache_enabled:
                    cache_key = self._get_cache_key(text)
                    cached_result = self._check_cache(cache_key)
                    if cached_result is not None:
                        self._cache_hits += 1
                        batch_results.append(cached_result)
                        continue
                    self._cache_misses += 1
                
                # Not cached
                batch_results.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)
            
            # Process uncached texts
            if uncached_texts:
                # Tokenize batch
                inputs = self.tokenizer(
                    uncached_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get predictions
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                # Process each result
                for i, text in enumerate(uncached_texts):
                    pos_prob = probs[i][0].item()
                    neg_prob = probs[i][1].item()
                    neu_prob = probs[i][2].item()
                    
                    sentiment_score = pos_prob - neg_prob
                    confidence = max(pos_prob, neg_prob, neu_prob)
                    
                    if pos_prob > neg_prob and pos_prob > neu_prob:
                        label = 'positive'
                    elif neg_prob > pos_prob and neg_prob > neu_prob:
                        label = 'negative'
                    else:
                        label = 'neutral'
                    
                    result = {
                        'sentiment_score': float(sentiment_score),
                        'confidence': float(confidence),
                        'label': label,
                        'probabilities': {
                            'positive': float(pos_prob),
                            'negative': float(neg_prob),
                            'neutral': float(neu_prob)
                        }
                    }
                    
                    # Cache result
                    if self.cache_enabled:
                        cache_key = self._get_cache_key(text)
                        self._store_cache(cache_key, result)
                    
                    # Store in batch results
                    batch_results[uncached_indices[i]] = result
            
            results.extend(batch_results)
        
        return results
    
    def aggregate_sentiment(
        self, 
        texts: List[str],
        method: str = 'weighted_mean',
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Aggregate sentiment across multiple texts.
        
        Useful for analyzing overall sentiment from multiple headlines or news articles.
        
        Args:
            texts: List of texts to analyze
            method: Aggregation method ('mean', 'weighted_mean', 'median')
            batch_size: Batch size for processing
        
        Returns:
            Dictionary with aggregated metrics:
            - sentiment_score: Aggregated score
            - avg_confidence: Average confidence
            - positive_ratio: % of positive texts
            - negative_ratio: % of negative texts
            - neutral_ratio: % of neutral texts
        
        Example:
            >>> headlines = fetch_daily_headlines('AAPL')
            >>> agg = analyzer.aggregate_sentiment(headlines)
            >>> print(f"Overall sentiment: {agg['sentiment_score']:.2f}")
        """
        if not texts:
            return {
                'sentiment_score': 0.0,
                'avg_confidence': 0.0,
                'positive_ratio': 0.0,
                'negative_ratio': 0.0,
                'neutral_ratio': 0.0
            }
        
        # Analyze all texts
        results = self.analyze_batch(texts, batch_size=batch_size)
        
        # Extract metrics
        scores = [r['sentiment_score'] for r in results]
        confidences = [r['confidence'] for r in results]
        labels = [r['label'] for r in results]
        
        # Aggregate sentiment score
        if method == 'mean':
            agg_score = np.mean(scores)
        elif method == 'weighted_mean':
            # Weight by confidence
            weights = np.array(confidences)
            agg_score = np.average(scores, weights=weights)
        elif method == 'median':
            agg_score = np.median(scores)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        # Calculate label ratios
        total = len(labels)
        positive_count = sum(1 for l in labels if l == 'positive')
        negative_count = sum(1 for l in labels if l == 'negative')
        neutral_count = sum(1 for l in labels if l == 'neutral')
        
        return {
            'sentiment_score': float(agg_score),
            'avg_confidence': float(np.mean(confidences)),
            'positive_ratio': positive_count / total,
            'negative_ratio': negative_count / total,
            'neutral_ratio': neutral_count / total,
            'num_texts': total
        }
    
    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache performance statistics.
        
        Returns:
            Dictionary with cache hits, misses, and hit rate
        """
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0
        
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'total_queries': total
        }
    
    def clear_cache(self):
        """Clear the sentiment cache."""
        if hasattr(self, '_cache'):
            self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("Cache cleared")
    
    # ===================================================================
    # PRIVATE METHODS
    # ===================================================================
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    @lru_cache(maxsize=1000)
    def _check_cache(self, cache_key: str) -> Optional[Dict]:
        """Check if result is cached (uses LRU cache decorator)."""
        # This method is cached by lru_cache decorator
        # We use a separate cache dict to store actual results
        if not hasattr(self, '_cache'):
            self._cache = {}
        return self._cache.get(cache_key)
    
    def _store_cache(self, cache_key: str, result: Dict):
        """Store result in cache."""
        if not hasattr(self, '_cache'):
            self._cache = {}
        self._cache[cache_key] = result
    
    def _empty_result(self) -> Dict[str, Union[float, Dict[str, float]]]:
        """Return empty/neutral result for invalid text."""
        return {
            'sentiment_score': 0.0,
            'confidence': 0.0,
            'label': 'neutral',
            'probabilities': {
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0
            }
        }


def example_usage():
    """Example usage of SentimentAnalyzer."""
    
    print("\n" + "="*70)
    print("  FinBERT SENTIMENT ANALYZER - EXAMPLE USAGE")
    print("="*70 + "\n")
    
    # Initialize analyzer
    print("Initializing analyzer...")
    analyzer = SentimentAnalyzer()
    
    # Single text analysis
    print("\n1. Single Text Analysis:")
    print("-" * 70)
    
    text = "Apple stock surges 5% on better-than-expected iPhone sales"
    result = analyzer.analyze_sentiment(text)
    
    print(f"Text: {text}")
    print(f"\nSentiment Score: {result['sentiment_score']:.3f}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Label: {result['label']}")
    print(f"Probabilities:")
    for label, prob in result['probabilities'].items():
        print(f"  {label}: {prob:.3f}")
    
    # Batch analysis
    print("\n2. Batch Analysis:")
    print("-" * 70)
    
    headlines = [
        "Revenue beats analyst estimates by 10%",
        "CEO resigns amid fraud investigation",
        "Company maintains steady growth trajectory",
        "Stock plummets on disappointing earnings",
        "New product launch exceeds expectations"
    ]
    
    results = analyzer.analyze_batch(headlines, show_progress=True)
    
    print("\nResults:")
    for headline, result in zip(headlines, results):
        score = result['sentiment_score']
        emoji = "📈" if score > 0.3 else "📉" if score < -0.3 else "➡️"
        print(f"{emoji} {score:+.2f} | {headline[:50]}")
    
    # Aggregate analysis
    print("\n3. Aggregate Sentiment:")
    print("-" * 70)
    
    agg = analyzer.aggregate_sentiment(headlines)
    
    print(f"Overall Sentiment: {agg['sentiment_score']:+.3f}")
    print(f"Average Confidence: {agg['avg_confidence']:.3f}")
    print(f"\nDistribution:")
    print(f"  Positive: {agg['positive_ratio']:.1%}")
    print(f"  Negative: {agg['negative_ratio']:.1%}")
    print(f"  Neutral:  {agg['neutral_ratio']:.1%}")
    
    # Cache statistics
    print("\n4. Cache Performance:")
    print("-" * 70)
    
    # Analyze same headlines again to test cache
    _ = analyzer.analyze_batch(headlines)
    
    stats = analyzer.get_cache_stats()
    print(f"Cache Hits: {stats['cache_hits']}")
    print(f"Cache Misses: {stats['cache_misses']}")
    print(f"Hit Rate: {stats['hit_rate']:.1%}")
    
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    example_usage()
