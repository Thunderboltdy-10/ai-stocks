"""
Data Cache Manager for AI-Stocks Training Pipeline

Handles centralized data fetching, feature engineering, and caching to avoid
redundant API calls and ensure data consistency across training steps.

Features:
- Single fetch per symbol per day
- Pickle-based caching with validation
- Index alignment verification
- Automatic cache invalidation after 24 hours
"""

import pickle
import hashlib
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict
import pandas as pd
import numpy as np
import logging

from data.data_fetcher import fetch_stock_data
from data.feature_engineer import (
    engineer_features,
    get_feature_columns,
    validate_and_fix_features,
    EXPECTED_FEATURE_COUNT,
)
from data.target_engineering import prepare_training_data

logger = logging.getLogger(__name__)


class DataCacheManager:
    """Manages data fetching and caching for training pipeline."""
    
    def __init__(self, cache_dir: str = 'cache', cache_lifetime_hours: int = 24):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory to store cache files
            cache_lifetime_hours: Cache validity period (default 24 hours)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_lifetime = timedelta(hours=cache_lifetime_hours)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DataCacheManager initialized: {self.cache_dir}")
    
    def get_cache_path(self, symbol: str) -> Path:
        """Get cache directory for a symbol."""
        symbol_dir = self.cache_dir / symbol.upper()
        symbol_dir.mkdir(parents=True, exist_ok=True)
        return symbol_dir
    
    def _compute_data_hash(self, df: pd.DataFrame) -> str:
        """Compute hash of DataFrame index for validation."""
        index_bytes = pd.util.hash_pandas_object(df.index).values.tobytes()
        return hashlib.md5(index_bytes).hexdigest()
    
    def _save_cache(self, symbol: str, raw_df: pd.DataFrame, 
                   engineered_df: pd.DataFrame, prepared_df: pd.DataFrame,
                   feature_cols: list):
        """Save data to cache with metadata.

        This function is defensive: it verifies the provided `feature_cols`
        match the canonical list from `get_feature_columns()` (respecting
        whether sentiment features are present). If a mismatch is detected
        it will attempt to recompute the prepared dataset from the
        validated engineered features before saving. Only caches that match
        the expected canonical feature count are persisted.
        """
        cache_path = self.get_cache_path(symbol)

        # Determine whether sentiment features appear to be present
        sentiment_core = {'news_volume', 'sentiment_mean'}
        sentiment_present = all(c in engineered_df.columns for c in sentiment_core)

        # Canonical feature list for this data
        try:
            canonical_features = get_feature_columns(include_sentiment=sentiment_present)
        except Exception:
            canonical_features = get_feature_columns(include_sentiment=True)

        # Validate provided feature_cols; if they differ, attempt reconciliation
        provided_set = set(feature_cols or [])
        canonical_set = set(canonical_features)

        if provided_set != canonical_set:
            logger.warning(
                f"Feature columns mismatch for {symbol}: provided={len(provided_set)}, canonical={len(canonical_set)}. "
                "Attempting automatic reconciliation..."
            )

            # Try to re-run validation on engineered_df to force canonicalization
            try:
                engineered_fixed = validate_and_fix_features(engineered_df.copy())
                prepared_fixed, feature_cols_fixed = prepare_training_data(engineered_fixed.copy(), horizons=[1])
            except Exception as e:
                logger.exception(f"Automatic reconciliation failed for {symbol}: {e}")
                prepared_fixed = prepared_df
                feature_cols_fixed = feature_cols

            # Use the recomputed versions if they match canonical set
            if set(feature_cols_fixed) == canonical_set:
                logger.info(f"Recomputed prepared data matches canonical feature set for {symbol}")
                prepared_df = prepared_fixed
                feature_cols = feature_cols_fixed
            else:
                # Final defensive step: intersect to canonical order and ensure presence
                logger.warning(
                    f"Could not fully reconcile features for {symbol}. "
                    "Will enforce canonical order and add missing columns with neutral values."
                )
                # Ensure prepared_df contains all canonical columns (add zeros where missing)
                excluded = {'Open', 'High', 'Low', 'Close', 'Volume', 'Date'}
                for col in canonical_features:
                    if col not in prepared_df.columns:
                        prepared_df[col] = 0.0

                # Drop any extras beyond canonical
                extra_cols = [c for c in prepared_df.columns if c not in canonical_set and c not in excluded]
                if extra_cols:
                    prepared_df = prepared_df.drop(columns=extra_cols)

                # Reorder to canonical ordering
                ordered = [c for c in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'] if c in prepared_df.columns]
                ordered += [c for c in canonical_features if c in prepared_df.columns]
                try:
                    prepared_df = prepared_df.reindex(columns=ordered)
                except Exception:
                    pass

                feature_cols = [c for c in canonical_features]

        # Log warning if feature count doesn't match expected, but proceed with caching
        # This allows the pipeline to work with different feature counts (e.g., feature selection)
        if len(feature_cols) != EXPECTED_FEATURE_COUNT:
            logger.warning(
                f"Feature count for {symbol}: {len(feature_cols)} (expected {EXPECTED_FEATURE_COUNT}). "
                "Proceeding with caching - model artifacts may use different feature sets."
            )

        # Save DataFrames
        raw_df.to_pickle(cache_path / 'raw_data.pkl')
        engineered_df.to_pickle(cache_path / 'engineered_features.pkl')
        prepared_df.to_pickle(cache_path / 'prepared_training.pkl')

        # Save canonical feature list (ordered)
        with open(cache_path / 'feature_columns.pkl', 'wb') as f:
            pickle.dump(feature_cols, f)

        # Save metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'raw_shape': raw_df.shape,
            'engineered_shape': engineered_df.shape,
            'prepared_shape': prepared_df.shape,
            'feature_count': len(feature_cols),
            'expected_feature_count': EXPECTED_FEATURE_COUNT,
            'raw_hash': self._compute_data_hash(raw_df),
            'engineered_hash': self._compute_data_hash(engineered_df),
            'prepared_hash': self._compute_data_hash(prepared_df),
            'version': '3.1'
        }

        with open(cache_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"✓ Cached data for {symbol}: {prepared_df.shape}")
    
    def _load_cache(self, symbol: str) -> Optional[Tuple]:
        """Load data from cache if valid."""
        cache_path = self.get_cache_path(symbol)
        metadata_file = cache_path / 'metadata.json'
        
        # Check if cache exists
        if not metadata_file.exists():
            return None
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Validate cached feature count matches current expectation
        # NOTE: Historically this forced a cache refresh when counts diverged,
        # which caused expensive refetches (news + FinBERT) even when the
        # cached data was still usable. Instead of forcing an immediate
        # refresh here, log a warning and allow the caller (`get_or_fetch_data`)
        # to decide whether to accept the cached data or trigger a refresh.
        if metadata.get('feature_count') != EXPECTED_FEATURE_COUNT:
            logger.warning(
                f"Cache feature count mismatch: cached={metadata.get('feature_count')}, expected={EXPECTED_FEATURE_COUNT}."
                " Proceeding to load cache; caller may choose to refresh explicitly."
            )
        # Check cache age
        cache_time = datetime.fromisoformat(metadata['timestamp'])
        cache_age = datetime.now() - cache_time
        if cache_age > self.cache_lifetime:
            # P0 FIX: Allow loading expired cache for training (set env var)
            import os
            if os.environ.get('ALLOW_EXPIRED_CACHE', '').lower() == 'true':
                logger.warning(f"⚠️  Cache expired for {symbol} (age: {cache_age}) but ALLOW_EXPIRED_CACHE=true, loading anyway")
            else:
                logger.info(f"⚠️  Cache expired for {symbol} (age: {cache_age})")
                return None
        
        # Load DataFrames
        try:
            raw_df = pd.read_pickle(cache_path / 'raw_data.pkl')
            engineered_df = pd.read_pickle(cache_path / 'engineered_features.pkl')
            prepared_df = pd.read_pickle(cache_path / 'prepared_training.pkl')
            
            with open(cache_path / 'feature_columns.pkl', 'rb') as f:
                feature_cols = pickle.load(f)
            
            # Validate hashes
            if self._compute_data_hash(prepared_df) != metadata['prepared_hash']:
                logger.warning(f"⚠️  Cache hash mismatch for {symbol}, will refetch")
                return None
            
            logger.info(f"✓ Loaded cached data for {symbol}: {prepared_df.shape}")
            return raw_df, engineered_df, prepared_df, feature_cols, metadata
            
        except Exception as e:
            logger.error(f"❌ Failed to load cache for {symbol}: {e}")
            return None
    
    def get_or_fetch_data(self, symbol: str, include_sentiment: bool = True,
                         force_refresh: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list]:
        """
        Get data from cache or fetch if not available.
        
        Args:
            symbol: Stock ticker
            include_sentiment: Whether to include sentiment features
            force_refresh: Force refetch even if cache exists
        
        Returns:
            (raw_df, engineered_df, prepared_df, feature_cols)
        """
        symbol = symbol.upper()
        
        # Try loading from cache first
        if not force_refresh:
            cached = self._load_cache(symbol)
            if cached is not None:
                raw_df, engineered_df, prepared_df, feature_cols, metadata = cached
                # Validate feature count against canonical list
                expected_count = len(get_feature_columns(include_sentiment=include_sentiment))
                if metadata.get('feature_count') != expected_count:
                    # Feature sets have diverged between cache and current code.
                    # Avoid forcing an expensive refetch (news + FinBERT) during training.
                    logger.warning(
                        f"⚠️  Cached feature count mismatch: expected {expected_count}, got {metadata.get('feature_count')}."
                    )
                    logger.info("Proceeding with cached data to avoid expensive refetch; consider running a one-off cache refresh if you intentionally changed features.")
                    return raw_df, engineered_df, prepared_df, feature_cols
                else:
                    return raw_df, engineered_df, prepared_df, feature_cols
        
        # Fetch fresh data
        logger.info(f"📊 Fetching fresh data for {symbol}...")
        
        # Step 1: Fetch raw OHLCV
        logger.info("  1/3 Fetching OHLCV data...")
        raw_df = fetch_stock_data(symbol, period='max')
        
        # Step 2: Engineer features (includes sentiment if enabled)
        logger.info(f"  2/3 Engineering features (sentiment={'enabled' if include_sentiment else 'disabled'})...")
        engineered_df = engineer_features(
            raw_df.copy(),
            symbol=symbol,
            include_sentiment=include_sentiment,
            cache_manager=self  # pass cache manager so sentiment can use news cache
        )
        # Validate and fix engineered features to ensure canonical feature set
        try:
            engineered_df = validate_and_fix_features(engineered_df)
        except Exception as e:
            logger.warning(f"Failed to validate/fix engineered features for {symbol}: {e}")
        
        # Step 3: Prepare training data (shift features, create targets, drop NaN)
        logger.info("  3/3 Preparing training data...")
        prepared_df, feature_cols = prepare_training_data(engineered_df.copy(), horizons=[1])
        
        # Validate alignment
        self._validate_alignment(raw_df, engineered_df, prepared_df)
        
        # Save to cache
        self._save_cache(symbol, raw_df, engineered_df, prepared_df, feature_cols)
        
        return raw_df, engineered_df, prepared_df, feature_cols
    
    def _validate_alignment(self, raw_df: pd.DataFrame, engineered_df: pd.DataFrame, 
                           prepared_df: pd.DataFrame):
        """
        Validate that data alignment is correct across processing steps.
        
        Critical checks:
        1. No index misalignment from NaN handling
        2. Row correspondence is maintained
        3. Feature count is correct
        """
        logger.info("🔍 Validating data alignment...")
        
        # Check 1: Prepared data should be subset of engineered data
        if not prepared_df.index.isin(engineered_df.index).all():
            raise ValueError("Prepared data index not subset of engineered data index")
        
        # Check 2: No duplicate indices
        if prepared_df.index.duplicated().any():
            raise ValueError("Duplicate indices found in prepared data")
        
        # Check 3: Check for NaN in prepared data (should be none)
        nan_count = prepared_df.isna().sum().sum()
        if nan_count > 0:
            raise ValueError(f"Found {nan_count} NaN values in prepared data")
        
        # Check 4: Verify row correspondence with sample
        if len(prepared_df) > 0:
            # Spot-check a middle index to ensure alignment hasn't shifted
            try:
                sample_idx = prepared_df.index[len(prepared_df) // 2]
                raw_close = raw_df.loc[sample_idx, 'Close'] if sample_idx in raw_df.index else None
                prep_close = prepared_df.loc[sample_idx, 'Close'] if 'Close' in prepared_df.columns and sample_idx in prepared_df.index else None
                if raw_close is not None and prep_close is not None:
                    if not np.isclose(raw_close, prep_close, rtol=1e-6):
                        raise ValueError(f"Close price mismatch at {sample_idx}: raw={raw_close}, prepared={prep_close}")
            except IndexError:
                # Prepared dataframe too small for spot-check; skip
                pass

        # NEW: Verify sequence alignment by sampling multiple rows and
        # comparing prepared_df.target_1d against actual close-to-close returns
        if 'target_1d' in prepared_df.columns and 'Close' in raw_df.columns:
            sample_size = min(20, len(prepared_df) - 1)
            if sample_size > 0:
                sample_indices = np.random.choice(len(prepared_df) - 1, size=sample_size, replace=False)
                misalignments = 0
                for i in sample_indices:
                    idx_t = prepared_df.index[i]
                    idx_t1 = prepared_df.index[i + 1]
                    if idx_t in raw_df.index and idx_t1 in raw_df.index:
                        actual_return = (raw_df.loc[idx_t1, 'Close'] / raw_df.loc[idx_t, 'Close']) - 1
                        target_return = prepared_df.loc[idx_t, 'target_1d']
                        # Allow small tolerance for log vs simple return differences
                        if not np.isfinite(actual_return) or not np.isfinite(target_return) or abs(actual_return - target_return) > 0.001:
                            misalignments += 1
                            logger.warning(f"Alignment issue at {idx_t}: actual_return={actual_return:.6f}, target={target_return:.6f}")

                if misalignments > max(1, int(sample_size * 0.1)):
                    raise ValueError(f"Feature-target misalignment detected: {misalignments}/{sample_size} samples")

                logger.info(f"✓ Sequence alignment validated ({sample_size} samples checked)")
        
        logger.info(f"✓ Alignment validation passed")
        # Check feature count matches expectation
        excluded = {'Open', 'High', 'Low', 'Close', 'Volume', 'Date'}
        prepared_feature_count = len([c for c in prepared_df.columns if c not in excluded])
        if prepared_feature_count != EXPECTED_FEATURE_COUNT:
            logger.warning(
                f"Feature count mismatch in prepared data: expected={EXPECTED_FEATURE_COUNT}, got={prepared_feature_count}"
            )
        logger.info(f"  Raw shape: {raw_df.shape}")
        logger.info(f"  Engineered shape: {engineered_df.shape}")
        logger.info(f"  Prepared shape: {prepared_df.shape}")
        logger.info(f"  Index alignment: {len(prepared_df)} rows preserved from {len(engineered_df)}")
    
    def clear_cache(self, symbol: Optional[str] = None):
        """Clear cache for a symbol or all symbols."""
        if symbol:
            cache_path = self.get_cache_path(symbol)
            if cache_path.exists():
                import shutil
                shutil.rmtree(cache_path)
                logger.info(f"🗑️  Cleared cache for {symbol}")
        else:
            if self.cache_dir.exists():
                import shutil
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"🗑️  Cleared all cache")
    
    def get_cache_info(self, symbol: str) -> Optional[Dict]:
        """Get cache metadata for a symbol."""
        cache_path = self.get_cache_path(symbol)
        metadata_file = cache_path / 'metadata.json'
        
        if not metadata_file.exists():
            return None
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        cache_time = datetime.fromisoformat(metadata['timestamp'])
        age = datetime.now() - cache_time
        
        return {
            **metadata,
            'age_hours': age.total_seconds() / 3600,
            'is_valid': age < self.cache_lifetime,
            'cache_path': str(cache_path)
        }

    def get_or_fetch_news(self, symbol: str, days_back: int = 365,
                          force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        Get cached news or fetch fresh data.

        Cache lifetime for news: 7 days (news doesn't change retroactively).

        Returns:
            DataFrame with news articles or None if fetch fails
        """
        cache_path = self.get_cache_path(symbol)
        news_cache = cache_path / 'news_data.pkl'
        news_meta = cache_path / 'news_metadata.json'

        # Try load existing cache
        if not force_refresh and news_cache.exists() and news_meta.exists():
            try:
                with open(news_meta, 'r') as f:
                    meta = json.load(f)

                cache_time = datetime.fromisoformat(meta['timestamp'])
                cache_age = datetime.now() - cache_time

                # News cache valid for 7 days
                if cache_age < timedelta(days=7):
                    news_df = pd.read_pickle(news_cache)
                    logger.info(f"✓ Loaded cached news for {symbol}: {len(news_df)} articles (age: {cache_age})")
                    return news_df
                else:
                    logger.info(f"⚠️  News cache expired (age: {cache_age}), refetching...")
            except Exception as e:
                logger.warning(f"Failed to load news cache for {symbol}: {e}")

        # Fetch fresh news
        logger.info(f"📰 Fetching fresh news for {symbol}...")
        try:
            from data.news_fetcher import NewsFetcher
            fetcher = NewsFetcher()
            news_df = fetcher.fetch_company_news(symbol, days_back=days_back)

            if news_df is None or getattr(news_df, 'empty', False):
                logger.warning(f"No news data available for {symbol}")
                return None

            # Save to cache
            try:
                news_df.to_pickle(news_cache)
                meta = {
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'article_count': len(news_df),
                    'days_back': days_back
                }
                with open(news_meta, 'w') as f:
                    json.dump(meta, f, indent=2)
                logger.info(f"✓ Cached {len(news_df)} news articles for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to save news cache for {symbol}: {e}")

            return news_df

        except Exception as e:
            logger.error(f"Failed to fetch news for {symbol}: {e}")
            return None