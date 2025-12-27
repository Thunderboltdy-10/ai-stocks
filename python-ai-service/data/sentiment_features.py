"""
Sentiment Feature Engineering

This module creates comprehensive sentiment features from financial news data.
Integrates with NewsFetcher and SentimentAnalyzer to generate trading signals
based on news sentiment analysis.

Features:
- Daily sentiment aggregation (mean, std, max, min)
- Sentiment technical indicators (SMA, momentum, RSI)
- Sentiment-price divergence detection
- News volume impact analysis
- Sentiment acceleration metrics
- Sentiment regime classification

Usage:
    from data.sentiment_features import SentimentFeatureEngineer
    from data.news_fetcher import NewsFetcher
    from data.sentiment_analyzer import SentimentAnalyzer
    
    # Fetch news
    fetcher = NewsFetcher()
    news_df = fetcher.fetch_company_news('AAPL', days_back=365)
    
    # Create sentiment features
    engineer = SentimentFeatureEngineer()
    sentiment_features = engineer.create_all_features(news_df, price_df)
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
import logging

# Load environment variables
try:
    from dotenv import load_dotenv
    # Look in project root (two levels up from python-ai-service/data/)
    env_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env')
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_nsis_from_daily(df: pd.DataFrame, sentiment_col: str = 'sentiment_mean', news_vol_col: str = 'news_volume', window: int = 5, decay: float = 0.3) -> pd.Series:
    """
    Compute News Sentiment Impact Score (NSIS) from an already-aggregated daily
    sentiment DataFrame. This preserves the previous behaviour when callers have
    an aggregated `sentiment_df`.

    Returns a pandas Series with NSIS values (first rows filled with 0).
    """
    if sentiment_col not in df.columns:
        raise ValueError(f"Missing sentiment column: {sentiment_col}")
    if news_vol_col not in df.columns:
        raise ValueError(f"Missing news volume column: {news_vol_col}")

    sentiment = df[sentiment_col].fillna(0.0).astype(float)
    news_vol = df[news_vol_col].fillna(1.0).astype(float)

    contrib = sentiment * np.log1p(news_vol)

    # weights: oldest -> newest order for rolling.apply (aligns with x order)
    weights = np.exp(-decay * np.arange(window))[::-1]

    def dot_weights(x: np.ndarray) -> float:
        w = weights[-len(x):]
        return float(np.dot(x, w))

    nsis = contrib.rolling(window=window, min_periods=1).apply(dot_weights, raw=True)
    nsis = nsis.fillna(0.0).replace([np.inf, -np.inf], 0.0)

    return nsis


def compute_nsis(news_df: pd.DataFrame, decay_factor: float = 0.95, volume_weight: bool = True) -> pd.DataFrame:
    """
    News Sentiment Impact Score (NSIS) computed from raw news articles.

    Args:
        news_df: DataFrame containing at least columns `date` and `sentiment`.
        decay_factor: Decay factor applied within each day's articles (higher -> faster decay).
        volume_weight: If True, boost scores on days with more articles.

    Returns:
        DataFrame with columns `date`, `nsis`, `nsis_fast`, `nsis_slow`.
    """
    # New approach: compute per-article weighted sentiment, then aggregate to daily NSIS
    if news_df is None or news_df.empty:
        return pd.DataFrame(columns=['date', 'nsis', 'nsis_fast', 'nsis_slow', 'nsis_normalized'])

    df = news_df.copy()
    # Ensure a usable datetime column exists
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    elif 'datetime' in df.columns:
        df['date'] = pd.to_datetime(df['datetime'], errors='coerce')
    else:
        raise ValueError("news_df must contain a 'date' or 'datetime' column")

    df = df.dropna(subset=['date'])
    df['date'] = df['date'].dt.normalize()

    # Use 'sentiment' or 'sentiment_score' or compute neutral 0.0
    if 'sentiment' in df.columns:
        df['sentiment_val'] = pd.to_numeric(df['sentiment'], errors='coerce').fillna(0.0)
    elif 'sentiment_score' in df.columns:
        df['sentiment_val'] = pd.to_numeric(df['sentiment_score'], errors='coerce').fillna(0.0)
    else:
        df['sentiment_val'] = 0.0

    # Time-decay within day: newest has higher weight
    def weighted_avg(group):
        n = len(group)
        weights = np.exp(-decay_factor * np.arange(n)[::-1])
        vals = group['sentiment_val'].values
        try:
            return float(np.average(vals, weights=weights))
        except Exception:
            return float(np.mean(vals) if len(vals) > 0 else 0.0)

    daily_nsis = df.sort_values('date').groupby('date').apply(weighted_avg).rename('nsis')
    result = pd.DataFrame(daily_nsis)
    result['nsis_fast'] = result['nsis'].rolling(window=5, min_periods=1).mean()
    result['nsis_slow'] = result['nsis'].rolling(window=20, min_periods=1).mean()

    # Normalized (z-score) over a longer window
    mean = result['nsis'].rolling(window=60, min_periods=1).mean()
    std = result['nsis'].rolling(window=60, min_periods=1).std().replace(0, np.nan)
    result['nsis_normalized'] = ((result['nsis'] - mean) / (std + 1e-8)).fillna(0.0)

    result.index.name = 'date'
    result = result.reset_index()
    result['date'] = pd.to_datetime(result['date']).dt.date
    return result


class SentimentFeatureEngineer:
    """
    Engineer sentiment features from financial news data.
    
    Creates technical indicators, divergence signals, and regime classifications
    based on sentiment scores from FinBERT analysis.
    
    Attributes:
        sentiment_analyzer: Optional SentimentAnalyzer instance
        sentiment_threshold_positive: Threshold for positive sentiment (default: 0.2)
        sentiment_threshold_negative: Threshold for negative sentiment (default: -0.2)
    """
    
    def __init__(
        self, 
        sentiment_analyzer=None,
        sentiment_threshold_positive: float = 0.2,
        sentiment_threshold_negative: float = -0.2
    ):
        """
        Initialize sentiment feature engineer.
        
        Args:
            sentiment_analyzer: Optional SentimentAnalyzer instance (lazy-loaded if None)
            sentiment_threshold_positive: Score above this is considered positive
            sentiment_threshold_negative: Score below this is considered negative
        """
        self._sentiment_analyzer = sentiment_analyzer
        self.sentiment_threshold_positive = sentiment_threshold_positive
        self.sentiment_threshold_negative = sentiment_threshold_negative
    
    @property
    def sentiment_analyzer(self):
        """Lazy-load sentiment analyzer only when needed."""
        if self._sentiment_analyzer is None:
            try:
                from data.sentiment_analyzer import SentimentAnalyzer
                logger.info("Initializing FinBERT sentiment analyzer...")
                self._sentiment_analyzer = SentimentAnalyzer()
            except Exception as e:
                logger.error(f"Failed to load sentiment analyzer: {e}")
                raise
        return self._sentiment_analyzer

    def analyze_sentiment(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run per-article sentiment analysis and attach `sentiment_score` and `confidence`.
        Returns a copy of `news_df` with the new columns. Defensive: handles exceptions
        and returns an empty DataFrame if analysis fails for all articles.
        """
        if news_df is None or news_df.empty:
            logger.warning("analyze_sentiment: empty news_df")
            return pd.DataFrame()

        df = news_df.copy()

        # Normalize candidate text column
        text_cols = ['summary', 'headline', 'title', 'text']
        text_col = next((c for c in text_cols if c in df.columns), None)
        if text_col is None:
            logger.error("analyze_sentiment: no text column found in news_df")
            return pd.DataFrame()

        scores = []
        confs = []
        for i, row in df.iterrows():
            text = row.get(text_col, None)
            if not text or pd.isna(text):
                scores.append(0.0)
                confs.append(0.0)
                continue
            try:
                res = self.sentiment_analyzer.analyze_sentiment(text)
                scores.append(float(res.get('sentiment_score', 0.0)))
                confs.append(float(res.get('confidence', 0.0)))
            except Exception as e:
                logger.warning(f"analyze_sentiment: failed for article index {i}: {e}")
                scores.append(0.0)
                confs.append(0.0)

        df['sentiment_score'] = scores
        df['confidence'] = confs

        nonzero = (df['sentiment_score'] != 0.0).sum()
        logger.info(f"After sentiment analysis: {len(df)} articles, {nonzero} non-zero scores")

        return df
    
    def aggregate_daily_sentiment(
        self, 
        news_df: pd.DataFrame,
        price_df: Optional[pd.DataFrame] = None,
        use_summary: bool = False
    ) -> pd.DataFrame:
        """
        Aggregate news sentiment by date with proper price data alignment.
        
        Key improvements:
        - Match news dates to trading days (not calendar days)
        - Handle timezone mismatches
        - Forward-fill weekends/holidays
        - Ensure complete date coverage
        
        Args:
            news_df: DataFrame with 'date' and 'headline' columns
            price_df: Optional price DataFrame to align dates with
            use_summary: Whether to analyze summary text instead of headlines
        
        Returns:
            DataFrame with daily sentiment metrics indexed by date
        
        Example:
            >>> daily_sentiment = engineer.aggregate_daily_sentiment(news_df, price_df)
            >>> print(daily_sentiment[['sentiment_mean', 'news_volume']].head())
        """
        if news_df is None or len(news_df) == 0:
            logger.warning("Empty news DataFrame provided")
            return pd.DataFrame()

        logger.info(f"Aggregating sentiment for {len(news_df)} news articles...")

        # Work on a copy and require a 'date' column (created by analyze_sentiment)
        df = news_df.copy()
        if 'date' not in df.columns:
            logger.error("aggregate_daily_sentiment: 'date' column missing from news_df")
            return pd.DataFrame()

        # Normalize to midnight timestamps for grouping consistency
        df['date'] = pd.to_datetime(df['date']).dt.normalize()

        # Prepare aggregation mapping
        agg_map = {
            'sentiment_score': ['mean', 'std', 'min', 'max', 'median'],
            'confidence': 'mean',
            'positive': 'sum',
            'negative': 'sum',
            'neutral': 'sum'
        }

        # Ensure per-article positive/negative/neutral flags exist; else derive from sentiment_score
        if not set(['positive', 'negative', 'neutral']).issubset(df.columns):
            if 'sentiment_score' in df.columns:
                df['positive'] = (pd.to_numeric(df['sentiment_score'], errors='coerce') > self.sentiment_threshold_positive).astype(int)
                df['negative'] = (pd.to_numeric(df['sentiment_score'], errors='coerce') < self.sentiment_threshold_negative).astype(int)
                df['neutral'] = (((pd.to_numeric(df['sentiment_score'], errors='coerce') >= self.sentiment_threshold_negative) &
                                  (pd.to_numeric(df['sentiment_score'], errors='coerce') <= self.sentiment_threshold_positive))).astype(int)
            else:
                df['positive'] = 0
                df['negative'] = 0
                df['neutral'] = 0

        try:
            grouped = df.groupby('date').agg(agg_map)

            # Flatten columns
            grouped.columns = [
                'sentiment_mean', 'sentiment_std', 'sentiment_min', 'sentiment_max', 'sentiment_median',
                'confidence_mean', 'positive_count', 'negative_count', 'neutral_count'
            ]

            # News volume per day
            news_volume = df.groupby('date').size().rename('news_volume')
            daily = grouped.join(news_volume)
            daily = daily.reset_index()

            logger.info(f"After aggregation: {len(daily)} unique days")

            if daily.empty:
                logger.error("Aggregation produced empty DataFrame!")
                return pd.DataFrame()

            total = daily['positive_count'] + daily['negative_count'] + daily['neutral_count']
            daily['positive_ratio'] = daily['positive_count'] / (total + 1e-8)
            daily['negative_ratio'] = daily['negative_count'] / (total + 1e-8)
            daily['neutral_ratio'] = daily['neutral_count'] / (total + 1e-8)

            daily = daily.drop(columns=['positive_count', 'negative_count', 'neutral_count'])

            # If price data provided, align to trading dates
            trading_dates = None
            if price_df is not None and not price_df.empty:
                if isinstance(price_df.index, pd.DatetimeIndex):
                    trading_dates = pd.to_datetime(price_df.index)
                elif 'Date' in price_df.columns:
                    trading_dates = pd.to_datetime(price_df['Date'])

            if trading_dates is not None:
                full_idx = pd.DatetimeIndex(pd.to_datetime(trading_dates)).normalize()
                daily = daily.set_index('date')
                daily.index = pd.to_datetime(daily.index)
                daily = daily.reindex(full_idx)

                # Forward-fill small gaps (news persists over weekends/holidays)
                # NOTE: No bfill() to prevent look-ahead bias
                daily = daily.ffill(limit=3)

                # Fill remaining NaNs with neutral defaults
                daily = daily.fillna({
                    'sentiment_mean': 0.0,
                    'sentiment_std': 0.0,
                    'sentiment_max': 0.0,
                    'sentiment_min': 0.0,
                    'sentiment_median': 0.0,
                    'confidence_mean': 0.5,
                    'news_volume': 0,
                    'positive_ratio': 0.33,
                    'negative_ratio': 0.33,
                    'neutral_ratio': 0.34
                })

                logger.info(f"✓ Aligned to {len(daily)} trading days")
                logger.info(f"   Days with news: {(daily['news_volume'] > 0).sum()}")

            else:
                # When no trading alignment, keep date as datetime index
                daily = daily.set_index('date')
                daily.index = pd.to_datetime(daily.index)

            return daily
        except Exception as e:
            logger.error(f"Error during aggregation: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
        # Filter news to match price data date range if provided
        if min_date is not None and max_date is not None:
            news_df = news_df[
                (news_df['trade_date'] >= min_date) & 
                (news_df['trade_date'] <= max_date)
            ]
            logger.info(f"News articles in range: {len(news_df)}")
        
        # Choose text column
        text_column = 'summary' if use_summary and 'summary' in news_df.columns else next((c for c in ['headline', 'title', 'summary', 'text'] if c in news_df.columns), None)
        if text_column is None and 'sentiment_score' not in news_df.columns:
            logger.error("aggregate_daily_sentiment: no text column and no sentiment_score column available")
            return pd.DataFrame()

        # Aggregate by date, prefer pre-computed per-article sentiment if present
        daily_sentiment = []

        for date, group in news_df.groupby('trade_date'):
            # Prefer pre-computed sentiment_score
            if 'sentiment_score' in group.columns:
                sentiments = pd.to_numeric(group['sentiment_score'], errors='coerce').dropna().astype(float).values
                confidences = pd.to_numeric(group.get('confidence', pd.Series([0.0]*len(group))), errors='coerce').fillna(0.0).values
                total_count = len(group)
            else:
                texts = group[text_column].tolist()
                sentiments = []
                confidences = []
                for text in texts:
                    if not text or pd.isna(text):
                        continue
                    try:
                        res = self.sentiment_analyzer.analyze_sentiment(text)
                        sentiments.append(res.get('sentiment_score', 0.0))
                        confidences.append(res.get('confidence', 0.0))
                    except Exception as e:
                        logger.warning(f"aggregate_daily_sentiment: per-article analyze failed: {e}")
                        continue
                sentiments = np.array(sentiments, dtype=float) if len(sentiments) > 0 else np.array([])
                confidences = np.array(confidences, dtype=float) if len(confidences) > 0 else np.array([])
                total_count = len(group)

            if len(sentiments) == 0:
                # still add a zero-volume day entry (will be filled later)
                daily_sentiment.append({
                    'date': pd.Timestamp(date),
                    'sentiment_mean': 0.0,
                    'sentiment_std': 0.0,
                    'sentiment_max': 0.0,
                    'sentiment_min': 0.0,
                    'sentiment_median': 0.0,
                    'confidence_mean': 0.0,
                    'news_volume': 0,
                    'positive_ratio': 0.0,
                    'negative_ratio': 0.0,
                    'neutral_ratio': 0.0
                })
                continue

            positive_count = int((sentiments > self.sentiment_threshold_positive).sum())
            negative_count = int((sentiments < self.sentiment_threshold_negative).sum())
            neutral_count = int(((sentiments >= self.sentiment_threshold_negative) & (sentiments <= self.sentiment_threshold_positive)).sum())

            daily_sentiment.append({
                'date': pd.Timestamp(date),
                'sentiment_mean': float(np.mean(sentiments)),
                'sentiment_std': float(np.std(sentiments)) if len(sentiments) > 1 else 0.0,
                'sentiment_max': float(np.max(sentiments)),
                'sentiment_min': float(np.min(sentiments)),
                'sentiment_median': float(np.median(sentiments)),
                'confidence_mean': float(np.mean(confidences)) if len(confidences) > 0 else 0.0,
                'news_volume': int(total_count),
                'positive_ratio': float(positive_count / max(1, total_count)),
                'negative_ratio': float(negative_count / max(1, total_count)),
                'neutral_ratio': float(neutral_count / max(1, total_count))
            })
        
        if len(daily_sentiment) == 0:
            logger.warning("No sentiment data could be aggregated")
            return pd.DataFrame()
        
        sentiment_df = pd.DataFrame(daily_sentiment)
        sentiment_df.set_index('date', inplace=True)
        
        # If price data provided, align to trading days
        if trading_dates is not None:
            # Create complete date range matching price data
            complete_index = pd.DatetimeIndex(trading_dates)
            sentiment_df = sentiment_df.reindex(complete_index)
            
            # Forward-fill sentiment for weekends/holidays (max 3 days)
            sentiment_df = sentiment_df.ffill(limit=3)
            
            # NOTE: No bfill() - fill remaining NaNs at start with neutral values instead
            # This prevents look-ahead bias
            
            # Fill remaining NaNs with neutral sentiment
            sentiment_df = sentiment_df.fillna({
                'sentiment_mean': 0.0,
                'sentiment_std': 0.0,
                'sentiment_max': 0.0,
                'sentiment_min': 0.0,
                'sentiment_median': 0.0,
                'confidence_mean': 0.5,
                'news_volume': 0,
                'positive_ratio': 0.33,
                'negative_ratio': 0.33,
                'neutral_ratio': 0.34
            })
            
            logger.info(f"✓ Aligned to {len(sentiment_df)} trading days")
            logger.info(f"   Days with news: {(sentiment_df['news_volume'] > 0).sum()}")
        else:
            sentiment_df = sentiment_df.sort_values('date').reset_index(drop=True)
            logger.info(f"✓ Aggregated sentiment for {len(sentiment_df)} trading days")
        
            # Compute NSIS from raw news (grouped by date) and merge
            try:
                nsis_df = compute_nsis(news_df)
                if not nsis_df.empty:
                    # Ensure index alignment: nsis_df uses Timestamp in 'date' column
                    nsis_df['date'] = pd.to_datetime(nsis_df['date']).dt.date
                    sentiment_df_reset = sentiment_df.reset_index()
                    sentiment_df_reset['date'] = pd.to_datetime(sentiment_df_reset['date']).dt.date
                    merged = sentiment_df_reset.merge(nsis_df, on='date', how='left')
                    merged = merged.set_index('date')
                    # If merged contains nsis columns, use them; else keep original
                    sentiment_df = merged
                    # Fill missing nsis values with sentiment_mean
                    if 'nsis' in sentiment_df.columns:
                        sentiment_df['nsis'] = sentiment_df['nsis'].fillna(sentiment_df.get('sentiment_mean', 0.0))
                    else:
                        sentiment_df['nsis'] = sentiment_df.get('sentiment_mean', 0.0)
                    # Ensure nsis_fast/slow exist
                    sentiment_df['nsis_fast'] = sentiment_df.get('nsis_fast', sentiment_df['nsis'])
                    sentiment_df['nsis_slow'] = sentiment_df.get('nsis_slow', sentiment_df['nsis'])
            except Exception as e:
                logger.warning(f"Failed to compute NSIS from raw news: {e}")
                # Fallback: compute NSIS from daily aggregates
                try:
                    sentiment_df['nsis'] = compute_nsis_from_daily(sentiment_df, 'sentiment_mean', 'news_volume', window=5, decay=0.3)
                    sentiment_df['nsis_fast'] = compute_nsis_from_daily(sentiment_df, 'sentiment_mean', 'news_volume', window=5, decay=0.5)
                    sentiment_df['nsis_slow'] = compute_nsis_from_daily(sentiment_df, 'sentiment_mean', 'news_volume', window=5, decay=0.1)
                except Exception:
                    sentiment_df['nsis'] = sentiment_df.get('sentiment_mean', 0.0)
                    sentiment_df['nsis_fast'] = sentiment_df['nsis']
                    sentiment_df['nsis_slow'] = sentiment_df['nsis']

            return sentiment_df
    
    def create_sentiment_technical_indicators(
        self, 
        sentiment_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create technical indicators from sentiment scores.
        
        Applies traditional technical analysis to sentiment data:
        - Moving averages (SMA 5, 20)
        - Momentum (SMA crossover)
        - RSI (sentiment overbought/oversold)
        - Volatility (rolling std)
        
        Args:
            sentiment_df: DataFrame with 'sentiment_mean' column
        
        Returns:
            DataFrame with added technical indicator columns
        
        Example:
            >>> sentiment_df = engineer.create_sentiment_technical_indicators(daily_sentiment)
            >>> print(sentiment_df[['sentiment_sma_5', 'sentiment_momentum']].head())
        """
        if sentiment_df.empty:
            return sentiment_df

        # Preserve existing columns and append technical indicator columns
        df = sentiment_df.copy()

        # Ensure sentiment_mean and sentiment_std exist
        if 'sentiment_mean' not in df.columns:
            df['sentiment_mean'] = 0.0
        if 'sentiment_std' not in df.columns:
            df['sentiment_std'] = 0.0

        # Moving averages - canonical names
        df['sentiment_sma_5'] = df['sentiment_mean'].rolling(window=5, min_periods=1).mean()
        df['sentiment_sma_20'] = df['sentiment_mean'].rolling(window=20, min_periods=1).mean()

        # Momentum and acceleration (use diffs over 5 days)
        df['sentiment_momentum'] = df['sentiment_mean'].diff(5).fillna(0.0)
        df['sentiment_acceleration'] = df['sentiment_momentum'].diff(5).fillna(0.0)

        # Volatility (smoothed std of daily sentiment)
        df['sentiment_volatility'] = df['sentiment_std'].rolling(window=10, min_periods=1).mean().fillna(0.0)

        # RSI on sentiment_mean
        delta = df['sentiment_mean'].diff()
        gain = delta.clip(lower=0).rolling(window=14, min_periods=1).mean()
        loss = -delta.clip(upper=0).rolling(window=14, min_periods=1).mean()
        rs = gain / (loss + 1e-8)
        df['sentiment_rsi'] = 100 - (100 / (1 + rs))

        logger.info("✓ Created sentiment technical indicators (canonical)")
        return df
    
    def create_sentiment_price_divergence(
        self, 
        sentiment_df: pd.DataFrame,
        price_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Detect divergence between sentiment and price movement.
        
        Divergences often predict reversals:
        - Price up + sentiment down = potential reversal down
        - Price down + sentiment up = potential reversal up
        
        Args:
            sentiment_df: DataFrame with sentiment features
            price_df: DataFrame with 'date' and 'close' or 'returns' columns
        
        Returns:
            DataFrame with divergence indicators
        
        Example:
            >>> features = engineer.create_sentiment_price_divergence(sentiment_df, price_df)
            >>> print(features[['price_up_sentiment_down', 'price_down_sentiment_up']].sum())
        """
        if sentiment_df.empty or price_df.empty:
            return sentiment_df
        
        df = sentiment_df.copy()
        
        # Merge with price data
        price_df = price_df.copy()
        
        # Ensure date columns
        if 'date' not in price_df.columns:
            price_df = price_df.reset_index()
        
        price_df['date'] = pd.to_datetime(price_df['date'])
        
        # Calculate returns if not present. Accept either 'close' or 'Close' (and other case variants).
        if 'returns' not in price_df.columns:
            close_col = None
            # Direct matches (common variants)
            if 'close' in price_df.columns:
                close_col = 'close'
            elif 'Close' in price_df.columns:
                close_col = 'Close'
            else:
                # Case-insensitive match for other variants (e.g., 'Close ', 'close_price')
                lowered = {c.lower(): c for c in price_df.columns}
                if 'close' in lowered:
                    close_col = lowered['close']

            if close_col:
                # Compute returns from the detected close column
                price_df['returns'] = price_df[close_col].pct_change()
            else:
                # Price column not found; this is non-actionable — use debug log and skip divergence
                logger.debug("Price column not found in price_df (expected 'close' or 'returns'); skipping sentiment-price divergence")
                return df
        
        # Merge on date
        df = df.merge(
            price_df[['date', 'returns']], 
            on='date', 
            how='left'
        )
        
        # Fill missing returns
        df['returns'] = df['returns'].fillna(0)
        
        # Divergence signals
        # Price up but sentiment deteriorating
        df['price_up_sentiment_down'] = (
            (df['returns'] > 0.01) & 
            (df['sentiment_momentum'] < -0.1)
        ).astype(int)
        
        # Price down but sentiment improving
        df['price_down_sentiment_up'] = (
            (df['returns'] < -0.01) & 
            (df['sentiment_momentum'] > 0.1)
        ).astype(int)
        
        # Strong divergence (larger thresholds)
        df['strong_bearish_divergence'] = (
            (df['returns'] > 0.02) & 
            (df['sentiment_momentum'] < -0.2)
        ).astype(int)
        
        df['strong_bullish_divergence'] = (
            (df['returns'] < -0.02) & 
            (df['sentiment_momentum'] > 0.2)
        ).astype(int)
        
        # Drop returns column (not a feature)
        df = df.drop(columns=['returns'])
        
        logger.info("✓ Created sentiment-price divergence features")
        
        return df
    
    def create_news_volume_impact(
        self, 
        sentiment_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create features based on news volume impact.
        
        High news volume + strong sentiment = more reliable signal.
        Uses 80th percentile as threshold for "high volume".
        
        Args:
            sentiment_df: DataFrame with 'news_volume' and 'sentiment_mean'
        
        Returns:
            DataFrame with news volume impact features
        
        Example:
            >>> features = engineer.create_news_volume_impact(sentiment_df)
            >>> print(features['high_volume_positive'].sum())
        """
        if sentiment_df.empty:
            return sentiment_df
        
        df = sentiment_df.copy()
        
        # Calculate volume percentiles
        volume_p80 = df['news_volume'].quantile(0.80)
        volume_p20 = df['news_volume'].quantile(0.20)
        
        # High volume signals
        df['high_volume_positive'] = (
            (df['news_volume'] > volume_p80) & 
            (df['sentiment_mean'] > 0.3)
        ).astype(int)
        
        df['high_volume_negative'] = (
            (df['news_volume'] > volume_p80) & 
            (df['sentiment_mean'] < -0.3)
        ).astype(int)
        
        # Low volume (less reliable)
        df['low_volume_day'] = (df['news_volume'] < volume_p20).astype(int)
        
        # Volume surge (compared to 5-day average)
        df['news_volume_ma5'] = df['news_volume'].rolling(window=5, min_periods=1).mean()
        df['sent_volume_surge'] = (df['news_volume'] > df['news_volume_ma5'] * 1.5).astype(int)
        
        # Combined signal: volume surge + strong sentiment
        df['surge_positive'] = (
            (df['sent_volume_surge'] == 1) & 
            (df['sentiment_mean'] > 0.2)
        ).astype(int)
        
        df['surge_negative'] = (
            (df['sent_volume_surge'] == 1) & 
            (df['sentiment_mean'] < -0.2)
        ).astype(int)
        
        # Drop intermediate columns
        df = df.drop(columns=['news_volume_ma5'])
        
        logger.info("✓ Created news volume impact features")
        
        return df
    
    def create_sentiment_regime(
        self, 
        sentiment_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Classify sentiment regime (bullish/bearish/neutral).
        
        Regime classification helps adapt trading strategy:
        - Bullish regime: Favor long positions
        - Bearish regime: Favor short/defensive positions
        - Neutral regime: Use other signals
        
        Args:
            sentiment_df: DataFrame with sentiment indicators
        
        Returns:
            DataFrame with regime classification columns
        
        Example:
            >>> features = engineer.create_sentiment_regime(sentiment_df)
            >>> print(features['sentiment_regime'].value_counts())
        """
        if sentiment_df.empty:
            return sentiment_df

        df = sentiment_df.copy()
        result = pd.DataFrame(index=df.index)

        bullish_condition = (df['sentiment_sma_20'] > 0.2)
        bearish_condition = (df['sentiment_sma_20'] < -0.2)

        result['bullish_regime'] = bullish_condition.astype(int)
        result['bearish_regime'] = bearish_condition.astype(int)
        result['neutral_regime'] = (~(bullish_condition | bearish_condition)).astype(int)
        result['regime_strength'] = df['sentiment_sma_20'].abs()

        logger.info("✓ Created sentiment regime features (canonical)")
        return result
    
    def create_all_features(
        self, 
        news_df: pd.DataFrame,
        price_df: Optional[pd.DataFrame] = None,
        use_summary: bool = False
    ) -> pd.DataFrame:
        """
        Create complete sentiment feature set.
        
        This is the main function that orchestrates all feature engineering steps:
        1. Aggregate daily sentiment
        2. Create technical indicators
        3. Detect price divergence (if price_df provided)
        4. Analyze news volume impact
        5. Classify sentiment regime
        
        Args:
            news_df: DataFrame with news data from NewsFetcher
            price_df: Optional DataFrame with price data for divergence detection
            use_summary: Whether to use summary text instead of headlines
        
        Returns:
            DataFrame with complete sentiment feature set (11+ features)
        
        Example:
            >>> from data.news_fetcher import NewsFetcher
            >>> fetcher = NewsFetcher()
            >>> news_df = fetcher.fetch_company_news('AAPL', days_back=365)
            >>> 
            >>> engineer = SentimentFeatureEngineer()
            >>> features = engineer.create_all_features(news_df, price_df)
            >>> print(f"Created {len(features.columns)} features")
        """
        logger.info("Creating complete sentiment feature set...")

        if news_df is None or news_df.empty:
            logger.warning("Empty or None news DataFrame provided")
            # Return zero-filled features matching price_df index if available
            if price_df is not None and not price_df.empty:
                return self._create_zero_filled_features(price_df.index)
            return pd.DataFrame()

        logger.info(f"Input: {len(news_df)} news articles; price days: {0 if price_df is None else (0 if price_df.empty else len(price_df))}")

        # Step 0: per-article sentiment analysis (defensive)
        news_with_sentiment = self.analyze_sentiment(news_df)
        logger.info(f"After sentiment analysis: {0 if news_with_sentiment is None else len(news_with_sentiment)} articles")
        if news_with_sentiment is None or news_with_sentiment.empty:
            logger.error("CRITICAL: Sentiment analysis returned empty DataFrame")
            if price_df is not None and not price_df.empty:
                return self._create_zero_filled_features(price_df.index)
            return pd.DataFrame()

        # Step 1: Aggregate daily sentiment with price alignment
        sentiment_df = self.aggregate_daily_sentiment(
            news_with_sentiment,
            price_df=price_df,
            use_summary=use_summary
        )

        logger.info(f"After daily aggregation: {0 if sentiment_df is None else len(sentiment_df)} days with sentiment")
        if sentiment_df is None or sentiment_df.empty:
            logger.error("CRITICAL: Daily aggregation returned empty DataFrame")
            if price_df is not None and not price_df.empty:
                return self._create_zero_filled_features(price_df.index)
            # fallback to zero-filled features using unique news dates
            try:
                # prefer 'date' created by analyze_sentiment
                if 'date' in news_with_sentiment.columns:
                    dates = pd.to_datetime(news_with_sentiment['date']).dt.normalize().unique()
                else:
                    dates = pd.to_datetime(news_with_sentiment['datetime']).dt.normalize().unique()
                idx = pd.DatetimeIndex(dates)
                return self._create_zero_filled_features(idx)
            except Exception:
                return pd.DataFrame()

        # Replace flat sentiment_mean with NSIS (weighted, decayed, volume-adjusted)
        try:
            # If we already have daily aggregated sentiment (sentiment_mean + news_volume),
            # compute NSIS from the daily aggregates which is the safe, intended path.
            if 'sentiment_mean' in sentiment_df.columns and 'news_volume' in sentiment_df.columns:
                sentiment_df['nsis'] = compute_nsis_from_daily(
                    sentiment_df, 'sentiment_mean', 'news_volume', window=5, decay=0.3
                )
                sentiment_df['nsis_fast'] = compute_nsis_from_daily(
                    sentiment_df, 'sentiment_mean', 'news_volume', window=5, decay=0.5
                )
                sentiment_df['nsis_slow'] = compute_nsis_from_daily(
                    sentiment_df, 'sentiment_mean', 'news_volume', window=5, decay=0.1
                )
            else:
                # Fall back to computing NSIS from raw news articles if available
                nsis_df = compute_nsis(news_df) if news_df is not None else pd.DataFrame()
                if not nsis_df.empty:
                    # nsis_df may contain a 'date' column; align/merge by date
                    nsis_df = nsis_df.copy()
                    nsis_df['date'] = pd.to_datetime(nsis_df['date']).dt.date
                    if 'date' in sentiment_df.index.names or isinstance(sentiment_df.index, pd.DatetimeIndex):
                        # align by index if sentiment_df indexed by Timestamp
                        sentiment_df_reset = sentiment_df.reset_index()
                        sentiment_df_reset['date'] = pd.to_datetime(sentiment_df_reset['date']).dt.date
                        merged = sentiment_df_reset.merge(nsis_df, on='date', how='left')
                        merged = merged.set_index('date')
                        sentiment_df['nsis'] = merged.get('nsis', sentiment_df.get('sentiment_mean', 0.0))
                        sentiment_df['nsis_fast'] = merged.get('nsis_fast', sentiment_df.get('sentiment_mean', 0.0))
                        sentiment_df['nsis_slow'] = merged.get('nsis_slow', sentiment_df.get('sentiment_mean', 0.0))
                    else:
                        # simple assignment if nsis_df index aligns
                        for col in ['nsis', 'nsis_fast', 'nsis_slow']:
                            if col in nsis_df.columns:
                                sentiment_df[col] = nsis_df[col].values

            # Normalized NSIS: z-score over 20-day rolling window
            nsis_mean = sentiment_df['nsis'].rolling(window=20, min_periods=1).mean()
            nsis_std = sentiment_df['nsis'].rolling(window=20, min_periods=1).std().replace(0, np.nan)
            sentiment_df['nsis_normalized'] = ((sentiment_df['nsis'] - nsis_mean) / nsis_std).fillna(0.0)

            # Overwrite sentiment_mean with nsis so downstream code uses weighted score
            sentiment_df['sentiment_mean'] = sentiment_df['nsis']
        except Exception as e:
            logger.warning(f"Failed to compute NSIS: {e}. Falling back to sentiment_mean.")
        
        # Step 2: Technical indicators
        sentiment_df = self.create_sentiment_technical_indicators(sentiment_df)
        
        # Step 3: Price divergence (if price data provided)
        if price_df is not None and not price_df.empty:
            sentiment_df = self.create_sentiment_price_divergence(sentiment_df, price_df)
        
        # Step 4: News volume impact
        sentiment_df = self.create_news_volume_impact(sentiment_df)
        
        # Step 5: Sentiment regime
        sentiment_df = self.create_sentiment_regime(sentiment_df)
        # Map produced columns to the canonical sentiment feature names expected
        # by the rest of the pipeline (avoid surprising extra/missing columns).
        try:
            from data.feature_engineer import get_sentiment_feature_columns
            canonical_cols = get_sentiment_feature_columns()
        except Exception:
            # Fallback canonical list (must match feature_engineer.get_sentiment_feature_columns())
            canonical_cols = [
                'sentiment_mean','sentiment_std','news_volume','news_volume_normalized','positive_ratio','negative_ratio',
                'sentiment_sma_5','sentiment_sma_20','sentiment_momentum','sentiment_acceleration','sentiment_volatility','sentiment_rsi',
                'price_up_sentiment_down','price_down_sentiment_up','strong_bearish_divergence','strong_bullish_divergence',
                'high_volume_positive','high_volume_negative','low_volume_day','volume_surge','surge_positive','surge_negative',
                'bullish_regime','bearish_regime','neutral_regime','regime_strength'
            ]

        # Build canonical DataFrame, filling from available columns or zeros
        canon_df = pd.DataFrame(index=sentiment_df.index)
        for col in canonical_cols:
            if col in sentiment_df.columns:
                canon_df[col] = sentiment_df[col]
            else:
                # Accept close equivalents
                if col == 'sentiment_mean' and 'nsis' in sentiment_df.columns:
                    canon_df[col] = sentiment_df['nsis']
                elif col == 'news_volume_normalized' and 'news_volume' in sentiment_df.columns:
                    canon_df[col] = np.log1p(sentiment_df['news_volume'])
                elif col == 'volume_surge' and 'sent_volume_surge' in sentiment_df.columns:
                    canon_df[col] = sentiment_df['sent_volume_surge']
                else:
                    # default zeros
                    canon_df[col] = 0.0

        # Ensure integer types where appropriate
        if 'news_volume' in canon_df.columns:
            try:
                canon_df['news_volume'] = canon_df['news_volume'].fillna(0).astype(int)
            except Exception:
                canon_df['news_volume'] = canon_df['news_volume'].fillna(0).round().astype(int)

        # Attach any remaining helper columns from sentiment_df under their names
        # (these will be dropped later by canonical validators)
        # Finally return canonical DataFrame
        logger.info(f"✓ Created {len(canon_df.columns)} sentiment features (canonical)")
        return canon_df
    
    def get_feature_columns(self) -> List[str]:
        """
        Get list of sentiment feature column names.
        
        Returns list of all features created by this class for use in
        machine learning models.
        
        Returns:
            List of feature column names
        """
        return [
            # Daily aggregates (NSIS replaces flat sentiment_mean)
            'nsis',
            'nsis_fast',
            'nsis_slow',
            'nsis_normalized',
            'sentiment_std',
            'sentiment_max',
            'sentiment_min',
            'sentiment_median',
            'news_volume',
            'positive_ratio',
            'negative_ratio',
            'neutral_ratio',
            'confidence_mean',
            
            # Technical indicators
            'sentiment_sma_5',
            'sentiment_sma_20',
            'sentiment_ema_5',
            'sentiment_ema_20',
            'sentiment_momentum',
            'sentiment_acceleration',
            'sentiment_volatility',
            'sentiment_rsi',
            
            # Price divergence (if price data provided)
            'price_up_sentiment_down',
            'price_down_sentiment_up',
            'strong_bearish_divergence',
            'strong_bullish_divergence',
            
            # News volume impact
            'high_volume_positive',
            'high_volume_negative',
            'low_volume_day',
            'sent_volume_surge',
            'surge_positive',
            'surge_negative',
            
            # Regime classification
            'bullish_regime',
            'bearish_regime',
            'neutral_regime',
            'regime_strength'
        ]
    
    # ===================================================================
    # PRIVATE METHODS
    # ===================================================================
    
    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI) for sentiment.
        
        RSI measures momentum and overbought/oversold conditions.
        Applied to sentiment scores instead of prices.
        
        Args:
            series: Sentiment score series
            period: RSI period (default: 14)
        
        Returns:
            RSI values (0-100)
        """
        # Calculate price changes
        delta = series.diff()
        
        # Separate gains and losses
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        
        # Calculate average gain/loss
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def validate_sentiment_features(
        sentiment_df: pd.DataFrame, 
        expected_rows: int
    ) -> bool:
        """
        Validate sentiment data quality.
        
        Checks:
        - Row count matches expected
        - NaN percentage is acceptable
        - News coverage is reasonable
        
        Args:
            sentiment_df: DataFrame with sentiment features
            expected_rows: Expected number of rows (trading days)
        
        Returns:
            True if validation passes, False otherwise
        """
        if len(sentiment_df) != expected_rows:
            logger.error(
                f"Sentiment row count mismatch: expected {expected_rows}, "
                f"got {len(sentiment_df)}"
            )
            return False
        
        # Check NaN percentage
        nan_pct = (sentiment_df.isna().sum() / len(sentiment_df) * 100).to_dict()
        high_nan = {k: v for k, v in nan_pct.items() if v > 10}
        
        if high_nan:
            logger.warning(f"High NaN percentage in sentiment features: {high_nan}")
        
        # Check if we have reasonable news coverage
        if 'news_volume' in sentiment_df.columns:
            days_with_news = (sentiment_df['news_volume'] > 0).sum()
            coverage = days_with_news / len(sentiment_df) * 100
            
            logger.info(
                f"News coverage: {coverage:.1f}% "
                f"({days_with_news}/{len(sentiment_df)} days)"
            )
            
            if coverage < 20:
                logger.warning(
                    f"Low news coverage: only {coverage:.1f}% of days have news. "
                    "This is normal for short time periods or less-covered stocks."
                )
        
        return True

    def _create_zero_filled_features(self, index) -> pd.DataFrame:
        """
        Create a zero-filled sentiment features DataFrame matching the provided index.
        """
        try:
            from data.feature_engineer import get_sentiment_feature_columns
            cols = get_sentiment_feature_columns()
        except Exception:
            cols = self.get_feature_columns()
        try:
            idx = pd.DatetimeIndex(index)
        except Exception:
            # Try to coerce
            idx = pd.to_datetime(index)
        zero_df = pd.DataFrame(0.0, index=idx, columns=cols)
        # Ensure integer columns are integers where appropriate
        if 'news_volume' in zero_df.columns:
            zero_df['news_volume'] = zero_df['news_volume'].astype(int)
        logger.warning("No sentiment-price alignment; returning zero-filled sentiment features")
        return zero_df


def example_usage():
    """Example usage of SentimentFeatureEngineer."""
    
    print("\n" + "="*70)
    print("  SENTIMENT FEATURE ENGINEERING - EXAMPLE USAGE")
    print("="*70 + "\n")
    
    # Check for API key
    api_key = os.getenv('FINNHUB_API_KEY')
    if not api_key:
        print("⚠️  FINNHUB_API_KEY environment variable not set")
        print("   Set it in .env file or export FINNHUB_API_KEY='your_key'")
        return
    
    try:
        from data.news_fetcher import NewsFetcher
        
        # Fetch news
        print("1. Fetching news data...")
        print("-" * 70)
        
        fetcher = NewsFetcher(api_key=api_key)
        symbol = 'AAPL'
        news_df = fetcher.fetch_company_news(symbol, days_back=90)
        
        print(f"✓ Fetched {len(news_df)} articles")
        
        # Create sentiment features
        print("\n2. Creating sentiment features...")
        print("-" * 70)
        
        engineer = SentimentFeatureEngineer()
        sentiment_features = engineer.create_all_features(news_df)
        
        print(f"✓ Created {len(sentiment_features.columns)} features")
        print(f"✓ Feature range: {sentiment_features['date'].min()} to {sentiment_features['date'].max()}")
        
        # Display sample features
        print("\n3. Sample Features:")
        print("-" * 70)
        
        feature_cols = [
            'date', 
            'sentiment_mean', 
            'news_volume',
            'sentiment_momentum',
            'bullish_regime'
        ]
        
        print(sentiment_features[feature_cols].head(10).to_string(index=False))
        
        # Summary statistics
        print("\n4. Feature Statistics:")
        print("-" * 70)
        
        print(f"Average sentiment: {sentiment_features['sentiment_mean'].mean():.3f}")
        print(f"Sentiment volatility: {sentiment_features['sentiment_std'].mean():.3f}")
        print(f"Average news volume: {sentiment_features['news_volume'].mean():.1f} articles/day")
        print(f"Bullish regime days: {sentiment_features['bullish_regime'].sum()}")
        print(f"Bearish regime days: {sentiment_features['bearish_regime'].sum()}")
        
        # Feature list
        print("\n5. All Feature Columns:")
        print("-" * 70)
        
        for i, col in enumerate(engineer.get_feature_columns(), 1):
            print(f"  {i:2d}. {col}")
        
        print("\n" + "="*70 + "\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    example_usage()
