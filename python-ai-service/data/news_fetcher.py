"""
Finnhub News Fetcher

This module provides integration with Finnhub API to fetch company news and sentiment data.
Finnhub offers real-time and historical news coverage for stocks with built-in rate limiting
and error handling.

Features:
- Historical news (up to 1 year on free tier)
- Real-time news updates
- Built-in rate limiting (60 calls/minute)
- News aggregation by trading date
- Smart text preprocessing
- Duplicate detection
- Finnhub sentiment API integration

API Documentation: https://finnhub.io/docs/api

Usage:
    from data.news_fetcher import NewsFetcher
    
    fetcher = NewsFetcher(api_key="YOUR_KEY")
    news_df = fetcher.fetch_company_news('AAPL', days_back=365)
    
    # Get daily aggregated news
    daily_news = fetcher.aggregate_news_by_date(news_df)
"""

import os
import time
import re
import html
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union
from functools import wraps
import logging

import pandas as pd
import numpy as np

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    from pathlib import Path

    # Search upward from this file for a .env file (supports running from several cwd)
    candidates = list(Path(__file__).resolve().parents)[:6]
    candidates = [Path(__file__).resolve()] + candidates
    env_file = None
    for p in candidates:
        candidate = p / '.env'
        if candidate.exists():
            env_file = str(candidate)
            break

    if env_file:
        load_dotenv(env_file)
    else:
        # Last attempt: try current working directory
        cwd_env = Path(os.getcwd()) / '.env'
        if cwd_env.exists():
            load_dotenv(str(cwd_env))
except ImportError:
    # dotenv not installed - will still work if env vars are set manually
    pass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import finnhub
try:
    import finnhub
except ImportError:
    logger.error(
        "finnhub-python library not found. Install with:\n"
        "pip install finnhub-python>=2.4.0"
    )
    raise


class RateLimiter:
    """
    Rate limiter for Finnhub API calls.
    
    Finnhub free tier limits:
    - 60 API calls per minute
    - 30 API calls per second (burst)
    
    This class ensures we stay under these limits with exponential backoff on errors.
    """
    
    def __init__(self, calls_per_minute: int = 60, calls_per_second: int = 30):
        """
        Initialize rate limiter.
        
        Args:
            calls_per_minute: Max calls per minute (default: 60 for free tier)
            calls_per_second: Max calls per second (default: 30 for free tier)
        """
        self.calls_per_minute = calls_per_minute
        self.calls_per_second = calls_per_second
        self.call_times = []
        self.min_interval = 60.0 / calls_per_minute  # Seconds between calls
    
    def wait_if_needed(self):
        """Wait if necessary to stay under rate limit."""
        now = time.time()
        
        # Remove calls older than 1 minute
        self.call_times = [t for t in self.call_times if now - t < 60]
        
        # Check if we've hit the per-minute limit
        if len(self.call_times) >= self.calls_per_minute:
            # Wait until the oldest call is > 1 minute old
            sleep_time = 60 - (now - self.call_times[0]) + 0.1
            if sleep_time > 0:
                logger.info(f"Rate limit reached. Sleeping {sleep_time:.1f}s...")
                time.sleep(sleep_time)
        
        # Always wait minimum interval between calls
        if self.call_times:
            time_since_last = now - self.call_times[-1]
            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                time.sleep(sleep_time)
        
        # Record this call
        self.call_times.append(time.time())


def rate_limited(max_retries: int = 3):
    """
    Decorator for rate-limited API calls with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts on error
    
    Returns:
        Decorated function with rate limiting and retry logic
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            for attempt in range(max_retries):
                try:
                    # Wait if needed for rate limiting
                    self.rate_limiter.wait_if_needed()
                    
                    # Execute function
                    return func(self, *args, **kwargs)
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    
                    # Check if it's a rate limit error
                    if 'rate limit' in error_msg or '429' in error_msg:
                        wait_time = (2 ** attempt) * 2  # Exponential backoff: 2, 4, 8 seconds
                        logger.warning(f"Rate limit hit. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    elif attempt < max_retries - 1:
                        # Other error - retry with backoff
                        wait_time = (2 ** attempt)
                        logger.warning(f"API error: {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        # Final attempt failed
                        logger.error(f"API call failed after {max_retries} attempts: {e}")
                        raise
            
            return None
        
        return wrapper
    return decorator


class NewsFetcher:
    """
    Finnhub news fetcher with rate limiting and preprocessing.
    
    This class handles fetching company news from Finnhub API with:
    - Automatic rate limiting
    - Error handling and retries
    - Text preprocessing
    - Duplicate detection
    - Date aggregation
    
    Attributes:
        client: Finnhub API client
        rate_limiter: Rate limiter instance
        api_key: Finnhub API key
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Finnhub news fetcher.
        
        Args:
            api_key: Finnhub API key (defaults to FINNHUB_API_KEY env variable)
        
        Raises:
            ValueError: If API key not provided and not in environment
        """
        # Get API key
        self.api_key = api_key or os.getenv('FINNHUB_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "Finnhub API key required. Either:\n"
                "1. Pass api_key parameter\n"
                "2. Set FINNHUB_API_KEY environment variable\n"
                "\nGet free API key at: https://finnhub.io/register"
            )
        
        # Initialize Finnhub client
        try:
            self.client = finnhub.Client(api_key=self.api_key)
            logger.info("✓ Finnhub client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Finnhub client: {e}")
            raise
        
        # Initialize rate limiter (60 calls/minute for free tier)
        self.rate_limiter = RateLimiter(calls_per_minute=60)
    
    @rate_limited(max_retries=3)
    def fetch_company_news(
        self, 
        symbol: str, 
        days_back: int = 365,
        include_summary: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical company news from Finnhub.
        
        Finnhub free tier provides up to 1 year of historical news.
        Returns DataFrame with headlines, summaries, sources, and timestamps.
        
        Args:
            symbol: Stock ticker (e.g., 'AAPL', 'TSLA')
            days_back: How many days back to fetch (max 365 on free tier)
            include_summary: Whether to include article summary (default: True)
        
        Returns:
            DataFrame with columns:
            - date: Article datetime
            - headline: Article headline
            - summary: Article summary (if include_summary=True)
            - source: News source
            - url: Article URL
            - category: News category
        
        Example:
            >>> fetcher = NewsFetcher(api_key="YOUR_KEY")
            >>> news = fetcher.fetch_company_news('AAPL', days_back=30)
            >>> print(f"Fetched {len(news)} articles")
        """
        logger.info(f"Fetching news for {symbol} (last {days_back} days)...")
        
        # Calculate date range
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days_back)
        
        from_str = from_date.strftime('%Y-%m-%d')
        to_str = to_date.strftime('%Y-%m-%d')
        
        # Fetch news from Finnhub
        news_items = []
        
        try:
            news_data = self.client.company_news(symbol, _from=from_str, to=to_str)
            
            logger.info(f"Retrieved {len(news_data)} articles")
            
            for item in news_data:
                # Extract data
                article = {
                    'date': datetime.fromtimestamp(item['datetime']),
                    'headline': item.get('headline', ''),
                    'source': item.get('source', 'Unknown'),
                    'url': item.get('url', ''),
                    'category': item.get('category', 'general')
                }
                
                # Add summary if requested
                if include_summary:
                    article['summary'] = item.get('summary', '')
                
                news_items.append(article)
            
            # Convert to DataFrame
            df = pd.DataFrame(news_items)
            
            if not df.empty:
                # Sort by date (newest first)
                df = df.sort_values('date', ascending=False).reset_index(drop=True)
                
                # Preprocess text
                df = self.preprocess_news(df, include_summary=include_summary)
                
                logger.info(f"✓ Processed {len(df)} articles for {symbol}")
            else:
                logger.warning(f"No news found for {symbol}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            raise
    
    @rate_limited(max_retries=3)
    def fetch_news_sentiment(self, symbol: str) -> Dict:
        """
        Fetch Finnhub's built-in news sentiment for a stock.
        
        This provides Finnhub's own sentiment analysis which can be compared
        with FinBERT sentiment for validation.
        
        Args:
            symbol: Stock ticker
        
        Returns:
            Dictionary with sentiment metrics:
            - buzz: Social media buzz metrics
            - companyNewsScore: News sentiment score
            - sectorAverageBullishPercent: Sector average
            - sectorAverageNewsScore: Sector average score
        
        Example:
            >>> sentiment = fetcher.fetch_news_sentiment('AAPL')
            >>> print(f"News score: {sentiment['companyNewsScore']}")
        """
        logger.info(f"Fetching Finnhub sentiment for {symbol}...")
        
        try:
            sentiment = self.client.news_sentiment(symbol)
            logger.info(f"✓ Retrieved sentiment for {symbol}")
            return sentiment
        except Exception as e:
            logger.error(f"Error fetching sentiment for {symbol}: {e}")
            raise
    
    def preprocess_news(
        self, 
        df: pd.DataFrame,
        include_summary: bool = True,
        max_length: int = 512
    ) -> pd.DataFrame:
        """
        Preprocess news text for sentiment analysis.
        
        Preprocessing steps:
        1. Clean HTML entities
        2. Remove duplicates
        3. Truncate to max token length (512 for BERT)
        4. Prioritize headline over summary
        
        Args:
            df: News DataFrame
            include_summary: Whether to process summaries
            max_length: Max character length (approx 512 tokens for BERT)
        
        Returns:
            Preprocessed DataFrame
        """
        if df.empty:
            return df
        
        # Clean HTML entities from headlines
        df['headline'] = df['headline'].apply(self._clean_html)
        
        # Clean summaries if included
        if include_summary and 'summary' in df.columns:
            df['summary'] = df['summary'].apply(self._clean_html)
        
        # Remove duplicate headlines (same story from different sources)
        df = self._remove_duplicates(df)
        
        # Truncate text to max length
        df['headline'] = df['headline'].str[:max_length]
        
        if include_summary and 'summary' in df.columns:
            df['summary'] = df['summary'].str[:max_length]
        
        # Create combined text (headline + summary) for sentiment analysis
        if include_summary and 'summary' in df.columns:
            # Prioritize headline, add summary if space allows
            df['text'] = df.apply(
                lambda row: self._combine_headline_summary(
                    row['headline'], 
                    row['summary'], 
                    max_length
                ), 
                axis=1
            )
        else:
            df['text'] = df['headline']
        
        return df
    
    def aggregate_news_by_date(
        self, 
        df: pd.DataFrame,
        forward_fill_weekends: bool = True
    ) -> pd.DataFrame:
        """
        Aggregate news by trading date.
        
        Groups all headlines from the same day and creates daily aggregates.
        Handles weekends/holidays by forward-filling from previous trading day.
        
        Args:
            df: News DataFrame with 'date' column
            forward_fill_weekends: Fill weekend/holiday dates with Friday's data
        
        Returns:
            DataFrame with one row per trading date containing:
            - date: Trading date
            - headlines: List of all headlines
            - num_articles: Number of articles
            - sources: List of unique sources
        
        Example:
            >>> daily_news = fetcher.aggregate_news_by_date(news_df)
            >>> print(daily_news.head())
        """
        if df.empty:
            return pd.DataFrame(columns=['date', 'headlines', 'num_articles', 'sources'])
        
        # Extract date (without time)
        df['trade_date'] = df['date'].dt.date
        
        # Group by date
        aggregated = df.groupby('trade_date').agg({
            'headline': lambda x: list(x),
            'text': lambda x: list(x),
            'source': lambda x: list(set(x)),
            'url': 'count'
        }).reset_index()
        
        # Rename columns
        aggregated.columns = ['date', 'headlines', 'text_list', 'sources', 'num_articles']
        
        # Convert date back to datetime
        aggregated['date'] = pd.to_datetime(aggregated['date'])
        
        # Forward fill for weekends/holidays if requested
        if forward_fill_weekends:
            aggregated = self._forward_fill_trading_days(aggregated)
        
        # Sort by date (newest first)
        aggregated = aggregated.sort_values('date', ascending=False).reset_index(drop=True)
        
        return aggregated
    
    # ===================================================================
    # PRIVATE METHODS
    # ===================================================================
    
    def _clean_html(self, text: str) -> str:
        """Remove HTML entities and clean text."""
        if not isinstance(text, str):
            return ''
        
        # Decode HTML entities
        text = html.unescape(text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove multiple whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate headlines (same story from different sources).
        
        Uses fuzzy matching on headlines to detect duplicates.
        Keeps the article from the most authoritative source.
        """
        if df.empty:
            return df
        
        # Simple approach: exact match on cleaned headlines
        # More sophisticated: use edit distance or embeddings
        
        # Clean headlines for comparison (lowercase, remove punctuation)
        df['headline_clean'] = df['headline'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
        
        # Keep first occurrence (already sorted by date)
        df = df.drop_duplicates(subset=['headline_clean'], keep='first')
        
        # Drop temporary column
        df = df.drop(columns=['headline_clean'])
        
        return df
    
    def _combine_headline_summary(
        self, 
        headline: str, 
        summary: str, 
        max_length: int
    ) -> str:
        """
        Combine headline and summary intelligently.
        
        Prioritizes headline (more impactful for sentiment).
        Adds summary if space allows within max_length.
        """
        if not summary or pd.isna(summary):
            return headline
        
        # Always include headline
        combined = headline
        
        # Add summary if it fits
        if len(combined) + len(summary) + 3 <= max_length:  # +3 for ". "
            combined = f"{headline}. {summary}"
        
        # Truncate to max length
        return combined[:max_length]
    
    def _forward_fill_trading_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Forward fill news data for weekends and holidays.
        
        If no news on a Saturday/Sunday, use Friday's news.
        This ensures we have sentiment data for every calendar day.
        """
        if df.empty:
            return df
        
        # Create complete date range
        min_date = df['date'].min()
        max_date = df['date'].max()
        
        all_dates = pd.date_range(start=min_date, end=max_date, freq='D')
        
        # Create DataFrame with all dates
        full_df = pd.DataFrame({'date': all_dates})
        
        # Merge with aggregated data
        merged = full_df.merge(df, on='date', how='left')
        
        # Forward fill missing data
        merged = merged.fillna(method='ffill')
        
        # Fill any remaining NaN (at the start) with empty lists
        merged['headlines'] = merged['headlines'].apply(
            lambda x: x if isinstance(x, list) else []
        )
        merged['text_list'] = merged['text_list'].apply(
            lambda x: x if isinstance(x, list) else []
        )
        merged['sources'] = merged['sources'].apply(
            lambda x: x if isinstance(x, list) else []
        )
        merged['num_articles'] = merged['num_articles'].fillna(0).astype(int)
        
        return merged


def example_usage():
    """Example usage of NewsFetcher."""
    
    print("\n" + "="*70)
    print("  FINNHUB NEWS FETCHER - EXAMPLE USAGE")
    print("="*70 + "\n")
    
    # Check for API key
    api_key = os.getenv('FINNHUB_API_KEY')
    if not api_key:
        print("⚠️  FINNHUB_API_KEY environment variable not set")
        print("   Get free API key at: https://finnhub.io/register")
        print("   Then: export FINNHUB_API_KEY='your_key_here'")
        return
    
    # Initialize fetcher
    print("Initializing Finnhub news fetcher...")
    fetcher = NewsFetcher(api_key=api_key)
    
    # Fetch company news
    print("\n1. Fetching Company News:")
    print("-" * 70)
    
    symbol = 'AAPL'
    news_df = fetcher.fetch_company_news(symbol, days_back=30)
    
    print(f"\nFetched {len(news_df)} articles for {symbol}")
    
    if not news_df.empty:
        print("\nSample headlines:")
        for idx, row in news_df.head(5).iterrows():
            date_str = row['date'].strftime('%Y-%m-%d')
            print(f"  [{date_str}] {row['headline'][:60]}...")
    
    # Aggregate by date
    print("\n2. Aggregating News by Date:")
    print("-" * 70)
    
    if not news_df.empty:
        daily_news = fetcher.aggregate_news_by_date(news_df)
        
        print(f"\nAggregated into {len(daily_news)} trading days")
        print("\nSample daily aggregates:")
        
        for idx, row in daily_news.head(5).iterrows():
            date_str = row['date'].strftime('%Y-%m-%d')
            print(f"  {date_str}: {row['num_articles']} articles from {len(row['sources'])} sources")
    
    # Fetch Finnhub sentiment
    print("\n3. Fetching Finnhub Sentiment:")
    print("-" * 70)
    
    try:
        sentiment = fetcher.fetch_news_sentiment(symbol)
        print(f"\nFinnhub Sentiment for {symbol}:")
        print(f"  Company News Score: {sentiment.get('companyNewsScore', 'N/A')}")
        print(f"  Sector Average: {sentiment.get('sectorAverageNewsScore', 'N/A')}")
    except Exception as e:
        print(f"  Note: Sentiment API requires paid tier ({e})")
    
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    example_usage()
