import logging
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)


class TFTDatasetBuilder:
    """Helper to prepare dataframes for PyTorch Forecasting TimeSeriesDataSet.

    Primary responsibilities:
    - Ensure required columns exist and raise clear errors otherwise.
    - Parse and sort timestamps.
    - Forward-fill target missing values and drop remaining NaNs.
    - Create a per-group sequential `time_idx` integer column.
    - Validate monotonic increasing time index per group.
    - Log dataset statistics for debugging.
    """

    @staticmethod
    def create_dataset(
        df: pd.DataFrame,
        date_col: str = "Date",
        group_col: str = "symbol",
        target_col: str = "Close",
        time_idx_col: str = "time_idx",
        start_time_index: int = 0,
        dropna_after_ffill: bool = True,
    ) -> Dict[str, object]:
        """Prepare dataframe and return the processed dataframe and stats.

        Args:
            df: Input pandas DataFrame containing at least the date, group and target columns.
            date_col: Name of the date column (will be converted to datetime).
            group_col: Column name to group by (e.g. symbol / ticker). Must exist.
            target_col: Name of the target column that may contain NaNs.
            time_idx_col: Name for the output integer time index column.
            start_time_index: Starting integer for time_idx (default 0).
            dropna_after_ffill: If True, drop rows that still have NaN in target after forward-fill.

        Returns:
            A dict with keys: `data` -> processed DataFrame, `stats` -> dict of dataset statistics.

        Raises:
            ValueError: If required columns are missing or dates within a group are not strictly increasing.
        """

        # Defensive copy
        if not isinstance(df, pd.DataFrame):
            raise ValueError("`df` must be a pandas DataFrame")

        missing = [c for c in (date_col, group_col, target_col) if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required column(s): {missing}. Please provide columns: `date_col`, `group_col`, and `target_col`."
            )

        df = df.copy()

        # Ensure datetime
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except Exception as e:
            raise ValueError(f"Failed to parse `{date_col}` as datetime: {e}") from e

        # Sort by group and date
        df = df.sort_values([group_col, date_col]).reset_index(drop=True)

        # Forward-fill target within each group
        df[target_col] = df.groupby(group_col)[target_col].apply(lambda s: s.ffill())

        # Optionally drop rows that still have NaN target after forward-fill
        if dropna_after_ffill:
            before = len(df)
            df = df.dropna(subset=[target_col]).reset_index(drop=True)
            after = len(df)
            logger.debug("Dropped %d rows with missing `%s` after forward-fill", before - after, target_col)

        # Validate dates are strictly increasing within each group
        bad_groups = []
        for g, group_df in df.groupby(group_col):
            dates = group_df[date_col]
            # Strictly increasing means each element greater than previous
            if not dates.is_monotonic_increasing or dates.duplicated().any():
                # More specific check: if any non-increasing or duplicated
                if any(dates.diff().dropna() <= pd.Timedelta(0)) or dates.duplicated().any():
                    bad_groups.append(g)

        if bad_groups:
            raise ValueError(
                f"Found non-strictly-increasing or duplicate {date_col} values for group(s): {bad_groups}. "
                "Ensure each group's timestamps are unique and strictly increasing (or aggregate / deduplicate first)."
            )

        # Create sequential time_idx per group
        df[time_idx_col] = df.groupby(group_col).cumcount() + int(start_time_index)

        # Validate time_idx is strictly increasing per group (no duplicates, increasing)
        bad_timeidx_groups = []
        for g, group_df in df.groupby(group_col):
            tidx = group_df[time_idx_col]
            if not tidx.is_monotonic_increasing or tidx.duplicated().any():
                bad_timeidx_groups.append(g)

        if bad_timeidx_groups:
            raise ValueError(
                f"Created `{time_idx_col}` but found non-monotonic or duplicated indices in groups: {bad_timeidx_groups}."
            )

        # Collect stats
        stats = {
            "date_min": df[date_col].min(),
            "date_max": df[date_col].max(),
            "sample_count": len(df),
            "group_count": df[group_col].nunique(),
            "feature_count": max(0, df.shape[1] - len({date_col, group_col, target_col, time_idx_col})),
        }

        logger.info(
            "TFT dataset prepared: rows=%d, groups=%d, date_range=%s to %s, features=%d",
            stats["sample_count"],
            stats["group_count"],
            stats["date_min"],
            stats["date_max"],
            stats["feature_count"],
        )

        # Add a bit more debug detail
        logger.debug("Top 5 groups by row count: %s", df[group_col].value_counts().head().to_dict())

        return {"data": df, "stats": stats}
"""
TFT dataset builder: convert pandas DataFrame(s) into
`pytorch_forecasting.TimeSeriesDataSet` objects for Temporal Fusion Transformer.

Prepares data for TFT training using 147 canonical features (93 technical + 20 new + 34 sentiment).

Features and behavior:
- Handles static features (sector, market_cap, beta)
- Handles time-varying known and unknown features
- Supports multi-symbol datasets (group_id = 'symbol')
- Provides train/validation dataset creation and dataloaders

Usage example in repository docs.
"""

from typing import List, Dict, Optional
import logging
import pandas as pd
import numpy as np
import torch
try:
    from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer
    from pytorch_forecasting.data import NaNLabelEncoder
except Exception:
    # Allow module to be imported in environments without pytorch_forecasting
    TimeSeriesDataSet = None
    GroupNormalizer = None
    NaNLabelEncoder = None

# canonical feature helpers
from data.feature_engineer import get_feature_columns, EXPECTED_FEATURE_COUNT

logger = logging.getLogger(__name__)

# Example metadata (can be extended or passed into prepare_dataframe)
# Symbol metadata is optional and not used for feature selection.
SYMBOL_METADATA = {}


class TFTDatasetBuilder:
    """Build TimeSeriesDataSet suitable for PyTorch Forecasting's TFT.

    Args:
        max_encoder_length: number of historical steps to provide to encoder
        max_prediction_length: forecast horizon
        target_col: name of the target column in prepared DataFrame
        quantiles: quantiles used downstream (kept for metadata)
    """
    def __init__(self, max_encoder_length: int = 60, max_prediction_length: int = 10,
                 target_col: str = 'returns', quantiles: List[float] = [0.1, 0.5, 0.9]):
        self.max_encoder_length = int(max_encoder_length)
        self.max_prediction_length = int(max_prediction_length)
        self.target_col = target_col
        self.quantiles = list(quantiles)

        # Definitions used when building TimeSeriesDataSet
        # Keep static_reals empty to avoid including metadata fields (e.g. market_cap)
        # that may be missing for many symbols and cause validation failures.
        self.static_categoricals = ['symbol', 'sector']
        self.static_reals = []

        self.time_varying_known_categoricals = ['day_of_week', 'month', 'quarter']
        # Use a filled vix level plus a missing-flag so TFT doesn't fail on NA-heavy series
        self.time_varying_known_reals = ['vix_level_filled', 'vix_level_missing', 'days_to_earnings']

        # time_varying_unknown_reals will be determined at dataset creation time
        self.time_varying_unknown_reals = None

    def prepare_dataframe(self, df: pd.DataFrame, symbol: str, metadata: Optional[Dict] = None) -> pd.DataFrame:
        """Prepare a single-symbol DataFrame for TFT ingestion.

        The input `df` should contain engineered features (123 features) and a datetime index
        or a `Date` column. This function will add `time_idx`, `symbol`, `date`, static
        metadata columns, and known time-varying features.
        """
        df = df.copy()

        # Ensure datetime index and a 'date' column
        if isinstance(df.index, pd.DatetimeIndex):
            df['date'] = pd.to_datetime(df.index).tz_localize(None)
            df = df.reset_index(drop=True)
        elif 'Date' in df.columns:
            df['date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        else:
            raise ValueError('DataFrame must have a DatetimeIndex or a Date column')

        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)

        # time_idx: consecutive integer per symbol starting at 0
        df['time_idx'] = np.arange(len(df), dtype=np.int32)

        # symbol column
        df['symbol'] = symbol

        # Static metadata: only sector is used as a categorical. Do not create
        # numeric metadata columns such as `market_cap` here (they may be missing)
        # because PyTorch Forecasting validates static_reals and will fail if
        # many NA/inf values are present.
        meta = metadata if metadata is not None else SYMBOL_METADATA.get(symbol, {})
        df['sector'] = meta.get('sector', 'Unknown')

        # Time-varying known features
        # Use string/categorical values for these to satisfy PyTorch Forecasting expectations
        # (it requires categorical variables to be string/categorical types)
        df['day_of_week'] = df['date'].dt.day_name().astype(str)
        df['month'] = df['date'].dt.month_name().astype(str)
        df['quarter'] = df['date'].dt.quarter.apply(lambda q: f'Q{q}').astype(str)

        # days_to_earnings: optional; if missing, fill with -1
        if 'days_to_earnings' not in df.columns:
            df['days_to_earnings'] = -1
        else:
            df['days_to_earnings'] = df['days_to_earnings'].fillna(-1).astype(int)

        # vix_level: prefer existing column, otherwise NaN (user can merge externally)
        if 'vix' in df.columns:
            df['vix_level'] = df['vix'].astype(float)
        else:
            df['vix_level'] = np.nan
        # create a filled version and an indicator for missing values
        # NOTE: No bfill() to prevent look-ahead bias - use ffill() + neutral value fallback
        df['vix_level_filled'] = df['vix_level'].ffill().fillna(20.0).astype(float)  # 20 is historical VIX mean
        df['vix_level_missing'] = df['vix_level'].isna().astype(int)
        # drop original vix_level to avoid it being interpreted as an unknown real
        # with many NAs by TimeSeriesDataSet validation
        if 'vix_level' in df.columns:
            df.drop(columns=['vix_level'], inplace=True)

        # Ensure target column exists (or map from 'returns')
        if self.target_col not in df.columns:
            if 'returns' in df.columns:
                df[self.target_col] = df['returns']
            else:
                raise ValueError(f"Target column '{self.target_col}' not found")

        # Drop rows where target is NA or infinite — TimeSeriesDataSet does not allow NA targets.
        # After dropping, reindex and recompute `time_idx` so that indices are consecutive per symbol.
        # This prevents the dataset constructor from failing due to NA/inf target values.
        # Explicitly handle infs by replacing with NaN first, then dropping
        if self.target_col in df.columns:
            df[self.target_col] = df[self.target_col].replace([np.inf, -np.inf], np.nan)
            
        is_finite_target = df[self.target_col].notna()
        if not is_finite_target.all():
            n_bad = (~is_finite_target).sum()
            print(f"Dropping {n_bad} rows with non-finite target '{self.target_col}'")
            df = df[is_finite_target].reset_index(drop=True)
            # recompute date ordering and time_idx
            if 'date' in df.columns:
                df = df.sort_values('date').reset_index(drop=True)
            df['time_idx'] = np.arange(len(df), dtype=np.int32)
            # If after dropping we have too few rows, raise an error
            if len(df) < (self.max_encoder_length + self.max_prediction_length):
                raise ValueError(f"Not enough data for symbol {symbol} after dropping non-finite targets: {len(df)} rows")

        # Verify canonical feature list is present and set as time_varying_unknown_reals
        canonical_features = get_feature_columns(include_sentiment=True)
        missing = set(canonical_features) - set(df.columns)
        if missing:
            raise ValueError(
                f"TFT requires all {len(canonical_features)} canonical features.\n"
                f"Missing {len(missing)} features: {missing}\n"
                f"Ensure engineer_features(include_sentiment=True) was called."
            )

        # Store time-varying features (use ALL canonical features)
        # Do NOT include metadata fields such as market_cap here; canonical_features
        # is authoritative for the model inputs and contains only the 147 features
        # produced by `engineer_features()`.
        self.time_varying_unknown_reals = canonical_features

        # Verify count
        assert len(self.time_varying_unknown_reals) == EXPECTED_FEATURE_COUNT, \
            f"TFT feature count mismatch: {len(self.time_varying_unknown_reals)} != {EXPECTED_FEATURE_COUNT}"

        print(f"✓ TFT prepared with {EXPECTED_FEATURE_COUNT} features for {symbol}")

        return df

    def _infer_unknown_reals(self, combined: pd.DataFrame):
        """Infer time-varying unknown real feature names from combined DataFrame."""
        reserved = set(self.static_categoricals + self.static_reals +
                       self.time_varying_known_categoricals + self.time_varying_known_reals +
                       ['time_idx', 'date', 'symbol', self.target_col])
        # All numeric columns not in reserved are treated as unknown reals
        candidate = [c for c in combined.columns if c not in reserved]
        # filter numeric dtypes
        unknown_reals = [c for c in candidate if pd.api.types.is_numeric_dtype(combined[c])]
        return unknown_reals

    def create_dataset(self, df_list: List[pd.DataFrame], split: str = 'train',
                       reference_dataset: Optional[TimeSeriesDataSet] = None) -> TimeSeriesDataSet:
        """Create a TimeSeriesDataSet from prepared DataFrames.

        If `split == 'train'` this creates and returns a training TimeSeriesDataSet.
        If `split == 'val'`, a validation dataset is created via
        `TimeSeriesDataSet.from_dataset(reference_dataset, combined_df, predict=True)`
        and `reference_dataset` must not be None.
        """
        # Concatenate all symbol DataFrames
        if isinstance(df_list, list):
            combined = pd.concat(df_list, ignore_index=True)
        else:
            combined = df_list

        # Verify time_varying_unknown_reals is set and correct
        if self.time_varying_unknown_reals is None:
            raise ValueError("time_varying_unknown_reals not set. Call prepare_dataframe() first.")

        if len(self.time_varying_unknown_reals) != EXPECTED_FEATURE_COUNT:
            raise AssertionError(
                f"TFT requires {EXPECTED_FEATURE_COUNT} features, "
                f"got {len(self.time_varying_unknown_reals)}"
            )

        # Ensure combined DataFrame only contains the columns TFT will use.
        # This prevents optional metadata fields (e.g. 'market_cap') from being
        # interpreted as model inputs and causing NA/infinite-value validation errors.
        required_cols = ['time_idx', 'symbol', self.target_col] + \
                        self.time_varying_unknown_reals + self.time_varying_known_categoricals + self.time_varying_known_reals + self.static_categoricals + self.static_reals
        # Keep only intersection preserving order and remove duplicates (keep first occurrence)
        seen = set()
        required_unique = []
        for c in required_cols:
            if c in combined.columns and c not in seen:
                required_unique.append(c)
                seen.add(c)

        # Preserve any original Date/Date columns if present so we can build time_idx reliably
        preserve_date_cols = []
        if 'Date' in combined.columns:
            preserve_date_cols.append('Date')
        if 'date' in combined.columns:
            # prefer lowercase 'date' if both present
            if 'date' not in preserve_date_cols:
                preserve_date_cols.append('date')

        # Ensure preserved date columns are kept in the required set and appear first
        required_unique = preserve_date_cols + [c for c in required_unique if c not in preserve_date_cols]

        extra = [c for c in combined.columns if c not in required_unique]
        if extra:
            print(f"Dropping {len(extra)} extra columns from TFT dataset: {extra[:10]}...")
            combined = combined[required_unique]

        # Ensure there is a proper `time_idx` column. We try to preserve any
        # provided time information (Date/date columns or DatetimeIndex). If
        # missing, create a consecutive integer index per symbol (or overall
        # if no symbol column). We also ensure monotonic increasing order per
        # group by sorting by date/index when necessary.
        if 'time_idx' not in combined.columns:
            # If a Date/date column exists, use it to sort and then assign cumcount
            if 'symbol' in combined.columns:
                if 'date' in combined.columns or 'Date' in combined.columns:
                    dt_col = 'date' if 'date' in combined.columns else 'Date'
                    # sort by symbol then date to ensure ordering
                    combined = combined.sort_values(['symbol', dt_col]).reset_index(drop=True)
                    combined['time_idx'] = combined.groupby('symbol').cumcount()
                else:
                    # No date available: rely on current row order but ensure integer idx per group
                    combined['time_idx'] = combined.groupby('symbol').cumcount()
            else:
                # Single-group dataset: try to infer ordering from a DatetimeIndex if present
                if isinstance(combined.index, pd.DatetimeIndex):
                    combined = combined.reset_index(drop=False)
                    # keep original index as 'Date' if not present
                    if 'date' not in combined.columns and 'Date' not in combined.columns:
                        combined.rename(columns={'index': 'date'}, inplace=True)
                    combined = combined.sort_values('date').reset_index(drop=True)
                    combined['time_idx'] = np.arange(len(combined), dtype=np.int32)
                else:
                    # Fallback: simple arange
                    combined['time_idx'] = np.arange(len(combined), dtype=np.int32)
        else:
            # coerce to integer index if present
            combined['time_idx'] = combined['time_idx'].astype(int)

        # Verify monotonic increasing time_idx per group and fix by sorting when possible
        bad_timeidx_groups = []
        if 'symbol' in combined.columns:
            for g, group_df in combined.groupby('symbol'):
                tidx = group_df['time_idx']
                if not tidx.is_monotonic_increasing or tidx.duplicated().any():
                    bad_timeidx_groups.append(g)
                    # Attempt to sort by date if available to fix ordering
                    if 'date' in group_df.columns or 'Date' in group_df.columns:
                        dt_col = 'date' if 'date' in group_df.columns else 'Date'
                        combined.loc[combined['symbol'] == g] = group_df.sort_values(dt_col).values
                    else:
                        # As a last resort, reassign sequential time_idx for this group
                        idxs = group_df.index
                        combined.loc[idxs, 'time_idx'] = np.arange(len(idxs), dtype=np.int32)
        else:
            tidx = combined['time_idx']
            if (not tidx.is_monotonic_increasing) or tidx.duplicated().any():
                # Try to sort by 'date' if available
                if 'date' in combined.columns or 'Date' in combined.columns:
                    dt_col = 'date' if 'date' in combined.columns else 'Date'
                    combined = combined.sort_values(dt_col).reset_index(drop=True)
                    combined['time_idx'] = np.arange(len(combined), dtype=np.int32)
                else:
                    raise ValueError("Found non-monotonic or duplicated 'time_idx' in single-group dataset")

        # Re-check for any groups still bad and raise if we couldn't fix them
        remaining_bad = []
        if 'symbol' in combined.columns:
            for g, group_df in combined.groupby('symbol'):
                tidx = group_df['time_idx']
                if not tidx.is_monotonic_increasing or tidx.duplicated().any():
                    remaining_bad.append(g)
        else:
            tidx = combined['time_idx']
            if (not tidx.is_monotonic_increasing) or tidx.duplicated().any():
                remaining_bad = ['<single_group>']

        if remaining_bad:
            raise ValueError(
                f"Created/validated 'time_idx' but found non-monotonic or duplicated indices in groups: {remaining_bad}."
            )

        # Log time_idx range for debugging
        try:
            logger.info("time_idx range: [%s, %s]", combined['time_idx'].min(), combined['time_idx'].max())
        except Exception:
            # Fallback to print if logger misconfigured in some environments
            print(f"time_idx range: [{combined['time_idx'].min()}, {combined['time_idx'].max()}]")

        # Ensure 'time_idx' is considered a time-varying known real for TFT
        if 'time_idx' not in self.time_varying_known_reals:
            self.time_varying_known_reals = ['time_idx'] + list(self.time_varying_known_reals)

        # SAFETY NET: Fill any remaining NaNs in features with 0.0 to prevent validation failure
        # This is critical because TimeSeriesDataSet checks for NaNs in all continuous variables
        # and will crash if any are found, even if they are just a few.
        # We warn if we find any.
        if not combined.empty:
            # Check for NaNs in reals
            reals_to_check = self.time_varying_unknown_reals + self.time_varying_known_reals + self.static_reals
            # Filter to only those present
            reals_to_check = [c for c in reals_to_check if c in combined.columns]
            
            if reals_to_check:
                # Fast check
                if combined[reals_to_check].isna().any().any():
                    nan_counts = combined[reals_to_check].isna().sum()
                    bad_cols = nan_counts[nan_counts > 0]
                    print(f"⚠️  Warning: Found NaNs in TFT dataset features (filling with 0):")
                    for c, count in bad_cols.items():
                        print(f"   - {c}: {count} NaNs")
                    
                    # Fill with 0
                    combined[reals_to_check] = combined[reals_to_check].fillna(0.0)
                    
            # Also check target again just in case
            if self.target_col in combined.columns and combined[self.target_col].isna().any():
                 print(f"⚠️  Warning: Found NaNs in target '{self.target_col}' (filling with 0)")
                 combined[self.target_col] = combined[self.target_col].fillna(0.0)

        # Create dataset parameters (include known/unknown/ static definitions expected by PyTorch Forecasting)
        dataset_params = {
            'data': combined,
            'time_idx': 'time_idx',
            # Use a cleaned numeric target column to avoid hidden dtype/encoding issues
            # created elsewhere in the pipeline (duplicate names, object dtypes, masked arrays).
            # We create `__tft_target__` below and point the dataset at it.
            'target': '__tft_target__',
            'group_ids': ['symbol'],
            'min_encoder_length': 1,
            'max_encoder_length': self.max_encoder_length,
            'min_prediction_length': 1,
            'max_prediction_length': self.max_prediction_length,
            'static_categoricals': self.static_categoricals,
            'static_reals': self.static_reals,
            'time_varying_known_categoricals': self.time_varying_known_categoricals,
            'time_varying_known_reals': self.time_varying_known_reals,
            'time_varying_unknown_reals': self.time_varying_unknown_reals,
            'add_relative_time_idx': True,
            'add_target_scales': True,
            'add_encoder_length': True,
            'target_normalizer': GroupNormalizer(groups=['symbol'], transformation='softplus'),
            # Allow missing timesteps
            'allow_missing_timesteps': True,
        }

        # Defensive: coerce target to numeric, replace inf, and handle NaNs
        if self.target_col in combined.columns:
            # Diagnostics: report current non-finite count
            initial_nonfinite = int(~np.isfinite(pd.to_numeric(combined[self.target_col], errors='coerce')).sum())
            print(f"Debug: initial non-finite '{self.target_col}' (pre-coerce): {initial_nonfinite} / {len(combined)}")

            # Coerce to numeric in case of unexpected dtypes (assignment avoids chained-assignment warnings)
            combined[self.target_col] = pd.to_numeric(combined[self.target_col], errors='coerce')

            # Replace infinities with NaN using assignment (avoid inplace on a slice)
            combined[self.target_col] = combined[self.target_col].replace([np.inf, -np.inf], np.nan)

            # Count non-finite targets after coercion
            nonfinite_count = int(combined[self.target_col].isna().sum())
            total = len(combined)
            pct = nonfinite_count / max(1, total)
            print(f"Debug: post-coerce non-finite '{self.target_col}': {nonfinite_count} / {total} ({pct:.2%})")

            # If only a tiny fraction are bad, drop them; otherwise fill with neutral 0.0
            if nonfinite_count > 0:
                if pct <= 0.05:
                    print(f"⚠️  Dropping {nonfinite_count} ({pct:.2%}) non-finite '{self.target_col}' rows before dataset creation.")
                    combined = combined.loc[combined[self.target_col].notna()].reset_index(drop=True)
                else:
                    print(f"⚠️  Filling {nonfinite_count} ({pct:.2%}) non-finite '{self.target_col}' values with 0.0 to allow dataset creation.")
                    combined[self.target_col] = combined[self.target_col].fillna(0.0)

            # Show final count
            final_nonfinite = int(combined[self.target_col].isna().sum())
            print(f"Debug: final non-finite '{self.target_col}' after handling: {final_nonfinite} / {len(combined)}")

        # Create an explicit cleaned numeric target column used by TFT to avoid
        # any issues caused by duplicate column names or non-numeric dtypes.
        # This column is guaranteed to be finite (filled with 0.0 if necessary).
        try:
            combined['__tft_target__'] = pd.to_numeric(combined[self.target_col], errors='coerce')
            combined['__tft_target__'] = combined['__tft_target__'].replace([np.inf, -np.inf], np.nan)
            # If any non-finite remain, fill them with 0.0 (safe default for training)
            nf = int(combined['__tft_target__'].isna().sum())
            if nf > 0:
                print(f"⚠️  Filling {nf} remaining non-finite values in cleaned target with 0.0")
                combined['__tft_target__'] = combined['__tft_target__'].fillna(0.0)
        except Exception as e:
            raise RuntimeError(f"Could not construct cleaned TFT target column: {e}")

        # Drop groups (symbols) that don't have enough rows to form at least one sample
        min_rows_required = self.max_encoder_length + self.max_prediction_length
        if 'symbol' in combined.columns:
            group_counts = combined.groupby('symbol').size()
            bad_groups = group_counts[group_counts < min_rows_required].index.tolist()
            if bad_groups:
                print(f"⚠️  Dropping symbols with insufficient rows (< {min_rows_required}): {bad_groups}")
                combined = combined[~combined['symbol'].isin(bad_groups)].reset_index(drop=True)

        if combined.empty:
            raise ValueError('No data remaining for TimeSeriesDataSet after dropping non-finite targets or small groups')

        # Create dataset
        if reference_dataset is None:
            dataset = TimeSeriesDataSet(**dataset_params)
            print(f"✓ Created TFT dataset: {len(dataset)} samples, {EXPECTED_FEATURE_COUNT} features")
        else:
            dataset = TimeSeriesDataSet.from_dataset(reference_dataset, combined)

        return dataset

    def create_dataloaders(self, train_dataset: TimeSeriesDataSet, val_dataset: TimeSeriesDataSet,
                           batch_size: int = 32, num_workers: int = 0):
        """Return train and validation dataloaders using the dataset helpers."""
        train_dataloader = train_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=num_workers)
        val_dataloader = val_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=num_workers)
        return train_dataloader, val_dataloader


__all__ = ['TFTDatasetBuilder', 'SYMBOL_METADATA']
