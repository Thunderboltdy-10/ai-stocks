"""
Forward Simulator: True Forward-Looking Simulation

Nuclear Redesign: Step-by-step simulation with NO look-ahead bias.

This simulator strictly enforces temporal causality:
- At each time step T, ONLY data up to T-1 is available
- Features are engineered from historical data only
- Predictions are made for T, executed at T's open
- Results recorded at T's close

This is the gold standard for validating trading strategies before
deployment to production.

Author: AI-Stocks Nuclear Redesign
Date: December 2025
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SimulationStep:
    """Single step in forward simulation.

    Attributes:
        date: Date of this step
        prediction: Model prediction for this day's return
        position: Position taken (based on prediction)
        entry_price: Price at entry (open)
        exit_price: Price at exit (close)
        actual_return: Actual return realized
        pnl: Profit/loss for this step
        cumulative_pnl: Cumulative P&L up to this step
        features_date: Date of latest features used (should be T-1)
    """
    date: datetime
    prediction: float
    position: float
    entry_price: float
    exit_price: float
    actual_return: float
    pnl: float
    cumulative_pnl: float
    features_date: datetime

    def has_look_ahead(self) -> bool:
        """Check if this step has look-ahead bias."""
        return self.features_date >= self.date


@dataclass
class ForwardSimulationResults:
    """Results from forward simulation.

    Attributes:
        steps: List of simulation steps
        total_return: Total cumulative return
        sharpe_ratio: Annualized Sharpe ratio
        max_drawdown: Maximum drawdown
        win_rate: Percentage of winning trades
        profit_factor: Gross profit / gross loss
        direction_accuracy: Percentage of correct direction predictions
        has_look_ahead_bias: Whether any step had look-ahead bias
    """
    steps: List[SimulationStep]
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    direction_accuracy: float
    has_look_ahead_bias: bool

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "=" * 60,
            "FORWARD SIMULATION RESULTS",
            "=" * 60,
            f"Total Steps: {len(self.steps)}",
            f"Total Return: {self.total_return:.2%}",
            f"Sharpe Ratio: {self.sharpe_ratio:.2f}",
            f"Max Drawdown: {self.max_drawdown:.2%}",
            f"Win Rate: {self.win_rate:.2%}",
            f"Profit Factor: {self.profit_factor:.2f}",
            f"Direction Accuracy: {self.direction_accuracy:.2%}",
            "",
        ]

        if self.has_look_ahead_bias:
            lines.append("WARNING: Look-ahead bias detected in simulation!")
        else:
            lines.append("No look-ahead bias detected.")

        lines.append("=" * 60)

        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert steps to DataFrame."""
        return pd.DataFrame([
            {
                'date': s.date,
                'prediction': s.prediction,
                'position': s.position,
                'entry_price': s.entry_price,
                'exit_price': s.exit_price,
                'actual_return': s.actual_return,
                'pnl': s.pnl,
                'cumulative_pnl': s.cumulative_pnl,
                'features_date': s.features_date,
            }
            for s in self.steps
        ])


class ForwardSimulator:
    """
    True forward-looking simulation with NO look-ahead bias.

    At each time step T:
    1. Access ONLY data up to T-1 (yesterday)
    2. Engineer features from historical data
    3. Make prediction for T (today)
    4. Execute at T's open
    5. Record result at T's close

    Example:
        >>> simulator = ForwardSimulator(predictor, feature_engineer)
        >>> results = simulator.simulate(data, start_date='2024-01-01')
        >>> print(results.summary())
    """

    def __init__(
        self,
        predictor: Any,
        feature_engineer: Callable[[pd.DataFrame], pd.DataFrame],
        sequence_length: int = 60,
        position_sizer: Optional[Callable[[float], float]] = None,
    ):
        """
        Initialize forward simulator.

        Args:
            predictor: Model or predictor with .predict() method
            feature_engineer: Function to engineer features from OHLCV data
            sequence_length: Sequence length for model input
            position_sizer: Optional function to convert prediction to position
        """
        self.predictor = predictor
        self.feature_engineer = feature_engineer
        self.sequence_length = sequence_length
        self.position_sizer = position_sizer or self._default_position_sizer

    def _default_position_sizer(self, prediction: float) -> float:
        """Default position sizing based on prediction magnitude."""
        # +1% prediction -> 50% long, +2% -> 100% long (capped)
        # -1% prediction -> 25% short, -2% -> 50% short (capped)
        if prediction > 0:
            position = min(prediction * 50, 1.0)
        else:
            position = max(prediction * 25, -0.5)
        return position

    def simulate(
        self,
        data: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        initial_capital: float = 10000.0,
        verbose: bool = True,
    ) -> ForwardSimulationResults:
        """
        Run forward simulation.

        Args:
            data: OHLCV DataFrame with DatetimeIndex
            start_date: Simulation start date (default: after warmup period)
            end_date: Simulation end date (default: last available date)
            initial_capital: Starting capital
            verbose: Print progress

        Returns:
            ForwardSimulationResults with all metrics
        """
        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'Date' in data.columns:
                data = data.set_index('Date')
            data.index = pd.to_datetime(data.index)

        # Determine simulation range
        all_dates = sorted(data.index.unique())

        # Need warmup period for feature engineering
        warmup_end = min(len(all_dates) - 1, self.sequence_length + 100)
        sim_start_idx = warmup_end

        if start_date is not None:
            start_dt = pd.to_datetime(start_date)
            for i, d in enumerate(all_dates):
                if d >= start_dt:
                    sim_start_idx = max(i, sim_start_idx)
                    break

        if end_date is not None:
            end_dt = pd.to_datetime(end_date)
            sim_end_idx = len(all_dates) - 1
            for i, d in enumerate(all_dates):
                if d > end_dt:
                    sim_end_idx = i - 1
                    break
        else:
            sim_end_idx = len(all_dates) - 1

        sim_dates = all_dates[sim_start_idx:sim_end_idx + 1]

        if verbose:
            logger.info(f"Forward simulation: {sim_dates[0]} to {sim_dates[-1]}")
            logger.info(f"Total steps: {len(sim_dates)}")

        # Run simulation
        steps = []
        cumulative_pnl = 0.0
        has_look_ahead = False

        for i, T in enumerate(sim_dates):
            if verbose and i % 50 == 0:
                logger.info(f"Step {i}/{len(sim_dates)}: {T.date()}")

            # Get index of T in original data
            t_idx = all_dates.index(T)

            # CRITICAL: Use data ONLY up to T-1 (yesterday)
            historical_data = data.loc[all_dates[:t_idx]]  # Up to but NOT including T
            features_date = all_dates[t_idx - 1] if t_idx > 0 else all_dates[0]

            # Check for look-ahead bias
            if features_date >= T:
                has_look_ahead = True
                logger.warning(f"Look-ahead detected at {T}: features_date={features_date}")

            # Engineer features from historical data
            try:
                features = self.feature_engineer(historical_data)

                # Get last sequence_length rows
                if len(features) < self.sequence_length:
                    logger.warning(f"Insufficient data at {T}, skipping")
                    continue

                features_seq = features.iloc[-self.sequence_length:].values

                # Make prediction
                prediction = self._get_prediction(features_seq)

            except Exception as e:
                logger.warning(f"Feature engineering failed at {T}: {e}")
                continue

            # Calculate position
            position = self.position_sizer(prediction)

            # Get prices for T
            try:
                day_data = data.loc[T]
                if isinstance(day_data, pd.DataFrame):
                    day_data = day_data.iloc[0]

                entry_price = day_data.get('Open', day_data.get('open', 0))
                exit_price = day_data.get('Close', day_data.get('close', entry_price))
                actual_return = (exit_price - entry_price) / entry_price if entry_price > 0 else 0

            except Exception as e:
                logger.warning(f"Price data missing at {T}: {e}")
                continue

            # Calculate P&L
            pnl = position * actual_return
            cumulative_pnl += pnl

            steps.append(SimulationStep(
                date=T,
                prediction=prediction,
                position=position,
                entry_price=entry_price,
                exit_price=exit_price,
                actual_return=actual_return,
                pnl=pnl,
                cumulative_pnl=cumulative_pnl,
                features_date=features_date,
            ))

        # Calculate metrics
        results = self._calculate_metrics(steps, has_look_ahead)

        if verbose:
            logger.info(f"\n{results.summary()}")

        return results

    def _get_prediction(self, features_seq: np.ndarray) -> float:
        """Get prediction from model."""
        # Reshape for batch dimension if needed
        if len(features_seq.shape) == 2:
            features_seq = features_seq.reshape(1, *features_seq.shape)

        # Get prediction
        pred = self.predictor.predict(features_seq)

        # Handle different output shapes
        if hasattr(pred, 'numpy'):
            pred = pred.numpy()

        pred = np.asarray(pred).flatten()

        return float(pred[-1]) if len(pred) > 0 else 0.0

    def _calculate_metrics(
        self,
        steps: List[SimulationStep],
        has_look_ahead: bool,
    ) -> ForwardSimulationResults:
        """Calculate simulation metrics."""
        if not steps:
            return ForwardSimulationResults(
                steps=[],
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                direction_accuracy=0.0,
                has_look_ahead_bias=has_look_ahead,
            )

        # Extract arrays
        pnls = np.array([s.pnl for s in steps])
        predictions = np.array([s.prediction for s in steps])
        actuals = np.array([s.actual_return for s in steps])
        cumulative_pnls = np.array([s.cumulative_pnl for s in steps])

        # Total return
        total_return = steps[-1].cumulative_pnl if steps else 0.0

        # Sharpe ratio (annualized)
        if len(pnls) > 1 and np.std(pnls) > 0:
            sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Max drawdown
        peak = np.maximum.accumulate(cumulative_pnls + 1)  # +1 to start from capital
        drawdown = (peak - (cumulative_pnls + 1)) / peak
        max_drawdown = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

        # Win rate
        wins = np.sum(pnls > 0)
        total_trades = np.sum(np.abs(pnls) > 1e-6)
        win_rate = wins / total_trades if total_trades > 0 else 0.0

        # Profit factor
        gross_profit = np.sum(pnls[pnls > 0])
        gross_loss = np.abs(np.sum(pnls[pnls < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        # Direction accuracy
        correct_direction = np.sign(predictions) == np.sign(actuals)
        direction_accuracy = np.mean(correct_direction)

        return ForwardSimulationResults(
            steps=steps,
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            direction_accuracy=direction_accuracy,
            has_look_ahead_bias=has_look_ahead,
        )


def run_forward_simulation(
    symbol: str,
    predictor: Any,
    start_date: str,
    end_date: Optional[str] = None,
    data: Optional[pd.DataFrame] = None,
) -> ForwardSimulationResults:
    """
    Convenience function to run forward simulation.

    Args:
        symbol: Stock symbol
        predictor: Model or predictor
        start_date: Simulation start date
        end_date: Optional end date
        data: Optional pre-loaded data

    Returns:
        ForwardSimulationResults
    """
    from data.data_fetcher import fetch_stock_data
    from data.feature_engineer import engineer_features

    # Load data if not provided
    if data is None:
        data = fetch_stock_data(symbol, period='max')

    # Create simulator
    simulator = ForwardSimulator(
        predictor=predictor,
        feature_engineer=engineer_features,
    )

    # Run simulation
    return simulator.simulate(
        data=data,
        start_date=start_date,
        end_date=end_date,
    )
