"""
RL-Based Position Sizer using PPO (Proximal Policy Optimization)

This module implements a reinforcement learning-based position sizing system
that learns to optimize position sizes based on:
    - Model prediction confidence
    - Market volatility regime
    - Recent portfolio performance
    - Risk constraints

The RL agent replaces the traditional rule-based position sizing with a
learned policy that adapts to market conditions.

Key components:
1. TradingEnv - Gymnasium environment for position sizing
2. PPOPositionSizer - Wrapper class for inference
3. Training utilities

Reference papers:
    - Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
    - Moody & Saffell (2001) "Learning to Trade via Direct Reinforcement"
"""

from __future__ import annotations

import os
import pickle
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Stable-Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

logger = logging.getLogger(__name__)


# =============================================================================
# TRADING ENVIRONMENT
# =============================================================================

class PositionSizingEnv(gym.Env):
    """
    Gymnasium environment for RL-based position sizing.
    
    The agent observes:
        - Model predicted return (magnitude)
        - Model predicted direction (sign)
        - Model confidence (based on prediction variance)
        - Current volatility (rolling std of returns)
        - Volatility regime (low/normal/high encoded)
        - Recent PnL (rolling sum of recent returns)
        - Days since last trade
        - Current position (-1 to 1)
    
    The agent outputs:
        - Position size: continuous value in [-1, 1]
        - -1 = full short, 0 = flat, 1 = full long
    
    Reward is shaped to encourage:
        - Risk-adjusted returns (Sharpe-like)
        - Penalize large drawdowns
        - Encourage trading when signal is strong
    
    Args:
        predictions: Array of predicted returns [N,]
        actual_returns: Array of actual returns [N,]
        volatilities: Array of realized volatilities [N,]
        max_position: Maximum position size (default: 1.0)
        transaction_cost: Cost per trade as fraction (default: 0.001)
        risk_free_rate: Annualized risk-free rate (default: 0.04)
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        predictions: np.ndarray,
        actual_returns: np.ndarray,
        volatilities: Optional[np.ndarray] = None,
        max_position: float = 1.0,
        transaction_cost: float = 0.001,
        risk_free_rate: float = 0.04,
        lookback_pnl: int = 20,
        **kwargs
    ):
        super().__init__()
        
        self.predictions = np.asarray(predictions, dtype=np.float32)
        self.actual_returns = np.asarray(actual_returns, dtype=np.float32)
        
        # Calculate volatilities if not provided
        if volatilities is None:
            self.volatilities = self._compute_rolling_vol(self.actual_returns)
        else:
            self.volatilities = np.asarray(volatilities, dtype=np.float32)
        
        self.max_position = max_position
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate
        self.lookback_pnl = lookback_pnl
        
        self.n_steps = len(self.predictions)
        
        # Observation space: 8 features
        # [pred_return, pred_sign, confidence, volatility, vol_regime, recent_pnl, days_since_trade, current_position]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(8,),
            dtype=np.float32
        )
        
        # Action space: continuous position size
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()
    
    def _compute_rolling_vol(self, returns: np.ndarray, window: int = 20) -> np.ndarray:
        """Compute rolling volatility."""
        vol = np.zeros_like(returns)
        for i in range(window, len(returns)):
            vol[i] = np.std(returns[i-window:i])
        vol[:window] = vol[window] if len(vol) > window else 0.01
        return np.clip(vol, 0.001, 0.1)
    
    def _get_vol_regime(self, vol: float) -> float:
        """Encode volatility regime: -1 (low), 0 (normal), 1 (high)."""
        if vol < 0.01:
            return -1.0  # Low vol
        elif vol > 0.03:
            return 1.0   # High vol
        return 0.0       # Normal
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = self.lookback_pnl  # Start after lookback period
        self.current_position = 0.0
        self.days_since_trade = 0
        self.equity_curve = [1.0]
        self.positions_history = []
        self.returns_history = []
        self.cumulative_pnl = 0.0
        
        obs = self._get_observation()
        return obs, {}
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation vector."""
        t = self.current_step
        
        # Model prediction
        pred_return = self.predictions[t]
        pred_sign = np.sign(pred_return)
        
        # Confidence based on prediction magnitude (higher = more confident)
        confidence = np.clip(np.abs(pred_return) / 0.03, 0.0, 1.0)
        
        # Current volatility
        volatility = self.volatilities[t]
        vol_regime = self._get_vol_regime(volatility)
        
        # Recent PnL
        recent_returns = self.returns_history[-self.lookback_pnl:] if len(self.returns_history) >= self.lookback_pnl else self.returns_history
        recent_pnl = np.sum(recent_returns) if recent_returns else 0.0
        
        obs = np.array([
            pred_return * 100,  # Scale for numerical stability
            pred_sign,
            confidence,
            volatility * 100,
            vol_regime,
            recent_pnl * 100,
            min(self.days_since_trade / 10.0, 1.0),  # Normalized days since trade
            self.current_position,
        ], dtype=np.float32)
        
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Position size in [-1, 1]
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Extract position from action
        new_position = float(np.clip(action[0], -1.0, 1.0)) * self.max_position
        
        # Calculate position change and transaction cost
        position_change = abs(new_position - self.current_position)
        transaction_cost = position_change * self.transaction_cost
        
        # Get actual return for this step
        actual_return = self.actual_returns[self.current_step]
        
        # Calculate PnL from position
        position_pnl = self.current_position * actual_return - transaction_cost
        
        # Update state
        self.returns_history.append(position_pnl)
        self.positions_history.append(new_position)
        self.cumulative_pnl += position_pnl
        
        # Update equity curve
        new_equity = self.equity_curve[-1] * (1 + position_pnl)
        self.equity_curve.append(new_equity)
        
        # Track days since trade
        if position_change > 0.1:
            self.days_since_trade = 0
        else:
            self.days_since_trade += 1
        
        self.current_position = new_position
        self.current_step += 1
        
        # Calculate reward
        reward = self._calculate_reward(position_pnl, actual_return, new_position)
        
        # Check termination
        terminated = self.current_step >= self.n_steps - 1
        truncated = False
        
        # Get next observation
        if not terminated:
            obs = self._get_observation()
        else:
            obs = np.zeros(8, dtype=np.float32)
        
        info = {
            'pnl': position_pnl,
            'position': new_position,
            'cumulative_pnl': self.cumulative_pnl,
            'equity': new_equity,
        }
        
        return obs, reward, terminated, truncated, info
    
    def _calculate_reward(
        self,
        pnl: float,
        actual_return: float,
        position: float
    ) -> float:
        """
        Calculate shaped reward.
        
        Reward components:
        1. PnL component - primary driver
        2. Direction bonus - extra reward for correct direction
        3. Drawdown penalty - penalize large drawdowns
        4. Risk adjustment - prefer stable returns
        """
        # Base PnL reward (scaled)
        reward = pnl * 100
        
        # Direction bonus: reward correct direction bets
        if actual_return != 0 and position != 0:
            direction_correct = np.sign(actual_return) == np.sign(position)
            if direction_correct:
                reward += 0.1 * abs(position)  # Bonus for being right
        
        # Drawdown penalty
        if len(self.equity_curve) > 1:
            peak = max(self.equity_curve)
            current = self.equity_curve[-1]
            drawdown = (peak - current) / peak
            if drawdown > 0.1:  # 10% drawdown threshold
                reward -= drawdown * 0.5
        
        # Penalize excessive trading (churning)
        if len(self.positions_history) >= 2:
            n_recent = min(5, len(self.positions_history) - 1)
            recent_changes = sum(
                abs(self.positions_history[-i] - self.positions_history[-i-1])
                for i in range(1, n_recent + 1)
            )
            if recent_changes > 2.0:  # Too much position changing
                reward -= 0.05
        
        return reward
    
    def render(self, mode='human'):
        """Render current state."""
        if mode == 'human':
            print(f"Step {self.current_step}: "
                  f"Position={self.current_position:.2f}, "
                  f"PnL={self.cumulative_pnl:.4f}, "
                  f"Equity={self.equity_curve[-1]:.4f}")


# =============================================================================
# PPO POSITION SIZER
# =============================================================================

class PPOPositionSizer:
    """
    PPO-based position sizer for inference.
    
    This class wraps a trained PPO model for deployment in the
    trading pipeline. It provides methods for:
    - Loading trained models
    - Getting position sizes from observations
    - Training new models
    
    Args:
        model_path: Path to saved model (optional)
        device: Device for inference ('auto', 'cpu', 'cuda')
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'auto'
    ):
        self.model: Optional[PPO] = None
        self.vec_normalize: Optional[VecNormalize] = None
        self.device = device
        
        if model_path is not None:
            self.load(model_path)
    
    def load(self, model_path: str) -> None:
        """Load trained model from path."""
        model_path = Path(model_path)
        
        # Load PPO model
        self.model = PPO.load(str(model_path), device=self.device)
        logger.info(f"Loaded PPO model from {model_path}")
        
        # Load VecNormalize stats if available
        vec_normalize_path = model_path.with_suffix('.vecnormalize.pkl')
        if vec_normalize_path.exists():
            with open(vec_normalize_path, 'rb') as f:
                self.vec_normalize = pickle.load(f)
            logger.info(f"Loaded VecNormalize from {vec_normalize_path}")
    
    def save(self, model_path: str) -> None:
        """Save trained model to path."""
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.model is not None:
            self.model.save(str(model_path))
            logger.info(f"Saved PPO model to {model_path}")
        
        if self.vec_normalize is not None:
            vec_normalize_path = model_path.with_suffix('.vecnormalize.pkl')
            with open(vec_normalize_path, 'wb') as f:
                pickle.dump(self.vec_normalize, f)
    
    def get_position_size(
        self,
        pred_return: float,
        volatility: float,
        recent_pnl: float = 0.0,
        current_position: float = 0.0,
        days_since_trade: int = 0,
    ) -> float:
        """
        Get position size from trained model.
        
        Args:
            pred_return: Predicted return from forecasting model
            volatility: Current realized volatility
            recent_pnl: Recent cumulative PnL
            current_position: Current position size
            days_since_trade: Days since last significant trade
            
        Returns:
            Recommended position size in [-1, 1]
        """
        if self.model is None:
            # Fallback to rule-based if no model
            return self._fallback_position(pred_return, volatility)
        
        # Construct observation
        pred_sign = np.sign(pred_return)
        confidence = np.clip(np.abs(pred_return) / 0.03, 0.0, 1.0)
        vol_regime = self._get_vol_regime(volatility)
        
        obs = np.array([
            pred_return * 100,
            pred_sign,
            confidence,
            volatility * 100,
            vol_regime,
            recent_pnl * 100,
            min(days_since_trade / 10.0, 1.0),
            current_position,
        ], dtype=np.float32)
        
        # Normalize observation if vec_normalize available
        if self.vec_normalize is not None:
            obs = self.vec_normalize.normalize_obs(obs.reshape(1, -1)).flatten()
        
        # Get action from model (deterministic for inference)
        action, _ = self.model.predict(obs, deterministic=True)
        
        return float(np.clip(action[0], -1.0, 1.0))
    
    def _get_vol_regime(self, vol: float) -> float:
        """Encode volatility regime."""
        if vol < 0.01:
            return -1.0
        elif vol > 0.03:
            return 1.0
        return 0.0
    
    def _fallback_position(self, pred_return: float, volatility: float) -> float:
        """Rule-based fallback position sizing."""
        # Base position from prediction
        base_position = np.sign(pred_return) * np.clip(abs(pred_return) / 0.02, 0.0, 1.0)
        
        # Scale down in high volatility
        vol_scalar = np.clip(0.02 / max(volatility, 0.001), 0.5, 1.5)
        
        return base_position * vol_scalar
    
    def train(
        self,
        predictions: np.ndarray,
        actual_returns: np.ndarray,
        volatilities: Optional[np.ndarray] = None,
        total_timesteps: int = 100_000,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        verbose: int = 1,
        save_path: Optional[str] = None,
        eval_freq: int = 10_000,
    ) -> Dict[str, Any]:
        """
        Train PPO model on historical data.
        
        Args:
            predictions: Historical model predictions
            actual_returns: Historical actual returns
            volatilities: Historical volatilities
            total_timesteps: Total training timesteps
            learning_rate: PPO learning rate
            n_steps: Steps per update
            batch_size: Minibatch size
            n_epochs: Epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_range: PPO clip range
            ent_coef: Entropy coefficient
            verbose: Verbosity level
            save_path: Path to save best model
            eval_freq: Evaluation frequency
            
        Returns:
            Training history dict
        """
        # Create training environment
        def make_env():
            env = PositionSizingEnv(
                predictions=predictions,
                actual_returns=actual_returns,
                volatilities=volatilities,
            )
            env = Monitor(env)
            return env
        
        # Vectorized environment
        vec_env = DummyVecEnv([make_env])
        
        # Normalize observations
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
        )
        self.vec_normalize = vec_env
        
        # Create PPO model
        self.model = PPO(
            'MlpPolicy',
            vec_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            verbose=verbose,
            device=self.device,
            tensorboard_log=save_path + '/tensorboard' if save_path else None,
        )
        
        # Callbacks
        callbacks = []
        
        if save_path is not None:
            # Checkpoint callback
            checkpoint_callback = CheckpointCallback(
                save_freq=eval_freq,
                save_path=save_path,
                name_prefix='ppo_position_sizer',
            )
            callbacks.append(checkpoint_callback)
        
        # Train
        logger.info(f"Training PPO position sizer for {total_timesteps} timesteps...")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
        )
        
        # Save final model
        if save_path is not None:
            final_path = Path(save_path) / 'ppo_position_sizer_final'
            self.save(str(final_path))
        
        # Compute training stats
        returns = vec_env.get_original_reward()
        
        return {
            'total_timesteps': total_timesteps,
            'final_reward_mean': np.mean(returns) if len(returns) > 0 else 0.0,
            'final_reward_std': np.std(returns) if len(returns) > 0 else 0.0,
        }


# =============================================================================
# HYBRID POSITION SIZER (RL + Rule-based)
# =============================================================================

class HybridPositionSizer:
    """
    Hybrid position sizer combining RL and rule-based approaches.
    
    This class provides a robust position sizing system that:
    1. Uses RL when available and confident
    2. Falls back to rule-based in edge cases
    3. Applies risk constraints
    
    Args:
        rl_model_path: Path to trained RL model
        max_position: Maximum allowed position size
        min_confidence_for_rl: Minimum RL confidence to use RL
        volatility_scalar: Whether to scale by volatility
    """
    
    def __init__(
        self,
        rl_model_path: Optional[str] = None,
        max_position: float = 1.0,
        min_confidence_for_rl: float = 0.3,
        volatility_scalar: bool = True,
        base_volatility: float = 0.015,
    ):
        self.max_position = max_position
        self.min_confidence_for_rl = min_confidence_for_rl
        self.volatility_scalar = volatility_scalar
        self.base_volatility = base_volatility
        
        # Initialize RL sizer
        self.rl_sizer: Optional[PPOPositionSizer] = None
        if rl_model_path is not None and Path(rl_model_path).exists():
            self.rl_sizer = PPOPositionSizer(model_path=rl_model_path)
    
    def get_position(
        self,
        pred_return: float,
        pred_confidence: float,
        volatility: float,
        recent_pnl: float = 0.0,
        current_position: float = 0.0,
        days_since_trade: int = 0,
    ) -> Tuple[float, str]:
        """
        Get position size with method indicator.
        
        Args:
            pred_return: Predicted return
            pred_confidence: Model confidence [0, 1]
            volatility: Current volatility
            recent_pnl: Recent cumulative PnL
            current_position: Current position
            days_since_trade: Days since last trade
            
        Returns:
            (position_size, method) - method is 'rl' or 'rule'
        """
        # Try RL first if available and confident enough
        if (self.rl_sizer is not None and 
            self.rl_sizer.model is not None and
            pred_confidence >= self.min_confidence_for_rl):
            
            position = self.rl_sizer.get_position_size(
                pred_return=pred_return,
                volatility=volatility,
                recent_pnl=recent_pnl,
                current_position=current_position,
                days_since_trade=days_since_trade,
            )
            method = 'rl'
        else:
            # Rule-based fallback
            position = self._rule_based_position(
                pred_return, pred_confidence, volatility
            )
            method = 'rule'
        
        # Apply volatility scaling
        if self.volatility_scalar and volatility > 0:
            vol_scalar = self.base_volatility / max(volatility, 0.005)
            vol_scalar = np.clip(vol_scalar, 0.5, 2.0)
            position *= vol_scalar
        
        # Apply max position constraint
        position = np.clip(position, -self.max_position, self.max_position)
        
        return position, method
    
    def _rule_based_position(
        self,
        pred_return: float,
        confidence: float,
        volatility: float,
    ) -> float:
        """
        Rule-based position sizing.
        
        Kelly-inspired sizing with confidence weighting.
        """
        # Direction from prediction sign
        direction = np.sign(pred_return)
        
        # Magnitude from prediction size (capped at reasonable level)
        magnitude = np.clip(abs(pred_return) / 0.02, 0.0, 1.0)
        
        # Weight by confidence
        confidence_weight = confidence ** 0.5  # Square root for less aggressive scaling
        
        # Base position
        position = direction * magnitude * confidence_weight
        
        # Reduce in extreme volatility
        if volatility > 0.04:  # Very high vol
            position *= 0.5
        elif volatility < 0.005:  # Very low vol (suspicious)
            position *= 0.7
        
        return position


# =============================================================================
# TESTING
# =============================================================================

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    
    print("=" * 60)
    print("RL Position Sizer Test")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    # Simulated predictions (with some skill)
    actual_returns = np.random.randn(n_samples) * 0.02
    noise = np.random.randn(n_samples) * 0.01
    predictions = actual_returns * 0.3 + noise  # 30% signal + noise
    
    print(f"\nSynthetic data:")
    print(f"  Samples: {n_samples}")
    print(f"  Return mean: {actual_returns.mean():.6f}")
    print(f"  Return std: {actual_returns.std():.6f}")
    print(f"  Prediction corr: {np.corrcoef(predictions, actual_returns)[0,1]:.4f}")
    
    # Test environment
    print("\nTesting PositionSizingEnv...")
    env = PositionSizingEnv(
        predictions=predictions,
        actual_returns=actual_returns,
    )
    
    obs, info = env.reset()
    print(f"  Observation shape: {obs.shape}")
    print(f"  Action space: {env.action_space}")
    
    # Run random episode
    total_reward = 0
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated:
            break
    
    print(f"  Random policy total reward: {total_reward:.4f}")
    
    # Test PPO training (short)
    print("\nTesting PPO training (short run)...")
    sizer = PPOPositionSizer()
    
    history = sizer.train(
        predictions=predictions[:500],
        actual_returns=actual_returns[:500],
        total_timesteps=2000,  # Very short for testing
        verbose=0,
    )
    
    print(f"  Training completed")
    
    # Test inference
    print("\nTesting inference...")
    position = sizer.get_position_size(
        pred_return=0.01,
        volatility=0.015,
        recent_pnl=0.02,
        current_position=0.3,
    )
    print(f"  Position for +1% prediction: {position:.4f}")
    
    position = sizer.get_position_size(
        pred_return=-0.02,
        volatility=0.03,
        recent_pnl=-0.01,
        current_position=0.0,
    )
    print(f"  Position for -2% prediction (high vol): {position:.4f}")
    
    # Test hybrid sizer
    print("\nTesting HybridPositionSizer...")
    hybrid = HybridPositionSizer(max_position=1.0)
    
    pos, method = hybrid.get_position(
        pred_return=0.015,
        pred_confidence=0.7,
        volatility=0.02,
    )
    print(f"  High confidence position: {pos:.4f} (method: {method})")
    
    pos, method = hybrid.get_position(
        pred_return=0.005,
        pred_confidence=0.2,
        volatility=0.02,
    )
    print(f"  Low confidence position: {pos:.4f} (method: {method})")
    
    print("\n" + "=" * 60)
    print("RL Position Sizer test PASSED!")
    print("=" * 60)
