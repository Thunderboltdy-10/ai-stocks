# Research Notes (2026-02-23)

This note captures primary-source research used to guide the current architecture and next iterations.

## Primary sources reviewed

1. XGBoost (Chen & Guestrin, 2016)
- https://arxiv.org/abs/1603.02754
- Used for GPU-accelerated tabular alpha modeling baseline.

2. LightGBM (Ke et al., 2017)
- https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html
- Informs histogram/leaf-wise boosting tradeoffs.

3. Deep Learning in Asset Pricing (Gu, Kelly, Xiu, 2020)
- https://www.nber.org/papers/w25398
- Motivates nonlinear signal extraction and strict out-of-sample validation discipline.

4. Volatility-Managed Portfolios (Moreira & Muir, 2017)
- https://www.nber.org/papers/w22208
- Basis for volatility-targeting overlay in execution layer.

5. Time-Series Momentum (Moskowitz, Ooi, Pedersen, 2012)
- https://www.sciencedirect.com/science/article/pii/S0304405X11002613
- Supports regime/trend-aware execution overlays.

6. Financial Machine Learning / purged CV methodology (Lopez de Prado)
- Book reference: *Advances in Financial Machine Learning* (Wiley, 2018)
- Basis for purged temporal splitting and leakage-aware validation logic.

7. Trend-following + volatility scaling robustness literature (Kim, Tse, Wald, 2016)
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2691996
- Used as a cautionary and design reference for scaling/trend overlays.

## Implemented from research

- GPU-first GBM training with strict leakage controls.
- Metadata-driven position sizing priors (training target distribution).
- Volatility-targeting overlay (`vol_target_annual=0.25`) in backtest and live prediction path.
- Model-quality gate + passive fallback to avoid forcing weak models across symbols.

## Next research-driven implementation targets

1. Regime-expert mixture of GBMs
- Separate experts for trend/uptrend/downtrend/high-volatility regimes.
- Gate with regime posterior from engineered features; validate with purged WFE.

2. Conformal risk controls
- Prediction-interval-aware position clipping (reduce size when uncertainty widens).

3. Cost-aware objective in tuning
- Integrate turnover-cost penalties directly into Optuna objective, not only post-hoc backtests.

4. Cross-asset feature enrichment
- Explicit market/sector context (SPY/QQQ/sector ETF relative momentum and correlation) with strict shift/lag handling.

