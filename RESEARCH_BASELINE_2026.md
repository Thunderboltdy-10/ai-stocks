# Research Baseline 2026

This project should earn trust in three stages:

1. Safe baseline
2. Cross-symbol evidence
3. Riskier architecture exploration

## Safe baseline

Use the production GBM-first path before adding new model families:

- `training/train_gbm.py`
- purged walk-forward validation
- holdout-aware quality scoring
- multi-window backtests
- core-vs-holdout generalization gates
- inventory-aware execution backtests
- realized forward holdout simulation

Why this is the baseline:

- It is the most inspectable path in the repo.
- It is the easiest path to falsify across symbols and windows.
- It already supports daily and intraday routing without introducing another fragile stack.

## Acceptance standard

A change is not accepted because one symbol looks good.

It should survive:

- diversified daily basket (`daily15`)
- diversified intraday basket (`intraday10`)
- long and short windows
- holdout gate checks on alpha, hit rate, Sharpe, and generalization gap

Recent gate tightening in `python-ai-service/scripts/run_generalization_gate.py` raises the bar from “not obviously broken” to “positive holdout edge with smaller room for luck”.

## Workflow

Primary workflow entrypoint:

```bash
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate ai-stocks
cd /home/thunderboltdy/ai-stocks/python-ai-service
python scripts/run_research_workflow.py --workflow full --overwrite
```

The `/ai` page exposes the same workflow through the Research Ops panel.

## What stays experimental

The repo can still explore stronger sequence models, but only after the safe path clears holdout:

- PatchTST: https://arxiv.org/abs/2211.14730
- Temporal Fusion Transformer: https://arxiv.org/abs/1912.09363
- Conformalized Quantile Regression: https://papers.nips.cc/paper_files/paper/2019/hash/5103c3584b063c431bd12689b5e76fb-Abstract.html

Use these as research branches of thought, not as the first thing trusted in routing.

## Hyperparameter stance

Hyperparameters should be chosen by cost-aware walk-forward behavior, not only directional accuracy.

Prefer:

- Optuna TPE / pruning: https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html
- XGBoost GPU hist tree method for scalable sweeps: https://xgboost.readthedocs.io/en/stable/parameter.html

## Non-negotiables

- no forward leakage
- no acceptance from a tiny symbol basket
- no acceptance from one lucky window
- no trust in synthetic forward paths over realized holdout behavior
