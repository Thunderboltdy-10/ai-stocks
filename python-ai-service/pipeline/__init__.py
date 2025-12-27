"""
Production Pipeline Module for AI-Stocks

This module provides an end-to-end production pipeline that:
1. Separates VALIDATION phase from PRODUCTION phase
2. Runs walk-forward validation on all models (LSTM, GBM, xLSTM-TS)
3. Computes aggregate WFE (Walk Forward Efficiency)
4. Only proceeds to production training if WFE > 50%
5. Trains production models on ALL data after validation passes
6. Trains stacking meta-learner
7. Saves complete ensemble for deployment

Key Classes:
- ProductionPipeline: Main orchestrator
- ValidationReport: Validation phase results
- ProductionModel: Trained production model container

Author: AI-Stocks Nuclear Redesign
Date: December 2025
"""

from .production_pipeline import (
    ProductionPipeline,
    ValidationReport,
    ProductionModel,
    PipelineConfig,
)

__all__ = [
    'ProductionPipeline',
    'ValidationReport',
    'ProductionModel',
    'PipelineConfig',
]
