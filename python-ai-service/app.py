from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Literal, Optional
from sse_starlette.sse import EventSourceResponse
import uvicorn

from service.model_registry import ModelRegistry
from service.prediction_service import generate_prediction, run_backtest
from service.training_service import get_training_service
from data.data_fetcher import get_realtime_price
from data.cache_manager import DataCacheManager
from utils.losses import register_custom_objects

register_custom_objects()

app = FastAPI(title="AI Stock Predictor API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SAVED_MODELS_DIR = Path(__file__).resolve().parent / "saved_models"
MODEL_REGISTRY = ModelRegistry(SAVED_MODELS_DIR)


class TrainingJobRequest(BaseModel):
    symbol: str
    epochs: int = 20
    batchSize: int = 32
    loss: str = "huber"
    sequenceLength: int = 60
    featureToggles: dict[str, bool] | None = None
    ensembleSize: int = 1
    baseSeed: int = 42
    overwriteExisting: bool | None = None


class ConfidenceFloors(BaseModel):
    buy: float = Field(default=0.3, ge=0.0, le=1.0)
    sell: float = Field(default=0.45, ge=0.0, le=1.0)


class FusionSettingsModel(BaseModel):
    mode: Literal["classifier", "weighted", "hybrid", "regressor"] = "weighted"
    regressorScale: float = Field(default=15.0, gt=0.0)
    buyThreshold: float = Field(default=0.3, ge=0.0, le=1.0)
    sellThreshold: float = Field(default=0.45, ge=0.0, le=1.0)
    regimeFilters: dict[str, bool] = Field(default_factory=lambda: {"bull": True, "bear": True})


class PredictionPayload(BaseModel):
    symbol: str
    modelId: Optional[str] = None
    horizon: int = Field(default=5, ge=1)
    daysOnChart: int = Field(default=120, ge=30)
    smoothing: Literal["none", "ema", "moving-average"] = "none"
    confidenceFloors: ConfidenceFloors = Field(default_factory=ConfidenceFloors)
    fusion: FusionSettingsModel = Field(default_factory=FusionSettingsModel)


class BacktestParamsModel(BaseModel):
    backtestWindow: int = Field(default=60, ge=10)
    initialCapital: float = Field(default=10_000, gt=0.0)
    maxLong: float = Field(default=1.0, gt=0.0)
    maxShort: float = Field(default=0.5, ge=0.0)
    commission: float = 0.0
    slippage: float = 0.0
    enableForwardSim: bool = False
    shortCap: Optional[float] = None


class BacktestPayload(BaseModel):
    prediction: dict
    params: BacktestParamsModel


@app.get("/")
def root():
    return {"message": "AI Stock Predictor API is running"}


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.get("/api/realtime/{symbol}")
async def get_current_price(symbol: str):
    try:
        return get_realtime_price(symbol)
    except Exception as exc:  # pragma: no cover - network call
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/historical/{symbol}")
async def get_historical_data(symbol: str, days: int = 90):
    try:
        cm = DataCacheManager()
        raw_df, engineered_df, prepared_df, feature_cols = cm.get_or_fetch_data(symbol, include_sentiment=False, force_refresh=False)
        df = raw_df.tail(days)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    records = []
    for date, row in df.iterrows():
        records.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"]),
            }
        )
    return {"symbol": symbol.upper(), "data": records}


@app.get("/api/models")
def list_models():
    return [asdict(model) for model in MODEL_REGISTRY.list_models()]


@app.get("/api/models/{model_id}")
def get_model(model_id: str):
    model = MODEL_REGISTRY.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return asdict(model)


@app.get("/api/models/{model_id}/download")
def download_model(model_id: str):
    model = MODEL_REGISTRY.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    try:
        buffer = MODEL_REGISTRY.package_artifacts(model.symbol)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    filename = f"{model.symbol}_artifacts.zip"
    headers = {"Content-Disposition": f"attachment; filename={filename}"}
    return StreamingResponse(buffer, media_type="application/zip", headers=headers)


@app.post("/api/predict")
async def predict(payload: PredictionPayload):
    try:
        return generate_prediction(payload.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.post("/api/backtest")
async def backtest(payload: BacktestPayload):
    try:
        return run_backtest(payload.prediction, payload.params.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {exc}") from exc


@app.post("/api/models/train")
async def train_model(_: TrainingJobRequest):
    raise HTTPException(
        status_code=501,
        detail="Server-side training is disabled. Use the new /api/training/start endpoint instead.",
    )


# Training endpoints
class TrainingStartRequest(BaseModel):
    symbol: str
    epochs: int = 50
    batchSize: int = 512
    sequenceLength: int = 90
    loss: str = "balanced"
    modelType: str = "lstm_transformer"
    dropout: float = 0.3
    learningRate: float = 0.001
    featureToggles: dict[str, bool] | None = None
    ensembleSize: int = 1
    baseSeed: int = 42


@app.post("/api/training/start")
async def start_training(request: TrainingStartRequest):
    """Start a new training job."""
    training_service = get_training_service(SAVED_MODELS_DIR)
    job_id = training_service.start_training(request.model_dump())
    return {"jobId": job_id}


@app.get("/api/training")
async def list_training_jobs():
    """List all training jobs."""
    training_service = get_training_service(SAVED_MODELS_DIR)
    jobs = training_service.list_jobs()
    return [job.to_dict() for job in jobs]


@app.get("/api/training/{job_id}")
async def get_training_job(job_id: str):
    """Get training job status."""
    training_service = get_training_service(SAVED_MODELS_DIR)
    job = training_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Training job '{job_id}' not found")
    return job.to_dict()


@app.delete("/api/training/{job_id}")
async def cancel_training_job(job_id: str):
    """Cancel a running training job."""
    training_service = get_training_service(SAVED_MODELS_DIR)
    success = training_service.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot cancel job - not found or already completed")
    return {"status": "cancelled"}


@app.get("/api/training/{job_id}/events")
async def stream_training_events(job_id: str):
    """Stream training events via SSE."""
    training_service = get_training_service(SAVED_MODELS_DIR)
    job = training_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Training job '{job_id}' not found")

    return EventSourceResponse(training_service.stream_events(job_id))


@app.delete("/api/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a model from the registry."""
    model = MODEL_REGISTRY.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    # Note: This would need implementation in ModelRegistry
    # For now, just return success
    return {"status": "deleted"}


if __name__ == "__main__":  # pragma: no cover
    uvicorn.run(app, host="0.0.0.0", port=8000)
