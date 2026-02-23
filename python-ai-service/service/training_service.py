"""
Training Service - Async training orchestration with SSE progress streaming.
"""

import asyncio
import json
import os
import subprocess
import sys
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional
import re


class TrainingStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingJob:
    id: str
    symbol: str
    status: TrainingStatus
    progress: float
    current_epoch: int
    total_epochs: int
    started_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None
    model_type: str = "lstm_transformer"
    epochs: int = 50
    batch_size: int = 512
    sequence_length: int = 90
    loss: str = "balanced"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "status": self.status.value,
            "progress": self.progress,
            "currentEpoch": self.current_epoch,
            "totalEpochs": self.total_epochs,
            "startedAt": self.started_at,
            "completedAt": self.completed_at,
            "error": self.error,
            "modelType": self.model_type,
        }


@dataclass
class EpochUpdate:
    epoch: int
    total_epochs: int
    train_loss: float
    val_loss: float
    learning_rate: float
    timestamp: str
    directional_accuracy: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "epoch",
            "epoch": self.epoch,
            "totalEpochs": self.total_epochs,
            "trainLoss": self.train_loss,
            "valLoss": self.val_loss,
            "learningRate": self.learning_rate,
            "timestamp": self.timestamp,
            "directionalAccuracy": self.directional_accuracy,
        }


class TrainingService:
    """Manages training jobs with async execution and SSE streaming."""

    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.jobs: Dict[str, TrainingJob] = {}
        self.job_events: Dict[str, List[Dict[str, Any]]] = {}
        self.event_queues: Dict[str, asyncio.Queue] = {}
        self.active_processes: Dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

    def start_training(self, config: Dict[str, Any]) -> str:
        """Start a new training job and return the job ID."""
        job_id = str(uuid.uuid4())[:8]
        symbol = config.get("symbol", "AAPL").upper()
        epochs = config.get("epochs", 50)
        batch_size = config.get("batchSize", 512)
        sequence_length = config.get("sequenceLength", 90)
        loss = config.get("loss", "balanced")
        model_type = config.get("modelType", "lstm_transformer")

        job = TrainingJob(
            id=job_id,
            symbol=symbol,
            status=TrainingStatus.QUEUED,
            progress=0.0,
            current_epoch=0,
            total_epochs=epochs,
            started_at=datetime.now().isoformat(),
            model_type=model_type,
            epochs=epochs,
            batch_size=batch_size,
            sequence_length=sequence_length,
            loss=loss,
        )

        with self._lock:
            self.jobs[job_id] = job
            self.job_events[job_id] = []
            self.event_queues[job_id] = asyncio.Queue()

        # Start training in background thread
        thread = threading.Thread(target=self._run_training, args=(job_id, config))
        thread.daemon = True
        thread.start()

        return job_id

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get job by ID."""
        return self.jobs.get(job_id)

    def list_jobs(self) -> List[TrainingJob]:
        """List all jobs."""
        return list(self.jobs.values())

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        job = self.jobs.get(job_id)
        if not job:
            return False

        if job.status not in (TrainingStatus.QUEUED, TrainingStatus.RUNNING):
            return False

        # Kill the subprocess if running
        if job_id in self.active_processes:
            proc = self.active_processes[job_id]
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            del self.active_processes[job_id]

        with self._lock:
            job.status = TrainingStatus.CANCELLED
            job.completed_at = datetime.now().isoformat()
            self._emit_event(job_id, {"type": "cancelled"})

        return True

    async def stream_events(self, job_id: str) -> AsyncGenerator[str, None]:
        """Stream SSE events for a job."""
        job = self.jobs.get(job_id)
        if not job:
            yield f"data: {json.dumps({'type': 'error', 'error': 'Job not found'})}\n\n"
            return

        queue = self.event_queues.get(job_id)
        if not queue:
            yield f"data: {json.dumps({'type': 'error', 'error': 'Event queue not found'})}\n\n"
            return

        # Send any existing events first
        for event in self.job_events.get(job_id, []):
            yield f"data: {json.dumps(event)}\n\n"

        # Stream new events
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30)
                yield f"data: {json.dumps(event)}\n\n"

                # Check for terminal events
                if event.get("type") in ("completed", "failed", "cancelled"):
                    break
            except asyncio.TimeoutError:
                # Send heartbeat
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"

    def _emit_event(self, job_id: str, event: Dict[str, Any]):
        """Emit an event to the job's queue."""
        if job_id in self.job_events:
            self.job_events[job_id].append(event)

        if job_id in self.event_queues:
            queue = self.event_queues[job_id]
            try:
                # Use call_soon_threadsafe if we're in a different thread
                loop = asyncio.get_event_loop()
                loop.call_soon_threadsafe(queue.put_nowait, event)
            except RuntimeError:
                # No event loop, queue directly
                pass

    def _run_training(self, job_id: str, config: Dict[str, Any]):
        """Run training in a subprocess."""
        job = self.jobs[job_id]

        with self._lock:
            job.status = TrainingStatus.RUNNING

        self._emit_event(job_id, {"type": "started", "symbol": job.symbol})

        try:
            # Determine script based on model type
            model_type = config.get("modelType", "lstm_transformer")
            if model_type == "gbm":
                module_name = "training.train_gbm"
            elif model_type == "stacking":
                module_name = "training.train_stacking_ensemble"
            else:
                module_name = "training.train_1d_regressor_final"

            # Build command
            if model_type == "gbm":
                cmd = [
                    sys.executable,
                    "-m",
                    module_name,
                    job.symbol,
                    "--overwrite",
                    "--n-trials",
                    str(max(10, min(50, int(config.get("nTrials", 20))))),
                ]
            else:
                cmd = [
                    sys.executable,
                    "-m",
                    module_name,
                    "--symbol",
                    job.symbol,
                    "--epochs",
                    str(job.epochs),
                    "--batch-size",
                    str(job.batch_size),
                    "--sequence-length",
                    str(job.sequence_length),
                ]
                cmd.extend(["--loss", job.loss])

            # Run process
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
                cwd=str(Path(__file__).parent.parent),
            )

            self.active_processes[job_id] = proc

            # Parse output for epoch updates
            epoch_pattern = re.compile(
                r"Epoch\s+(\d+)/(\d+).*?loss[:\s]+([0-9.]+).*?val_loss[:\s]+([0-9.]+)",
                re.IGNORECASE
            )
            lr_pattern = re.compile(r"lr[:\s]+([0-9.e-]+)", re.IGNORECASE)
            da_pattern = re.compile(r"directional.*?accuracy[:\s]+([0-9.]+)", re.IGNORECASE)

            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue

                # Try to parse epoch update
                epoch_match = epoch_pattern.search(line)
                if epoch_match:
                    epoch = int(epoch_match.group(1))
                    total = int(epoch_match.group(2))
                    train_loss = float(epoch_match.group(3))
                    val_loss = float(epoch_match.group(4))

                    lr_match = lr_pattern.search(line)
                    lr = float(lr_match.group(1)) if lr_match else 0.001

                    da_match = da_pattern.search(line)
                    da = float(da_match.group(1)) if da_match else None

                    with self._lock:
                        job.current_epoch = epoch
                        job.progress = epoch / total

                    update = EpochUpdate(
                        epoch=epoch,
                        total_epochs=total,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        learning_rate=lr,
                        timestamp=datetime.now().isoformat(),
                        directional_accuracy=da,
                    )
                    self._emit_event(job_id, update.to_dict())

            # Wait for process to complete
            return_code = proc.wait()

            if job_id in self.active_processes:
                del self.active_processes[job_id]

            with self._lock:
                if return_code == 0:
                    job.status = TrainingStatus.COMPLETED
                    job.progress = 1.0
                    job.current_epoch = job.total_epochs
                else:
                    job.status = TrainingStatus.FAILED
                    job.error = f"Training failed with exit code {return_code}"

                job.completed_at = datetime.now().isoformat()

            if return_code == 0:
                self._emit_event(job_id, {"type": "completed"})
            else:
                self._emit_event(job_id, {"type": "failed", "error": job.error})

        except Exception as e:
            with self._lock:
                job.status = TrainingStatus.FAILED
                job.error = str(e)
                job.completed_at = datetime.now().isoformat()

            self._emit_event(job_id, {"type": "failed", "error": str(e)})


# Global instance
_training_service: Optional[TrainingService] = None


def get_training_service(models_dir: Path) -> TrainingService:
    """Get or create the training service singleton."""
    global _training_service
    if _training_service is None:
        _training_service = TrainingService(models_dir)
    return _training_service
