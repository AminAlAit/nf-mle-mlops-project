"""FastAPI inference microservice for NYC Green Taxi ride duration prediction."""

import csv
import threading
from pathlib import Path
from typing import Any, Dict

import joblib
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, make_asgi_app

ROOT_DIR            = Path(__file__).resolve().parents[1]
MODEL_PATH          = ROOT_DIR / "models" / "model.joblib"
REQUEST_LOG_PATH    = ROOT_DIR / "monitoring" / "requests.csv"


# Pydantic schemas
class RideFeatures(BaseModel):
    PULocationID: str
    DOLocationID: str
    trip_distance: float


class PredictionResponse(BaseModel):
    duration_minutes: float


# Model service and audit logger
class ModelService:
    """Singleton service to load and cache the scikit-learn model."""
    _instance: Any = None

    @classmethod
    def get_model(cls) -> Any:
        if cls._instance is None:
            if not MODEL_PATH.exists():
                raise HTTPException(status_code=503, detail="Model artifact not found on disk")
            cls._instance = joblib.load(MODEL_PATH)
        return cls._instance


class RequestLogger:
    """Thread-safe request audit logger for Evidently drift detection."""
    _lock = threading.Lock()

    def __init__(self, log_path: Path = REQUEST_LOG_PATH):
        self.log_path = log_path

    def log(self, features: RideFeatures, duration: float) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "PULocationID": features.PULocationID,
            "DOLocationID": features.DOLocationID,
            "trip_distance": features.trip_distance,
            "duration": duration,
        }
        with self._lock:
            file_exists = self.log_path.exists()
            with self.log_path.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)


def get_model() -> Any:
    """FastAPI dependency provider for model instance."""
    return ModelService.get_model()


def get_logger() -> RequestLogger:
    """FastAPI dependency provider for request logger."""
    return RequestLogger()


# Application and Prometheus metrics
app = FastAPI(
    title="Green Taxi Ride Duration API",
    description="Real-time inference API for NYC Green Taxi duration predictions.",
    version="1.0.0",
)

# Prometheus metrics
prediction_counter   = Counter("taxi_predictions_total", "Total number of predictions served")
api_requests_counter = Counter("api_prediction_requests_total", "Total prediction requests for Grafana compatibility")
prediction_latency   = Histogram("api_prediction_latency_seconds", "Prediction latency in seconds")

# Expose /metrics endpoint for Prometheus scraping
app.mount("/metrics", make_asgi_app())


@app.on_event("startup")
def preload_model() -> None:
    """Preload model weights into RAM on application startup."""
    try:
        ModelService.get_model()
    except Exception:
        # Allow service to start even if model is compiled in later step
        pass


# Endpoints
@app.get("/health")
def health(model: Any = Depends(get_model)) -> Dict[str, Any]:
    """Healthcheck endpoint verifying server status and model readiness."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(
    features: RideFeatures,
    model: Any = Depends(get_model),
    logger: RequestLogger = Depends(get_logger),
) -> PredictionResponse:
    """Serve validated trip duration predictions and record operational telemetry."""
    if features.trip_distance <= 0:
        raise HTTPException(status_code=400, detail="Trip distance must be positive")

    try:
        payload = [features.model_dump()]

        with prediction_latency.time():
            duration = float(model.predict(payload)[0])
            prediction_counter.inc()
            api_requests_counter.inc()
            logger.log(features, duration)

        return PredictionResponse(duration_minutes=round(duration, 4))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
