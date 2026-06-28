"""
FastAPI Model Serving — Credit Card Fraud Detection
=====================================================
Serves predictions from a trained RandomForest model via REST API.
Logs all prediction requests to Redis (for drift detection downstream)
with an in-memory fallback when Redis is unavailable.

Endpoints:
    POST /predict   — classify a transaction as fraud/legitimate
    GET  /health    — liveness check for container orchestration
    GET  /metrics   — Prometheus-format metrics (request count, latency, predictions)

Why FastAPI?
- Async-first design handles concurrent prediction requests efficiently
- Automatic OpenAPI docs at /docs for easy integration testing
- Pydantic models enforce input validation before hitting the model
- Industry standard for ML serving (used by major ML platforms)
"""

import json
import os
import time
from collections import deque
from contextlib import asynccontextmanager

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

# ── Global state ─────────────────────────────────────────────
# Loaded at startup, shared across requests.
# Using module-level state is fine for a single-process uvicorn worker;
# for multi-worker production, you'd use a model registry or shared memory.
model = None
scaler = None
redis_client = None

# In-memory prediction log — fallback when Redis is unavailable.
# Bounded deque prevents memory leaks; drift detector only needs recent window.
prediction_log: deque = deque(maxlen=10000)

# Simple metrics counters — a minimal Prometheus-compatible approach.
# In production, you'd use prometheus_client library with histograms.
metrics_store = {
    "request_count": 0,
    "prediction_fraud": 0,
    "prediction_legit": 0,
    "total_latency_ms": 0.0,
    "errors": 0,
}


# ── Pydantic models ─────────────────────────────────────────
# These enforce input/output contracts. Invalid requests get a 422 with details
# before any model inference happens — fail fast, fail clearly.


class PredictionRequest(BaseModel):
    """Input features for a single transaction prediction."""

    amount: float = Field(..., description="Transaction amount in dollars")
    time: int = Field(..., description="Seconds elapsed from first transaction")
    v1: float = Field(..., description="PCA component 1")
    v2: float = Field(..., description="PCA component 2")
    v3: float = Field(..., description="PCA component 3")
    v4: float = Field(..., description="PCA component 4")
    v5: float = Field(..., description="PCA component 5")
    v6: float = Field(..., description="PCA component 6")
    v7: float = Field(..., description="PCA component 7")
    v8: float = Field(..., description="PCA component 8")
    v9: float = Field(..., description="PCA component 9")
    v10: float = Field(..., description="PCA component 10")

    model_config = {
        "json_schema_extra": {
            "example": {
                "amount": 150.0,
                "time": 3600,
                "v1": -1.35,
                "v2": 1.19,
                "v3": 0.27,
                "v4": 0.16,
                "v5": 0.45,
                "v6": 0.06,
                "v7": -0.68,
                "v8": 0.09,
                "v9": -0.25,
                "v10": -0.17,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Prediction result with probability and feature echo."""

    prediction: int = Field(..., description="0 = legitimate, 1 = fraud")
    probability: float = Field(..., description="Probability of fraud (0.0 to 1.0)")
    label: str = Field(..., description="Human-readable label")


class HealthResponse(BaseModel):
    """Service health status."""

    status: str
    model_loaded: bool
    redis_connected: bool


# ── Startup / shutdown ───────────────────────────────────────


def load_model() -> None:
    """
    Load the trained model and scaler from disk.
    Checks multiple paths to support both local dev and Docker container layouts.
    """
    global model, scaler

    # Search paths in priority order: Docker mount → local dev → relative
    model_paths = [
        "/models/model.joblib",
        os.path.join(os.path.dirname(__file__), "..", "models", "model.joblib"),
        "models/model.joblib",
    ]

    scaler_paths = [
        "/models/scaler.joblib",
        os.path.join(os.path.dirname(__file__), "..", "models", "scaler.joblib"),
        "models/scaler.joblib",
    ]

    for path in model_paths:
        if os.path.exists(path):
            model = joblib.load(path)
            print(f"Model loaded from: {path}")
            break
    else:
        print("WARNING: No model file found — /predict will return 503")

    for path in scaler_paths:
        if os.path.exists(path):
            scaler = joblib.load(path)
            print(f"Scaler loaded from: {path}")
            break
    else:
        print("WARNING: No scaler file found — predictions will use unscaled features")


def connect_redis() -> None:
    """
    Attempt Redis connection for prediction logging.
    Non-blocking: falls back to in-memory deque if Redis is down.
    Redis Streams give us an append-only log that the drift detector can consume.
    """
    global redis_client
    try:
        import redis

        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        redis_client = redis.from_url(redis_url, decode_responses=True)
        redis_client.ping()
        print(f"Redis connected: {redis_url}")
    except Exception as e:
        redis_client = None
        print(f"Redis unavailable ({e}) — using in-memory prediction log")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle — load model and connect to Redis."""
    load_model()
    connect_redis()
    yield


# ── App ──────────────────────────────────────────────────────

app = FastAPI(
    title="ML Fraud Detection API",
    description="Real-time credit card fraud prediction with drift monitoring",
    version="1.0.0",
    lifespan=lifespan,
)


def log_prediction(features: dict, prediction: int, probability: float) -> None:
    """
    Log a prediction to Redis stream (or in-memory fallback).
    The drift detector reads these logs to compare current data distributions
    against the training reference — this is how we detect silent model degradation.
    """
    entry = {
        "timestamp": time.time(),
        "features": json.dumps(features),
        "prediction": prediction,
        "probability": probability,
    }

    if redis_client is not None:
        try:
            # XADD appends to a Redis Stream — ordered, persistent, consumer-group ready.
            # maxlen ~10000 caps memory usage; the ~ means approximate trimming (faster).
            redis_client.xadd("predictions", entry, maxlen=10000)
            return
        except Exception:
            pass

    # Fallback: bounded deque keeps recent predictions in memory
    prediction_log.append(entry)


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Classify a transaction as fraudulent or legitimate.

    Takes 12 features (amount, time, v1-v10), scales them using the training
    scaler, and runs inference through the RandomForest model. Returns the
    binary prediction plus the fraud probability score.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.monotonic()

    # Convert request to feature array in the same order as training
    features = {
        "v1": request.v1,
        "v2": request.v2,
        "v3": request.v3,
        "v4": request.v4,
        "v5": request.v5,
        "v6": request.v6,
        "v7": request.v7,
        "v8": request.v8,
        "v9": request.v9,
        "v10": request.v10,
        "amount": request.amount,
        "time": request.time,
    }

    feature_array = np.array(
        [
            [
                features["v1"],
                features["v2"],
                features["v3"],
                features["v4"],
                features["v5"],
                features["v6"],
                features["v7"],
                features["v8"],
                features["v9"],
                features["v10"],
                features["amount"],
                features["time"],
            ]
        ]
    )

    # Apply the same scaling transform used during training.
    # Without this, feature magnitudes would be different and predictions meaningless.
    if scaler is not None:
        feature_array = scaler.transform(feature_array)

    prediction = int(model.predict(feature_array)[0])
    probability = float(model.predict_proba(feature_array)[0][1])

    # Update metrics
    elapsed_ms = (time.monotonic() - start_time) * 1000
    metrics_store["request_count"] += 1
    metrics_store["total_latency_ms"] += elapsed_ms
    if prediction == 1:
        metrics_store["prediction_fraud"] += 1
    else:
        metrics_store["prediction_legit"] += 1

    # Log for drift detection
    log_prediction(features, prediction, probability)

    return PredictionResponse(
        prediction=prediction,
        probability=round(probability, 4),
        label="fraud" if prediction == 1 else "legitimate",
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Liveness check for container orchestration.
    Returns model and Redis connection status so orchestrators know
    if the service is ready to accept prediction requests.
    """
    redis_ok = False
    if redis_client is not None:
        try:
            redis_client.ping()
            redis_ok = True
        except Exception:
            pass

    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        redis_connected=redis_ok,
    )


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> PlainTextResponse:
    """
    Prometheus-compatible metrics endpoint.

    Returns plain text in Prometheus exposition format so standard monitoring
    tools can scrape these without any additional configuration. Metrics include:
    - Total prediction requests
    - Predictions by class (fraud vs legitimate)
    - Average inference latency
    - Error count
    """
    req_count = metrics_store["request_count"]
    avg_latency = (
        metrics_store["total_latency_ms"] / req_count if req_count > 0 else 0.0
    )

    # In-memory prediction log size — useful for debugging fallback behavior
    log_size = len(prediction_log)

    lines = [
        "# HELP fraud_detector_requests_total Total prediction requests",
        "# TYPE fraud_detector_requests_total counter",
        f"fraud_detector_requests_total {req_count}",
        "",
        "# HELP fraud_detector_predictions_total Predictions by class",
        "# TYPE fraud_detector_predictions_total counter",
        f'fraud_detector_predictions_total{{class="fraud"}} {metrics_store["prediction_fraud"]}',
        f'fraud_detector_predictions_total{{class="legit"}} {metrics_store["prediction_legit"]}',
        "",
        "# HELP fraud_detector_latency_avg_ms Average inference latency in milliseconds",
        "# TYPE fraud_detector_latency_avg_ms gauge",
        f"fraud_detector_latency_avg_ms {avg_latency:.2f}",
        "",
        "# HELP fraud_detector_errors_total Total error count",
        "# TYPE fraud_detector_errors_total counter",
        f"fraud_detector_errors_total {metrics_store['errors']}",
        "",
        "# HELP fraud_detector_prediction_log_size In-memory prediction log size",
        "# TYPE fraud_detector_prediction_log_size gauge",
        f"fraud_detector_prediction_log_size {log_size}",
        "",
    ]

    return PlainTextResponse("\n".join(lines), media_type="text/plain")
