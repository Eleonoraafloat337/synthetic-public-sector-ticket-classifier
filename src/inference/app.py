"""FastAPI application for ticket classification."""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field, field_validator

from inference.audit import build_audit_record, emit_audit_record
from inference.logging import configure_logging
from inference.model import TicketClassifier
from monitoring.drift import DriftDetector
from ticket_classifier.config import get_settings

settings = get_settings()
configure_logging(settings.log_level)

REQUEST_COUNTER = Counter("classify_requests_total", "Total classify requests", ["endpoint"])
LABEL_COUNTER = Counter("classify_label_total", "Predicted label distribution", ["label"])
LOW_CONFIDENCE_COUNTER = Counter("classify_low_confidence_total", "Low-confidence predictions")
LATENCY_HISTOGRAM = Histogram(
    "classify_latency_seconds",
    "Classification request latency with p50/p95/p99 available via histogram_quantile",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

classifier = TicketClassifier(settings.model_dir, settings.model_version, settings.confidence_threshold)
drift_detector = DriftDetector(window_size=settings.drift_window, threshold=settings.drift_threshold)
started_at = time.monotonic()


class ClassifyRequest(BaseModel):
    """Single classification request."""

    text: Annotated[str, Field(min_length=1, max_length=settings.max_input_chars)]
    include_scores: bool = False

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        value.encode("utf-8")
        return value


class BatchClassifyRequest(BaseModel):
    """Batch classification request with a hard limit for latency control."""

    texts: Annotated[list[Annotated[str, Field(min_length=1, max_length=settings.max_input_chars)]], Field(min_length=1, max_length=50)]
    include_scores: bool = False

    @field_validator("texts")
    @classmethod
    def texts_must_not_be_blank(cls, values: list[str]) -> list[str]:
        for value in values:
            if not value.strip():
                raise ValueError("texts must not contain blanks")
            value.encode("utf-8")
        return values


class ClassificationResponse(BaseModel):
    """API response for one ticket."""

    label: str
    confidence: float
    model_version: str
    latency_ms: int
    below_threshold: bool = False
    scores: dict[str, float] | None = None


@asynccontextmanager
async def lifespan(_: FastAPI) -> Any:
    classifier.load()
    yield


app = FastAPI(title="Public Sector Ticket Classifier", version=settings.model_version, lifespan=lifespan)


def _response_payload(prediction: Any, latency_ms: int, include_scores: bool) -> ClassificationResponse:
    return ClassificationResponse(
        label=prediction.label,
        confidence=round(prediction.confidence, 6),
        model_version=settings.model_version,
        latency_ms=latency_ms,
        below_threshold=prediction.below_threshold,
        scores=prediction.scores if include_scores else None,
    )


@app.post("/classify", response_model=ClassificationResponse)
async def classify(request: ClassifyRequest, http_request: Request) -> ClassificationResponse:
    """Classify one ticket and emit a privacy-preserving audit event."""
    start = time.perf_counter()
    REQUEST_COUNTER.labels(endpoint="/classify").inc()
    predictions = classifier.classify_batch([request.text])
    latency_ms = int((time.perf_counter() - start) * 1000)
    prediction = predictions[0]
    LABEL_COUNTER.labels(label=prediction.label).inc()
    drift_detector.observe(prediction.label)
    if prediction.confidence < settings.low_confidence_threshold:
        LOW_CONFIDENCE_COUNTER.inc()
    LATENCY_HISTOGRAM.labels(endpoint="/classify").observe(latency_ms / 1000)
    request_id = http_request.headers.get("x-request-id", str(uuid.uuid4()))
    emit_audit_record(
        build_audit_record(
            input_length=len(request.text),
            label=prediction.label,
            confidence=prediction.confidence,
            below_threshold=prediction.below_threshold,
            model_version=settings.model_version,
            request_id=request_id,
        ),
    )
    return _response_payload(prediction, latency_ms, request.include_scores)


@app.post("/classify/batch", response_model=list[ClassificationResponse])
async def classify_batch(request: BatchClassifyRequest) -> list[ClassificationResponse]:
    """Classify up to 50 tickets in a single batched model call."""
    start = time.perf_counter()
    REQUEST_COUNTER.labels(endpoint="/classify/batch").inc()
    predictions = classifier.classify_batch(request.texts)
    latency_ms = int((time.perf_counter() - start) * 1000)
    LATENCY_HISTOGRAM.labels(endpoint="/classify/batch").observe(latency_ms / 1000)
    responses: list[ClassificationResponse] = []
    for prediction in predictions:
        LABEL_COUNTER.labels(label=prediction.label).inc()
        drift_detector.observe(prediction.label)
        if prediction.confidence < settings.low_confidence_threshold:
            LOW_CONFIDENCE_COUNTER.inc()
        responses.append(_response_payload(prediction, latency_ms, request.include_scores))
    return responses


@app.get("/health")
async def health() -> dict[str, object]:
    """Health probe consumed by Azure Container Apps."""
    if not classifier.loaded:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok", "model_loaded": classifier.loaded, "model_version": settings.model_version, "uptime_seconds": int(time.monotonic() - started_at)}


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    """Return Prometheus metrics."""
    return PlainTextResponse(generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)


@app.get("/metrics/drift")
async def drift_metrics() -> dict[str, object]:
    """Return current label-distribution drift."""
    return drift_detector.snapshot()
