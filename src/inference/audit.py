"""Prediction audit logging that never records raw request text."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any


AUDIT_LOGGER = logging.getLogger("ticket_classifier.audit")


def build_audit_record(
    *,
    input_length: int,
    label: str,
    confidence: float,
    below_threshold: bool,
    model_version: str,
    request_id: str,
) -> dict[str, Any]:
    """Build an audit event without including raw user text."""
    return {
        "event": "classification",
        "timestamp": datetime.now(UTC).isoformat(),
        "input_length": input_length,
        "label": label,
        "confidence": round(confidence, 6),
        "below_threshold": below_threshold,
        "model_version": model_version,
        "request_id": request_id,
    }


def emit_audit_record(record: dict[str, Any]) -> None:
    """Write the audit event as JSON to stdout via logging."""
    AUDIT_LOGGER.info(json.dumps(record, separators=(",", ":"), sort_keys=True))
