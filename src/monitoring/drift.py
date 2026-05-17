"""Simple label-distribution drift monitoring."""

from __future__ import annotations

import logging
import math
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque

from ticket_classifier.config import DEFAULT_DRIFT_THRESHOLD, DEFAULT_DRIFT_WINDOW
from ticket_classifier.labels import LABELS

LOGGER = logging.getLogger(__name__)


def jensen_shannon_divergence(p: list[float], q: list[float]) -> float:
    """Return JS divergence using natural logarithms."""
    midpoint = [(left + right) / 2.0 for left, right in zip(p, q, strict=True)]
    return 0.5 * _kl_divergence(p, midpoint) + 0.5 * _kl_divergence(q, midpoint)


def _kl_divergence(p: list[float], q: list[float]) -> float:
    return sum(left * math.log(left / right) for left, right in zip(p, q, strict=True) if left > 0 and right > 0)


def normalize(counts: dict[str, int] | Counter[str]) -> list[float]:
    """Normalize counts over known labels plus uncertain."""
    ordered_labels = [*LABELS, "uncertain"]
    total = sum(counts.get(label, 0) for label in ordered_labels)
    if total == 0:
        return [1.0 / len(ordered_labels)] * len(ordered_labels)
    return [counts.get(label, 0) / total for label in ordered_labels]


@dataclass
class DriftDetector:
    """Track recent predictions and compare them to a baseline distribution."""

    baseline_counts: dict[str, int] = field(default_factory=lambda: {label: 1 for label in LABELS})
    window_size: int = DEFAULT_DRIFT_WINDOW
    threshold: float = DEFAULT_DRIFT_THRESHOLD
    predictions: Deque[str] = field(init=False)

    def __post_init__(self) -> None:
        self.predictions = deque(maxlen=self.window_size)

    def observe(self, label: str) -> None:
        """Record a predicted label for later drift checks."""
        self.predictions.append(label)

    def snapshot(self) -> dict[str, object]:
        """Return current JS score and per-label shift."""
        recent_counts = Counter(self.predictions)
        baseline = normalize(self.baseline_counts)
        recent = normalize(recent_counts)
        js_score = jensen_shannon_divergence(baseline, recent)
        labels = [*LABELS, "uncertain"]
        shift = {label: recent[index] - baseline[index] for index, label in enumerate(labels)}
        if js_score > self.threshold:
            LOGGER.warning(
                "label_distribution_drift",
                extra={"extra": {"event": "drift_warning", "js_divergence": js_score}},
            )
        return {"js_divergence": js_score, "per_label_shift": shift, "window_size": len(self.predictions)}
