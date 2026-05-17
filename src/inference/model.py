"""Model loading and batched text classification."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ticket_classifier.labels import LABELS

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Prediction:
    """One normalized classification prediction."""

    label: str
    confidence: float
    scores: dict[str, float]
    below_threshold: bool


class TicketClassifier:
    """Load a Hugging Face model once and serve batched predictions.

    If no local model artifacts exist, a deterministic keyword fallback keeps
    local smoke tests and health probes operational without downloading models.
    """

    def __init__(self, model_dir: Path, model_version: str, confidence_threshold: float) -> None:
        self.model_dir = model_dir
        self.model_version = model_version
        self.confidence_threshold = confidence_threshold
        self.labels = self._load_labels(model_dir)
        self.tokenizer: Any | None = None
        self.model: Any | None = None
        self.device: str = "cpu"
        self.loaded = False

    def load(self) -> None:
        """Load model artifacts once at application startup."""
        if self.loaded:
            return
        try:
            if (self.model_dir / "config.json").exists():
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
                self.model.eval()
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model.to(self.device)
                LOGGER.info("model_loaded", extra={"extra": {"model_dir": str(self.model_dir), "device": self.device}})
            else:
                LOGGER.warning("model_artifacts_missing_using_keyword_fallback", extra={"extra": {"model_dir": str(self.model_dir)}})
        finally:
            self.loaded = True

    def classify_batch(self, texts: list[str]) -> list[Prediction]:
        """Classify texts using one batched forward pass when a model exists."""
        if self.model is None or self.tokenizer is None:
            return [self._keyword_prediction(text) for text in texts]
        import torch

        encoded = self.tokenizer(texts, truncation=True, padding=True, max_length=160, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = self.model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        predictions: list[Prediction] = []
        for row in probabilities:
            scores = {label: float(row[index]) for index, label in enumerate(self.labels)}
            predictions.append(self._finalize(scores))
        return predictions

    def _finalize(self, scores: dict[str, float]) -> Prediction:
        label = max(scores, key=scores.get)
        confidence = scores[label]
        below_threshold = confidence < self.confidence_threshold
        return Prediction(
            label="uncertain" if below_threshold else label,
            confidence=confidence,
            scores=scores,
            below_threshold=below_threshold,
        )

    def _keyword_prediction(self, text: str) -> Prediction:
        lower = text.lower()
        keyword_map = {
            "login_issue": ("login", "password", "locked", "sign in", "account access"),
            "policy_question": ("policy", "rule", "guidance", "allowed", "procedure"),
            "incident_report": ("incident", "accident", "hazard", "spill", "injury"),
            "training_request": ("training", "course", "workshop", "learning", "induction"),
            "data_privacy": ("privacy", "data breach", "personal information", "consent", "records request"),
            "document_access": ("document", "file", "folder", "sharepoint", "record"),
            "procurement_request": ("purchase", "procure", "license", "vendor", "hardware"),
            "compliance_audit": ("audit", "compliance", "fisma", "essential 8", "evidence"),
            "identity_management": ("mfa", "service account", "role", "offboard", "provision"),
            "infrastructure_fault": ("network", "vpn", "server", "outage", "cloud"),
            "grievance_hr": ("grievance", "complaint", "misconduct", "harassment", "workplace"),
            "change_management": ("change", "cab", "release", "rollback", "deployment"),
        }
        raw_scores = {label: 0.01 for label in self.labels}
        for label, keywords in keyword_map.items():
            raw_scores[label] = 0.2 + sum(0.18 for keyword in keywords if keyword in lower)
        total = sum(raw_scores.values())
        scores = {label: value / total for label, value in raw_scores.items()}
        return self._finalize(scores)

    def _load_labels(self, model_dir: Path) -> list[str]:
        label_map = model_dir / "label_map.json"
        if not label_map.exists():
            return list(LABELS)
        payload = json.loads(label_map.read_text(encoding="utf-8"))
        labels = payload.get("labels")
        return labels if isinstance(labels, list) and all(isinstance(item, str) for item in labels) else list(LABELS)
