"""Core classifier wrapper around Hugging Face Transformers artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class ClassificationResult:
    """One ticket classification result."""

    label: str
    confidence: float
    scores: dict[str, float]


class TicketClassifier:
    """Load and run a Hugging Face sequence-classification model."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.eval()

    def predict(self, text: str) -> ClassificationResult:
        """Classify one non-empty ticket string."""
        if not text.strip():
            raise ValueError("text must not be empty")

        encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=160)
        with torch.no_grad():
            logits = self.model(**encoded).logits[0]
            probabilities = torch.softmax(logits, dim=-1)

        id_to_label = self.model.config.id2label
        scores = {id_to_label[index]: float(score) for index, score in enumerate(probabilities)}
        label = max(scores, key=scores.get)
        return ClassificationResult(label=label, confidence=scores[label], scores=scores)
